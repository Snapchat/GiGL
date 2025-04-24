import pathlib
import re
import tempfile
import typing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from tempfile import _TemporaryFileWrapper as TemporaryFileWrapper  # type: ignore
from typing import IO, AnyStr, Dict, Iterable, List, Optional, Tuple, Union

import google.cloud.exceptions as google_exceptions
import google.cloud.storage as storage

from gigl.common import GcsUri, LocalUri
from gigl.common.collections.itertools import batch
from gigl.common.logger import Logger
from gigl.common.utils.local_fs import remove_file_if_exist
from gigl.common.utils.retry import retry

logger = Logger()

UPLOAD_RETRY_DEADLINE_S = 60 * 60 * 2  # limit of 2 hours maximum to upload something

# No more than 100 calls should be included in a single batch request.
# The total batch request payload must be less than 10MB
# More reading: https://cloud.google.com/storage/docs/batch#overview
_BLOB_BATCH_SIZE = 80


@retry(deadline_s=UPLOAD_RETRY_DEADLINE_S)
def _upload_file_to_gcs(
    source_file_path: LocalUri,
    dest_gcs_path: GcsUri,
    project: str,
    gcs_utils_client: Optional[storage.Client] = None,
):
    (
        bucket_name,
        destination_blob_name,
    ) = GcsUtils.get_bucket_and_blob_path_from_gcs_path(dest_gcs_path)
    local_storage_client = gcs_utils_client
    if local_storage_client is None:
        local_storage_client = storage.Client(project=project)
    bucket = local_storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_path.uri)


def _pickling_safe_upload_file_to_gcs(obj: Tuple[Tuple[LocalUri, GcsUri], str]):
    file_paths, project = obj
    source_file_path, dest_gcs_path = file_paths
    storage_client = storage.Client(project=project)
    _upload_file_to_gcs(
        source_file_path=source_file_path,
        dest_gcs_path=dest_gcs_path,
        project=project,
        gcs_utils_client=storage_client,
    )


def _upload_files_to_gcs_parallel(
    project: str, local_file_path_to_gcs_path_map: Dict[LocalUri, GcsUri]
):
    with ProcessPoolExecutor(max_workers=None) as executor:
        results = executor.map(
            _pickling_safe_upload_file_to_gcs,
            zip(
                local_file_path_to_gcs_path_map.items(),
                [project] * len(local_file_path_to_gcs_path_map),
            ),
        )
        list(results)  # wait for all uploads to finish


class GcsUtils:
    """Utility class for interacting with Google Cloud Storage (GCS)."""

    def __init__(self, project: Optional[str] = None) -> None:
        """
        Initialize the GcsUtils instance.

        Args:
            project (Optional[str]): The GCP project ID. Defaults to None.
        """
        self.__storage_client = storage.Client(project=project)

    def upload_from_string(self, gcs_path: GcsUri, content: str) -> None:
        bucket_name, blob_name = self.get_bucket_and_blob_path_from_gcs_path(gcs_path)
        bucket = self.__storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_string(content)

    def upload_from_filelike(
        self,
        gcs_path: GcsUri,
        filelike: IO[AnyStr],
        content_type: str = "application/octet-stream",
    ) -> None:
        """Uploads a file-like object to GCS.

        A "filelike" object is one that satisfies the typing.IO interface, e.g contains read(), write(), etc.
        The prototypical example of this is the object returned by open(),
        but we also use io.BytesIO as an in-memory buffer which also satisfies the typing.IO interface.

        Args:
            gcs_path (GcsUri): The GCS path to upload the file to.
            filelike (IO[AnyStr]): The file-like object to upload.
            content_type (str): The content type of the file. Defaults to "application/octet-stream".
        """
        bucket_name, blob_name = self.get_bucket_and_blob_path_from_gcs_path(gcs_path)
        bucket = self.__storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_file(filelike, content_type=content_type)

    def read_from_gcs(self, gcs_path: GcsUri) -> str:
        bucket_name, blob_name = self.get_bucket_and_blob_path_from_gcs_path(gcs_path)
        bucket = self.__storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        return blob.download_as_text()

    def upload_files_to_gcs(
        self,
        local_file_path_to_gcs_path_map: Dict[LocalUri, GcsUri],
        parallel: bool = True,
    ) -> None:
        """
        Upload files from local paths to their subsequent provided GCS paths.

        Args:
            local_file_path_to_gcs_path_map (Dict[LocalUri, GcsUri]): A dictionary mapping local file paths to GCS paths.
            parallel (bool): Flag indicating whether to upload files in parallel. Defaults to True.
        """
        if parallel:
            _upload_files_to_gcs_parallel(
                project=self.__storage_client.project,
                local_file_path_to_gcs_path_map=local_file_path_to_gcs_path_map,
            )
        else:
            for (
                source_file_path,
                dest_gcs_path,
            ) in local_file_path_to_gcs_path_map.items():
                _upload_file_to_gcs(
                    source_file_path=source_file_path,
                    dest_gcs_path=dest_gcs_path,
                    project=self.__storage_client.project,
                    gcs_utils_client=self.__storage_client,
                )

    def download_file_from_gcs_to_temp_file(
        self, gcs_path: GcsUri
    ) -> TemporaryFileWrapper:
        f = tempfile.NamedTemporaryFile()
        dest_file_path = LocalUri(str(f.name))
        self.download_file_from_gcs(gcs_path=gcs_path, dest_file_path=dest_file_path)
        return f

    def download_file_from_gcs(
        self, gcs_path: GcsUri, dest_file_path: LocalUri
    ) -> None:
        bucket_name, blob_path = GcsUtils.get_bucket_and_blob_path_from_gcs_path(
            gcs_path
        )
        bucket = self.__storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        self.__download_blob_from_gcs(blob, dest_file_path)

    @staticmethod
    def __download_blob_from_gcs(blob: storage.Blob, dest_file_path: LocalUri):
        dest_file_path_str: str = dest_file_path.uri
        pathlib.Path(dest_file_path_str).parent.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"Downloading gs://{blob.bucket.name}/{blob.name} to {dest_file_path_str}"
        )
        blob.download_to_filename(dest_file_path_str)

    def list_uris_with_gcs_path_pattern(
        self,
        gcs_path: GcsUri,
        suffix: Optional[str] = None,
        pattern: Optional[str] = None,
    ) -> List[GcsUri]:
        """
        List GCS URIs with a given suffix or pattern.

        Ex:
        gs://bucket-name/dir/file1.txt
        gs://bucket-name/dir/foo.txt
        gs://bucket-name/dir/file.json

        list_uris_with_gcs_path_pattern(gcs_path=gs://bucket-name/dir, suffix=".txt") -> [gs://bucket-name/dir/file1.txt, gs://bucket-name/dir/foo.txt]
        list_uris_with_gcs_path_pattern(gcs_path=gs://bucket-name/dir, pattern="file.*") -> [gs://bucket-name/dir/file1.txt, gs://bucket-name/dir/file.json]

        Args:
            gcs_path (GcsUri): The GCS path to list URIs from.
            suffix (Optional[str]): The suffix to filter URIs by. If None (the default), then no filtering on suffix will be done.
            pattern (Optional[str]): The regex to filter URIs by. If None (the default), then no filtering on the pattern will be done.

        Returns:
            List[GcsUri]: A list of GCS URIs that match the given suffix or pattern.
        """
        if suffix and pattern:
            logger.warning(
                f"Attempting to filter uris with both suffix ({suffix}) and pattern ({pattern}). This is odd, are you usre you want to do so?"
            )
        blobs = self.__list_file_blobs_at_gcs_path(gcs_path=gcs_path)
        if suffix:
            blobs = [blob for blob in blobs if blob.name.endswith(suffix)]
        if pattern:
            matcher = re.compile(pattern)
            blobs = [blob for blob in blobs if matcher.match(blob.name)]
        gcs_uris = [GcsUri.join("gs://", blob.bucket.name, blob.name) for blob in blobs]
        return gcs_uris

    def __list_file_blobs_at_gcs_path(self, gcs_path: GcsUri) -> List[storage.Blob]:
        bucket_name, prefix = self.get_bucket_and_blob_path_from_gcs_path(gcs_path)
        blobs = self.__storage_client.list_blobs(
            bucket_or_name=bucket_name, prefix=prefix
        )  # Get list of blobs
        file_blobs = [
            blob for blob in blobs if not blob.name.endswith("/")
        ]  # Filter out directories
        return file_blobs

    def download_files_from_gcs_paths_to_local_paths(
        self, file_map: Dict[GcsUri, LocalUri]
    ):
        """
        Downloads files from GCS path to local path.
        :param file_map: mapping of GCS path -> local path
        :return:
        """
        blobs_and_paths = []

        for gcs_path, local_path in file_map.items():
            file_blobs = self.__list_file_blobs_at_gcs_path(gcs_path)
            if len(file_blobs):
                blob = file_blobs[0]
                blobs_and_paths.append((blob, local_path))
            else:
                logger.info(f"Could not find and download {gcs_path}.")

        with ThreadPoolExecutor(max_workers=None) as executor:
            executor.map(
                lambda params: self.__download_blob_from_gcs(*params), blobs_and_paths
            )

    def download_files_from_gcs_paths_to_local_dir(
        self, gcs_paths: List[GcsUri], local_path_dir: LocalUri
    ) -> None:
        for gcs_path in gcs_paths:
            file_blobs = self.__list_file_blobs_at_gcs_path(gcs_path)

            file_blob_fully_qualified_paths = [
                f"gs://{blob.bucket.name}/{blob.name}" for blob in file_blobs
            ]
            local_dest_paths = [
                LocalUri(uri=qualified_path.replace(gcs_path.uri, local_path_dir.uri))
                for qualified_path in file_blob_fully_qualified_paths
            ]
            logger.info(f"Downloading blobs: {file_blobs}")

            with ThreadPoolExecutor(max_workers=None) as executor:
                executor.map(
                    lambda params: self.__download_blob_from_gcs(*params),
                    zip(file_blobs, local_dest_paths),
                )

    @staticmethod
    def get_bucket_and_blob_path_from_gcs_path(
        gcs_path: GcsUri,
    ) -> Tuple[str, str]:
        gcs_path_str: str = gcs_path.uri
        gcs_parts: List[str] = gcs_path_str.split(
            "/"
        )  # "gs://bucket-name/file/path" -> ['gs:', '', 'bucket-name', 'file', 'path']
        bucket_name, blob_name = gcs_parts[2], "/".join(gcs_parts[3:])
        return bucket_name, blob_name

    def does_gcs_file_exist(self, gcs_path: GcsUri) -> bool:
        bucket_name, blob_name = self.get_bucket_and_blob_path_from_gcs_path(gcs_path)
        blob = self.__storage_client.get_bucket(bucket_name).blob(blob_name)
        return blob.exists()

    def delete_gcs_file_if_exist(self, gcs_path: GcsUri) -> None:
        bucket_name, blob_name = self.get_bucket_and_blob_path_from_gcs_path(gcs_path)
        blob = self.__storage_client.get_bucket(bucket_name).blob(blob_name)
        if blob.exists():
            blob.delete()
            logger.info(f"Deleted GCS file '{gcs_path}'")
        else:
            logger.info(
                f"Attempted to delete GCS file but file does not exist '{gcs_path}'"
            )

    def count_blobs_in_gcs_path(
        self, gcs_path: GcsUri, suffix: Optional[str] = None
    ) -> int:
        bucket_name, blob_name = self.get_bucket_and_blob_path_from_gcs_path(gcs_path)
        matching_blobs = list(
            self.__storage_client.get_bucket(bucket_name).list_blobs(prefix=blob_name)
        )
        if suffix:
            matching_blobs = [
                blob for blob in matching_blobs if blob.name.endswith(suffix)
            ]
        return len(matching_blobs)

    @staticmethod
    def __delete_gcs_blob(blob: storage.Blob) -> None:
        try:
            blob.delete()
            logger.info(f"Deleted file '{blob.name}'")
        except google_exceptions.NotFound:
            logger.info(f"Could not delete {blob.name}; not found")
        except Exception as e:
            logger.exception(f"Could not delete {blob.name}; {repr(e)}")

    def delete_files_in_bucket_dir(self, gcs_path: GcsUri) -> None:
        bucket_name, blob_name = self.get_bucket_and_blob_path_from_gcs_path(gcs_path)
        matching_blobs = list(
            self.__storage_client.get_bucket(bucket_name).list_blobs(prefix=blob_name)
        )
        logger.info(f"bucket {bucket_name}, prefix {blob_name}")
        self.delete_files(gcs_files=matching_blobs)
        logger.info(f"Files deleted in '{gcs_path}'")

    def delete_files(self, gcs_files: Iterable[Union[GcsUri, storage.Blob]]) -> None:
        matching_blobs: List[storage.Blob] = list()
        for gcs_file in gcs_files:
            if not isinstance(gcs_file, storage.Blob):
                bucket_name, blob_name = self.get_bucket_and_blob_path_from_gcs_path(
                    gcs_file
                )
                blob = self.__storage_client.get_bucket(bucket_name).blob(blob_name)
            else:
                blob = gcs_file
            matching_blobs.append(blob)

        batched_blobs_to_delete: List[List[storage.Blob]] = batch(
            list_of_items=matching_blobs, chunk_size=_BLOB_BATCH_SIZE
        )

        def __batch_delete_blobs(blobs: List[storage.Blob]):
            logger.info(f"Will delete ({len(blobs)}) gcs files")
            with self.__storage_client.batch():
                for blob in blobs:
                    blob.delete()

        with ThreadPoolExecutor(max_workers=None) as executor:
            executor.map(__batch_delete_blobs, batched_blobs_to_delete)

    def copy_gcs_path(self, src_gcs_path: GcsUri, dst_gcs_path: GcsUri):
        src_bucket_name, src_prefix = self.get_bucket_and_blob_path_from_gcs_path(
            gcs_path=src_gcs_path
        )
        src_bucket = self.__storage_client.bucket(bucket_name=src_bucket_name)
        dst_bucket_name, dst_prefix = self.get_bucket_and_blob_path_from_gcs_path(
            gcs_path=dst_gcs_path
        )
        dst_bucket = self.__storage_client.bucket(bucket_name=dst_bucket_name)

        blobs = list(
            self.__storage_client.list_blobs(
                bucket_or_name=src_bucket_name, prefix=src_prefix
            )
        )
        logger.info(
            f"Will copy {len(blobs)} files from {src_gcs_path} to {dst_gcs_path}"
        )

        def __batch_copy_blobs(
            src_bucket: storage.Bucket,
            dst_bucket: storage.Bucket,
            src_prefix: str,
            dst_prefix: str,
            src_blobs: List[storage.Blob],
        ):
            dst_blob_names: List[str] = [
                src_blob.name.replace(src_prefix, dst_prefix, 1)
                for src_blob in src_blobs
            ]
            with self.__storage_client.batch():
                logger.debug(
                    f"Will copy {len(src_blobs)} files from {src_bucket}://{src_prefix} to {dst_bucket}://{dst_prefix}."
                )
                for src_blob, dst_blob_name in zip(src_blobs, dst_blob_names):
                    src_bucket.copy_blob(
                        blob=src_blob,
                        destination_bucket=dst_bucket,
                        new_name=dst_blob_name,
                    )

        batched_blobs_to_copy = batch(list_of_items=blobs, chunk_size=_BLOB_BATCH_SIZE)
        with ThreadPoolExecutor(max_workers=None) as executor:
            futures = []
            for src_blobs_batch in batched_blobs_to_copy:
                futures.append(
                    executor.submit(
                        __batch_copy_blobs,
                        src_bucket,
                        dst_bucket,
                        src_prefix,
                        dst_prefix,
                        src_blobs_batch,
                    )
                )
            for future in futures:
                future.result()
        logger.info(f"Finished copying files from {src_gcs_path} to {dst_gcs_path}.")

    def __delete_irrelevant_lifecycle_rules(self, bucket: storage.Bucket):
        """
        Iterate over lifecycle rules and only keep relevant ones.
        """

        def should_keep_lifecycle_rule_delete(
            rule: storage.bucket.LifecycleRuleDelete,
        ) -> bool:
            # Only keep LifecycleRuleDelete if it has
            # (i) no prefix, or (ii) a condition matches an existing prefix.
            conditions = rule.get("condition")
            prefixes = conditions.get("matchesPrefix") if conditions else []
            if prefixes:
                does_any_prefix_currently_exist = False
                for prefix in prefixes:
                    blobs = bucket.list_blobs(prefix=prefix)
                    if any(blobs):
                        does_any_prefix_currently_exist = True
                        break
                if not does_any_prefix_currently_exist:
                    return False
            return True

        rules_to_keep = []
        bucket.reload()
        for rule in bucket.lifecycle_rules:
            # If encountering a LifecycleRuleDelete, check if it should be kept.
            if isinstance(
                rule, storage.bucket.LifecycleRuleDelete
            ) and not should_keep_lifecycle_rule_delete(rule=rule):
                logger.info(f"Will delete lifecycle rule: {rule}")
                continue
            rules_to_keep.append(rule)

        bucket.lifecycle_rules = rules_to_keep
        bucket.patch()
        logger.info(f"Preserved {len(rules_to_keep)} lifecycle rules.")

    def add_bucket_lifecycle_rule_with_prefix(
        self,
        gcs_path: GcsUri,
        days_to_expire: int,
        should_delete_irrelevant_lifecycle_rules=False,
    ) -> None:
        bucket_name, prefix = self.get_bucket_and_blob_path_from_gcs_path(
            gcs_path=gcs_path
        )
        bucket = self.__storage_client.bucket(bucket_name=bucket_name)
        bucket.reload()
        logger.info(
            f"Bucket {bucket_name} has {len(list(bucket.lifecycle_rules))} lifecycle rules."
        )
        if should_delete_irrelevant_lifecycle_rules:
            self.__delete_irrelevant_lifecycle_rules(bucket=bucket)
        bucket.add_lifecycle_delete_rule(age=days_to_expire, matches_prefix=[prefix])
        bucket.patch()
        logger.info(
            f"Set lifecycle rule to expire files at {gcs_path} after {days_to_expire} days."
        )

    # TODO: move this to somewhere more appropriate
    def close_upload_delete_and_push_to_gcs(
        self, local_file_handle: typing.TextIO, gcs_file_path: GcsUri
    ) -> None:
        local_file_path = LocalUri(local_file_handle.name)
        local_file_handle.close()
        self.upload_files_to_gcs(
            local_file_path_to_gcs_path_map={local_file_path: gcs_file_path},
            parallel=False,
        )
        remove_file_if_exist(local_file_path)
        logger.info(f"Moved {local_file_path} to {gcs_file_path}")
