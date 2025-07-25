import shutil
import tempfile
from collections.abc import Mapping
from tempfile import _TemporaryFileWrapper as TemporaryFileWrapper  # type: ignore
from typing import IO, AnyStr, Optional, Sequence, Tuple, Type, Union, cast

from gigl.common import GcsUri, HttpUri, LocalUri, Uri, UriFactory
from gigl.common.logger import Logger
from gigl.common.utils.gcs import GcsUtils
from gigl.common.utils.http import HttpUtils
from gigl.common.utils.local_fs import (
    FileSystemEntity,
    copy_files,
    count_files_with_uri_prefix,
    create_file_symlinks,
    does_path_exist,
    list_at_path,
    remove_file_or_folder_if_exist,
)

logger = Logger()


class FileLoader:
    def __init__(self, project: Optional[str] = None):
        gcs_utils = GcsUtils(project)
        self.__gcs_utils = gcs_utils
        self.__unsupported_uri_message = (
            f"{self.__class__.__name__} does not support Uris of this type."
        )

    @staticmethod
    def __get_uri_map_schema(
        uri_map: Mapping[Uri, Uri]
    ) -> Tuple[Optional[Type[Uri]], Optional[Type[Uri]]]:
        uniform_src_type: Optional[Type[Uri]] = None
        uniform_dst_type: Optional[Type[Uri]] = None
        src_types: list[Type[Uri]] = [uri.__class__ for uri in uri_map.keys()]
        dst_types: list[Type[Uri]] = [uri.__class__ for uri in uri_map.values()]
        if all([src_types[0] == x for x in src_types]):
            uniform_src_type = src_types[0]
        if all([dst_types[0] == x for x in dst_types]):
            uniform_dst_type = dst_types[0]
        return uniform_src_type, uniform_dst_type

    def load_directories(
        self,
        source_to_dest_directory_map: dict[Uri, Uri],
    ):
        for dir_uri_src, dir_uri_dst in source_to_dest_directory_map.items():
            self.load_directory(dir_uri_src=dir_uri_src, dir_uri_dst=dir_uri_dst)

    def load_directory(self, dir_uri_src: Uri, dir_uri_dst: Uri):
        uri_map_schema = self.__get_uri_map_schema(uri_map={dir_uri_src: dir_uri_dst})

        if uri_map_schema == (GcsUri, LocalUri):
            dir_uri_src = cast(GcsUri, dir_uri_src)
            dir_uri_dst = cast(LocalUri, dir_uri_dst)
            self.__gcs_utils.download_files_from_gcs_paths_to_local_dir(
                gcs_paths=[dir_uri_src], local_path_dir=dir_uri_dst
            )
        elif uri_map_schema == (LocalUri, GcsUri):
            dir_uri_src = cast(LocalUri, dir_uri_src)
            dir_uri_dst = cast(GcsUri, dir_uri_dst)
            local_paths: list[LocalUri] = list_at_path(
                local_path=dir_uri_src, file_system_entity=FileSystemEntity.FILE
            )
            gcs_paths: list[GcsUri] = [
                GcsUri.join(dir_uri_dst, local_fn.uri)
                for local_fn in list_at_path(dir_uri_src, names_only=True)
            ]
            local_file_path_to_gcs_path_map: dict[LocalUri, GcsUri] = {
                src: dst for src, dst in zip(local_paths, gcs_paths)
            }
            self.load_files(
                source_to_dest_file_uri_map=cast(
                    dict[Uri, Uri], local_file_path_to_gcs_path_map
                )
            )
        elif uri_map_schema == (LocalUri, LocalUri):
            dir_uri_src = cast(LocalUri, dir_uri_src)
            dir_uri_dst = cast(LocalUri, dir_uri_dst)

            local_src_paths: list[LocalUri] = list_at_path(
                local_path=dir_uri_src, file_system_entity=FileSystemEntity.FILE
            )
            local_dst_paths: list[LocalUri] = [
                LocalUri.join(dir_uri_dst, local_src_fn)
                for local_src_fn in list_at_path(
                    local_path=dir_uri_src,
                    names_only=True,
                    file_system_entity=FileSystemEntity.FILE,
                )
            ]
            source_to_dest_file_uri_map = {
                src: dst for src, dst in zip(local_src_paths, local_dst_paths)
            }
            self.load_files(
                source_to_dest_file_uri_map=cast(
                    dict[Uri, Uri], source_to_dest_file_uri_map
                )
            )
        else:
            raise TypeError(self.__unsupported_uri_message)

    def load_files(
        self,
        source_to_dest_file_uri_map: Mapping[Uri, Uri],
        should_create_symlinks_if_possible: bool = True,
    ) -> None:
        uri_map_schema = self.__get_uri_map_schema(uri_map=source_to_dest_file_uri_map)

        if uri_map_schema == (GcsUri, LocalUri):
            logger.info("Downloading from GCS to Local")
            self.__gcs_utils.download_files_from_gcs_paths_to_local_paths(
                file_map=cast(dict[GcsUri, LocalUri], source_to_dest_file_uri_map)
            )
        elif uri_map_schema == (LocalUri, GcsUri):
            logger.info("Uploading from Local to GCS")
            self.__gcs_utils.upload_files_to_gcs(
                local_file_path_to_gcs_path_map=cast(
                    dict[LocalUri, GcsUri], source_to_dest_file_uri_map
                ),
                parallel=True,
            )
        elif uri_map_schema == (LocalUri, LocalUri):
            logger.info("Copying from Local to Local")
            local_source_to_link_path_map = source_to_dest_file_uri_map
            if should_create_symlinks_if_possible:
                logger.info("Will create symlinks")
                create_file_symlinks(
                    local_source_to_link_path_map=cast(
                        dict[LocalUri, LocalUri], local_source_to_link_path_map
                    ),
                    should_overwrite=True,
                )
            else:
                logger.info("Will copy files")
                copy_files(
                    local_source_to_local_dst_path_map=cast(
                        dict[LocalUri, LocalUri], local_source_to_link_path_map
                    ),
                    should_overwrite=True,
                )
        elif uri_map_schema == (HttpUri, LocalUri):
            logger.info("Downloading from HTTP to Local")
            HttpUtils.download_files_from_http(
                http_to_local_path_map=cast(
                    dict[HttpUri, LocalUri], source_to_dest_file_uri_map
                ),
            )
        else:
            for file_uri_src, file_uri_dst in source_to_dest_file_uri_map.items():
                self.load_file(
                    file_uri_src=file_uri_src,
                    file_uri_dst=file_uri_dst,
                    should_create_symlinks_if_possible=should_create_symlinks_if_possible,
                )

    def load_file(
        self,
        file_uri_src: Uri,
        file_uri_dst: Uri,
        should_create_symlinks_if_possible: bool = True,
    ) -> None:
        uri_map_schema = self.__get_uri_map_schema(uri_map={file_uri_src: file_uri_dst})
        uri_map = {file_uri_src: file_uri_dst}

        if uri_map_schema == (GcsUri, LocalUri):
            self.__gcs_utils.download_file_from_gcs(
                gcs_path=cast(GcsUri, file_uri_src),
                dest_file_path=cast(LocalUri, file_uri_dst),
            )
        elif uri_map_schema == (LocalUri, GcsUri):
            self.__gcs_utils.upload_files_to_gcs(
                local_file_path_to_gcs_path_map=cast(dict[LocalUri, GcsUri], uri_map),
                parallel=False,
            )
        elif uri_map_schema == (LocalUri, LocalUri):
            local_source_to_link_path_map = {file_uri_src: file_uri_dst}
            if should_create_symlinks_if_possible:
                create_file_symlinks(
                    local_source_to_link_path_map=cast(
                        dict[LocalUri, LocalUri], local_source_to_link_path_map
                    ),
                    should_overwrite=True,
                )
            else:
                copy_files(
                    local_source_to_local_dst_path_map=cast(
                        dict[LocalUri, LocalUri], local_source_to_link_path_map
                    ),
                    should_overwrite=True,
                )
        elif uri_map_schema == (HttpUri, LocalUri):
            HttpUtils.download_files_from_http(
                http_to_local_path_map=cast(dict[HttpUri, LocalUri], uri_map),
            )
        else:
            logger.warning(f"Unsupported uri_map_schema: {uri_map_schema}")
            raise TypeError(self.__unsupported_uri_message)

    def load_from_filelike(self, uri: Uri, filelike: IO[AnyStr]) -> None:
        """
        Load a file from a file-like object into the specified URI.

        Args:
            uri (Uri): The URI to load the file into.
            filelike (IO[AnyStr]): A file-like object containing the data to be loaded.
        """
        if isinstance(uri, GcsUri):
            self.__gcs_utils.upload_from_filelike(gcs_path=uri, filelike=filelike)
        elif isinstance(uri, LocalUri):
            ptr = filelike.tell()
            first = filelike.read(1)
            filelike.seek(ptr)  # Reset the file pointer after reading
            if isinstance(first, bytes):
                with open(uri.uri, "wb") as dest:
                    shutil.copyfileobj(filelike, dest)
            else:
                with open(uri.uri, "w", encoding="utf-8") as dest:
                    shutil.copyfileobj(filelike, dest)

        else:
            raise NotImplementedError(
                f"Cannot load file from buffer to URI {uri.uri} of type {type(uri)}; {self.__unsupported_uri_message}"
            )

    def load_to_temp_file(
        self,
        file_uri_src: Uri,
        delete: bool = False,
        should_create_symlinks_if_possible: bool = True,
    ) -> TemporaryFileWrapper:
        temp_file_handle = tempfile.NamedTemporaryFile(delete=delete)
        temp_file_path = LocalUri(str(temp_file_handle.name))
        self.load_file(
            file_uri_src=file_uri_src,
            file_uri_dst=temp_file_path,
            should_create_symlinks_if_possible=should_create_symlinks_if_possible,
        )
        return temp_file_handle

    def count_assets(self, uri_prefix: Uri, suffix: Optional[str] = None) -> int:
        if isinstance(uri_prefix, GcsUri):
            return self.__gcs_utils.count_blobs_in_gcs_path(
                gcs_path=uri_prefix, suffix=suffix
            )
        elif isinstance(uri_prefix, LocalUri):
            return count_files_with_uri_prefix(uri_prefix=uri_prefix, suffix=suffix)
        else:
            raise TypeError(
                f"Uri type not supported, got {uri_prefix} in type {type(uri_prefix)}"
            )

    def does_uri_exist(self, uri: Union[str, Uri]) -> bool:
        """""
        Check if a URI exists

        Args:
            uri (Union[str, Uri]): uri to check
        Returns:
            bool: True if URI exists, False otherwise
        """ ""

        _uri = UriFactory.create_uri(uri=uri) if isinstance(uri, str) else uri

        exists: bool
        if type(_uri) == GcsUri:
            exists = self.__gcs_utils.does_gcs_file_exist(gcs_path=_uri)
        elif type(_uri) == LocalUri:
            exists = does_path_exist(cast(LocalUri, _uri))
        elif type(_uri) == HttpUri:
            exists = HttpUtils.does_http_path_resolve(http_path=cast(HttpUri, _uri))
        else:
            raise NotImplementedError(f"{self.__unsupported_uri_message} : {_uri}")
        return exists

    def delete_files(self, uris: list[Uri]) -> None:
        """
        Recursively delete files in the specified URIs.

        Args:
            uris (list[Uri]): URIs to delete
        Returns
            None
        """
        for uri in uris:
            if isinstance(uri, LocalUri):
                remove_file_or_folder_if_exist(local_path=uri)
            elif isinstance(uri, GcsUri):
                self.__gcs_utils.delete_files_in_bucket_dir(gcs_path=uri)
            else:
                raise NotImplementedError(
                    f"Cannot delete URI {uri.uri} of type {type(uri)}; {self.__unsupported_uri_message}"
                )

    def list_children(self, uri: Uri, pattern: Optional[str] = None) -> Sequence[Uri]:
        """
        List all children of the given URI.
        Args:
            uri (Uri): The URI to list children of.
            pattern (Optional[str]): Optional regex to match. If not provided then all children will be returned.
        Returns:
            list[Uri]: A list of URIs for the children of the given URI.
        """
        if isinstance(uri, GcsUri):
            return self.__gcs_utils.list_uris_with_gcs_path_pattern(
                gcs_path=uri, pattern=pattern
            )
        elif isinstance(uri, LocalUri):
            return list_at_path(local_path=uri, regex=pattern)
        else:
            raise NotImplementedError(
                f"Cannot list children of URI {uri.uri} of type {type(uri)}; {self.__unsupported_uri_message}"
            )
