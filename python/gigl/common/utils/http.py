import pathlib
from concurrent.futures import ThreadPoolExecutor

import requests

from gigl.common import HttpUri, LocalUri
from gigl.common.logger import Logger

logger = Logger()


class HttpUtils:
    @staticmethod
    def does_http_path_resolve(http_path: HttpUri) -> bool:
        """
        Checks if an HTTP(S) URL resolves to a valid file.
        """
        response = requests.head(http_path.uri)
        return response.status_code == 200

    @staticmethod
    def download_file_from_http(http_path: HttpUri, dest_file_path: LocalUri):
        """
        Downloads a file from an HTTP(S) URL to a local file path.

        Args:
            http_path (HttpUri): The HTTP(S) URL to download from.
            dest_file_path (LocalUri): The local file path to save the downloaded file.
        """
        logger.info(f"Downloading file from {http_path.uri} to {dest_file_path.uri}")
        response = requests.get(http_path.uri)
        response.raise_for_status()

        pathlib.Path(dest_file_path.uri).parent.mkdir(parents=True, exist_ok=True)
        with open(dest_file_path.uri, "wb") as f:
            f.write(response.content)

    @staticmethod
    def download_files_from_http(http_to_local_path_map: dict[HttpUri, LocalUri]):
        """
        Downloads a list of files from an HTTP(S) URL to a list of local file paths.

        Args:
            http_to_local_path_map (dict[HttpUri, LocalUri]): A dictionary mapping HTTP(S) URLs to local file paths.
        """
        with ThreadPoolExecutor() as executor:
            results = executor.map(
                HttpUtils.download_file_from_http,
                http_to_local_path_map.keys(),
                http_to_local_path_map.values(),
            )
            list(
                results
            )  # wait for all downloads to finish - also throws exceptions from threads, if any failed
