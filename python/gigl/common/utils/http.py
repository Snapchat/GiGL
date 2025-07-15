from concurrent.futures import ProcessPoolExecutor
from typing import Dict

import requests

from gigl.common import HttpUri, LocalUri


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
        response = requests.get(http_path.uri)
        response.raise_for_status()
        with open(dest_file_path.uri, "wb") as f:
            f.write(response.content)

    @staticmethod
    def download_files_from_http(http_to_local_path_map: Dict[HttpUri, LocalUri]):
        """
        Downloads a list of files from an HTTP(S) URL to a list of local file paths.

        Args:
            http_to_local_path_map (Dict[HttpUri, LocalUri]): A dictionary mapping HTTP(S) URLs to local file paths.
        """
        with ProcessPoolExecutor() as executor:
            executor.map(
                HttpUtils.download_file_from_http,
                http_to_local_path_map.keys(),
                http_to_local_path_map.values(),
            )
