from __future__ import annotations

from pathlib import Path
from typing import List, Type, Union

from gigl.common.types.uri.gcs_uri import GcsUri
from gigl.common.types.uri.http_uri import HttpUri
from gigl.common.types.uri.local_uri import LocalUri
from gigl.common.types.uri.uri import Uri


class UriFactory:
    """
    Factory class to create the proper concrete Uri instance given an input.
    """

    @staticmethod
    def create_uri(uri: Union[str, Path, Uri]) -> Uri:
        """
        Create a Uri object based on the given URI string, path, or existing Uri object.

        Args:
            uri (Union[str, Path, Uri]): The URI string, path, or existing Uri object.

        Returns:
            Uri: A created Uri object based on the given input.
        """
        uri_types: List[Type[Uri]] = [GcsUri, HttpUri, LocalUri]
        for subcls in uri_types:
            if subcls.is_valid(uri=uri, raise_exception=False):
                return subcls(uri=uri)
        raise TypeError(
            f"{UriFactory.__name__} only supports {[cls.__name__ for cls in uri_types]} types."
        )
