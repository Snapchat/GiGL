from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from gigl.common.types.uri.uri import Uri


class GcsUri(Uri):
    """
    Represents a Google Cloud Storage (GCS) URI.

    Args:
        uri (Union[str, Path, GcsUri]): The URI string or path to initialize the GcsUri object.
    """

    def __init__(self, uri: Union[str, Path, GcsUri]) -> None:
        self.is_valid(uri=self._token_to_string(uri), raise_exception=True)
        super().__init__(uri=uri)

    @property
    def bucket(self) -> str:
        return self.uri.split("/")[2]

    @property
    def path(self) -> str:
        return "/".join(self.uri.split("/")[3:])

    @classmethod
    def _has_valid_prefix(cls, uri: Union[str, Path, Uri]) -> bool:
        return cls._token_to_string(uri).startswith("gs://")

    @classmethod
    def _has_no_backslash(cls, uri: Union[str, Path, Uri]) -> bool:
        return cls._token_to_string(uri).count("\\") == 0

    @classmethod
    def is_valid(
        cls, uri: Union[str, Path, Uri], raise_exception: Optional[bool] = False
    ) -> bool:
        """
        Check if the given URI is valid.

        Args:
            uri (Union[str, Path, Uri]): The URI to be validated.
            raise_exception (Optiona[bool]): Whether to raise an exception if the URI is invalid.
                Defaults to False.

        Returns:
            bool: True if the URI is valid, False otherwise.
        """
        has_valid_prefix: bool = cls._has_valid_prefix(uri=uri)
        has_no_backslash: bool = cls._has_no_backslash(uri=uri)
        if raise_exception and not has_valid_prefix:
            raise TypeError(f"{cls.__name__} must start with gs://; got {uri}")
        if raise_exception and not has_no_backslash:
            raise TypeError(f"{cls.__name__} does not support backslashes; got {uri}")
        return True if (has_valid_prefix and has_no_backslash) else False

    @classmethod
    def join(
        cls, token: Union[str, Path, Uri], *tokens: Union[str, Path, Uri]
    ) -> GcsUri:
        """
        Joins multiple URI tokens together and returns a new GcsUri object.

        Args:
            token (Union[str, Path, Uri]): The first URI token to join.
            *tokens (Union[str, Path, Uri]): Additional URI tokens to join.

        Returns:
            GcsUri: A new GcsUri object representing the joined URI.
        """
        joined_uri = super().join(token, *tokens)
        uri = cls(uri=joined_uri.uri)
        return uri

    def __repr__(self) -> str:
        return self.uri
