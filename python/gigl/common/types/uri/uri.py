from __future__ import annotations

import os
from pathlib import Path
from typing import Any, List, Union


class Uri(object):
    """
    A URI; currently supports GCS ('gs://foo/bar'), HTTP ('http://abc.com/xyz') or local ('/foo/bar').
    """

    @property
    def uri(self):
        return self.__uri

    def __init__(self, uri: Union[str, Path, Uri]):
        self.__uri = self._token_to_string(uri)

    @staticmethod
    def _token_to_string(token: Union[str, Path, Uri]) -> str:
        if isinstance(token, str):
            return token
        elif isinstance(token, Uri):
            return token.uri
        elif isinstance(token, Path):
            return str(token)
        return ""

    @classmethod
    def join(cls, token: Union[str, Path, Uri], *tokens: Union[str, Path, Uri]) -> Uri:
        """
        Join multiple tokens to create a new Uri instance.

        Args:
            token: The first token to join.
            tokens: Additional tokens to join.

        Returns:
            A new Uri instance representing the joined URI.

        """
        token = cls._token_to_string(token)
        token_strs: List[str] = [cls._token_to_string(token) for token in tokens]
        joined_tmp_path = os.path.join(token, *token_strs)
        joined_path = Uri(joined_tmp_path)
        return joined_path

    @classmethod
    def is_valid(
        cls, uri: Union[str, Path, Uri], raise_exception: bool = False
    ) -> bool:
        """
        Check if the given URI is valid.

        Args:
            uri: The URI to check.
            raise_exception: Whether to raise an exception if the URI is invalid.

        Returns:
            bool: True if the URI is valid, False otherwise.
        """
        raise NotImplementedError(
            f"Subclasses of {cls.__name__} are responsible"
            f" for implementing custom is_valid logic."
        )

    def get_basename(self) -> str:
        """
        The base name is the final component of the path, effectively extracting the file or directory name from a full path string.
        i.e. get_basename("/foo/bar.txt") -> bar.txt
        get_basename("gs://bucket/foo") -> foo
        """
        return self.uri.split("/")[-1]

    def __repr__(self) -> str:
        return self.uri

    def __hash__(self) -> int:
        return hash(self.uri)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Uri):
            return self.uri == other.uri
        return False
