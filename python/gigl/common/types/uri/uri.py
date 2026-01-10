from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from typing_extensions import Self


class Uri(object):
    """
    A URI; currently supports GCS ('gs://foo/bar'), HTTP ('http://abc.com/xyz') or local ('/foo/bar').
    """

    @property
    def uri(self):
        return self.__uri

    def __init__(self, uri: _URI_LIKE):
        self.__uri = self._token_to_string(uri)

    @staticmethod
    def _token_to_string(token: _URI_LIKE) -> str:
        if isinstance(token, str):
            return token
        elif isinstance(token, Uri):
            return token.uri
        elif isinstance(token, Path):
            return str(token)
        return ""

    # TODO(kmonte): You should not be able to join a Uri with a Uri of a different type.
    # *or* join HTTP on HTTP or GCS on GCS.
    # This is not backwards compatible, so come around to this later.
    @classmethod
    def join(cls, token: _URI_LIKE, *tokens: _URI_LIKE) -> Self:
        """
        Join multiple tokens to create a new Uri instance.

        Args:
            token: The first token to join.
            tokens: Additional tokens to join.

        Returns:
            A new Uri instance representing the joined URI.

        """
        token = cls._token_to_string(token)
        token_strs: list[str] = [cls._token_to_string(token) for token in tokens]
        joined_tmp_path = os.path.join(token, *token_strs)
        joined_path = cls(joined_tmp_path)
        return joined_path

    @classmethod
    def is_valid(cls, uri: _URI_LIKE, raise_exception: bool = False) -> bool:
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

    def __truediv__(self, other: _URI_LIKE) -> Self:
        if isinstance(other, Uri) and not isinstance(other, type(self)):
            raise TypeError(
                f"Cannot use '/' operator to join {type(self).__name__} with {type(other).__name__}"
            )
        return self.join(self, other)


_URI_LIKE = str | Path | Uri
