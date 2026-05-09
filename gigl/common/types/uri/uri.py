from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Union

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

    @classmethod
    def _is_absolute(cls, token: _URI_LIKE) -> bool:
        token_str = cls._token_to_string(token)

        # Note: "://" is used to detect GcsUri and HttpUri prefixes, but will incorrectly
        # classify relative LocalUri path's containing "://" as absolute
        return Path(token_str).is_absolute() or "://" in token_str

    @classmethod
    def _has_matching_uri_type(
        cls, token: _URI_LIKE, allow_base_uri_join: bool = False
    ) -> bool:
        token_is_path_like = not isinstance(token, Uri)
        if token_is_path_like: # str and Path tokens do not have a Uri type to match.
            return True

        token_matches_join_type = token.__class__ is cls
        if token_matches_join_type: # Same concrete Uri type, e.g. GcsUri.join(GcsUri(...)).
            return True

        base_join_with_concrete_token = allow_base_uri_join and cls is Uri
        if base_join_with_concrete_token: # Uri.join(GcsUri(...), "suffix") stays supported for compatibility.
            return True

        # Remaining Uri tokens are cross-type mixes, e.g. GcsUri with HttpUri.
        return False

    @classmethod
    def join(cls, token: _URI_LIKE, *tokens: _URI_LIKE) -> Self:
        """
        Join multiple tokens to create a new Uri instance.
        The first token may be an absolute or relative path. Every additional token must be relative.

        Note:
            - Rejecting suffix strings containing "://" may break callers that
              intentionally store URI-looking strings inside paths.
            - Rejecting absolute LocalUri suffix tokens is stricter than os.path.join,
              which implicitly discards earlier tokens on the join path.
            - Concrete Uri joins cannot mix Uri token types; base Uri.join keeps
              the first token generic for compatibility.

        Args:
            token: The first token to join.
            tokens: Additional tokens to join.

        Returns:
            A new Uri instance representing the joined URI.

        """
        # Keep base Uri.join generic for existing callers that pass concrete Uri instances.
        if not cls._has_matching_uri_type(token, allow_base_uri_join=True):
            raise TypeError(
                f"Cannot join {cls.__name__} with {token.__class__.__name__}"
            )

        for suffix in tokens:
            if not cls._has_matching_uri_type(suffix):
                raise TypeError(
                    f"Cannot join {cls.__name__} with {suffix.__class__.__name__}"
                )

            if cls._is_absolute(suffix):
                raise TypeError(
                    f"URI join suffixes must be relative; got absolute path: {suffix}"
                )

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
        return self.join(self, other)


_URI_LIKE = Union[str, Path, Uri]
