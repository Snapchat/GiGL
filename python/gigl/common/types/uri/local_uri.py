from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

from gigl.common.types.uri.uri import Uri


class LocalUri(Uri, os.PathLike):
    """Represents a local URI (Uniform Resource Identifier) that extends the `Uri` class and implements the `os.PathLike` interface."""

    @classmethod
    def join(
        cls, token: Union[str, Path, Uri], *tokens: Union[str, Path, Uri]
    ) -> LocalUri:
        """Joins multiple URI tokens together and returns a new `LocalUri` object.

        Args:
            token (Union[str, Path, Uri]): The first URI token to join.
            *tokens (Union[str, Path, Uri]): Additional URI tokens to join.

        Returns:
            LocalUri: A new `LocalUri` object representing the joined URI.
        """
        joined_uri = super().join(token, *tokens)
        return cls(uri=joined_uri)

    @classmethod
    def is_valid(
        cls, uri: Union[str, Path, Uri], raise_exception: Optional[bool] = False
    ) -> bool:
        """Checks if the given URI is valid.

        Args:
            uri (Union[str, Path, Uri]): The URI to check.
            raise_exception (Optional[bool]): Whether to raise an exception if the URI is invalid. Defaults to False.

        Returns:
            bool: True if the URI is valid, False otherwise.
        """
        return True  # Default

    def absolute(self) -> LocalUri:
        """Returns an absolute `LocalUri` object.

        Returns:
            LocalUri: An absolute `LocalUri` object.
        """
        return LocalUri(uri=Path(self.uri).absolute())

    def __repr__(self) -> str:
        """Returns a string representation of the `LocalUri` object.

        Returns:
            str: The string representation of the `LocalUri` object.
        """
        return self.uri

    def __fspath__(self):
        """Return the file system path representation of the object.
        Needed for `os.PathLike` interface.

        Returns:
            str: The file system path representation of the object.
        """
        return str(self)
