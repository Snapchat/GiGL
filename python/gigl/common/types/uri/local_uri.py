from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

from gigl.common.types.uri.uri import Uri


class LocalUri(Uri, os.PathLike):
    """Represents a local URI (Uniform Resource Identifier) that extends the `Uri` class and implements the `os.PathLike` interface."""

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
        is_valid: bool = True
        # Check if path starts with known remote schemes
        if str(uri).startswith(("gs://", "http://", "https://")):
            is_valid = False
            if raise_exception:
                raise ValueError(f"URI {uri} is not a valid local path")
        return is_valid

    def absolute(self) -> LocalUri:
        """Returns an absolute `LocalUri` object.

        Returns:
            LocalUri: An absolute `LocalUri` object.
        """
        return LocalUri(uri=Path(self.uri).absolute())

    def __fspath__(self):
        """Return the file system path representation of the object.
        Needed for `os.PathLike` interface.

        Returns:
            str: The file system path representation of the object.
        """
        return str(self)
