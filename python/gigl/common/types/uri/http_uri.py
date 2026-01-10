from __future__ import annotations

from pathlib import Path

from gigl.common.types.uri.uri import Uri


class HttpUri(Uri):
    """Represents an HTTP URI."""

    def __init__(self, uri: str | Path | HttpUri) -> None:
        self._has_valid_prefix(uri=uri)
        self._has_no_backslash(uri=uri)
        super().__init__(uri=uri)

    @classmethod
    def _has_valid_prefix(cls, uri: str | Path | Uri) -> bool:
        """Check if the URI has a valid prefix (http:// or https://)."""
        uri_str = cls._token_to_string(uri)

        return uri_str.startswith("http://") or uri_str.startswith("https://")

    @classmethod
    def _has_no_backslash(cls, uri: str | Path | Uri) -> bool:
        """Check if the URI has no backslashes."""
        return cls._token_to_string(uri).count("\\") == 0

    @classmethod
    def is_valid(cls, uri: str | Path | Uri, raise_exception: bool = False) -> bool:
        """Check if the URI is valid.

        Args:
            uri: The URI to check.
            raise_exception: Whether to raise an exception if the URI is invalid.

        Returns:
            bool: True if the URI is valid, False otherwise.
        """
        has_valid_prefix: bool = cls._has_valid_prefix(uri=uri)
        has_no_backslash: bool = cls._has_no_backslash(uri=uri)
        if raise_exception and not has_valid_prefix:
            raise TypeError(f"{cls.__name__} must start with http:// or https://")
        if raise_exception and not has_no_backslash:
            raise TypeError(f"{cls.__name__} does not support backslashes")
        return has_valid_prefix and has_no_backslash
