from typing import Final

_TRUE_VALUES: Final[frozenset[str]] = frozenset({"y", "yes", "t", "true", "on", "1"})
_FALSE_VALUES: Final[frozenset[str]] = frozenset({"n", "no", "f", "false", "off", "0"})


def str_to_bool(value: str) -> bool:
    """
    Converts a string representation of truth to a bool.

    Accepts and rejects the same spellings as `distutils.util.strtobool`, which is no
    longer part of the standard library, though the `ValueError` message keeps the
    caller's casing instead of lowercasing it:

      - True: "y", "yes", "t", "true", "on", "1"
      - False: "n", "no", "f", "false", "off", "0"

    Matching is case-insensitive. Whitespace is not stripped, so " true" raises rather
    than returning True.

    Example:
        >>> str_to_bool("TRUE")
        True
        >>> str_to_bool("off")
        False

    Args:
        value (str): The string to interpret as a boolean.

    Returns:
        bool: The truth value `value` spells.

    Raises:
        ValueError: If `value` is not one of the accepted spellings.
    """
    normalized_value = value.lower()
    if normalized_value in _TRUE_VALUES:
        return True
    if normalized_value in _FALSE_VALUES:
        return False
    raise ValueError(f"invalid truth value {value!r}")
