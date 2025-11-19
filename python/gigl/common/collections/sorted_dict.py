from collections.abc import Iterator, Mapping
from typing import Any, Protocol, TypeVar


class _Comparable(Protocol):
    """Protocol for types that support comparison operations."""

    def __lt__(self, other: Any) -> bool:
        """Less than comparison."""
        ...


KT = TypeVar("KT", bound=_Comparable)
VT = TypeVar("VT")


class SortedDict(Mapping[KT, VT]):
    """A dictionary that maintains sorted order of keys during iteration.

    This class extends the standard dictionary behavior by automatically sorting
    keys whenever the dictionary is iterated over or represented as a string.
    The sorting is lazy and only performed when needed for efficiency.

    Type Parameters:
        KT: The type of keys in the dictionary. Must support comparison operations
            (implement __lt__) for sorting to work correctly.
        VT: The type of values in the dictionary

    Example:
        >>> sd = SortedDict({'z': 1, 'a': 2, 'b': 3})
        >>> list(sd.keys())
        ['a', 'b', 'z']
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize a SortedDict with the same arguments as a standard dict.

        Args:
            *args: Positional arguments passed to dict constructor
            **kwargs: Keyword arguments passed to dict constructor
        """
        self.__dict: dict[KT, VT] = dict(*args, **kwargs)
        self.__needs_memoization: bool = True
        self.__sort_dict_if_needed()

    def __len__(self) -> int:
        """Return the number of items in the dictionary.

        Returns:
            int: The number of key-value pairs in the dictionary
        """
        return len(self.__dict)

    def __getitem__(self, key: KT) -> VT:
        """Get the value associated with the given key.

        Args:
            key: The key to look up

        Returns:
            VT: The value associated with the key

        Raises:
            KeyError: If the key is not found in the dictionary
        """
        return self.__dict[key]

    def __eq__(self, other: object) -> bool:
        """Check equality against another mapping object.

        Args:
            other: Another object to compare against, typically a Mapping

        Returns:
            bool: True if the other object is a Mapping with identical key-value
                pairs, False otherwise
        """
        if not isinstance(other, Mapping):
            return False

        if len(self) != len(other):
            return False

        for self_key, self_val in self.items():
            if self_key not in other:
                return False
            if self_val != other[self_key]:
                return False
        return True

    def __setitem__(self, key: KT, value: VT) -> None:
        """Set the value for a given key.

        If the key is new, the dictionary will be marked for re-sorting on the
        next iteration or representation.

        Args:
            key: The key to set
            value: The value to associate with the key
        """
        if key not in self.__dict:
            self.__needs_memoization = True
        self.__dict[key] = value

    def __delitem__(self, key: KT) -> None:
        """Delete the key-value pair for the given key.

        Args:
            key: The key to delete

        Raises:
            KeyError: If the key is not found in the dictionary
        """
        del self.__dict[key]

    def __sort_dict_if_needed(self) -> None:
        """Sort the dictionary by keys if new keys have been added.

        This method is called automatically before iteration or representation.
        It uses a lazy evaluation strategy to avoid unnecessary sorting operations.
        """
        if self.__needs_memoization:
            self.__dict = dict(sorted(self.__dict.items(), key=lambda item: item[0]))
            self.__needs_memoization = False

    def __iter__(self) -> Iterator[KT]:
        """Return an iterator over the sorted keys of the dictionary.

        Returns:
            Iterator[KT]: An iterator over the dictionary keys in sorted order
        """
        self.__sort_dict_if_needed()
        return iter(self.__dict)

    def __repr__(self) -> str:
        """Return a string representation of the sorted dictionary.

        Returns:
            str: A string representation showing the dictionary in sorted key order
        """
        self.__sort_dict_if_needed()
        return repr(self.__dict)
