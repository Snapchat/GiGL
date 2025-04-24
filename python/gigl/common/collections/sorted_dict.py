from collections.abc import Mapping
from typing import TypeVar

KT = TypeVar("KT")
VT = TypeVar("VT")


class SortedDict(Mapping[KT, VT]):
    """Sorted Dictionary implementation
    Given a dictionary, sorts it by keys when iterating over it
    """

    def __init__(self, *args, **kwargs):
        self.__dict = dict(*args, **kwargs)
        self.__needs_memoization = True
        self.__sort_dict_if_needed()

    def __len__(self):
        return len(self.__dict)

    def __getitem__(self, key):
        return self.__dict[key]

    def __eq__(self, other: object) -> bool:
        """Check equality against another object

        Args:
            other (object): Dictionary like object or SortedDict

        Returns:
            bool: Returns true if the current instance of frozen dict matches the
            other dictionary / frozen dict
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

    def __setitem__(self, key, value):
        if key not in self.__dict:
            self.__needs_memoization = True
        self.__dict[key] = value

    def __delitem__(self, key):
        del self.__dict[key]

    def __sort_dict_if_needed(self):
        if self.__needs_memoization:
            self.__dict = dict(sorted(self.__dict.items(), key=lambda item: item[0]))
            self.__needs_memoization = False

    def __iter__(self):
        self.__sort_dict_if_needed()
        return iter(self.__dict)

    def __repr__(self):
        self.__sort_dict_if_needed()
        return self.__dict.__repr__()
