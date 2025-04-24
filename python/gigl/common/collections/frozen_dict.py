from collections.abc import Mapping
from typing import TypeVar

KT = TypeVar("KT")
VT = TypeVar("VT")


class FrozenDict(Mapping[KT, VT]):
    """Frozen Dictionary implementation,
    Given a dictionary, freezes it, i.e. disallows any edits or deletion
    """

    def __init__(self, *args, **kwargs):
        self.__dict = dict(*args, **kwargs)
        self.__hash = None

    def __iter__(self):
        return iter(self.__dict)

    def __len__(self):
        return len(self.__dict)

    def __getitem__(self, key):
        return self.__dict[key]

    def __eq__(self, other: object) -> bool:
        """Check equality against another object

        Args:
            other (object): Dictionary like object or FrozenDict

        Returns:
            bool: Returns true if the current instance of frozen dict matches the
            other dictionary / frozen dict
        """
        if isinstance(other, FrozenDict):
            return hash(self) == hash(other)

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

    def __hash__(self):
        if self.__hash is None:
            self.__hash = hash(tuple(sorted(self.__dict.items())))
        return self.__hash

    def __repr__(self) -> str:
        return self.__dict.__repr__()
