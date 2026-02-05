from typing import Any

from absl.testing import absltest

from gigl.common.collections.sorted_dict import SortedDict
from tests.test_assets.test_case import TestCase


class TestSortedDict(TestCase):
    """Test suite for SortedDict collection class."""

    def setUp(self) -> None:
        """Set up test fixtures before each test method."""
        self.unsorted_data = {"z": 1, "a": 2, "m": 3, "b": 4}
        self.sorted_keys = ["a", "b", "m", "z"]

    def test_init_empty(self) -> None:
        """Test initialization of empty SortedDict."""
        sd: SortedDict[str, int] = SortedDict()
        self.assertEqual(len(sd), 0)
        self.assertEqual(list(sd.keys()), [])

    def test_init_with_dict(self) -> None:
        """Test initialization with a dictionary."""
        sd: SortedDict[str, int] = SortedDict(self.unsorted_data)
        self.assertEqual(len(sd), 4)
        self.assertEqual(list(sd.keys()), self.sorted_keys)

    def test_init_with_kwargs(self) -> None:
        """Test initialization with keyword arguments."""
        sd: SortedDict[str, int] = SortedDict(z=1, a=2, m=3, b=4)
        self.assertEqual(len(sd), 4)
        self.assertEqual(list(sd.keys()), self.sorted_keys)

    def test_init_with_items(self) -> None:
        """Test initialization with list of tuples."""
        items = [("z", 1), ("a", 2), ("m", 3), ("b", 4)]
        sd: SortedDict[str, int] = SortedDict(items)
        self.assertEqual(len(sd), 4)
        self.assertEqual(list(sd.keys()), self.sorted_keys)

    def test_iter_returns_sorted_keys(self) -> None:
        """Test that iteration returns keys in sorted order."""
        sd: SortedDict[str, int] = SortedDict(self.unsorted_data)
        keys = list(sd)
        self.assertEqual(keys, self.sorted_keys)

    def test_items_are_sorted_by_key(self) -> None:
        """Test that items() method returns items sorted by key."""
        sd: SortedDict[str, int] = SortedDict(self.unsorted_data)
        items = list(sd.items())
        expected = [("a", 2), ("b", 4), ("m", 3), ("z", 1)]
        self.assertEqual(items, expected)

    def test_getitem(self) -> None:
        """Test item access via bracket notation."""
        sd: SortedDict[str, int] = SortedDict(self.unsorted_data)
        self.assertEqual(sd["a"], 2)
        self.assertEqual(sd["z"], 1)
        self.assertEqual(sd["m"], 3)

    def test_getitem_missing_key_raises_keyerror(self) -> None:
        """Test that accessing missing key raises KeyError."""
        sd: SortedDict[str, int] = SortedDict(self.unsorted_data)
        with self.assertRaises(KeyError):
            _ = sd["nonexistent"]

    def test_setitem_new_key(self) -> None:
        """Test adding new key-value pair maintains sorted order."""
        sd: SortedDict[str, int] = SortedDict({"z": 1, "a": 2})
        sd["m"] = 3
        self.assertEqual(list(sd.keys()), ["a", "m", "z"])
        self.assertEqual(sd["m"], 3)

    def test_setitem_existing_key(self) -> None:
        """Test updating existing key-value pair."""
        sd: SortedDict[str, int] = SortedDict(self.unsorted_data)
        sd["a"] = 100
        self.assertEqual(sd["a"], 100)
        self.assertEqual(list(sd.keys()), self.sorted_keys)

    def test_setitem_multiple_additions_maintain_order(self) -> None:
        """Test that multiple additions maintain sorted order."""
        sd: SortedDict[str, int] = SortedDict()
        sd["z"] = 1
        sd["a"] = 2
        sd["m"] = 3
        sd["b"] = 4
        sd["x"] = 5
        self.assertEqual(list(sd.keys()), ["a", "b", "m", "x", "z"])

    def test_delitem(self) -> None:
        """Test deleting key-value pair."""
        sd: SortedDict[str, int] = SortedDict(self.unsorted_data)
        del sd["m"]
        self.assertEqual(len(sd), 3)
        self.assertEqual(list(sd.keys()), ["a", "b", "z"])
        with self.assertRaises(KeyError):
            _ = sd["m"]

    def test_delitem_missing_key_raises_keyerror(self) -> None:
        """Test that deleting missing key raises KeyError."""
        sd: SortedDict[str, int] = SortedDict(self.unsorted_data)
        with self.assertRaises(KeyError):
            del sd["nonexistent"]

    def test_len(self) -> None:
        """Test length calculation."""
        sd: SortedDict[str, int] = SortedDict()
        self.assertEqual(len(sd), 0)

        sd["a"] = 1
        self.assertEqual(len(sd), 1)

        sd["b"] = 2
        self.assertEqual(len(sd), 2)

        del sd["a"]
        self.assertEqual(len(sd), 1)

    def test_contains(self) -> None:
        """Test membership checking with 'in' operator."""
        sd: SortedDict[str, int] = SortedDict(self.unsorted_data)
        self.assertIn("a", sd)
        self.assertIn("z", sd)
        self.assertNotIn("nonexistent", sd)

    def test_get_method(self) -> None:
        """Test get() method with default value."""
        sd: SortedDict[str, int] = SortedDict(self.unsorted_data)
        self.assertEqual(sd.get("a"), 2)
        self.assertEqual(sd.get("nonexistent"), None)
        self.assertEqual(sd.get("nonexistent", "default"), "default")

    def test_eq_with_dict(self) -> None:
        """Test equality comparison with regular dict."""
        sd: SortedDict[str, int] = SortedDict(self.unsorted_data)
        regular_dict = {"a": 2, "b": 4, "m": 3, "z": 1}
        self.assertEqual(sd, regular_dict)

    def test_eq_with_sorted_dict(self) -> None:
        """Test equality comparison with another SortedDict."""
        sd1: SortedDict[str, int] = SortedDict(self.unsorted_data)
        sd2: SortedDict[str, int] = SortedDict({"a": 2, "z": 1, "b": 4, "m": 3})
        self.assertEqual(sd1, sd2)

    def test_eq_with_different_dict(self) -> None:
        """Test inequality with different dictionary."""
        sd: SortedDict[str, int] = SortedDict(self.unsorted_data)
        different_dict = {"a": 2, "b": 4}
        self.assertNotEqual(sd, different_dict)

    def test_eq_with_different_values(self) -> None:
        """Test inequality when values differ."""
        sd: SortedDict[str, int] = SortedDict(self.unsorted_data)
        different_dict = {"a": 2, "b": 4, "m": 99, "z": 1}
        self.assertNotEqual(sd, different_dict)

    def test_eq_with_non_mapping(self) -> None:
        """Test inequality with non-mapping objects."""
        sd: SortedDict[str, int] = SortedDict(self.unsorted_data)
        self.assertNotEqual(sd, "not a dict")
        self.assertNotEqual(sd, 42)
        self.assertNotEqual(sd, None)
        self.assertNotEqual(sd, [("a", 2)])

    def test_repr(self) -> None:
        """Test string representation shows sorted order."""
        sd: SortedDict[str, int] = SortedDict({"z": 1, "a": 2, "b": 3})
        repr_str = repr(sd)
        self.assertEqual(repr_str, "{'a': 2, 'b': 3, 'z': 1}")

    def test_numeric_keys(self) -> None:
        """Test SortedDict with numeric keys."""
        sd: SortedDict[int, str] = SortedDict(
            {5: "five", 1: "one", 10: "ten", 2: "two"}
        )
        self.assertEqual(list(sd.keys()), [1, 2, 5, 10])
        self.assertEqual(list(sd.values()), ["one", "two", "five", "ten"])

    def test_mixed_type_values(self) -> None:
        """Test SortedDict with various value types."""
        sd: SortedDict[str, Any] = SortedDict(
            {"a": 1, "b": "string", "c": [1, 2, 3], "d": {"nested": "dict"}}
        )
        self.assertEqual(list(sd.keys()), ["a", "b", "c", "d"])
        self.assertEqual(sd["a"], 1)
        self.assertEqual(sd["b"], "string")
        self.assertEqual(sd["c"], [1, 2, 3])
        self.assertEqual(sd["d"], {"nested": "dict"})

    def test_update_after_iteration(self) -> None:
        """Test that updates after iteration maintain sorted order."""
        sd: SortedDict[str, int] = SortedDict({"c": 3, "a": 1})

        # Iterate once
        first_keys = list(sd.keys())
        self.assertEqual(first_keys, ["a", "c"])

        # Add new key
        sd["b"] = 2

        # Iterate again
        second_keys = list(sd.keys())
        self.assertEqual(second_keys, ["a", "b", "c"])


if __name__ == "__main__":
    absltest.main()
