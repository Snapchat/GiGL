import unittest

from gigl.common.collections.frozen_dict import FrozenDict


class FrozenDictTest(unittest.TestCase):
    def test_frozen_dict_is_frozen(self):
        frozen_dict: FrozenDict[int, int] = FrozenDict()

        def assign_dict_value():
            frozen_dict[10] = 20  # type: ignore [index]

        self.assertRaises(Exception, assign_dict_value)

        frozen_dict = FrozenDict({10: 20, 30: 49})
        self.assertRaises(Exception, assign_dict_value)

    def test_equality(self):
        dict_1: FrozenDict[str, int] = FrozenDict(a=1, b=2)
        dict_2: FrozenDict[str, int] = FrozenDict({"a": 1, "b": 2})
        dict_3: FrozenDict[str, int] = FrozenDict({"a": 1, "b": 3})
        dict_4: FrozenDict[str, int] = FrozenDict({})
        self.assertEqual(dict_1, dict_2)
        self.assertNotEqual(dict_1, dict_3)
        self.assertNotEqual(dict_2, dict_3)
        self.assertFalse(dict_4)
        self.assertTrue(dict_1)
