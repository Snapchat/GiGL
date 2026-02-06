import unittest

import gigl.common.utils.local_fs as local_fs_utils
from gigl.common import LocalUri


class OsUtilsTest(unittest.TestCase):
    def test_create_empty_file_if_none_exists(self):
        path_to_temp_file: LocalUri = LocalUri.join(
            ".test_assets", "test_create_empty_file_if_none_exists.txt"
        )
        local_fs_utils.remove_file_if_exist(path_to_temp_file)

        # Create file and ensure it exists
        local_fs_utils.create_empty_file_if_none_exists(path_to_temp_file)
        self.assertTrue(local_fs_utils.does_path_exist(path_to_temp_file))

        text_to_write_to_file = """
        Never gonna give you up
        Never gonna let you down
        Never gonna run around and desert you
        Never gonna make you cry
        Never gonna say goodbye
        Never gonna tell a lie and hurt you
        """
        with open(path_to_temp_file.uri, "a") as f:  # append mode
            f.writelines(text_to_write_to_file)

        local_fs_utils.create_empty_file_if_none_exists(
            path_to_temp_file
        )  # should be a NOP
        with open(path_to_temp_file.uri, "r") as f:  # read mode
            read_lines = f.read()
            self.assertEqual(read_lines, text_to_write_to_file)
