import os
import pathlib

gigl_root_dir = pathlib.Path(__file__).parent.parent.parent.parent.parent


def change_working_dir_to_gigl_root():
    """
    Can be used inside notebooks to change the working directory to the GIGL root directory.
    This is useful for ensuring that relative imports and file paths work correctly no matter where the notebook is located.
    """
    os.chdir(gigl_root_dir)
    print(f"Changed working directory to: {gigl_root_dir}")
