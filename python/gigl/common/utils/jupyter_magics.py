import os
import pathlib
from difflib import unified_diff
from IPython.display import display, HTML
from gigl.common import Uri
import yaml


gigl_root_dir = pathlib.Path(__file__).parent.parent.parent.parent.parent


def change_working_dir_to_gigl_root():
    """
    Can be used inside notebooks to change the working directory to the GIGL root directory.
    This is useful for ensuring that relative imports and file paths work correctly no matter where the notebook is located.
    """
    os.chdir(gigl_root_dir)
    print(f"Changed working directory to: {gigl_root_dir}")



def sort_yaml_dict_recursively(obj: dict) -> dict:
    # We sort the json recursively as the GiGL proto serialization code does not guarantee order of original keys.
    # This is important for the diff to be stable and not show errors due to key/list order changes.
    if isinstance(obj, dict):
        return {k: sort_yaml_dict_recursively(obj[k]) for k in sorted(obj)}
    elif isinstance(obj, list):
        return [sort_yaml_dict_recursively(item) for item in obj]
    else:
        return obj

def show_colored_unified_diff(f1_lines, f2_lines, f1_name, f2_name):
    diff_lines = list(unified_diff(f1_lines, f2_lines, fromfile=f1_name, tofile=f2_name))
    html_lines = []
    for line in diff_lines:
        if line.startswith('+') and not line.startswith('+++'):
            color = '#228B22'  # green
        elif line.startswith('-') and not line.startswith('---'):
            color = '#B22222'  # red
        elif line.startswith('@'):
            color = '#1E90FF'  # blue
        else:
            color = "#000000"  # black
        html_lines.append(f'<pre style="margin:0; color:{color}; background-color:white;">{line.rstrip()}</pre>')
    display(HTML(''.join(html_lines)))

from gigl.src.common.utils.file_loader import FileLoader



def show_task_config_colored_unified_diff(f1_uri: Uri, f2_uri: Uri, f1_name: str, f2_name: str):
    """
    Displays a colored unified diff of two task config files.
    Args:
        f1_uri (Uri): URI of the first file.
        f2_uri (Uri): URI of the second file.
    """
    file_loader = FileLoader()
    frozen_task_config_file_contents: str
    template_task_config_file_contents: str

    with open(file_loader.load_to_temp_file(file_uri_src=f1_uri).name, 'r') as f:
        data = yaml.safe_load(f)
        # sort_keys by default
        frozen_task_config_file_contents = yaml.dump(sort_yaml_dict_recursively(data))

    with open(file_loader.load_to_temp_file(file_uri_src=f2_uri).name, 'r') as f:
        data = yaml.safe_load(f)
        template_task_config_file_contents = yaml.dump(sort_yaml_dict_recursively(data))

    show_colored_unified_diff(
        template_task_config_file_contents.splitlines(),
        frozen_task_config_file_contents.splitlines(),
        f1_name=f1_name,
        f2_name=f2_name
    )
