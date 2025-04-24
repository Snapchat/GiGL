"""
Assert that the YAML configuration files in specified directories can be parsed as GiglResourceConfig or GbmlConfig protos.

Note that this does not check the *contents* of the fields set, e.g. `python_class_path: not a valid path` will not be caught.
This script does a subset of what config_validator does, but is faster and can be used locally.
You may also put "# yaml-check: disable" at the top of a YAML file to ignore it.

Usage:
    python assert_yaml_configs_parse.py --directories <dir1> <dir2> ... [--ignore_regex <file1> <file2> ...]

Arguments:
    --directories, -d: List of directories to check for YAML config files.
    --ignore_regex, -i: List of regex patterns to ignore files. If a file path matches any of the regex patterns, it will be skipped.

Description:
    The script recursively searches through the specified directories for YAML files.
    It attempts to parse each YAML file as either a GiglResourceConfig or GbmlConfig based on the filename.
    If a file cannot be parsed, it logs the error and reports all invalid files at the end.
    If any of the ignore_regex matches the file path, or the first line of the file starts with "# yaml-check: disable",
    the file will be skipped.

Examples:
    To check all YAML files in the 'configs' directory:
        python assert_yaml_configs_parse.py --directories configs

    To check YAML files in 'configs' and 'more_configs' directories, ignoring 'foo/bar/ignore_this.yaml' and everything under qux/dir/:
        python assert_yaml_configs_parse.py --directories configs more_configs --ignore_regex foo/bar/ignore_this.yaml qux/dir/.*
"""

import argparse
import os
import re
from typing import Dict, List

from gigl.common import Uri, UriFactory
from gigl.common.logger import Logger
from gigl.common.utils.proto_utils import ProtoUtils
from snapchat.research.gbml.gbml_config_pb2 import GbmlConfig
from snapchat.research.gbml.gigl_resource_config_pb2 import GiglResourceConfig

logger = Logger()

_IGNORE_COMMENT = "# yaml-check: disable"


def assert_configs_parse(directories: List[str], ignore_regex: List[str] = []) -> None:
    proto_utils = ProtoUtils()
    invalid_configs: Dict[Uri, str] = {}
    logger.info(f"Checking if configs in {directories} are valid.")
    logger.info(f"Ignoring regex: {ignore_regex}")
    ignore = [re.compile(regex) for regex in ignore_regex]
    total = 0
    for directory in directories:
        for root, _, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                if (
                    file.endswith(".yaml")
                    and not any(r.match(file_path) for r in ignore)
                    and ("resource_config" in file or "task_config" in file)
                ):
                    with open(file_path, "r") as f:
                        if f.readline().strip().startswith(_IGNORE_COMMENT):
                            logger.info(
                                f"Ignored {file_path} due to the '{_IGNORE_COMMENT}' header."
                            )
                            continue
                    total += 1
                    yaml_file = UriFactory.create_uri(file_path)
                    try:
                        if "resource_config" in file:
                            proto_utils.read_proto_from_yaml(
                                yaml_file, GiglResourceConfig
                            )
                        elif "task_config" in file:
                            proto_utils.read_proto_from_yaml(yaml_file, GbmlConfig)
                        else:
                            continue
                        logger.info(f"{yaml_file} parsed successfully.")
                    except Exception as e:
                        invalid_configs[yaml_file] = str(e)

    logger.info(f"Checked {total} YAML files.")
    if invalid_configs:
        logger.error(f"Found {len(invalid_configs)} invalid YAML files:")
        for yaml_file, error in invalid_configs.items():
            logger.error(f"{yaml_file}: {error}")
        exit(-1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check if config files can be parsed.")
    parser.add_argument(
        "--directories",
        "-d",
        nargs="+",
        required=True,
        type=str,
        help="Directories to check for config files.",
    )
    parser.add_argument(
        "--ignore_regex",
        "-i",
        nargs="+",
        required=False,
        type=str,
        default=[],
        help="Regex to skip.",
    )
    args = parser.parse_args()
    assert_configs_parse(directories=args.directories, ignore_regex=args.ignore_regex)
