import enum
import glob
import os
import pathlib
import re
import shutil
from typing import Callable, Dict, List, Optional

from gigl.common import LocalUri
from gigl.common.logger import Logger

logger = Logger()


class FileSystemEntity(enum.Enum):
    """Class representing the entities in a file system."""

    FILE = enum.auto()
    DIRECTORY = enum.auto()


def delete_local_directory(local_path: LocalUri) -> None:
    """
    Deletes a local directory.

    Args:
        local_path (LocalUri): The path of the local directory to be deleted.

    Returns:
        None
    """
    local_path_str: str = local_path.uri
    shutil.rmtree(local_path_str, ignore_errors=True)
    logger.info(f"Deleted local directory at {local_path_str}")


def delete_and_create_local_path(local_path: LocalUri) -> None:
    """
    Deletes the existing local directory at the given path and creates a new one.

    Args:
        local_path (LocalUri): The path of the local directory to delete and create.

    Returns:
        None
    """
    delete_local_directory(local_path)
    pathlib.Path(local_path.uri).mkdir(parents=True, exist_ok=True)


def remove_file_if_exist(local_path: LocalUri) -> None:
    """
    Remove file with os if file exists.

    Args:
        local_path (LocalUri): The local path of the file to be removed.

    Returns:
        None
    """
    local_path_str: str = local_path.uri
    if does_path_exist(local_path):
        os.remove(local_path_str)
    logger.info(f"Deleted local file at {local_path_str}")


def does_path_exist(local_path: LocalUri) -> bool:
    """
    Check if a file exists.

    Args:
        local_path (LocalUri): The local path to check.

    Returns:
        bool: True if the file exists, False otherwise.
    """
    local_path_str: str = local_path.uri
    return os.path.exists(local_path_str)


def list_at_path(
    local_path: LocalUri,
    *,
    regex: Optional[str] = None,
    file_system_entity: Optional[FileSystemEntity] = None,
    names_only: bool = False,
) -> List[LocalUri]:
    """
    List all files and directories in the given local path.

    Args:
        local_path (LocalUri): The local path to search for files and directories.
        regex (Optional[str]): Optional regex to match. If not provided then all children will be returned.
        entity (Optional[FileSystemEntity]): Optional entity type to filter by. If not provided then all children will be returned.
        names_only (bool): If True, return only the base names of the files and directories. Defaults to False. e.g /path/to/file.txt -> file.txt

    Returns:
        List[LocalUri]: A list of local URIs for the files and directories in the given path.
    """
    children = os.listdir(local_path.uri)
    entity_filter: Callable[[str], bool]
    if file_system_entity == FileSystemEntity.FILE:
        entity_filter = lambda x: os.path.isfile(os.path.join(local_path.uri, x))
    elif file_system_entity == FileSystemEntity.DIRECTORY:
        entity_filter = lambda x: os.path.isdir(os.path.join(local_path.uri, x))
    else:
        entity_filter = lambda _: True

    pattern_filter: Callable[[str], bool]
    if regex:
        matcher = re.compile(regex)
        pattern_filter = lambda x: matcher.match(x) is not None
    else:
        pattern_filter = lambda _: True

    builder: Callable[[str], LocalUri]
    if names_only:
        builder = lambda x: LocalUri(x)
    else:
        builder = lambda x: LocalUri.join(local_path, x)

    return [
        builder(child)
        for child in children
        if pattern_filter(child) and entity_filter(child)
    ]


def count_files_with_uri_prefix(
    uri_prefix: LocalUri, suffix: Optional[str] = None
) -> int:
    """
    Count the number of files with a given URI prefix.

    Args:
        uri_prefix (LocalUri): The URI prefix to match.
        suffix (Optional[str]): The suffix to match. Defaults to None.

    Returns:
        int: The number of files with the given URI prefix.
    """
    matching_files = glob.glob(uri_prefix.uri + "*")
    if suffix:
        matching_files = [
            filename for filename in matching_files if filename.endswith(suffix)
        ]
    return len(matching_files)


def remove_folder_if_exist(local_path: LocalUri, ignore_errors: bool = True) -> None:
    """
    Remove a folder if it exists.

    Args:
        local_path (LocalUri): The local path of the folder to be removed.
        ignore_errors (bool): If True, ignore errors during removal. Defaults to True.

    Returns:
        None
    """
    if does_path_exist(local_path):
        uri_str: str = local_path.uri
        shutil.rmtree(uri_str, ignore_errors=ignore_errors)
        logger.info(f"Directory removed '{uri_str}'")


def create_empty_file_if_none_exists(local_path: LocalUri) -> None:
    """
    Create an empty file if it doesn't already exist.

    Args:
        local_path (LocalUri): The local path of the file to be created.

    Returns:
        None
    """
    os.makedirs(os.path.dirname(local_path.uri), exist_ok=True)
    if not does_path_exist(local_path):
        with open(local_path.uri, "w"):
            pass


def copy_files(
    local_source_to_local_dst_path_map: Dict[LocalUri, LocalUri],
    should_overwrite: Optional[bool] = False,
) -> None:
    """
    Copy files from source paths to destination paths.

    Args:
        local_source_to_local_dst_path_map (Dict[LocalUri, LocalUri]): A dictionary mapping source paths to destination paths.
        should_overwrite (Optional[bool]): If True, overwrite existing files at the destination paths. Defaults to False.

    Returns:
        None
    """
    for source_uri, dst_uri in local_source_to_local_dst_path_map.items():
        source_uri = source_uri.absolute()
        dst_uri = dst_uri.absolute()
        if source_uri != dst_uri:
            logger.info(f"Copying {source_uri} -> {dst_uri}...")
            if should_overwrite:
                remove_file_if_exist(dst_uri)
            try:
                pathlib.Path(dst_uri.uri).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source_uri.uri, dst_uri.uri)
            except FileExistsError as e:
                logger.error(f"Failed to copy {dst_uri} -> {source_uri}: {e}")
        else:
            logger.info(f"Cannot copy a file to itself: {source_uri}")


def create_file_symlinks(
    local_source_to_link_path_map: Dict[LocalUri, LocalUri],
    should_overwrite: Optional[bool] = False,
) -> None:
    """
    Create symlinks between source paths and link paths.

    Args:
        local_source_to_link_path_map (Dict[LocalUri, LocalUri]): A dictionary mapping source paths to link paths.
        should_overwrite (Optional[bool]): If True, overwrite existing links at the link paths. Defaults to False.

    Returns:
        None
    """
    for source_uri, link_uri in local_source_to_link_path_map.items():
        source_uri = source_uri.absolute()
        link_uri = link_uri.absolute()
        if source_uri != link_uri:
            logger.info(f"Creating symlink {link_uri} -> {source_uri}")
            if should_overwrite:
                remove_file_if_exist(link_uri)
            try:
                pathlib.Path(link_uri.uri).parent.mkdir(parents=True, exist_ok=True)
                pathlib.Path(link_uri.uri).symlink_to(source_uri.uri)
            except FileExistsError as e:
                logger.error(f"Failed to symlink {link_uri} -> {source_uri}: {e}")
        else:
            logger.info(f"Cannot symlink a file to itself: {source_uri}")


def append_line_to_file(file_path: LocalUri, line: str) -> None:
    """
    Append a line to a file if it doesn't already exist in the file.

    Args:
        file_path (LocalUri): The path of the file to append the line to.
        line (str): The line to append.

    Returns:
        None
    """
    create_empty_file_if_none_exists(file_path)
    with open(file_path.uri, "r") as f:
        lines = f.readlines()
        if line in lines:
            logger.info(f"Line already exists in {file_path}")
            return

    with open(file_path.uri, "a") as f:
        f.write(f"{line}\n")
    logger.info(f"Appended line to {file_path}")


def remove_line_from_file(file_path: LocalUri, line: str) -> None:
    """
    Remove a line from a file if it exists in the file.

    Args:
        file_path (LocalUri): The path of the file to remove the line from.
        line (str): The line to remove.

    Returns:
        None
    """
    if does_path_exist(file_path):
        with open(file_path.uri, "r") as f:
            lines = f.readlines()

        if (line + "\n") in lines:
            with open(file_path.uri, "w") as f:
                for line_in_file in lines:
                    if line_in_file != line + "\n":
                        f.write(line_in_file)
            logger.info(f"Removed line from {file_path}")
        else:
            logger.info(f"Line not found in {file_path}")
    else:
        logger.warning(f"File does not exist at {file_path}")


def remove_file_or_folder_if_exist(local_path: LocalUri) -> None:
    """
    Remove a file or folder if it exists.

    Args:
        local_path (LocalUri): The local path of the file or folder to be removed.

    Returns:
        None
    """
    if os.path.isdir(local_path.uri):
        remove_folder_if_exist(local_path)
    else:
        remove_file_if_exist(local_path)
