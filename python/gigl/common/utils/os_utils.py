import importlib
import subprocess
from io import TextIOWrapper
from typing import Optional

from gigl.common.logger import Logger

logger = Logger()


def num_lines_in_file(f: TextIOWrapper) -> int:
    f.seek(0)  # Reset read pointer
    return sum(1 for _ in f)


def import_obj(obj_path: str):
    """
    Given an object path by name (e.g. "module_a.module_b.object_c"), returns object_c.

    Args:
        obj_path (str): The object path by name.

    Returns:
        object: The imported object.
    """

    def _import_module(module_or_obj_name: str) -> tuple:
        """
        Given an object path by name (e.g. "module_a.module_b.object_c"), returns a handle to module_a.module_b and the object name ("object_c").

        Args:
            module_or_obj_name (str): The object path by name.

        Returns:
            tuple: A tuple containing the imported module and the relative object name.
        """

        parts = module_or_obj_name.split(".")
        for i in range(len(parts), 0, -1):
            partial_path = ".".join(parts[:i])
            logger.info(f"Will try importing module: {partial_path}")
            try:
                module = importlib.import_module(partial_path)
                relative_obj_name = ".".join(parts[i:])
                logger.info(
                    f"Successfully imported module = {module}, which potentially has object = {relative_obj_name}"
                )
                return module, relative_obj_name
            except ImportError as e:
                logger.info(f'Unable to import "{partial_path}".')
                logger.info(f"{e}")
        raise ImportError(module_or_obj_name)

    def _find_obj_in_module(module: object, relative_obj_name: str) -> object:
        """
        Given a module (e.g. module_a.module_b) and a relative object name (e.g. "object_c"), return a handle for object_c.

        Args:
            module (module): The module to search for the object.
            relative_obj_name (str): The relative object name.

        Returns:
            object: The found object.
        """

        obj = module
        for part in relative_obj_name.split("."):
            logger.info(f"Trying to access {part} in {obj}")
            try:
                obj = getattr(obj, part)
            except AttributeError as e:
                logger.error(f"Could not find the attribute {part} in {obj}")
                try:
                    module = importlib.import_module(obj_path)
                except ImportError as import_error:
                    logger.error(
                        f"Could not import module {obj_path} either. Import error: {import_error}"
                    )
                logger.error(e)
                raise e
        return obj

    module, relative_obj_name = _import_module(obj_path)
    return _find_obj_in_module(module, relative_obj_name)


def run_command_and_stream_stdout(cmd: str) -> Optional[int]:
    """
    Executes a command and streams the stdout output.

    Args:
        cmd (str): The command to be executed.

    Returns:
        Optional[int]: The return code of the command, or None if the command failed to execute.
    """
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    while True:
        output = process.stdout.readline()  # type: ignore
        if output == b"" and process.poll() is not None:
            break
        if output:
            print(output.strip())
    return_code: Optional[int] = process.poll()
    return return_code
