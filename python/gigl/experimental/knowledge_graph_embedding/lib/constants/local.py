from pathlib import Path

PROJECT_ROOT_DIR = Path(__file__).resolve().parent.parent.parent
HYDRA_ROOT_DIR = Path.joinpath(PROJECT_ROOT_DIR, "conf")  # "conf"
HYDRA_CONFIG_FILE_PATH = "config.yaml"
