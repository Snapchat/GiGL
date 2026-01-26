from pathlib import Path

PROJECT_ROOT_DIR = (
    Path(__file__).resolve().parent.parent.parent
)  # gigl/experimental/knowledge_graph_embedding/
HYDRA_ROOT_DIR = Path.joinpath(
    PROJECT_ROOT_DIR, "conf"
)  # gigl/experimental/knowledge_graph_embedding/conf
HYDRA_CONFIG_FILE_PATH = "config.yaml"
