import argparse
import os
import tempfile
from typing import Literal, Tuple

import hydra
from omegaconf import DictConfig

import gigl.experimental.knowledge_graph_embedding.lib.constants.gcs as gcs_constants
import gigl.experimental.knowledge_graph_embedding.lib.constants.local as local_constants
from gigl.common import GcsUri, LocalUri, UriFactory
from gigl.common.types.uri.uri import Uri
from gigl.env.pipelines_config import get_resource_config
from gigl.src.common.types import AppliedTaskIdentifier
from gigl.src.common.types.pb_wrappers.gigl_resource_config import (
    GiglResourceConfigWrapper,
)
from gigl.src.common.utils.file_loader import FileLoader


def build_modeling_and_resource_config_from_args(
    mode: Literal["enumerating", "training"]
) -> Tuple[AppliedTaskIdentifier, DictConfig, GiglResourceConfigWrapper]:
    """
    Build the modeling and resource config from command line arguments.

    Parses command line arguments to extract configuration URIs, loads the modeling
    configuration using Hydra, and loads the resource configuration. Handles both
    local and GCS URI sources, automatically downloading remote configs when needed.

    Args:
        mode: The execution mode, either "enumerating" (for data preprocessing)
            or "training" (for model training). Determines fallback config paths.

    Returns:
        Tuple[AppliedTaskIdentifier, DictConfig, GiglResourceConfigWrapper]: A tuple containing:
            - Applied task identifier for the current job
            - Hydra modeling configuration (training hyperparameters, model config, etc.)
            - Resource configuration wrapper (project info, compute resources, etc.)

    Raises:
        ValueError: If an unsupported mode is provided.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--applied_task_identifier",
        required=True,
        help="Applied Task Identifier for this job.",
    )
    parser.add_argument(
        "--modeling_config_uri",
        required=False,
        help="Path to local or GCS hydra config",
    )
    parser.add_argument(
        "--resource_config_uri",
        required=True,
        help="Path to the resource config file, which contains the project and other resource information.",
    )
    args, unknown = parser.parse_known_args()

    applied_task_identifier = AppliedTaskIdentifier(args.applied_task_identifier)

    # Load the resource config.
    resource_config: GiglResourceConfigWrapper = get_resource_config(
        resource_config_uri=UriFactory.create_uri(args.resource_config_uri)
    )

    # Load the hydra config.
    local_config_uri: LocalUri
    modeling_config_uri: Uri

    if not args.modeling_config_uri:
        if mode == "enumerating":
            # If no modeling config URI is provided, use the fallback local config for enumeration.
            modeling_config_uri = LocalUri.join(
                local_constants.HYDRA_ROOT_DIR, local_constants.HYDRA_CONFIG_FILE_PATH
            )
        elif mode == "training":
            # If no modeling config URI is provided, infer the enumerated config path from gcs_constants.
            modeling_config_uri = gcs_constants.get_enumerated_config_output_path(
                applied_task_identifier=applied_task_identifier,
            )
        else:
            raise ValueError(
                f"Unsupported mode: {mode}. Use 'enumerating' or 'training'."
            )
    else:
        modeling_config_uri = UriFactory.create_uri(args.modeling_config_uri)
    if isinstance(modeling_config_uri, GcsUri):
        # If the config is a GCS URI, we need to download it first.
        file_loader = FileLoader(project=resource_config.project)
        tfh = tempfile.NamedTemporaryFile(suffix=".yaml", delete=False)
        local_config_uri = LocalUri(tfh.name)
        file_loader.load_file(
            file_uri_src=modeling_config_uri, file_uri_dst=local_config_uri
        )
    elif isinstance(modeling_config_uri, LocalUri):
        # If the config is a local URI, we can use it directly.
        local_config_uri = modeling_config_uri

    config_dir = os.path.abspath(os.path.dirname(local_config_uri.uri))
    config_name = os.path.basename(local_config_uri.uri)

    with hydra.initialize_config_dir(config_dir=config_dir):
        modeling_config = hydra.compose(config_name=config_name, overrides=unknown)

    return applied_task_identifier, modeling_config, resource_config
