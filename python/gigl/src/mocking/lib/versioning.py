from __future__ import annotations

import json
from typing import Dict, NamedTuple

from gigl.common import Uri, UriFactory
from gigl.common.logger import Logger
from gigl.src.mocking.lib.constants import MOCKED_DATASET_ARTIFACT_METADATA_LOCAL_PATH

logger = Logger()


class MockedDatasetArtifactMetadata(NamedTuple):
    version: str
    frozen_gbml_config_uri: Uri

    @property
    def json_serializable(self):
        return {
            "version": self.version,
            "frozen_gbml_config_uri": self.frozen_gbml_config_uri.uri,
        }

    @staticmethod
    def from_dict(dict_repr: Dict[str, str]) -> MockedDatasetArtifactMetadata:
        return MockedDatasetArtifactMetadata(
            version=dict_repr["version"],
            frozen_gbml_config_uri=UriFactory.create_uri(
                uri=dict_repr["frozen_gbml_config_uri"]
            ),
        )


def get_mocked_dataset_artifact_metadata() -> Dict[str, MockedDatasetArtifactMetadata]:
    """
    Creates a dictionary of task names to mocked dataset artifact metadata.

    Returns:
        A dictionary of mocked dataset artifact metadata, where the keys are dataset names and the values are artifact metadata.
    """
    artifact_metadata_uri = MOCKED_DATASET_ARTIFACT_METADATA_LOCAL_PATH
    f = open(artifact_metadata_uri.uri, "r")
    metadata: Dict[str, Dict[str, str]] = {}
    try:
        metadata = json.load(fp=f)
    except json.JSONDecodeError as e:
        logger.error(
            f"Error parsing artifact metadata file at {artifact_metadata_uri.uri}."
        )
    artifact_metadata = {
        task_name: MockedDatasetArtifactMetadata.from_dict(
            dict_repr=metadata[task_name]
        )
        for task_name in metadata
    }
    return artifact_metadata


def update_mocked_dataset_artifact_metadata(
    task_name_to_artifact_metadata: Dict[str, MockedDatasetArtifactMetadata]
) -> None:
    """
    Update the mocked dataset artifact metadata with the given task names and metadata.

    Args:
        task_name_to_versions (Dict[str, MockedDatasetArtifactMetadata]): A dictionary containing task names and their corresponding metadata.

    Returns:
        None
    """
    current_artifact_metadata = get_mocked_dataset_artifact_metadata()
    for task_name, artifact_metadata in task_name_to_artifact_metadata.items():
        if not task_name in current_artifact_metadata:
            logger.info(
                f"{task_name} not found in artifact metadata tracker. Adding it."
            )
        logger.info(f"Updating metadata {artifact_metadata}.")
        current_artifact_metadata[task_name] = artifact_metadata

    serializable_current_artifact_metadata = {
        task_name: metadata.json_serializable
        for task_name, metadata in current_artifact_metadata.items()
    }
    json.dump(
        obj=serializable_current_artifact_metadata,
        fp=open(MOCKED_DATASET_ARTIFACT_METADATA_LOCAL_PATH.uri, "w"),
        indent=4,
    )
    logger.info("Updated artifact metadata tracker file.")
