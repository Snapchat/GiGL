from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from gigl.common import LocalUri
from gigl.common.logger import Logger

logger = Logger()


class KfpOutputViewers:
    def __init__(self) -> None:
        """
        To enable component ui visualizations, your component must have an output path called
        mlpipeline_ui_metadata and export a JSON-serialized dictionary to that path that contains
        metadata of what visualizations to render.

        Example:
        component.yaml
        >>> outputs:
        >>> - {name: mlpipeline_ui_metadata, type: UI metadata}
        >>> ...
        >>> command: [
        >>>    python, -m, some_script.py,
        >>>    --mlpipeline_ui_metadata_path, {outputPath: mlpipeline_ui_metadata},

        some_script.py
        >>> if __name__ == "__main__":
        >>> parser = argparse.ArgumentParser()
        >>> parser.add_argument(
        ...    "--mlpipeline_ui_metadata_path",
        ...    type=str,
        ...    help="The file path to output the visualization metadata to",
        ... )
        >>> args = parser.parse_args()
        >>> outputs = KfpOutputViewers()
        >>> outputs.add_confusion_matrix(
        ...     confusion_matrix=array([[2, 0, 0], [0, 0, 1], [1, 0, 2]])
        ... )
        >>> outputs.write_to_output_viewer_path(args.mlpipeline_ui_metadata_path)

        https://www.kubeflow.org/docs/components/pipelines/sdk/output-viewer/
        """
        self.__outputs_list: List[Dict[str, Any]] = []

    def add_confusion_matrix(
        self, confusion_matrix: np.ndarray, vocab: Optional[List[str]] = None
    ) -> None:
        if vocab is None:
            vocab = [str(i) for i in range(confusion_matrix.shape[0])]
        data = []
        for target_index, target_row in enumerate(confusion_matrix):
            for predicted_index, count in enumerate(target_row):
                data.append((vocab[target_index], vocab[predicted_index], count))

        df_confusion_matrix = pd.DataFrame(
            data, columns=["target", "predicted", "count"]
        )
        confusion_matrix_str = df_confusion_matrix.to_csv(
            columns=["target", "predicted", "count"], header=False, index=False
        )
        self.__outputs_list.append(
            {
                "type": "confusion_matrix",
                "format": "csv",
                "schema": [
                    {"name": "target", "type": "CATEGORY"},
                    {"name": "predicted", "type": "CATEGORY"},
                    {"name": "count", "type": "NUMBER"},
                ],
                "source": confusion_matrix_str,
                "storage": "inline",
                # Convert vocab to string because for boolean values we want "True|False" to match csv data.
                "labels": list(map(str, vocab)),
            }
        )

    def write_to_output_viewer_path(self, path: LocalUri) -> None:
        metadata = {"outputs": self.__outputs_list}
        Path(path.uri).parent.mkdir(parents=True, exist_ok=True)
        with open(path.uri, "w") as f:
            json.dump(metadata, f)
