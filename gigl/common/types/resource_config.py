from __future__ import annotations

from dataclasses import dataclass, field

from gigl.src.common.constants.components import GiGLComponents


@dataclass
class CommonPipelineComponentConfigs:
    cuda_container_image: str
    cpu_container_image: str
    dataflow_container_image: str
    # Additional job arguments for the pipeline components, by component.
    # Only SubgraphSampler supports additional_job_args, for now.
    additional_job_args: dict[GiGLComponents, dict[str, str]] = field(
        default_factory=dict
    )
