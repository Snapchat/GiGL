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
    # Environment variables baked into every GiGL-owned container at compile time.
    # Applied uniformly across all SPECED_COMPONENTS plus the GLT eligibility check
    # and log_metrics_to_ui tasks. The managed VertexNotificationEmailOp is excluded.
    env_vars: dict[str, str] = field(default_factory=dict)
