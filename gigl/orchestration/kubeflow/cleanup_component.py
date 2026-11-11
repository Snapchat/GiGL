"""KFP exit task that cancels the jobs a GiGL pipeline run launched.

The pipeline is compiled with this task in place of the email notification when
``ExitTask.CLEANUP`` is selected; see :func:`gigl.orchestration.kubeflow.kfp_pipeline.generate_pipeline`.
"""

from kfp import dsl


# The base image is a placeholder. ``add_task_resource_requirements`` sets the real
# GiGL cpu image at compile time, and that image already ships ``kfp``, so KFP
# must not pip install it at run time.
@dsl.component(base_image="python:3.11", install_kfp_package=False)
def cleanup_child_jobs_op(job_name: str, resource_config_uri: str) -> None:
    """Cancel every Vertex AI, Dataflow, and Dataproc child of ``job_name``.

    Runs on every pipeline exit, including cancels. A run that launched nothing
    exits cleanly.

    Args:
        job_name: The GiGL applied task identifier.
        resource_config_uri: URI of the resource config that gives project and region.
    """
    from gigl.common import UriFactory
    from gigl.src.common.job_cleanup import cancel_child_jobs

    cancel_child_jobs(
        job_name=job_name,
        resource_config_uri=UriFactory.create_uri(resource_config_uri),
    )
