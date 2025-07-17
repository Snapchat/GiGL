from typing import Callable, Optional, TypeVar

from kfp import dsl
from kfp.components import PythonComponent

_T = TypeVar("_T")


def _gigl_component(func: Callable, image: str, **kwargs) -> PythonComponent:
    """Decorator to create a GiGL component with a specified base image."""
    return dsl.component(
        func=func,
        base_image=image,
    )(**kwargs)


def ConfigValidator(base_image: str, **kwargs) -> PythonComponent:
    def config_validator(
        job_name: str,
        task_config_uri: str,
        start_at: str,
        resource_config_uri: str,
        stop_after: Optional[str] = None,
    ) -> None:
        from gigl.common import UriFactory
        from gigl.common.logger import Logger
        from gigl.src.validation_check.config_validator import kfp_validation_checks

        logger = Logger()
        logger.info("Starting config validation checks")
        kfp_validation_checks(
            job_name=job_name,
            task_config_uri=UriFactory.create_uri(task_config_uri),
            start_at=start_at,
            resource_config_uri=UriFactory.create_uri(resource_config_uri),
            stop_after=stop_after,
        )
        logger.info("Config validation checks completed successfully")

    return _gigl_component(config_validator, base_image, **kwargs)


def ConfigPopulator(base_image: str, **kwargs) -> PythonComponent:
    def config_populator(
        task_config_uri: str,
        resource_config_uri: str,
        job_name: str,
    ) -> str:
        from gigl.common import UriFactory
        from gigl.src.common.types import AppliedTaskIdentifier
        from gigl.src.config_populator.config_populator import ConfigPopulator

        config_populator = ConfigPopulator()
        frozen_gbml_config_uri = config_populator.run(
            applied_task_identifier=AppliedTaskIdentifier(job_name),
            task_config_uri=UriFactory.create_uri(task_config_uri),
            resource_config_uri=UriFactory.create_uri(resource_config_uri),
        )

        return frozen_gbml_config_uri.uri

    return _gigl_component(config_populator, base_image, **kwargs)


def SubgraphSampler(base_image: str, **kwargs) -> PythonComponent:
    def subgraph_sampler(
        task_config_uri: str,
        resource_config_uri: str,
        job_name: str,
        custom_worker_image_uri: str,
        **kwargs,
    ) -> None:
        from gigl.common import UriFactory
        from gigl.src.common.types import AppliedTaskIdentifier
        from gigl.src.subgraph_sampler.subgraph_sampler import SubgraphSampler

        subgraph_sampler = SubgraphSampler()
        subgraph_sampler.run(
            applied_task_identifier=AppliedTaskIdentifier(job_name),
            task_config_uri=UriFactory.create_uri(task_config_uri),
            resource_config_uri=UriFactory.create_uri(resource_config_uri),
            custom_worker_image_uri=custom_worker_image_uri,
            **kwargs,
        )

    return _gigl_component(subgraph_sampler, base_image, **kwargs)


# def Trainer
