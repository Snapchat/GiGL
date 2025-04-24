from kfp import dsl


def check_glt_backend_eligibility_component(
    task_config_uri: str, base_image: str
) -> bool:
    comp = dsl.component(
        func=_check_glt_backend_eligibility_component, base_image=base_image
    )
    comp.description = "Check whether to use GLT Backend"
    return comp(task_config_uri=task_config_uri).output


def _check_glt_backend_eligibility_component(
    task_config_uri: str,
) -> bool:
    """
    Used by KFP to check if GLT should be used as a backend for current run.
    Args:
        task_config_uri (str): Task config uri for current run
    Returns:
        bool: Whether to use GLT as a backend for current run ('True' or 'False')
    """

    # This is required to resolve below packages when containerized by KFP.
    import os
    import sys

    sys.path.append(os.getcwd())

    from gigl.common import UriFactory
    from gigl.src.common.types.pb_wrappers.gbml_config import GbmlConfigPbWrapper

    config = GbmlConfigPbWrapper.get_gbml_config_pb_wrapper_from_uri(
        gbml_config_uri=UriFactory.create_uri(task_config_uri)
    )
    return config.should_use_experimental_glt_backend
