from typing import Dict

# from gigl.common.services.kfp import KFPService
from gigl.common.types.wrappers.kfp_api import KfpTaskDetails

# TODO: This needs to update ot Vertex AI
# def get_task_details_from_kfp_pipeline(
#     kfp_service: KFPService, experiment_name: str, kfp_run_name: str
# ) -> Dict[str, KfpTaskDetails]:
#     pipeline_run_detail: Optional[
#         ApiRunDetailWrapper
#     ] = kfp_service.get_latest_run_with_name(
#         kfp_run_name=kfp_run_name, experiment_name=experiment_name
#     )
#     assert pipeline_run_detail is not None
#     return pipeline_run_detail.task_details_map


def assert_component_runtimes_match_expected_parameters(
    task_details_map: Dict[str, KfpTaskDetails],
    component_name_runtime_hr: Dict[str, int],
) -> None:
    for component_name, expected_runtime_hr in component_name_runtime_hr.items():
        relevant_task = task_details_map.get(component_name)
        if relevant_task is None:
            raise ValueError(
                f"Component {component_name} not found in pipeline runtime manifest: {task_details_map}"
            )

        t_start = relevant_task.started_at
        t_finish = relevant_task.finished_at
        runtime_sec = (t_finish - t_start).seconds
        expected_runtime_sec = expected_runtime_hr * 3600
        if runtime_sec > expected_runtime_sec:
            raise ValueError(
                f"Component {component_name} took longer than expected runtime of {expected_runtime_hr} hrs. Actual runtime was {t_finish- t_start}."
            )
