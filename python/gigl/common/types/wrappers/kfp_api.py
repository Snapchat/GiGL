import datetime
from typing import Dict, Optional

import kfp_server_api
from kfp_server_api.models.v2beta1_pipeline_task_detail import V2beta1PipelineTaskDetail
from kfp_server_api.models.v2beta1_run_details import V2beta1RunDetails
from kfp_server_api.models.v2beta1_runtime_config import V2beta1RuntimeConfig

from gigl.common.logger import Logger
from gigl.common.utils.func_tools import lru_cache

logger = Logger()


class KfpTaskDetails:
    """Convenience class to access relevant task specific properties more easily."""

    def __init__(self, pipeline_task_details: V2beta1PipelineTaskDetail) -> None:
        self._pipeline_task_details: V2beta1PipelineTaskDetail = pipeline_task_details

    @property
    def pod_name(self) -> Optional[str]:
        pod_name = self._pipeline_task_details.pod_name
        if pod_name is None:  # Get child pod name instead
            if len(self._pipeline_task_details.child_tasks) == 1:
                pod_name = self._pipeline_task_details.child_tasks[0].pod_name
            else:
                logger.warning(
                    f"Multiple child tasks found. Unable to determine pod name for pipeline_task: {self._pipeline_task_details}."
                )
        return pod_name

    @property
    def display_name(self) -> str:
        return self._pipeline_task_details.display_name

    @property
    def finished_at(self) -> datetime.datetime:
        return self._pipeline_task_details.end_time

    @property
    def started_at(self) -> datetime.datetime:
        return self._pipeline_task_details.start_time

    def __repr__(self) -> str:
        return f"""
        Task Details:
        - Display Name: {self.display_name}
        - Pod Name: {self.pod_name}
        - Started At: {self.started_at}
        - Finished At: {self.finished_at}
        - V2beta1PipelineTaskDetail: {self._pipeline_task_details}
        """


class ApiRunDetailWrapper:
    def __init__(self, api_run: kfp_server_api.V2beta1Run) -> None:
        self._api_run = api_run

    @property
    def api_run(self) -> kfp_server_api.V2beta1Run:
        return self._api_run

    @property
    def created_at(self) -> datetime.datetime:
        return self.api_run.created_at

    @property
    def finished_at(self) -> datetime.datetime:
        return self.api_run.finished_at

    @property
    @lru_cache(maxsize=1)
    def job_parameters(self) -> Dict[str, str]:
        parameters_dict: Dict[str, str] = {}

        runtime_config: V2beta1RuntimeConfig = self.api_run.runtime_config
        for name, val in runtime_config.parameters.items():
            parameters_dict[name] = val

        return parameters_dict

    @property
    @lru_cache(maxsize=1)
    def task_details_map(
        self,
    ) -> Dict[str, KfpTaskDetails]:
        """

        Returns:
            Dict[str, KfpTaskDetails]: Note that the keys are the display name of the KFP component
        """
        task_details_dict: Dict[str, KfpTaskDetails] = {}
        run_details: V2beta1RunDetails = self.api_run.run_details
        task: V2beta1PipelineTaskDetail
        for task in run_details.task_details:
            task_details_dict[task.display_name] = KfpTaskDetails(task)
        return task_details_dict
