import argparse
import contextlib
import multiprocessing as mp
import sys
import tempfile
import traceback
from distutils.util import strtobool
from typing import Any, Dict, Optional

import tensorflow as tf
import torch
import torch.distributed
import torch.nn.parallel

import gigl.src.common.utils.model as model_utils
from gigl.common import GcsUri, LocalUri, Uri, UriFactory
from gigl.common.logger import Logger
from gigl.common.metrics.decorators import flushes_metrics, profileit
from gigl.common.utils import os_utils, torch_training
from gigl.common.utils.local_fs import does_path_exist
from gigl.common.utils.torch_training import (
    get_distributed_backend,
    get_rank,
    is_distributed_available_and_initialized,
    should_distribute,
)
from gigl.src.common.constants.metrics import (
    TIMER_TRAINER_CLEANUP_ENV_S,
    TIMER_TRAINER_EXPORT_INFERENCE_ASSETS_S,
    TIMER_TRAINER_S,
    TIMER_TRAINER_SETUP_ENV_S,
)
from gigl.src.common.modeling_task_specs.utils.profiler_wrapper import (
    TMP_PROFILER_LOG_DIR_NAME,
    TorchProfiler,
)
from gigl.src.common.translators.model_eval_metrics_translator import (
    EvalMetricsCollectionTranslator,
)
from gigl.src.common.types.model_eval_metrics import EvalMetricsCollection
from gigl.src.common.types.pb_wrappers.gbml_config import GbmlConfigPbWrapper
from gigl.src.common.types.task_metadata import TaskMetadataType
from gigl.src.common.utils.file_loader import FileLoader
from gigl.src.common.utils.metrics_service_provider import (
    get_metrics_service_instance,
    initialize_metrics,
)
from gigl.src.common.utils.model import load_state_dict_from_uri
from gigl.src.common.utils.time import current_formatted_datetime
from gigl.src.training.v1.lib.base_trainer import BaseTrainer

logger = Logger()


@profileit(
    metric_name=TIMER_TRAINER_EXPORT_INFERENCE_ASSETS_S,
    get_metrics_service_instance_fn=get_metrics_service_instance,
)
def save_model(trainer: BaseTrainer, gbml_config_pb_wrapper: GbmlConfigPbWrapper):
    if get_rank() == 0:  # if not distributed, get_rank returns 0 by default
        model_save_path_uri = UriFactory.create_uri(
            gbml_config_pb_wrapper.shared_config.trained_model_metadata.trained_model_uri
        )
        scripted_model_save_path_uri = UriFactory.create_uri(
            gbml_config_pb_wrapper.shared_config.trained_model_metadata.scripted_model_uri
        )
        logger.info(
            f"Saving model to: {model_save_path_uri}, and scripted model to: {scripted_model_save_path_uri}"
        )

        model_utils.save_state_dict(
            model=trainer.model, save_to_path_uri=model_save_path_uri
        )

    if is_distributed_available_and_initialized():
        torch.distributed.barrier()
    # TODO: (svij-sc) Investigate if we can enable scripted model (torch.compile())
    # TODO: (yliu2-sc) If we enable scripted model, we will need to update config populator code for pretrained pipelines
    # model_utils.save_model(
    #     model=trainer.model,
    #     save_to_path_uri=scripted_model_save_path_uri,
    #     is_scripted=True,
    # )


def setup_model_device(
    model: torch.nn.Module,
    supports_distributed_training: bool,
    should_enable_find_unused_parameters: bool,
    device: torch.device,
):
    """
    Configures the model by setting it on device, syncing batch norm, and wrapping the model with DDP with the relevant flags, such as find_unused_parameters
    Args:
        model (torch.nn.Module): Model initialized for training
        supports_distributed_training (bool): Whether distributed training is supported, defined in the modeling task spec
        should_enable_find_unused_parameters (bool): Whether we allow for parameters to not receive gradient on backward pass in DDP
        device (torch.device): Torch device to set the model to
    """
    model = model.to(device=device)
    if is_distributed_available_and_initialized():
        if not supports_distributed_training:
            raise ValueError(
                f"Trying to instantiate distributed training with trainer instance that does not support distributed training"
            )
        # If device is 'cpu', DDP requires device_ids=None.
        device_ids = None if device == torch.device("cpu") else [device]
        # replace nn.BatchNormxD layers with its synced counterpart
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # set broadcast_buffers=False to avoid autograd error from BatchNorm
        # see https://github.com/pytorch/pytorch/issues/22095
        # and https://discuss.pytorch.org/t/distributeddataparallel-broadcast-buffers/21088 for details
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=device_ids,
            broadcast_buffers=False,
            find_unused_parameters=should_enable_find_unused_parameters,
        )
    return model


def generate_trainer_instance(
    gbml_config_pb_wrapper: GbmlConfigPbWrapper,
) -> BaseTrainer:
    kwargs: Dict[str, Any] = {}

    trainer_class_path: str = gbml_config_pb_wrapper.trainer_config.trainer_cls_path
    kwargs = dict(gbml_config_pb_wrapper.trainer_config.trainer_args)
    trainer: BaseTrainer
    try:
        trainer_cls = os_utils.import_obj(trainer_class_path)
        trainer = trainer_cls(**kwargs)
        assert isinstance(trainer, BaseTrainer)
    except Exception as e:
        logger.error(
            f"Could not instantiate class {trainer_class_path} with args {kwargs}: {e}"
        )
        raise e
    return trainer


def get_torch_profiler_instance(
    gbml_config_pb_wrapper: GbmlConfigPbWrapper,
) -> Optional[TorchProfiler]:
    should_enable_profiler = (
        gbml_config_pb_wrapper.profiler_config.should_enable_profiler
    )
    profiler_kwargs = dict(gbml_config_pb_wrapper.profiler_config.profiler_args)
    profiler = TorchProfiler(**profiler_kwargs) if should_enable_profiler else None
    return profiler


class GnnTrainingProcess:
    def __write_model_eval_metrics_to_uri(
        self,
        model_eval_metrics: EvalMetricsCollection,
        gbml_config_pb_wrapper: GbmlConfigPbWrapper,
    ):
        file_loader = FileLoader()
        tfh = tempfile.NamedTemporaryFile(delete=False)
        local_tfh_uri = LocalUri(tfh.name)
        eval_metrics_uri = UriFactory.create_uri(
            uri=gbml_config_pb_wrapper.shared_config.trained_model_metadata.eval_metrics_uri
        )

        EvalMetricsCollectionTranslator.write_kfp_metrics_to_pipeline_metric_path(
            eval_metrics=model_eval_metrics, path=local_tfh_uri
        )
        file_loader.load_file(file_uri_src=local_tfh_uri, file_uri_dst=eval_metrics_uri)
        logger.info(f"Wrote eval metrics to {eval_metrics_uri.uri}.")

    def __run_model_evaluation(
        self,
        trainer_instance: BaseTrainer,
        gbml_config_pb_wrapper: GbmlConfigPbWrapper,
        device: torch.device,
    ):
        model_eval_metrics: EvalMetricsCollection = trainer_instance.eval(
            gbml_config_pb_wrapper=gbml_config_pb_wrapper,
            device=device,
        )

        logger.info(f"Got model eval metrics: {model_eval_metrics}")

        # flushing offline metrics to Grafana dashboard
        metrics_instance = get_metrics_service_instance()
        if metrics_instance is not None:
            if (
                gbml_config_pb_wrapper.task_metadata_pb_wrapper.task_metadata_type
                == TaskMetadataType.NODE_ANCHOR_BASED_LINK_PREDICTION_TASK
            ):
                for metric in model_eval_metrics.metrics.values():
                    metrics_instance.add_gauge(
                        metric_name=metric.name, gauge=metric.value
                    )
                metrics_instance.flush_metrics()

        # flushing offline metrics to kubeflow pipelines
        self.__write_model_eval_metrics_to_uri(
            model_eval_metrics=model_eval_metrics,
            gbml_config_pb_wrapper=gbml_config_pb_wrapper,
        )

    def __run_training(
        self,
        trainer_instance: BaseTrainer,
        gbml_config_pb_wrapper: GbmlConfigPbWrapper,
        device: torch.device,
    ):
        trainer_instance.setup_for_training()
        logger.info(f"Starting training at {current_formatted_datetime()}")
        tensorboard_log_uri = (
            gbml_config_pb_wrapper.shared_config.trained_model_metadata.tensorboard_logs_uri
        )
        profiler = get_torch_profiler_instance(
            gbml_config_pb_wrapper=gbml_config_pb_wrapper
        )

        file_writer = None
        if gbml_config_pb_wrapper.trainer_config.should_log_to_tensorboard:
            file_writer = tf.summary.create_file_writer(tensorboard_log_uri)

        with file_writer.as_default() if file_writer else contextlib.nullcontext():
            with profiler.profiler_context() if profiler else contextlib.nullcontext() as prof:  # type: ignore
                trainer_instance.train(
                    gbml_config_pb_wrapper=gbml_config_pb_wrapper,
                    device=device,
                    profiler=prof,
                )
        if profiler:
            if does_path_exist(TMP_PROFILER_LOG_DIR_NAME):
                file_loader = FileLoader()
                profiler_specified_dir = (
                    gbml_config_pb_wrapper.profiler_config.profiler_log_dir
                )
                file_loader.load_directory(
                    dir_uri_src=TMP_PROFILER_LOG_DIR_NAME,
                    dir_uri_dst=(
                        GcsUri(profiler_specified_dir)
                        if GcsUri.is_valid(profiler_specified_dir)
                        else LocalUri(profiler_specified_dir)
                    ),
                )
            else:
                logger.info(
                    f"Profiler logs dir not found at {TMP_PROFILER_LOG_DIR_NAME}. Did profiler run successfully?"
                )
        save_model(
            trainer=trainer_instance, gbml_config_pb_wrapper=gbml_config_pb_wrapper
        )
        logger.info(f"Finished training at {current_formatted_datetime()}")

    def __run(
        self,
        task_config_uri: Uri,
        device: torch.device,
    ):
        gbml_config_pb_wrapper = (
            GbmlConfigPbWrapper.get_gbml_config_pb_wrapper_from_uri(
                gbml_config_uri=task_config_uri
            )
        )

        trainer_instance: BaseTrainer = generate_trainer_instance(
            gbml_config_pb_wrapper=gbml_config_pb_wrapper,
        )

        if gbml_config_pb_wrapper.shared_config.should_skip_training:
            pretrained_model_uri = (
                gbml_config_pb_wrapper.shared_config.trained_model_metadata.trained_model_uri
            )
            logger.info(
                f"Skip training. Load pretrained model from {pretrained_model_uri}"
            )
            model_state_dict = load_state_dict_from_uri(
                load_from_uri=UriFactory.create_uri(pretrained_model_uri)
            )
            model = trainer_instance.init_model(
                gbml_config_pb_wrapper=gbml_config_pb_wrapper,
                state_dict=model_state_dict,
            )
        else:
            model = trainer_instance.init_model(
                gbml_config_pb_wrapper=gbml_config_pb_wrapper,
            )

        trainer_args = gbml_config_pb_wrapper.trainer_config.trainer_args

        # If enabled, will initialize DDP with the find_unused_param flag as True, allowing for parameters to receive no gradient in the backwards pass.
        # This can occur in cases like conditional computation (https://intellabs.github.io/distiller/conditional_computation.html).
        # This can also occur when training on a heterogeneous graph when there is an expected node type missing from the batched input graph.
        # If all parameters are always expected to receive backprop in training, it is not recommended to enable this flag, as it can adversely affect
        # performance as a result of the extra traversal of the autograd graph every iteration.
        should_enable_find_unused_parameters = bool(
            strtobool(trainer_args.get("should_enable_find_unused_parameters", "False"))
        )

        trainer_instance.model = setup_model_device(
            model=model,
            supports_distributed_training=trainer_instance.supports_distributed_training,
            should_enable_find_unused_parameters=should_enable_find_unused_parameters,
            device=device,
        )

        # run training if not pretrained skip training
        if not gbml_config_pb_wrapper.shared_config.should_skip_training:
            self.__run_training(
                trainer_instance=trainer_instance,
                gbml_config_pb_wrapper=gbml_config_pb_wrapper,
                device=device,
            )
            if gbml_config_pb_wrapper.shared_config.should_skip_model_evaluation:
                logger.warning(
                    "Warning, should_skip_model_evaluation is set to "
                    + f"{gbml_config_pb_wrapper.shared_config.should_skip_model_evaluation}. "
                    + "We will skip evaluation. Are you sure you wanted this?"
                )

        if not gbml_config_pb_wrapper.shared_config.should_skip_model_evaluation:
            self.__run_model_evaluation(
                trainer_instance=trainer_instance,
                gbml_config_pb_wrapper=gbml_config_pb_wrapper,
                device=device,
            )

    @flushes_metrics(get_metrics_service_instance_fn=get_metrics_service_instance)
    @profileit(
        metric_name=TIMER_TRAINER_S,
        get_metrics_service_instance_fn=get_metrics_service_instance,
    )
    def run(self, task_config_uri: Uri, device: torch.device):
        try:
            self.__setup_training_env(device=device)
            self.__run(
                task_config_uri=task_config_uri,
                device=device,
            )
            self.__cleanup_training_env()

        except Exception as e:
            logger.error("Training failed due to a raised exception; which will follow")
            logger.error(e)
            logger.error(traceback.format_exc())
            logger.info("Cleaning up training environment...")
            self.__cleanup_training_env()
            sys.exit(f"System will now exit: {e}")

    @flushes_metrics(get_metrics_service_instance_fn=get_metrics_service_instance)
    @profileit(
        metric_name=TIMER_TRAINER_SETUP_ENV_S,
        get_metrics_service_instance_fn=get_metrics_service_instance,
    )
    def __setup_training_env(self, device: torch.device):
        use_cuda = device.type != "cpu"
        if should_distribute():
            distributed_backend = get_distributed_backend(use_cuda=use_cuda)
            logger.info(f"Using distributed PyTorch with {distributed_backend}")
            torch.distributed.init_process_group(backend=distributed_backend)
            logger.info("Successfully initiated distributed backend!")

    @flushes_metrics(get_metrics_service_instance_fn=get_metrics_service_instance)
    @profileit(
        metric_name=TIMER_TRAINER_CLEANUP_ENV_S,
        get_metrics_service_instance_fn=get_metrics_service_instance,
    )
    def __cleanup_training_env(self):
        logger.info("Cleaning up training environment.")
        if is_distributed_available_and_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    # Bug Fix: Random deadlocks with dataloaders
    # DDP Note (https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)
    # "If you plan on using this module with a nccl backend or a gloo backend (that uses Infiniband), together with a DataLoader that uses multiple workers, please change the multiprocessing start method to forkserver (Python 3 only) or spawn. Unfortunately Gloo (that uses Infiniband) and NCCL2 are not fork safe, and you will likely experience deadlocks if you don’t change this setting."
    mp.set_start_method("spawn")

    parser = argparse.ArgumentParser(description="Program to train a GBML model")
    parser.add_argument(
        "--job_name",
        type=str,
        help="Unique identifier for the job name",
    )
    parser.add_argument(
        "--task_config_uri",
        type=str,
        help="Gbml config uri",
    )
    parser.add_argument(
        "--use_cuda",
        action="store_true",
        default=False,
        help="Dictates whether or not to use CUDA training",
    )
    parser.add_argument(
        "--resource_config_uri",
        type=str,
        help="Runtime argument for resource and env specifications of each component",
    )
    args = parser.parse_args()

    if not args.job_name or not args.task_config_uri or not args.resource_config_uri:
        raise RuntimeError("Missing command-line arguments")

    use_cuda = args.use_cuda and torch.cuda.is_available()
    if use_cuda:
        logger.info("Using CUDA")
    device = torch.device("cuda" if use_cuda else "cpu")
    logger.info(f"Starting training with device: {device}")

    task_config_uri = UriFactory.create_uri(args.task_config_uri)
    logger.info(f"Will use the following config for training: {task_config_uri}")
    logger.info(
        f"World Size: {torch_training.get_world_size()}, Rank: {torch_training.get_rank()}, Should Distribute: {torch_training.should_distribute()}"
    )

    initialize_metrics(task_config_uri=task_config_uri, service_name=args.job_name)

    training_process = GnnTrainingProcess()
    training_process.run(task_config_uri=task_config_uri, device=device)
