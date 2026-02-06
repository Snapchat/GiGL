import torch.nn

import gigl.src.common.utils.model as model_utils
from gigl.common import UriFactory
from gigl.common.logger import Logger
from gigl.common.utils.os_utils import import_obj
from gigl.src.common.types.pb_wrappers.gbml_config import GbmlConfigPbWrapper
from gigl.src.common.types.task_metadata import TaskMetadataType
from snapchat.research.gbml import dataset_metadata_pb2, gbml_config_pb2

logger = Logger()


def train_model(
    gbml_config_pb: gbml_config_pb2.GbmlConfig,
):
    trainer_cls = import_obj(gbml_config_pb.trainer_config.trainer_cls_path)
    kwargs = dict(gbml_config_pb.trainer_config.trainer_args)
    trainer = trainer_cls(**kwargs)

    gbml_config_pb_wrapper = GbmlConfigPbWrapper(gbml_config_pb=gbml_config_pb)
    trainer.init_model(gbml_config_pb_wrapper=gbml_config_pb_wrapper)
    trainer.setup_for_training()

    dataset_metadata_pb_wrapper = gbml_config_pb_wrapper.dataset_metadata_pb_wrapper
    graph_metadata_pb_wrapper = gbml_config_pb_wrapper.graph_metadata_pb_wrapper
    task_metadata_pb_wrapper = gbml_config_pb_wrapper.task_metadata_pb_wrapper

    if task_metadata_pb_wrapper.task_metadata_type == TaskMetadataType.NODE_BASED_TASK:
        assert isinstance(
            dataset_metadata_pb_wrapper.output_metadata,
            dataset_metadata_pb2.SupervisedNodeClassificationDataset,
        ), f"Did not find {dataset_metadata_pb2.SupervisedNodeClassificationDataset.__name__} instance"
    elif (
        task_metadata_pb_wrapper.task_metadata_type
        == TaskMetadataType.NODE_ANCHOR_BASED_LINK_PREDICTION_TASK
    ):
        assert isinstance(
            dataset_metadata_pb_wrapper.output_metadata,
            dataset_metadata_pb2.NodeAnchorBasedLinkPredictionDataset,
        ), f"Did not find {dataset_metadata_pb2.NodeAnchorBasedLinkPredictionDataset.__name__} instance"
    else:
        raise NotImplementedError

    trainer.train(
        gbml_config_pb_wrapper=gbml_config_pb_wrapper,
        device=torch.device("cpu"),
    )

    model_save_path_uri = UriFactory.create_uri(
        gbml_config_pb.shared_config.trained_model_metadata.trained_model_uri
    )
    model_utils.save_state_dict(
        model=trainer.model, save_to_path_uri=model_save_path_uri
    )
    logger.info(f"Saved model to: {model_save_path_uri}.")
