import unittest

from gigl.common.logger import Logger
from gigl.src.common.types.pb_wrappers.gbml_config import GbmlConfigPbWrapper
from gigl.src.training.v1.lib.training_process import generate_trainer_instance
from snapchat.research.gbml import gbml_config_pb2
from tests.test_assets.test_modeling_spec import TestModelingTaskSpec

logger = Logger()


class GnnTrainerTest(unittest.TestCase):
    def test_can_generate_instance(self):
        optim_lr = 10
        optim_weight_decay = 20
        num_epochs = -1

        trainer_config = gbml_config_pb2.GbmlConfig.TrainerConfig(
            trainer_cls_path=(
                "tests." "test_assets." "test_modeling_spec." "TestModelingTaskSpec"
            ),
            trainer_args={
                "optim_lr": str(optim_lr),
                "optim_weight_decay": str(optim_weight_decay),
                "num_epochs": str(num_epochs),
            },
        )
        gbml_config_pb = gbml_config_pb2.GbmlConfig(trainer_config=trainer_config)

        trainer_instance = generate_trainer_instance(
            gbml_config_pb_wrapper=GbmlConfigPbWrapper(gbml_config_pb=gbml_config_pb),
        )

        assert isinstance(
            trainer_instance, TestModelingTaskSpec
        ), "trainer_instance must be an instance of TestModelingTaskSpec"
        self.assertEqual(trainer_instance._optim_lr, optim_lr)
        self.assertEqual(trainer_instance._optim_weight_decay, optim_weight_decay)
        self.assertEqual(trainer_instance._num_epochs, num_epochs)
