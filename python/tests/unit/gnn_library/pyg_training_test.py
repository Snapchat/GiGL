from typing import Dict, List

from gigl.common.logger import Logger
from gigl.src.applied_tasks.test_tasks.academic import (
    get_pyg_cora_dataset,
    log_stats_for_pyg_planetoid_dataset,
)
from gigl.src.common.types.graph_data import NodeType

logger = Logger()

import tempfile
import unittest

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from gigl.src.common.models.layers.decoder import DecoderType, LinkPredictionDecoder
from gigl.src.common.models.pyg.homogeneous import Transformer
from gigl.src.common.models.pyg.link_prediction import LinkPredictionGNN


class GCN(torch.nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_feats, h_feats)
        self.conv2 = GCNConv(h_feats, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        return x


class PygTrainingTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.__tmp_dir = tempfile.TemporaryDirectory()
        dataset = get_pyg_cora_dataset(store_at=self.__tmp_dir.name)
        log_stats_for_pyg_planetoid_dataset(dataset=dataset)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        data = dataset[0]

        self.data = data.to(self.device)
        self.node_types: List[NodeType] = [NodeType("test_node_type")]
        model = GCN(self.data.x.shape[1], 16, dataset.num_classes)
        self.model = model.to(device)
        logger.info(self.model)

        lp_model = LinkPredictionGNN(
            encoder=Transformer(in_dim=self.data.x.shape[1], hid_dim=16, out_dim=128),
            decoder=LinkPredictionDecoder(
                decoder_type=DecoderType.hadamard_MLP, decoder_channel_list=[128, 64, 1]
            ),
        )
        self.lp_model = lp_model.to(device)
        logger.info(self.lp_model)

        self.epochs = 10
        self.lr = 0.001

    def tearDown(self) -> None:
        super().tearDown()
        self.__tmp_dir.cleanup()

    def _train_model(self):
        self.model.train()
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr
        )  # Define optimizer.
        best_val_acc = 0
        best_test_acc = 0

        self.model.train()

        features = self.data.x
        edge_index = self.data.edge_index
        labels = self.data.y
        train_mask = self.data.train_mask
        val_mask = self.data.val_mask
        test_mask = self.data.test_mask

        for e in range(self.epochs):
            optimizer.zero_grad()
            logits = self.model(features, edge_index)
            pred = logits.argmax(1)
            loss = F.cross_entropy(logits[train_mask], labels[train_mask])
            train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
            val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
            test_acc = (pred[test_mask] == labels[test_mask]).float().mean()
            if best_val_acc < val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if e % 3 == 0:
                logger.info(
                    f"In epoch {e}, loss: {loss:.3f}, val acc: {val_acc:.3f} (best {best_val_acc:.3f}),"
                    f" test acc: {test_acc:.3f} (best {best_test_acc:.3f})"
                )

            self.assertGreater(best_test_acc, 0)

    def _train_step(self):
        features = self.data
        labels = self.data.y
        train_mask = self.data.train_mask
        optim = torch.optim.Adam(
            self.lp_model.parameters(), lr=self.lr
        )  # Define optimizer.

        # put model in train mode
        self.lp_model.train()
        out: Dict[NodeType, torch.Tensor] = self.lp_model(
            features, self.node_types, self.device
        )
        encoder_output: torch.Tensor = list(out.values())[0]

        decoder_input_1 = torch.clone(encoder_output)
        indexes = torch.randperm(decoder_input_1.shape[0])
        decoder_input_2 = decoder_input_1[indexes]

        scores = self.lp_model.decode(decoder_input_1, decoder_input_2)
        loss = F.cross_entropy(scores[train_mask], labels[train_mask])

        optim.zero_grad()
        loss.backward()
        optim.step()

    def test_pyg_training(self):
        self._train_model()

    def test_var_change(self):
        """Check if LinkPredictionGNN model params change during training"""

        params = [np for np in self.lp_model.named_parameters() if np[1].requires_grad]
        initial_params = [(name, p.clone()) for (name, p) in params]

        # run a training step
        self._train_step()

        # check if variables have changed
        for (_, p0), (name, p1) in zip(initial_params, params):
            if torch.equal(p0.to(self.device), p1.to(self.device)):
                raise AssertionError(
                    "{var_name} {msg}".format(var_name=name, msg="did not change")
                )
