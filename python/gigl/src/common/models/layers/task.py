import copy
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
from torch_geometric.nn import GraphConv

from gigl.common.logger import Logger
from gigl.src.common.modeling_task_specs.utils.infer import (  # type: ignore
    infer_root_embeddings,
    infer_training_batch,
)
from gigl.src.common.models.layers.count_min_sketch import (
    CountMinSketch,
    calculate_in_batch_candidate_sampling_probability,
)
from gigl.src.common.models.layers.loss import (
    AligmentLoss,
    BGRLLoss,
    FeatureReconstructionLoss,
    GBTLoss,
    GRACELoss,
    MarginLoss,
    ModelResultType,
    RetrievalLoss,
    SoftmaxLoss,
    TBGRLLoss,
    UniformityLoss,
    WhiteningDecorrelationLoss,
)
from gigl.src.common.models.pyg.graph.augmentations import (  # type: ignore
    get_augmented_graph,
)
from gigl.src.common.types.pb_wrappers.gbml_config import GbmlConfigPbWrapper
from gigl.src.common.types.task_inputs import NodeAnchorBasedLinkPredictionTaskInputs

logger = Logger()


class NodeAnchorBasedLinkPredictionBaseTask(ABC, nn.Module):
    @abstractmethod
    def forward(
        self,
        task_input: NodeAnchorBasedLinkPredictionTaskInputs,
        gbml_config_pb_wrapper: GbmlConfigPbWrapper,
        should_eval: bool,
        device: torch.device,
    ) -> Tuple[torch.Tensor, int]:
        raise NotImplementedError

    @property
    @abstractmethod
    def result_types(self) -> List[ModelResultType]:
        raise NotImplementedError

    @property
    def task_name(self) -> str:
        return self.__class__.__name__


class Softmax(NodeAnchorBasedLinkPredictionBaseTask):
    def __init__(
        self,
        softmax_temperature: float = 0.07,
    ):
        super(Softmax, self).__init__()
        self.loss = SoftmaxLoss(softmax_temperature=softmax_temperature)

    def forward(
        self,
        task_input: NodeAnchorBasedLinkPredictionTaskInputs,
        gbml_config_pb_wrapper: GbmlConfigPbWrapper,
        should_eval: bool,
        device: torch.device,
    ) -> Tuple[torch.Tensor, int]:
        assert len(task_input.batch_scores) > 0
        return self.loss(loss_input=task_input.batch_scores, device=device)

    @property
    def result_types(self) -> List[ModelResultType]:
        return [ModelResultType.batch_scores]


class Margin(NodeAnchorBasedLinkPredictionBaseTask):
    def __init__(
        self,
        margin: float = 0.5,
    ):
        super(Margin, self).__init__()
        self.loss = MarginLoss(margin=margin)

    def forward(
        self,
        task_input: NodeAnchorBasedLinkPredictionTaskInputs,
        gbml_config_pb_wrapper: GbmlConfigPbWrapper,
        should_eval: bool,
        device: torch.device,
    ) -> Tuple[torch.Tensor, int]:
        assert len(task_input.batch_scores) > 0
        return self.loss(loss_input=task_input.batch_scores, device=device)

    @property
    def result_types(self) -> List[ModelResultType]:
        return [ModelResultType.batch_scores]


class Retrieval(NodeAnchorBasedLinkPredictionBaseTask):
    def __init__(
        self,
        loss: Optional[nn.Module] = None,
        temperature: float = 0.07,
        remove_accidental_hits: bool = True,
        should_enable_candidate_sampling_correction: bool = False,
        count_min_sketch_width: int = 10000,
        count_min_sketch_depth: int = 10,
    ):
        super(Retrieval, self).__init__()
        self.should_enable_candidate_sampling_correction = (
            should_enable_candidate_sampling_correction
        )
        self.loss = RetrievalLoss(
            loss=loss,
            temperature=temperature,
            remove_accidental_hits=remove_accidental_hits,
        )
        if should_enable_candidate_sampling_correction:
            self.main_batch_cm_sketch = CountMinSketch(
                width=count_min_sketch_width,
                depth=count_min_sketch_depth,
            )
            self.random_neg_batch_cm_sketch = CountMinSketch(
                width=count_min_sketch_width,
                depth=count_min_sketch_depth,
            )
            logger.info(
                f"Retrieval loss is included in the tasks and candidate sampling correction is enabled, creating CountMinSketch objects for main and random negative batches with width = {count_min_sketch_width} and depth = {count_min_sketch_depth}"
            )

    def forward(
        self,
        task_input: NodeAnchorBasedLinkPredictionTaskInputs,
        gbml_config_pb_wrapper: GbmlConfigPbWrapper,
        should_eval: bool,
        device: torch.device,
    ) -> Tuple[torch.Tensor, int]:
        assert len(task_input.batch_combined_scores) > 0
        assert task_input.batch_embeddings is not None
        running_loss = torch.tensor(0.0, device=device)
        running_batch_size = 0
        for condensed_edge_type in task_input.batch_combined_scores:
            batch_combined_scores = task_input.batch_combined_scores[
                condensed_edge_type
            ]
            if self.should_enable_candidate_sampling_correction and not should_eval:
                positive_ids = batch_combined_scores.positive_ids
                hard_neg_ids = batch_combined_scores.hard_neg_ids
                random_neg_ids = batch_combined_scores.random_neg_ids

                # Compute the candidate sampling probability for each candidate node.
                self.main_batch_cm_sketch.add_torch_long_tensor(positive_ids)
                self.main_batch_cm_sketch.add_torch_long_tensor(hard_neg_ids)
                self.random_neg_batch_cm_sketch.add_torch_long_tensor(random_neg_ids)
                # Batch size is the sum of the number of positive, hard negative because they share the same cm sketch
                positive_candidate_sampling_probability = calculate_in_batch_candidate_sampling_probability(
                    frequency_tensor=self.main_batch_cm_sketch.estimate_torch_long_tensor(
                        positive_ids
                    ),
                    total_cnt=self.main_batch_cm_sketch.total(),
                    batch_size=positive_ids.numel() + hard_neg_ids.numel(),
                )
                hard_neg_candidate_sampling_probability = calculate_in_batch_candidate_sampling_probability(
                    frequency_tensor=self.main_batch_cm_sketch.estimate_torch_long_tensor(
                        hard_neg_ids
                    ),
                    total_cnt=self.main_batch_cm_sketch.total(),
                    batch_size=positive_ids.numel() + hard_neg_ids.numel(),
                )
                random_neg_candidate_sampling_probability = calculate_in_batch_candidate_sampling_probability(
                    frequency_tensor=self.random_neg_batch_cm_sketch.estimate_torch_long_tensor(
                        random_neg_ids
                    ),
                    total_cnt=self.random_neg_batch_cm_sketch.total(),
                    batch_size=random_neg_ids.numel(),
                )
                candidate_sampling_probability = torch.cat(
                    (
                        positive_candidate_sampling_probability,
                        hard_neg_candidate_sampling_probability,
                        random_neg_candidate_sampling_probability,
                    )
                ).to(device=device)
            else:
                candidate_sampling_probability = None
            loss, batch_size = self.loss(
                batch_combined_scores=batch_combined_scores,
                repeated_query_embeddings=task_input.batch_embeddings.repeated_query_embeddings[
                    condensed_edge_type
                ],
                candidate_sampling_probability=candidate_sampling_probability,
                device=device,
            )
            running_loss += loss
            running_batch_size += batch_size
        return running_loss, running_batch_size

    @property
    def result_types(self) -> List[ModelResultType]:
        return [ModelResultType.batch_combined_scores, ModelResultType.batch_embeddings]


class GRACE(NodeAnchorBasedLinkPredictionBaseTask):
    """
    Creates 2 augmented views of input graph with augmentations 1 and 2 and defines task-specific linear head for GRACE Loss
    """

    def __init__(
        self,
        encoder: nn.Module,
        temperature: float = 0.001,
        feat_drop_1: float = 0.3,
        edge_drop_1: float = 0.3,
        feat_drop_2: float = 0.3,
        edge_drop_2: float = 0.3,
    ):
        super(GRACE, self).__init__()
        self.encoder = encoder
        hid_dim = self.encoder.hid_dim
        out_dim = self.encoder.out_dim
        self.head = torch.nn.Sequential(
            torch.nn.Linear(out_dim, hid_dim),  # type: ignore
            torch.nn.ReLU(),
            torch.nn.Linear(hid_dim, out_dim),  # type: ignore
        )
        self.loss = GRACELoss(temperature=temperature)
        self.feat_drop_1 = feat_drop_1
        self.edge_drop_1 = edge_drop_1
        self.feat_drop_2 = feat_drop_2
        self.edge_drop_2 = edge_drop_2

    def forward(
        self,
        task_input: NodeAnchorBasedLinkPredictionTaskInputs,
        gbml_config_pb_wrapper: GbmlConfigPbWrapper,
        should_eval: bool,
        device: torch.device,
    ) -> Tuple[torch.Tensor, int]:
        main_batch = task_input.input_batch.main_batch
        augmented_graph_1 = get_augmented_graph(
            graph=main_batch.graph.to(device=device),
            edge_drop_ratio=self.edge_drop_1,  # type: ignore
            feat_drop_ratio=self.edge_drop_2,  # type: ignore
        )
        augmented_embeddings_1 = infer_root_embeddings(
            model=self.encoder,
            graph=augmented_graph_1,
            root_node_indices=main_batch.root_node_indices,
            gbml_config_pb_wrapper=gbml_config_pb_wrapper,
            device=device,
        )
        augmented_graph_2 = get_augmented_graph(
            graph=main_batch.graph.to(device=device),
            edge_drop_ratio=self.edge_drop_2,  # type: ignore
            feat_drop_ratio=self.feat_drop_2,  # type: ignore
        )
        augmented_embeddings_2 = infer_root_embeddings(
            model=self.encoder,
            graph=augmented_graph_2,
            root_node_indices=main_batch.root_node_indices,
            gbml_config_pb_wrapper=gbml_config_pb_wrapper,
            device=device,
        )
        h1 = self.head(augmented_embeddings_1)
        h2 = self.head(augmented_embeddings_2)
        return self.loss(h1=h1, h2=h2, device=device)

    @property
    def result_types(self) -> List[ModelResultType]:
        return [ModelResultType.input_batch]


class FeatureReconstruction(NodeAnchorBasedLinkPredictionBaseTask):
    """
    Masks out percentage of anchor nodes' features before attempting to recreate these embeddings
    """

    def __init__(
        self,
        encoder: nn.Module,
        alpha: float = 3.0,
        edge_drop: float = 0.3,
    ):
        super(FeatureReconstruction, self).__init__()
        self.encoder = encoder
        in_dim = self.encoder.in_dim
        out_dim = self.encoder.out_dim
        self.loss = FeatureReconstructionLoss(alpha=alpha)
        self.reconstruction_decoder = GraphConv(out_dim, in_dim)
        self.reconstruction_mask = torch.nn.Parameter(torch.zeros(1, in_dim))  # type: ignore
        self.reconstruction_enc_dec = torch.nn.Linear(out_dim, out_dim, bias=False)  # type: ignore
        self.edge_drop = edge_drop

    def forward(
        self,
        task_input: NodeAnchorBasedLinkPredictionTaskInputs,
        gbml_config_pb_wrapper: GbmlConfigPbWrapper,
        should_eval: bool,
        device: torch.device,
    ) -> Tuple[torch.Tensor, int]:
        # TODO (mkolodner) Update GraphMAE logic to work in both heterogeneous use case
        if gbml_config_pb_wrapper.graph_metadata_pb_wrapper.is_heterogeneous:
            raise NotImplementedError(
                "GraphMAE is not yet supported with heterogeneous graphs"
            )
        condensed_node_type = (
            gbml_config_pb_wrapper.graph_metadata_pb_wrapper.condensed_node_types[0]
        )
        main_batch = task_input.input_batch.main_batch
        augmented_graph = get_augmented_graph(
            graph=main_batch.graph.to(device=device),
            edge_drop_ratio=self.edge_drop,  # type: ignore
            feat_drop_ratio=0.0,
        )
        root_node_indices = main_batch.root_node_indices
        x_target = augmented_graph.x[root_node_indices].clone()
        augmented_clone = augmented_graph.clone()
        augmented_clone.x[root_node_indices] = 0
        augmented_clone.x[root_node_indices] += self.reconstruction_mask
        h = infer_training_batch(
            model=self.encoder,
            training_batch=augmented_clone,
            gbml_config_pb_wrapper=gbml_config_pb_wrapper,
            device=device,
        )[condensed_node_type]
        h = self.reconstruction_enc_dec(h)
        h[root_node_indices] = 0
        x_pred = self.reconstruction_decoder(h, augmented_graph.edge_index)[
            root_node_indices
        ]
        loss_obj = self.loss(x_target=x_target, x_pred=x_pred)
        return (
            loss_obj[0],
            loss_obj[1],
        )

    @property
    def result_types(self) -> List[ModelResultType]:
        return [ModelResultType.input_batch]


class WhiteningDecorrelation(NodeAnchorBasedLinkPredictionBaseTask):
    """
    Creates 2 augmented views of input graph with augmentations 1 and 2 and defines task-specific linear head for Whitening Decorrelation Loss
    """

    def __init__(
        self,
        encoder: nn.Module,
        lambd: float = 1e-3,
        feat_drop_1: float = 0.2,
        edge_drop_1: float = 0.2,
        feat_drop_2: float = 0.2,
        edge_drop_2: float = 0.2,
    ):
        super(WhiteningDecorrelation, self).__init__()
        self.encoder = encoder
        hid_dim = self.encoder.hid_dim
        out_dim = self.encoder.out_dim
        self.loss = WhiteningDecorrelationLoss(lambd=lambd)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(out_dim, hid_dim),  # type: ignore
            torch.nn.ReLU(),
            torch.nn.Linear(hid_dim, out_dim),  # type: ignore
        )
        self.feat_drop_1 = feat_drop_1
        self.edge_drop_1 = edge_drop_1
        self.feat_drop_2 = feat_drop_2
        self.edge_drop_2 = edge_drop_2

    def forward(
        self,
        task_input: NodeAnchorBasedLinkPredictionTaskInputs,
        gbml_config_pb_wrapper: GbmlConfigPbWrapper,
        should_eval: bool,
        device: torch.device,
    ) -> Tuple[torch.Tensor, int]:
        main_batch = task_input.input_batch.main_batch
        augmented_graph_1 = get_augmented_graph(
            graph=main_batch.graph.to(device=device),
            edge_drop_ratio=self.edge_drop_1,  # type: ignore
            feat_drop_ratio=self.feat_drop_1,  # type: ignore
        )
        augmented_embeddings_1 = infer_root_embeddings(
            model=self.encoder,
            graph=augmented_graph_1,
            root_node_indices=main_batch.root_node_indices,
            gbml_config_pb_wrapper=gbml_config_pb_wrapper,
            device=device,
        )
        augmented_graph_2 = get_augmented_graph(
            graph=main_batch.graph.to(device=device),
            edge_drop_ratio=self.edge_drop_2,  # type: ignore
            feat_drop_ratio=self.feat_drop_2,  # type: ignore
        )
        augmented_embeddings_2 = infer_root_embeddings(
            model=self.encoder,
            graph=augmented_graph_2,
            root_node_indices=main_batch.root_node_indices,
            gbml_config_pb_wrapper=gbml_config_pb_wrapper,
            device=device,
        )
        h1 = self.head(augmented_embeddings_1)
        h2 = self.head(augmented_embeddings_2)
        return self.loss(h1=h1, h2=h2, N=augmented_embeddings_1.shape[0], device=device)

    @property
    def result_types(self) -> List[ModelResultType]:
        return [ModelResultType.input_batch]


class GBT(NodeAnchorBasedLinkPredictionBaseTask):
    """
    Creates 2 augmented views of input graph with augmentations 1 and 2 and defines task-specific linear head for GBT loss
    """

    def __init__(
        self,
        encoder: nn.Module,
        feat_drop_1: float = 0.2,
        edge_drop_1: float = 0.2,
        feat_drop_2: float = 0.2,
        edge_drop_2: float = 0.2,
    ):
        super(GBT, self).__init__()
        self.encoder = encoder
        self.loss = GBTLoss()
        self.feat_drop_1 = feat_drop_1
        self.edge_drop_1 = edge_drop_1
        self.feat_drop_2 = feat_drop_2
        self.edge_drop_2 = edge_drop_2

    def forward(
        self,
        task_input: NodeAnchorBasedLinkPredictionTaskInputs,
        gbml_config_pb_wrapper: GbmlConfigPbWrapper,
        should_eval: bool,
        device: torch.device,
    ) -> Tuple[torch.Tensor, int]:
        main_batch = task_input.input_batch.main_batch
        augmented_graph_1 = get_augmented_graph(
            graph=main_batch.graph.to(device=device),
            edge_drop_ratio=self.edge_drop_1,  # type: ignore
            feat_drop_ratio=self.feat_drop_1,  # type: ignore
        )
        augmented_embeddings_1 = infer_root_embeddings(
            model=self.encoder,
            graph=augmented_graph_1,
            root_node_indices=main_batch.root_node_indices,
            gbml_config_pb_wrapper=gbml_config_pb_wrapper,
            device=device,
        )
        augmented_graph_2 = get_augmented_graph(
            graph=main_batch.graph.to(device=device),
            edge_drop_ratio=self.edge_drop_2,  # type: ignore
            feat_drop_ratio=self.feat_drop_2,  # type: ignore
        )
        augmented_embeddings_2 = infer_root_embeddings(
            model=self.encoder,
            graph=augmented_graph_2,
            root_node_indices=main_batch.root_node_indices,
            gbml_config_pb_wrapper=gbml_config_pb_wrapper,
            device=device,
        )
        return self.loss(
            z_a=augmented_embeddings_1, z_b=augmented_embeddings_2, device=device
        )

    @property
    def result_types(self) -> List[ModelResultType]:
        return [ModelResultType.input_batch]


class BGRL(NodeAnchorBasedLinkPredictionBaseTask):
    """
    Creates 2 augmented views of input graph with augmentations 1 and 2 and defines task-specific linear head and BGRL loss
    """

    def __init__(
        self,
        encoder: nn.Module,
        feat_drop_1: float = 0.8,
        edge_drop_1: float = 0.8,
        feat_drop_2: float = 0.1,
        edge_drop_2: float = 0.8,
    ):
        super(BGRL, self).__init__()
        self.encoder = encoder
        hid_dim = self.encoder.hid_dim
        out_dim = self.encoder.out_dim
        self.offline_encoder = copy.deepcopy(encoder)
        for param in self.offline_encoder.parameters():  # type: ignore
            param.requires_grad = False
        self.loss = BGRLLoss()
        self.head = torch.nn.Sequential(
            torch.nn.Linear(out_dim, hid_dim),  # type: ignore
            torch.nn.ReLU(),
            torch.nn.Linear(hid_dim, out_dim),  # type: ignore
        )
        self.feat_drop_1 = feat_drop_1
        self.edge_drop_1 = edge_drop_1
        self.feat_drop_2 = feat_drop_2
        self.edge_drop_2 = edge_drop_2

    def forward(
        self,
        task_input: NodeAnchorBasedLinkPredictionTaskInputs,
        gbml_config_pb_wrapper: GbmlConfigPbWrapper,
        should_eval: bool,
        device: torch.device,
    ) -> Tuple[torch.Tensor, int]:
        main_batch = task_input.input_batch.main_batch
        augmented_graph_1 = get_augmented_graph(
            graph=main_batch.graph.to(device=device),
            edge_drop_ratio=self.edge_drop_1,  # type: ignore
            feat_drop_ratio=self.feat_drop_1,  # type: ignore
        )
        augmented_graph_2 = get_augmented_graph(
            graph=main_batch.graph.to(device=device),
            edge_drop_ratio=self.edge_drop_2,  # type: ignore
            feat_drop_ratio=self.feat_drop_2,  # type: ignore
        )
        enc1 = infer_root_embeddings(
            model=self.encoder,
            graph=augmented_graph_1,
            root_node_indices=main_batch.root_node_indices,
            gbml_config_pb_wrapper=gbml_config_pb_wrapper,
            device=device,
        )
        enc2 = infer_root_embeddings(
            model=self.encoder,
            graph=augmented_graph_2,
            root_node_indices=main_batch.root_node_indices,
            gbml_config_pb_wrapper=gbml_config_pb_wrapper,
            device=device,
        )
        y1 = self.offline_encoder(augmented_graph_1)[main_batch.root_node_indices]
        y2 = self.offline_encoder(augmented_graph_2)[main_batch.root_node_indices]
        q1 = self.head(enc1)
        q2 = self.head(enc2)
        return self.loss(q1=q1, q2=q2, y1=y1, y2=y2)

    def update_offline_encoder(self, mm: float):
        for param_q, param_k in zip(
            self.encoder.parameters(), self.offline_encoder.parameters()
        ):
            param_k.data.mul_(mm).add_(param_q.data, alpha=1.0 - mm)

    @property
    def result_types(self) -> List[ModelResultType]:
        return [ModelResultType.input_batch]


class TBGRL(NodeAnchorBasedLinkPredictionBaseTask):
    """
    Creates 3 augmented views of input graph with positive augmentations 1, 2 and negative augmentation 3 and defines task-specific linear head and TBGRL loss
    """

    def __init__(
        self,
        encoder: nn.Module,
        neg_lambda: float = 0.12,
        feat_drop_1: float = 0.8,
        edge_drop_1: float = 0.8,
        feat_drop_2: float = 0.1,
        edge_drop_2: float = 0.8,
        feat_drop_neg: float = 0.95,
        edge_drop_neg: float = 0.95,
    ):
        super(TBGRL, self).__init__()
        self.encoder = encoder
        hid_dim = self.encoder.hid_dim
        out_dim = self.encoder.out_dim
        self.offline_encoder = copy.deepcopy(encoder)
        for param in self.offline_encoder.parameters():  # type: ignore
            param.requires_grad = False
        self.loss = TBGRLLoss(neg_lambda=neg_lambda)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(out_dim, hid_dim),  # type: ignore
            torch.nn.ReLU(),
            torch.nn.Linear(hid_dim, out_dim),  # type: ignore
        )

        self.feat_drop_1 = feat_drop_1
        self.edge_drop_1 = edge_drop_1
        self.feat_drop_2 = feat_drop_2
        self.edge_drop_2 = edge_drop_2
        self.feat_drop_neg = feat_drop_neg
        self.edge_drop_neg = edge_drop_neg

    def forward(
        self,
        task_input: NodeAnchorBasedLinkPredictionTaskInputs,
        gbml_config_pb_wrapper: GbmlConfigPbWrapper,
        should_eval: bool,
        device: torch.device,
    ) -> Tuple[torch.Tensor, int]:
        main_batch = task_input.input_batch.main_batch
        augmented_graph_1 = get_augmented_graph(
            graph=main_batch.graph.to(device=device),
            edge_drop_ratio=self.edge_drop_1,  # type: ignore
            feat_drop_ratio=self.feat_drop_1,  # type: ignore
        )
        augmented_graph_2 = get_augmented_graph(
            graph=main_batch.graph.to(device=device),
            edge_drop_ratio=self.edge_drop_2,  # type: ignore
            feat_drop_ratio=self.feat_drop_2,  # type: ignore
        )
        augmented_graph_3 = get_augmented_graph(
            graph=main_batch.graph.to(device=device),
            edge_drop_ratio=self.edge_drop_neg,  # type: ignore
            feat_drop_ratio=self.feat_drop_neg,  # type: ignore
            graph_perm=True,
        )
        enc1 = infer_root_embeddings(
            model=self.encoder,
            graph=augmented_graph_1,
            root_node_indices=main_batch.root_node_indices,
            gbml_config_pb_wrapper=gbml_config_pb_wrapper,
            device=device,
        )
        enc2 = infer_root_embeddings(
            model=self.encoder,
            graph=augmented_graph_2,
            root_node_indices=main_batch.root_node_indices,
            gbml_config_pb_wrapper=gbml_config_pb_wrapper,
            device=device,
        )
        neg_y = self.offline_encoder(augmented_graph_3)[main_batch.root_node_indices]
        y1 = self.offline_encoder(augmented_graph_1)[main_batch.root_node_indices]
        y2 = self.offline_encoder(augmented_graph_2)[main_batch.root_node_indices]
        q1 = self.head(enc1)
        q2 = self.head(enc2)
        return self.loss(q1=q1, q2=q2, y1=y1, y2=y2, neg_y=neg_y)

    def update_offline_encoder(self, mm: float):
        for param_q, param_k in zip(
            self.encoder.parameters(), self.offline_encoder.parameters()
        ):
            param_k.data.mul_(mm).add_(param_q.data, alpha=1.0 - mm)

    @property
    def result_types(self) -> List[ModelResultType]:
        return [ModelResultType.input_batch]


class DirectAU(NodeAnchorBasedLinkPredictionBaseTask):
    """
    DirectAU (https://arxiv.org/pdf/2206.12811.pdf) optimizes for representation quality in Collaborative Filtering from the
    perspective of alignment and uniformity on the hypersphere. It does so without the use of negative sampling and only uses the
    embeddings generated from the encoder.
    """

    def __init__(
        self, gamma: float = 1.0, alpha: float = 2.0, temperature: float = 2.0
    ):
        super(DirectAU, self).__init__()
        self.alignment_loss = AligmentLoss(alpha=alpha)
        self.uniformity_loss = UniformityLoss(temperature=temperature)
        self.gamma = gamma

    def forward(
        self,
        task_input: NodeAnchorBasedLinkPredictionTaskInputs,
        gbml_config_pb_wrapper: GbmlConfigPbWrapper,
        should_eval: bool,
        device: torch.device,
    ) -> Tuple[torch.Tensor, int]:
        assert task_input.batch_embeddings is not None
        batch_embeddings = task_input.batch_embeddings
        running_loss = torch.tensor(0.0, device=device)
        for condensed_edge_type in batch_embeddings.pos_embeddings:
            anchor_embeddings = batch_embeddings.repeated_query_embeddings[
                condensed_edge_type
            ]
            pos_embeddings = batch_embeddings.pos_embeddings[condensed_edge_type]
            running_loss += self.alignment_loss(
                user_embeddings=anchor_embeddings, item_embeddings=pos_embeddings
            )
            running_loss += self.gamma * self.uniformity_loss(
                user_embeddings=anchor_embeddings, item_embeddings=pos_embeddings
            )
        return running_loss, 1

    @property
    def result_types(self) -> List[ModelResultType]:
        return [ModelResultType.batch_embeddings]


class NodeAnchorBasedLinkPredictionTasks:
    def __init__(self) -> None:
        self._task_to_fn_map = nn.ModuleDict()
        self._task_to_weights_map: Dict[str, float] = {}
        self._result_types: Set[ModelResultType] = set()

    def _get_all_tasks(
        self,
    ) -> List[Tuple[NodeAnchorBasedLinkPredictionBaseTask, float]]:
        tasks_list: List[Tuple[NodeAnchorBasedLinkPredictionBaseTask, float]] = []
        for task in list(self._task_to_weights_map.keys()):
            fn = self._task_to_fn_map[task]
            weight = self._task_to_weights_map[task]
            tasks_list.append((fn, weight))
        return tasks_list

    def add_task(
        self, task: NodeAnchorBasedLinkPredictionBaseTask, weight: float
    ) -> None:
        self._task_to_fn_map[task.task_name] = task
        self._task_to_weights_map[task.task_name] = weight
        for result_type in task.result_types:
            self._result_types.add(result_type)

    def calculate_losses(
        self,
        batch_results: NodeAnchorBasedLinkPredictionTaskInputs,
        gbml_config_pb_wrapper: GbmlConfigPbWrapper,
        should_eval: bool,
        device: torch.device,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        loss_to_val_map: Dict[str, float] = {}
        loss_to_batch_size_map: Dict[str, int] = {}
        for task, weight in self._get_all_tasks():
            loss_val, batch_size = task(
                task_input=batch_results,
                gbml_config_pb_wrapper=gbml_config_pb_wrapper,
                should_eval=should_eval,
                device=device,
            )
            loss_to_val_map[task.task_name] = weight * loss_val
            loss_to_batch_size_map[task.task_name] = batch_size
        sample_wise_loss = torch.tensor(0.0, device=device)
        for loss_type in loss_to_val_map:
            cur_loss = loss_to_val_map[loss_type]
            sample_wise_loss += cur_loss / loss_to_batch_size_map[loss_type]
        final_loss: torch.Tensor
        final_loss_map: Dict[str, float]

        final_loss = sample_wise_loss
        final_loss_map = {
            key: float("{:.3f}".format(value / loss_to_batch_size_map[key]))
            for key, value in loss_to_val_map.items()
        }

        return final_loss, final_loss_map

    @property
    def result_types(self) -> Set[ModelResultType]:
        return self._result_types
