import itertools
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gigl.src.common.types.graph_data import CondensedEdgeType
from gigl.src.common.types.task_inputs import BatchCombinedScores, BatchScores


class ModelResultType(Enum):
    batch_scores = "batch_scores"
    batch_combined_scores = "batch_combined_scores"
    batch_embeddings = "batch_embeddings"
    input_batch = "input_batch"


class MarginLoss(nn.Module):
    """
    A loss layer built on top of the PyTorch implementation of the margin ranking loss.

    The loss function by default calculates the loss by
        margin_ranking_loss(pos_scores, hard_neg_scores, random_neg_scores, margin=margin, reduction='sum')

    It encourages the model to generate higher similarity scores for positive pairs than negative pairs by at least a margin.

    See: https://pytorch.org/docs/stable/generated/torch.nn.MarginRankingLoss.html for more information.
    """

    def __init__(
        self,
        margin: Optional[float] = None,
    ):
        super(MarginLoss, self).__init__()
        self.margin = margin

    def _calculate_margin_loss(
        self,
        pos_scores: torch.Tensor,
        hard_neg_scores: torch.Tensor,
        random_neg_scores: torch.Tensor,
        device: torch.device = torch.device("cpu"),
    ) -> Tuple[torch.Tensor, int]:
        all_neg_scores = torch.cat(
            (hard_neg_scores, random_neg_scores),
            dim=1,
        )  # shape=[1, num_hard_neg_nodes + num_random_neg_nodes]
        all_neg_scores_repeated = all_neg_scores.repeat(
            1, pos_scores.shape[1]
        )  # shape=[1, (num_hard_neg_nodes + num_random_neg_nodes) * num_pos_nodes]
        pos_scores_repeated = pos_scores.repeat_interleave(
            all_neg_scores.shape[1], dim=1
        )  # shape=[1, num_pos_nodes * (num_hard_neg_nodes + num_random_neg_nodes)]
        ys = torch.ones_like(pos_scores_repeated).to(
            device=device
        )  # shape=[1, num_pos_nodes * (num_hard_neg_nodes + num_random_neg_nodes)]

        loss = F.margin_ranking_loss(
            input1=pos_scores_repeated,
            input2=all_neg_scores_repeated,
            target=ys,
            margin=self.margin,  # type: ignore
            reduction="sum",
        )
        sample_size = pos_scores_repeated.numel()
        return loss, sample_size

    def forward(
        self,
        loss_input: List[Dict[CondensedEdgeType, BatchScores]],
        device: torch.device = torch.device("cpu"),
    ) -> Tuple[torch.Tensor, int]:
        batch_loss = torch.tensor(0.0).to(device=device)
        batch_size = 0
        # In case we have an empty list as input, avoids division by zero error
        if not len(loss_input):
            batch_size = 1
        for result_sample in loss_input:
            for condensed_edge_type in result_sample:
                if result_sample[condensed_edge_type].pos_scores.numel():
                    sample_loss, sample_size = self._calculate_margin_loss(
                        pos_scores=result_sample[condensed_edge_type].pos_scores,
                        hard_neg_scores=result_sample[
                            condensed_edge_type
                        ].hard_neg_scores,
                        random_neg_scores=result_sample[
                            condensed_edge_type
                        ].random_neg_scores,
                        device=device,
                    )
                    batch_loss += sample_loss
                    batch_size += sample_size
        return batch_loss, batch_size


class SoftmaxLoss(nn.Module):
    """
    A loss layer built on top of the PyTorch implementation of the softmax cross entropy loss.

    The loss function by default calculate the loss by
        cross_entropy(all_scores, ys, reduction='sum')

    See: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html for more information.
    """

    def __init__(
        self,
        softmax_temperature: Optional[float] = None,
    ):
        super(SoftmaxLoss, self).__init__()
        self.softmax_temperature = softmax_temperature

    def _calculate_softmax_loss(
        self,
        pos_scores: torch.Tensor,
        hard_neg_scores: torch.Tensor,
        random_neg_scores: torch.Tensor,
        device: torch.device,
    ) -> Tuple[torch.Tensor, int]:
        all_neg_scores = torch.cat(
            (hard_neg_scores, random_neg_scores),
            dim=1,
        ).squeeze()  # shape=[num_hard_neg_nodes + num_random_neg_nodes]
        all_neg_scores_repeated = all_neg_scores.repeat(
            pos_scores.shape[1], 1
        )  # shape=[num_pos_nodes, num_hard_neg_nodes + num_random_neg_nodes]
        all_scores = torch.cat(
            (
                pos_scores.reshape(-1, 1),
                all_neg_scores_repeated,
            ),
            dim=1,
        )  # shape=[num_pos_nodes, 1 + num_hard_neg_nodes + num_random_neg_nodes]
        ys = (
            torch.zeros(pos_scores.shape[1]).long().to(device=device)
        )  # shape=[num_pos_nodes]

        loss = F.cross_entropy(
            input=all_scores / self.softmax_temperature,
            target=ys,
            reduction="sum",
        )
        sample_size = pos_scores.shape[1]
        return loss, sample_size

    def forward(
        self,
        loss_input: List[Dict[CondensedEdgeType, BatchScores]],
        device: torch.device = torch.device("cpu"),
    ) -> Tuple[torch.Tensor, int]:
        batch_loss = torch.tensor(0.0).to(device=device)
        batch_size = 0
        # In case we have an empty list as input, avoids division by zero error
        if not len(loss_input):
            batch_size = 1
        for result_sample in loss_input:
            for condensed_edge_type in result_sample:
                if result_sample[condensed_edge_type].pos_scores.numel():
                    sample_loss, sample_size = self._calculate_softmax_loss(
                        pos_scores=result_sample[condensed_edge_type].pos_scores,
                        hard_neg_scores=result_sample[
                            condensed_edge_type
                        ].hard_neg_scores,
                        random_neg_scores=result_sample[
                            condensed_edge_type
                        ].random_neg_scores,
                        device=device,
                    )
                    batch_loss += sample_loss
                    batch_size += sample_size
        return batch_loss, batch_size


class RetrievalLoss(nn.Module):
    """
    A loss layer built on top of the tensorflow_recommenders implementation.
    https://www.tensorflow.org/recommenders/api_docs/python/tfrs/tasks/Retrieval

    The loss function by default calculates the loss by:
    ```
    cross_entropy(torch.mm(query_embeddings, candidate_embeddings.T), positive_indices, reduction='sum'),
    ```
    where the candidate embeddings are `torch.cat((positive_embeddings, random_negative_embeddings))`. It encourages the model to generate query embeddings that yield the highest similarity score with their own first hop compared with others' first hops and random negatives. We also filter out the cases where, in some rows, the query could accidentally treat its own positives as negatives.

    Args:
        loss (Optional[nn.Module]): Custom loss function to be used. If `None`, the default is `nn.CrossEntropyLoss(reduction="sum")`.
        temperature (Optional[float]): Temperature scaling applied to scores before computing cross-entropy loss. If not `None`, scores are divided by the temperature value.
        remove_accidental_hits (bool): Whether to remove accidental hits where the query's positive items are also present in the negative samples.
    """

    def __init__(
        self,
        loss: Optional[nn.Module] = None,
        temperature: Optional[float] = None,
        remove_accidental_hits: bool = False,
    ):
        super(RetrievalLoss, self).__init__()
        self._loss = loss if loss is not None else nn.CrossEntropyLoss(reduction="sum")
        self._temperature = temperature
        if self._temperature is not None and self._temperature < 1e-12:
            raise ValueError(
                f"The temperature is expected to be greater than 1e-12, however you provided {self._temperature}"
            )
        self._remove_accidental_hits = remove_accidental_hits

    def calculate_batch_retrieval_loss(
        self,
        scores: torch.Tensor,
        candidate_sampling_probability: Optional[torch.Tensor] = None,
        query_ids: Optional[torch.Tensor] = None,
        candidate_ids: Optional[torch.Tensor] = None,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """
        Args:
          scores: [num_queries, num_candidates] tensor of candidate and query embeddings similarity
          candidate_sampling_probability: [num_candidates], Optional tensor of candidate sampling probabilities.
            When given will be used to correct the logits toreflect the sampling probability of negative candidates.
          query_ids: [num_queries] Optional tensor containing query ids / anchor node ids.
          candidate_ids: [num_candidates] Optional tensor containing candidate ids.
          device: the device to set as default
        """
        num_queries: int = scores.shape[0]
        num_candidates: int = scores.shape[1]
        torch._assert(
            num_queries <= num_candidates,
            "Number of queries should be less than or equal to number of candidates in a batch",
        )

        labels = torch.eye(num_queries, num_candidates).to(
            device=device
        )  # [num_queries, num_candidates]
        duplicates = torch.zeros_like(labels).to(
            device=device
        )  # [num_queries, num_candidates]

        if self._temperature is not None:
            scores = scores / self._temperature

        # provide the corresponding candidate sampling probability to enable sampled softmax
        if candidate_sampling_probability is not None:
            scores = scores - torch.log(
                torch.clamp(
                    candidate_sampling_probability, min=1e-10
                )  # frequency can be used so only limit its lower bound here
            ).type(scores.dtype)

        # obtain a mask that indicates true labels for each query when using multiple positives per query
        if query_ids is not None:
            duplicates = torch.maximum(
                duplicates,
                self._mask_by_query_ids(
                    query_ids, num_queries, num_candidates, labels.dtype, device
                ),
            )  # [num_queries, num_candidates]

        # obtain a mask that indicates true labels for each query when random negatives contain positives in this batch
        if self._remove_accidental_hits:
            if candidate_ids is None:
                raise ValueError(
                    "When accidental hit removal is enabled, candidate ids must be supplied."
                )
            duplicates = torch.maximum(
                duplicates,
                self._mask_by_candidate_ids(
                    candidate_ids, num_queries, labels.dtype, device
                ),
            )  # [num_queries, num_candidates]

        if query_ids is not None or self._remove_accidental_hits:
            # mask out the extra positives in each row by setting their logits to min(scores.dtype)
            scores = scores + (duplicates - labels) * torch.finfo(scores.dtype).min

        return self._loss(scores, target=labels)

    def _mask_by_query_ids(
        self,
        query_ids: torch.Tensor,
        num_queries: int,
        num_candidates: int,
        dtype: torch.dtype,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """
        Args:
            query_ids: [num_queries] query ids / anchor node ids in the batch
            num_queries: number of queries / rows in the batch
            num_candidates: number of candidates / columns in the batch
            dtype: labels dtype
            device: the device to set as default
        """
        query_ids = torch.unsqueeze(query_ids, 1)  # [num_queries, 1]
        duplicates = torch.eq(query_ids, query_ids.T).type(
            dtype
        )  # [num_queries, num_queries]
        if num_queries < num_candidates:
            padding_zeros = torch.zeros(
                (num_queries, num_candidates - num_queries), dtype=dtype
            ).to(device=device)
            return torch.cat(
                (duplicates, padding_zeros), dim=1
            )  # [num_queries, num_candidates]
        return duplicates

    def _mask_by_candidate_ids(
        self,
        candidate_ids: torch.Tensor,
        num_queries: int,
        dtype: torch.dtype,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """
        Args:
            candidate_ids: [num_candidates] candidate ids in this batch
            num_queries: number of queries / rows in the batch
            dtype: labels dtype
            device: the device to set as default
        """
        positive_indices = torch.arange(num_queries).to(device=device)  # [num_queries]
        positive_candidate_ids = torch.gather(
            candidate_ids, 0, positive_indices
        ).unsqueeze(
            1
        )  # [num_queries, 1]
        all_candidate_ids = torch.unsqueeze(candidate_ids, 1)  # [num_candidates, 1]
        return torch.eq(positive_candidate_ids, all_candidate_ids.T).type(
            dtype
        )  # [num_queries, num_candidates]

    def forward(
        self,
        batch_combined_scores: BatchCombinedScores,
        repeated_query_embeddings: torch.FloatTensor,
        candidate_sampling_probability: Optional[torch.FloatTensor],
        device: torch.device = torch.device("cpu"),
    ) -> Tuple[torch.Tensor, int]:
        candidate_ids = torch.cat(
            (
                batch_combined_scores.positive_ids.to(device=device),
                batch_combined_scores.hard_neg_ids.to(device=device),
                batch_combined_scores.random_neg_ids.to(device=device),
            )
        )
        if repeated_query_embeddings.numel():  # type: ignore
            loss = self.calculate_batch_retrieval_loss(
                scores=batch_combined_scores.repeated_candidate_scores,
                candidate_sampling_probability=candidate_sampling_probability,
                query_ids=batch_combined_scores.repeated_query_ids,
                candidate_ids=candidate_ids,
                device=device,
            )
            batch_size = repeated_query_embeddings.shape[0]  # type: ignore
        else:
            loss = torch.tensor(0.0).to(device=device)
            batch_size = 1
        return loss, batch_size


class GRACELoss(nn.Module):
    """
    A loss class that implements the GRACE (https://arxiv.org/pdf/2006.04131.pdf) contrastive loss approach. We generate two graph views by
    corruption and learn node representations by maximizing the agreement of node representations in these two views. We introduce this to add an
    additional contrastive loss function for multi-task learning.
    """

    def __init__(
        self,
        temperature: Optional[float] = None,
    ):
        super(GRACELoss, self).__init__()
        self.temperature = temperature

    def forward(
        self,
        h1: torch.Tensor,
        h2: torch.Tensor,
        device: torch.device = torch.device("cpu"),
    ) -> Tuple[torch.Tensor, int]:
        """
        Args:
            h1 (torch.Tensor): First input tensor
            h2 (torch.Tensor): Second input tensor
            device (torch.device): the device to set as default

        Returns:
            Tuple[torch.Tensor, int]: The loss and the sample size
        """

        def sim_matrix(a: torch.Tensor, b: torch.Tensor, eps=1e-8) -> torch.Tensor:
            """
            Computes similarity between two vectors 'a' and 'b' by normalizing vectors before creating a cosine similarity matrix.
            """
            a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
            a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
            b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
            sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1)).to(device=device)
            return sim_mt

        def get_loss(h1: torch.Tensor, h2: torch.Tensor) -> torch.Tensor:
            """
            Uses cosine similarity matrices between intra-vew pairs and inter-view pairs to generate loss
            """
            f = lambda x: torch.exp(x / self.temperature)
            refl_sim = f(sim_matrix(h1, h1))  # intra-view pairs
            between_sim = f(sim_matrix(h1, h2))  # inter-view pairs
            x1 = refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()
            loss = -torch.log(between_sim.diag() / x1)
            return loss

        l1 = get_loss(h1, h2)
        l2 = get_loss(h2, h1)
        ret = (l1 + l2) * 0.5
        return ret.mean(), 1


class FeatureReconstructionLoss(nn.Module):
    """
    Computes SCE between original feature and reconstructed feature. See https://arxiv.org/pdf/2205.10803.pdf for more information about
    feature reconstruction. We use this as an auxiliary loss for training and improved generalization.
    """

    def __init__(
        self,
        alpha: float = 3.0,
    ):
        super(FeatureReconstructionLoss, self).__init__()
        self.alpha = alpha

    def forward(
        self,
        x_target: torch.Tensor,
        x_pred: torch.Tensor,
    ) -> Tuple[torch.Tensor, int]:
        x = F.normalize(x_target, p=2, dim=-1)  # SCE Loss Computation
        y = F.normalize(x_pred, p=2, dim=-1)
        loss = (1 - (x * y).sum(dim=-1)).pow_(self.alpha)
        loss = loss.mean()
        return loss, 1


class WhiteningDecorrelationLoss(nn.Module):
    """
    Utilizes canonical correlation analysis to compute similarity between augmented graphs as an auxiliary loss. See https://arxiv.org/pdf/2106.12484.pdf
    for more information.
    """

    def __init__(
        self,
        lambd: float = 1e-3,
    ):
        super(WhiteningDecorrelationLoss, self).__init__()
        self.lambd = lambd

    def forward(
        self,
        h1: torch.Tensor,
        h2: torch.Tensor,
        N: int,
        device: torch.device = torch.device("cpu"),
    ) -> Tuple[torch.Tensor, int]:
        """
        Args:
            h1 (torch.Tensor): First input tensor
            h2 (torch.Tensor): Second input tensor
            N (int): The number of samples
            device (torch.device): the device to set as default

        Returns:
            Tuple[torch.Tensor, int]: The loss and the sample size
        """
        z1 = (h1 - h1.mean(0)) / h1.std(0)
        z2 = (h2 - h2.mean(0)) / h2.std(0)

        c1 = torch.mm(z1.T, z1)
        c2 = torch.mm(z2.T, z2)

        c = (z1 - z2) / N
        c1 = c1 / N
        c2 = c2 / N

        loss_inv = torch.linalg.matrix_norm(c)
        iden = torch.tensor(np.eye(c1.shape[0])).to(device=device)
        loss_dec1 = torch.linalg.matrix_norm(iden - c1)
        loss_dec2 = torch.linalg.matrix_norm(iden - c2)
        return loss_inv + self.lambd * (loss_dec1 + loss_dec2), 1


class GBTLoss(nn.Module):
    """
    Computes the Barlow Twins loss on the two input matrices as an auxiliary loss.
    From the offical GBT implementation at:
    https://github.com/pbielak/graph-barlow-twins/blob/ec62580aa89bf3f0d20c92e7549031deedc105ab/gssl/loss.py
    """

    def __init__(
        self,
    ):
        super(GBTLoss, self).__init__()
        self.eps = 1e-15

    def forward(
        self,
        z_a: torch.Tensor,
        z_b: torch.Tensor,
        device: torch.device,
    ) -> Tuple[torch.Tensor, int]:
        """
        Args:
            z_a (torch.Tensor): First input matrix
            z_b (torch.Tensor): Second input matrix
            device (torch.device): the device to set as default

        Returns:
            Tuple[torch.Tensor, int]: The Barlow Twins loss and the sample size
        """
        batch_size = z_a.size(0)
        feature_dim = z_a.size(1)
        _lambda = 1 / feature_dim

        # Apply batch normalization
        z_a_norm = (z_a - z_a.mean(dim=0)) / (z_a.std(dim=0) + self.eps)
        z_b_norm = (z_b - z_b.mean(dim=0)) / (z_b.std(dim=0) + self.eps)
        # Cross-correlation matrix
        c = (z_a_norm.T @ z_b_norm) / batch_size

        # Loss function
        off_diagonal_mask = ~torch.eye(feature_dim).bool().to(device=device)
        loss = (1 - c.diagonal()).pow(2).sum() + _lambda * c[off_diagonal_mask].pow(
            2
        ).sum()
        return loss, 1


class BGRLLoss(nn.Module):
    """
    Leverages BGRL loss from https://arxiv.org/pdf/2102.06514.pdf, using an offline and online encoder to predict alternative augmentations of
    the input. The offline encoder is updated by an exponential moving average rather than traditional backpropogation. We use BGRL as an
    auxiliary loss for improved generalization.
    """

    def forward(
        self,
        q1: torch.Tensor,
        q2: torch.Tensor,
        y1: torch.Tensor,
        y2: torch.Tensor,
    ) -> Tuple[torch.Tensor, int]:
        loss = (
            2
            - F.cosine_similarity(q1, y2.detach(), dim=-1).mean()
            - F.cosine_similarity(q2, y1.detach(), dim=-1).mean()
        )
        return loss, 1


class TBGRLLoss(nn.Module):
    """
    TBGRL (https://arxiv.org/pdf/2211.14394.pdf) improves over BGRL by generating a third augmented graph as a negative sample,
    providing a cheap corruption that improves generalizability of the model in inductive settings. We use TBGRL as an auxiliary loss
    for improved generalization.
    """

    def __init__(
        self,
        neg_lambda: float = 0.12,
    ):
        super(TBGRLLoss, self).__init__()
        self.neg_lambda = neg_lambda

    def forward(
        self,
        q1: torch.Tensor,
        q2: torch.Tensor,
        y1: torch.Tensor,
        y2: torch.Tensor,
        neg_y: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, int]:
        sim1 = F.cosine_similarity(q1, y2.detach()).mean()
        sim2 = F.cosine_similarity(q2, y1.detach()).mean()
        neg_sim1 = F.cosine_similarity(q1, neg_y.detach()).mean()  # type: ignore
        neg_sim2 = F.cosine_similarity(q2, neg_y.detach()).mean()  # type: ignore
        loss = self.neg_lambda * (neg_sim1 + neg_sim2) - (1 - self.neg_lambda) * (  # type: ignore
            sim1 + sim2
        )
        return loss, 1


class AligmentLoss(nn.Module):
    """
    Taken from https://github.com/THUwangcy/DirectAU, AlignmentLoss increases the similarity of representations between positive user-item pairs.
    """

    def __init__(
        self,
        alpha: Optional[float] = 2.0,  # Should not tune this parameter
    ):
        super(AligmentLoss, self).__init__()
        self.alpha = alpha

    def forward(
        self, user_embeddings: torch.Tensor, item_embeddings: torch.Tensor
    ) -> torch.Tensor:
        return (user_embeddings - item_embeddings).norm(p=2, dim=1).pow(self.alpha).mean()  # type: ignore


class UniformityLoss(nn.Module):
    """
    Taken from https://github.com/THUwangcy/DirectAU, UniformityLoss measures how well the representations scatter on the hypersphere.
    """

    def __init__(
        self,
        temperature: float = 2.0,  # Should not tune this parameter
    ):
        super(UniformityLoss, self).__init__()
        self.temperature = temperature

    def forward(
        self, user_embeddings: torch.Tensor, item_embeddings: torch.Tensor
    ) -> torch.Tensor:
        user_uniformity = torch.pdist(user_embeddings, p=2).pow(2).mul(-self.temperature).exp().mean().log()  # type: ignore
        item_uniformity = torch.pdist(item_embeddings, p=2).pow(2).mul(-self.temperature).exp().mean().log()  # type: ignore
        return (user_uniformity + item_uniformity) / 2


# TODO Add Unit test for this loss
class KLLoss(nn.Module):
    """
    Calculates KL Divergence between two set of scores for the distribution loss.
    Taken from: https://github.com/snap-research/linkless-link-prediction/blob/main/src/main.py
    """

    def __init__(
        self,
        kl_temperature: float,
    ):
        super(KLLoss, self).__init__()
        self.kl_temperature = kl_temperature

    def forward(
        self,
        student_scores: torch.Tensor,
        teacher_scores: torch.Tensor,
    ) -> torch.Tensor:
        y_s = F.log_softmax(student_scores / self.kl_temperature, dim=-1)
        y_t = F.softmax(teacher_scores / self.kl_temperature, dim=-1)
        loss = (
            F.kl_div(y_s, y_t, size_average=False)
            * (self.kl_temperature**2)
            / y_s.size()[0]
        )
        return loss


# TODO Add Unit test for this loss
class LLPRankingLoss(nn.Module):
    """
    Calculates a margin-based rakning loss between two set of scores for the ranking loss in LLP.
    This differs from normal margin loss in that it prevents the student model from trying to
    differentiate miniscule differences in probabilities which the teacher may make w/ due to noise.
    Taken from: https://github.com/snap-research/linkless-link-prediction/blob/main/src/main.py
    """

    def __init__(
        self,
        margin: float,
    ):
        super(LLPRankingLoss, self).__init__()
        self.margin = margin
        self.margin_loss = nn.MarginRankingLoss(margin=margin)

    def forward(
        self,
        student_scores: torch.Tensor,
        teacher_scores: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        dim_pairs = [
            x for x in itertools.combinations(range(student_scores.shape[1]), r=2)
        ]
        pair_array = np.array(dim_pairs).T
        teacher_rank_list = torch.zeros((len(teacher_scores), pair_array.shape[1])).to(
            device
        )

        mask = teacher_scores[:, pair_array[0]] > (
            teacher_scores[:, pair_array[1]] + self.margin
        )
        teacher_rank_list[mask] = 1
        mask2 = teacher_scores[:, pair_array[0]] < (
            teacher_scores[:, pair_array[1]] - self.margin
        )
        teacher_rank_list[mask2] = -1
        first_rank_list = student_scores[:, pair_array[0]].squeeze()
        second_rank_list = student_scores[:, pair_array[1]].squeeze()
        return self.margin_loss(first_rank_list, second_rank_list, teacher_rank_list)
