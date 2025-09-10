from typing import List, Tuple, Union

import torch
import torch.nn.functional as F

# TODO(nshah-sc): Some of these functions don't require labels, and we should refactor them to not require them.
# The current implementation is parameterized as such to support some existing code.


def bpr_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """
    Computes Bayesian Personalized Ranking (BPR) loss.

    For each positive score s⁺ and its negatives s⁻₁, ..., s⁻_K, we compute:

    $$
    \mathcal{L}_\\text{BPR} = - \\frac{1}{BK} \sum_{i=1}^B \sum_{j=1}^K \log \sigma(s_i^+ - s_{ij}^-)
    $$

    Args:
        scores: Score tensor of shape [B, 1 + K], where B is the batch size and K is the number of negatives.
            1st column contains positive scores, and the rest are negative scores.
        labels: Label tensor of shape [B, 1 + K], where 1 indicates positive and 0 indicates negative.
            1st column contains positive labels, and the rest are negative labels.

    Returns:
        Scalar BPR loss
    """
    pos = scores[:, 0].unsqueeze(1)  # (B, 1)
    neg = scores[:, 1:]  # (B, K)

    diff = pos - neg  # (B, K)
    loss = -F.logsigmoid(diff).mean()  # scalar

    return loss


def infonce_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Computes InfoNCE contrastive loss.

    We treat each group of (1 positive + K negatives) as a (1 + K)-way classification:

    $$
    \mathcal{L}_\\text{InfoNCE} = - \\frac{1}{B} \sum_{i=1}^B \log \frac{\exp(s_i^+ / \\tau)}{\sum_{j=0}^{K} \exp(s_{ij} / \\tau)}
    $$

    Args:
        scores: Score tensor of shape [B, 1 + K], where B is the batch size and K is the number of negatives.
            1st column contains positive scores, and the rest are negative scores.
        labels: Label tensor of shape [B, 1 + K], where 1 indicates positive and 0 indicates negative.
            1st column contains positive labels, and the rest are negative labels.
        num_negatives: K, number of negatives per positive

    Returns:
        Scalar InfoNCE loss
    """
    scores = scores / temperature  # (B, 1 + K)
    loss = F.cross_entropy(
        scores, torch.zeros(scores.size(0), dtype=torch.long, device=scores.device)
    )
    return loss


def average_pos_neg_scores(
    scores: torch.Tensor, labels: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the average positive and negative scores from scores and labels.

    Args:
        scores: Score tensor.
        labels: Label tensor of corresponding shape.  1s indicate positive, 0s indicate negative.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (avg_pos_score, avg_neg_score).  These are scalars,
        and the tensors are detached from the computation graph since we don't want to backprop.
    """

    one_mask = labels == 1
    avg_pos_score = scores[one_mask].mean().detach()
    avg_neg_score = scores[~one_mask].mean().detach()
    return avg_pos_score, avg_neg_score


def hit_rate_at_k(
    scores: torch.Tensor,
    labels: torch.Tensor,
    ks: Union[int, List[int]],
) -> torch.Tensor:
    """
    Computes HitRate@K using pure tensor operations.

    HitRate@K is defined as:
        \[
        \text{HitRate@K} = \frac{1}{N} \sum_{i=1}^N \mathbb{1}\{ \text{positive in top-K} \}
        \]

    Args:
        scores: Score tensor of shape [B, 1 + K], where B is the batch size and K is the number of negatives.
            1st column contains positive scores, and the rest are negative scores.
        labels: Label tensor of shape [B, 1 + K], where 1 indicates positive and 0 indicates negative.
            1st column contains positive labels, and the rest are negative labels.
        ks: An integer or list of integers indicating K values.  Maximum K should be less than or equal
            to the number of negatives + 1.

    Returns:
        A tensor (if one K) or dict of tensors (if multiple Ks), each giving HitRate@K.
    """

    if isinstance(ks, int):
        ks = [ks]
    ks = torch.tensor(sorted(set(ks)), device=scores.device)

    # Get top max_k indices (shape B x max_k)
    max_k = ks.max().item()
    topk_indices = torch.topk(scores, k=max_k, dim=1).indices  # shape: (B, max_k)

    # Gather corresponding labels for top-k entries
    topk_labels = torch.gather(labels, dim=1, index=topk_indices)  # shape: (B, max_k)

    # For each k, compute hit (positive appeared in top-k)
    hits_at_k = (topk_labels.cumsum(dim=1) > 0).float()  # (B, max_k)
    hit_rates = hits_at_k[:, ks - 1].mean(dim=0)  # (len(ks),)
    return hit_rates


def mean_reciprocal_rank(
    scores: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """
    Computes Mean Reciprocal Rank (MRR) using pure tensor operations.

    MRR is defined as:
        \[
        \text{MRR} = \frac{1}{N} \sum_{i=1}^N \frac{1}{\text{rank}_i}
        \]
        where rank_i is the 1-based index of the positive in the sorted list.

    Args:
        scores: Score tensor of shape [B, 1 + K], where B is the batch size and K is the number of negatives.
            1st column contains positive scores, and the rest are negative scores.
        labels: Label tensor of shape [B, 1 + K], where 1 indicates positive and 0 indicates negative.
            1st column contains positive labels, and the rest are negative labels.

    Returns:
        Scalar tensor with MRR.
    """

    # Sort scores descending, get sort indices
    sorted_indices = torch.argsort(scores, dim=1, descending=True)

    # Use sort indices to reorder labels
    sorted_labels = torch.gather(labels, dim=1, index=sorted_indices)

    # Find the index of the positive label (label == 1) in sorted list
    reciprocal_ranks = 1.0 / (torch.argmax(sorted_labels, dim=1).float() + 1.0)  # (B,)
    return reciprocal_ranks.mean()
