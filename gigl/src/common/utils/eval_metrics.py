from typing import cast

import torch


def hit_rate_at_k(
    pos_scores: torch.FloatTensor, neg_scores: torch.FloatTensor, ks: torch.LongTensor
) -> torch.FloatTensor:
    """Computes Hit Rate @ K metrics for various Ks, evaluating 1+ positives against 1+ negatives.

    Args:
        pos_scores (torch.FloatTensor): Contains 1 or more positive sample scores.
        neg_scores (torch.FloatTensor): Contains 1 or more negative sample scores.
        ks (torch.LongTensor): k-values for which to compute hits.

    Returns:
        torch.FloatTensor: Hit rates corresponding to the requested ks.
    """
    max_k_requested = int(torch.max(ks).item())
    max_viable_k = 1 + neg_scores.numel()
    min_k_requested = torch.min(ks).item()
    assert (
        min_k_requested >= 1
    ), f"ks must be greater-or-equal to 1 (got {min_k_requested})"
    pos_scores_reshaped = pos_scores.view(-1, 1)
    neg_scores_reshaped = neg_scores.view(1, -1)
    num_pos_scores = pos_scores_reshaped.shape[0]
    neg_scores_repeated = neg_scores_reshaped.repeat(num_pos_scores, 1)
    all_scores = torch.hstack((pos_scores_reshaped, neg_scores_repeated))
    all_scores_sorted = torch.argsort(all_scores, dim=1, descending=True)
    one_hot_scores = all_scores_sorted == 0
    hit_indicators = torch.cumsum(one_hot_scores, dim=1)
    hit_rates = hit_indicators.float().mean(dim=0)
    hit_rates_padded = (
        torch.cat(
            (
                hit_rates,
                torch.ones(
                    size=(max_k_requested - hit_rates.numel(),), device=hit_rates.device
                ),
            )
        )
        if max_k_requested > max_viable_k
        else hit_rates
    )
    ks_adjusted = ks - 1  # subtract 1 since indices are 0-indexed
    hits_at_ks = torch.gather(input=hit_rates_padded, dim=0, index=ks_adjusted)
    return cast(torch.FloatTensor, hits_at_ks)


def mean_reciprocal_rank(
    pos_scores: torch.FloatTensor, neg_scores: torch.FloatTensor
) -> torch.FloatTensor:
    """Computes Mean Reciprocal Rank (MRR), evaluating 1+ positives against 1+ negatives.

    Args:
        pos_scores (torch.FloatTensor): Contains 1 or more positive sample scores.
        neg_scores (torch.FloatTensor): Contains 1 or more negative sample scores.

    Returns:
        torch.FloatTensor: Computed MRR score.
    """
    pos_scores_reshaped = pos_scores.view(-1, 1)
    neg_scores_reshaped = neg_scores.view(1, -1)
    num_pos_scores = pos_scores_reshaped.shape[0]
    neg_scores_repeated = neg_scores_reshaped.repeat(num_pos_scores, 1)
    all_scores = torch.hstack((pos_scores_reshaped, neg_scores_repeated))
    all_scores_sorted = torch.argsort(all_scores, dim=1, descending=True)
    _, unadjusted_ranks = torch.where(all_scores_sorted == 0)
    adjusted_ranks = unadjusted_ranks + 1  # +1 since ranks are 0-indexed here
    reciprocal_ranks = 1.0 / adjusted_ranks  # compute reciprocal
    mrr = torch.mean(reciprocal_ranks)
    return cast(torch.FloatTensor, mrr)
