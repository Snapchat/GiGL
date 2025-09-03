from typing import Optional

import torch
import torch.nn.functional as F
from applied_tasks.knowledge_graph_embedding.lib.model.types import (
    NegativeSamplingCorruptionType,
    SimilarityType,
)


def in_batch_relationwise_contrastive_similarity(
    src_embeddings: torch.Tensor,  # [B, D]
    dst_embeddings: torch.Tensor,  # [B, D]
    condensed_edge_types: torch.Tensor,  # [B]
    temperature: float = 1.0,
    scoring_function: SimilarityType = SimilarityType.COSINE,
    corrupt_side: NegativeSamplingCorruptionType = NegativeSamplingCorruptionType.DST,
    num_negatives: Optional[int] = None,  # Number of negatives to sample per instance
) -> torch.Tensor:
    """
    Computes relation-aware in-batch contrastive loss with optional sampled negatives.

    Args:
        src_embeddings: [B, D] Source node embeddings.
        dst_embeddings: [B, D] Destination node embeddings.
        condensed_edge_types: [B] Relation type IDs.
        temperature: Scaling factor for similarity.
        corrupt_side: "src", "dst", or "both".
        num_negatives: Number of in-batch negatives to sample per example (if None, use all).

    Returns:
        logits: [B, 1 + K] Logits for positive and negative pairs.
        labels: [B, 1 + K] Labels for positive and negative pairs. The first column is 1
                (positive), rest are 0 (negatives).
    """
    B, D = src_embeddings.shape

    # Compute similarity matrix between src and dst embeddings
    if scoring_function == SimilarityType.DOT:
        sim_matrix = src_embeddings @ dst_embeddings.T
    elif scoring_function == SimilarityType.COSINE:
        src_norm = F.normalize(src_embeddings, dim=1)
        dst_norm = F.normalize(dst_embeddings, dim=1)
        sim_matrix = src_norm @ dst_norm.T
    elif scoring_function == SimilarityType.EUCLIDEAN:
        # Negative squared L2 distance (to make it similar to scoring_function)
        src_sq = src_embeddings.pow(2).sum(dim=1, keepdim=True)  # [B, 1]
        dst_sq = dst_embeddings.pow(2).sum(dim=1, keepdim=True)  # [B, 1]
        sim_matrix = -(
            src_sq - 2 * src_embeddings @ dst_embeddings.T + dst_sq.T
        )  # [B, B]
    else:
        raise ValueError(f"Unsupported scoring_function: {scoring_function}")

    sim_matrix = sim_matrix / temperature  # [B, B]

    # Create mask indicating valid relations (same relation type)
    rel_mask = condensed_edge_types[:, None] == condensed_edge_types[None, :]  # [B, B]

    # Identity matrix for diagonal masking (positive pair)
    identity = torch.diag(torch.ones_like(condensed_edge_types, dtype=torch.bool))

    # Process based on corruption side: "src", "dst", or "both"
    if corrupt_side == NegativeSamplingCorruptionType.SRC:
        # In "src" corruption, we modify the source side, keeping the destination side fixed
        sim_matrix = sim_matrix.T  # [B, B] -> Now rows are dst, columns are src

    elif corrupt_side == NegativeSamplingCorruptionType.DST:
        # In "dst" corruption, we modify the destination side, keeping the source side fixed
        # No change to sim_matrix, this is the default behavior
        pass

    elif corrupt_side == NegativeSamplingCorruptionType.BOTH:
        # In "both" corruption, we randomly decide for each row whether to corrupt the src or dst
        # Randomly decide to corrupt "src" or "dst" for each example (50% chance each)
        is_src_corruption = torch.rand_like(
            condensed_edge_types, dtype=torch.float32
        ).bool()  # [B] Mask for src corruption

        # Corrupt the source (flip relation mask and similarity matrix for those cases)
        sim_matrix_src = sim_matrix.T  # [B, B] -> Now rows are dst, columns are src

        # Corrupt the destination (standard sim_matrix dst corruption)
        sim_matrix_dst = sim_matrix

        # Combine the two corruptions
        sim_matrix = torch.where(
            is_src_corruption.unsqueeze(1), sim_matrix_src, sim_matrix_dst
        )
    else:
        raise ValueError(
            f"Corruption type must be in {NegativeSamplingCorruptionType.SRC, NegativeSamplingCorruptionType.DST, NegativeSamplingCorruptionType.BOTH}; got {corrupt_side}."
        )

    # Mask invalid negatives (i.e., non-matching relations and diagonal)
    neg_mask = rel_mask & ~identity  # Mask for valid negative pairs

    # Get positive logits (diagonal of the similarity matrix)
    pos_logits = sim_matrix.diagonal().unsqueeze(1)  # [B, 1]

    # Mask the similarity matrix to only keep valid negatives
    logits_masked = sim_matrix.masked_fill(~neg_mask, float("-inf"))  # [B, B]

    if num_negatives is None:
        # If no negative sampling, use all valid negatives
        logits = torch.cat([pos_logits, logits_masked], dim=1)
        labels = torch.zeros_like(logits, dtype=torch.float)
        labels = labels.scatter(
            1, torch.zeros_like(condensed_edge_types, dtype=torch.long).view(-1, 1), 1
        )  # Set positive labels to 1 (first column)
        return logits, labels

    # ---- Fully tensorized negative sampling ----

    # Generate random scores for sampling negative pairs
    rand = torch.rand_like(logits_masked)  # [B, B]
    rand.masked_fill_(
        ~neg_mask, float("inf")
    )  # Set invalid positions to +inf so they won't be selected in topk

    # Sample negatives using topk: smallest random scores are selected
    K = num_negatives
    sampled_idx = rand.topk(K, dim=1, largest=False, sorted=False).indices  # [B, K]

    # Gather negative logits based on the sampled indices
    neg_logits = logits_masked.gather(1, sampled_idx)  # [B, K] gather negative logits

    # Concatenate positive logits with negative logits
    logits = torch.cat([pos_logits, neg_logits], dim=1)  # [B, 1 + K]

    labels = torch.zeros_like(logits, dtype=torch.float)
    labels = labels.scatter(
        1, torch.zeros_like(condensed_edge_types, dtype=torch.long).view(-1, 1), 1
    )  # Set positive labels to 1 (first column)

    return logits, labels


def against_batch_relationwise_contrastive_similarity(
    src_embeddings: torch.Tensor,  # [B, D]
    dst_embeddings: torch.Tensor,  # [B, D]
    condensed_edge_types: torch.Tensor,  # [B]
    batch_src_embeddings: torch.Tensor,  # [N, D]
    batch_dst_embeddings: torch.Tensor,  # [N, D]
    batch_condensed_edge_types: torch.Tensor,  # [N]
    temperature: float = 1.0,
    scoring_function: SimilarityType = SimilarityType.COSINE,
    corrupt_side: NegativeSamplingCorruptionType = NegativeSamplingCorruptionType.DST,
    num_negatives: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes relation-aware contrastive logits using an external batch of negatives.

    Args:
        src_embeddings: [B, D] Source node embeddings for the positive batch.
        dst_embeddings: [B, D] Destination node embeddings for the positive batch.
        condensed_edge_types: [B] Relation type IDs for the positive batch.
        batch_src_embeddings: [N, D] Source node embeddings for the negative batch.
        batch_dst_embeddings: [N, D] Destination node embeddings for the negative batch.
        batch_condensed_edge_types: [N] Relation type IDs for the negative batch.
        temperature: Scaling factor for similarity.
        scoring_function: Similarity scoring function to use ('dot', 'cosine', 'euclidean').
        corrupt_side: "src", "dst", or "both".
        num_negatives: Number of negatives to sample per instance. None means all negatives are used.

    Returns:
        logits: [B, 1 + K] similarity logits (positive + K negatives)
        labels: [B, 1 + K] one-hot labels (first col is positive)
    """

    B, D = src_embeddings.shape
    N = batch_src_embeddings.shape[0]

    # Precompute similarity matrix between [B] queries and [N] candidates
    if scoring_function == SimilarityType.DOT:
        sim_fn = lambda x, y: x @ y.T
    elif scoring_function == SimilarityType.COSINE:
        sim_fn = lambda x, y: F.normalize(x, dim=1) @ F.normalize(y, dim=1).T
    elif scoring_function == SimilarityType.EUCLIDEAN:

        def sim_fn(x, y):
            x_sq = x.pow(2).sum(dim=1, keepdim=True)  # [B, 1]
            y_sq = y.pow(2).sum(dim=1, keepdim=True)  # [N, 1]
            return -(x_sq - 2 * x @ y.T + y_sq.T)  # negative squared L2

    else:
        raise ValueError(
            f"Similarity type must be in {SimilarityType.COSINE, SimilarityType.DOT, SimilarityType.EUCLIDEAN}. Got {scoring_function}"
        )

    # Build masks for valid negatives per relation type
    # [B, N]: True where edge types match
    rel_mask = condensed_edge_types[:, None] == batch_condensed_edge_types[None, :]

    # Positive similarity
    pos_logits = (
        sim_fn(src_embeddings, dst_embeddings).diagonal().unsqueeze(1)
    )  # [B, 1]

    # Negative similarity matrix (B x N), depends on corruption side
    if corrupt_side == NegativeSamplingCorruptionType.SRC:
        neg_sim_matrix = sim_fn(batch_src_embeddings, dst_embeddings)  # [N, B]
        neg_sim_matrix = neg_sim_matrix.T  # [B, N]
    elif corrupt_side == NegativeSamplingCorruptionType.DST:
        neg_sim_matrix = sim_fn(src_embeddings, batch_dst_embeddings)  # [B, N]
    elif corrupt_side == NegativeSamplingCorruptionType.BOTH:
        is_src_corruption = torch.rand_like(
            condensed_edge_types, dtype=torch.float32
        ).bool()  # [B] Mask for src corruption
        neg_sim_src = sim_fn(batch_src_embeddings, dst_embeddings).T  # [B, N]
        neg_sim_dst = sim_fn(src_embeddings, batch_dst_embeddings)  # [B, N]
        neg_sim_matrix = torch.where(
            is_src_corruption.unsqueeze(1), neg_sim_src, neg_sim_dst
        )
    else:
        raise ValueError(f"Invalid corrupt_side: {corrupt_side}")

    neg_sim_matrix = neg_sim_matrix / temperature  # [B, N]

    # Mask invalid negatives (i.e., non-matching relations)
    logits_masked = neg_sim_matrix.masked_fill(~rel_mask, float("-inf"))  # [B, N]

    if num_negatives is None:
        # If no negative sampling, use all valid negatives
        logits = torch.cat([pos_logits, logits_masked], dim=1)
        labels = torch.zeros_like(logits, dtype=torch.float)
        labels = labels.scatter(
            1, torch.zeros_like(condensed_edge_types, dtype=torch.long).view(-1, 1), 1
        )  # Set positive labels to 1 (first column)
        return logits, labels

    # Sample K negatives per row from matching relation types
    rand = torch.rand_like(logits_masked)  # [B, N]
    rand.masked_fill_(~rel_mask, float("inf"))  # Prevent selecting mismatched negatives
    sampled_idx = rand.topk(num_negatives, dim=1, largest=False).indices  # [B, K]

    # Gather negative similarities
    neg_logits = logits_masked.gather(1, sampled_idx)  # [B, K] gather negative logits

    # Concatenate positive logits with negative logits
    logits = torch.cat([pos_logits, neg_logits], dim=1)  # [B, 1 + K]

    labels = torch.zeros_like(logits, dtype=torch.float)
    labels = labels.scatter(
        1, torch.zeros_like(condensed_edge_types, dtype=torch.long).view(-1, 1), 1
    )  # Set positive labels to 1 (first column)

    return logits, labels
