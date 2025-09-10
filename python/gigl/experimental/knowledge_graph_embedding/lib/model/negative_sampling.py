from typing import Optional

import torch
import torch.nn.functional as F

from gigl.experimental.knowledge_graph_embedding.lib.model.types import (
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
    Computes relation-aware in-batch contrastive similarity for knowledge graph embedding training.

    This function implements contrastive learning for knowledge graph embeddings, where the goal is
    to learn node representations such that related nodes (connected by edges) have similar embeddings
    while unrelated nodes have dissimilar embeddings.

    **Background:**
    In knowledge graph embedding, we represent entities (nodes) and relations (edge types) as dense
    vectors. For a triple (source, relation, destination), we want the embeddings of source and
    destination to be similar when they're connected by that relation type.

    **Contrastive Learning:**
    This function uses "in-batch" contrastive learning, meaning it creates negative examples by
    pairing each positive example with other examples in the same batch that have the same relation
    type but different source/destination nodes. This is computationally efficient since it reuses
    embeddings already computed for the batch.

    **Relation-aware:**
    The "relation-aware" aspect means that negative examples are only created within the same
    relation type. For example, if we have a positive example (person_A, "lives_in", city_B),
    we only create negatives with other "lives_in" relations, not with "works_for" relations.

    Args:
        src_embeddings (torch.Tensor): [B, D] Source node embeddings for each positive example.
            B is the batch size, D is the embedding dimension. Each row represents the embedding
            of a source entity in a knowledge graph triple.

        dst_embeddings (torch.Tensor): [B, D] Destination node embeddings for each positive example.
            Each row represents the embedding of a destination entity that should be similar to
            its corresponding source entity.

        condensed_edge_types (torch.Tensor): [B] Integer relation type IDs for each positive example.
            This identifies which relation type connects each source-destination pair. Examples
            within the same relation type can be used as negatives for each other.

        temperature (float, optional): Temperature parameter for scaling similarities before softmax.
            Lower values (< 1.0) make the model more confident in its predictions by sharpening
            the probability distribution. Higher values (> 1.0) make predictions more uniform.
            Defaults to 1.0.

        scoring_function (SimilarityType, optional): Function used to compute similarity between
            embeddings. Options are:
            - COSINE: Cosine similarity (angle between vectors, normalized)
            - DOT: Dot product (unnormalized, sensitive to magnitude)
            - EUCLIDEAN: Negative squared Euclidean distance (closer = more similar)
            Defaults to SimilarityType.COSINE.

        corrupt_side (NegativeSamplingCorruptionType, optional): Which side of the triple to corrupt
            when creating negative examples:
            - DST: Replace destination nodes (e.g., (person_A, "lives_in", wrong_city))
            - SRC: Replace source nodes (e.g., (wrong_person, "lives_in", city_B))
            - BOTH: Randomly choose to corrupt either source or destination for each example
            Defaults to NegativeSamplingCorruptionType.DST.

        num_negatives (Optional[int], optional): Number of negative examples to sample per positive
            example. If None, uses all valid negatives in the batch (can be computationally expensive
            for large batches). Setting a specific number (e.g., 10) makes training more efficient.
            Defaults to None.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing:

        logits (torch.Tensor): [B, 1 + K] Similarity scores for positive and negative pairs.
            The first column contains similarities for the true positive pairs. The remaining
            K columns contain similarities for the negative pairs. Higher values indicate
            higher similarity.

        labels (torch.Tensor): [B, 1 + K] Binary labels corresponding to the logits.
            The first column is all 1s (indicating positive pairs), and the remaining
            columns are all 0s (indicating negative pairs). Used for computing contrastive loss.

    Example:
        >>> # Batch of 3 examples with 2D embeddings
        >>> src_emb = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        >>> dst_emb = torch.tensor([[1.0, 0.1], [0.1, 1.0], [1.1, 1.0]])
        >>> relations = torch.tensor([0, 0, 1])  # First two are same relation type
        >>>
        >>> logits, labels = in_batch_relationwise_contrastive_similarity(
        ...     src_emb, dst_emb, relations, num_negatives=1
        ... )
        >>> # logits.shape: [3, 2] (positive + 1 negative per example)
        >>> # labels.shape: [3, 2] with first column all 1s, second column all 0s
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
    positive_src_embeddings: torch.Tensor,  # [B, D]
    positive_dst_embeddings: torch.Tensor,  # [B, D]
    positive_condensed_edge_types: torch.Tensor,  # [B]
    negative_batch_src_embeddings: torch.Tensor,  # [N, D]
    negative_batch_dst_embeddings: torch.Tensor,  # [N, D]
    batch_condensed_edge_types: torch.Tensor,  # [N]
    temperature: float = 1.0,
    scoring_function: SimilarityType = SimilarityType.COSINE,
    corrupt_side: NegativeSamplingCorruptionType = NegativeSamplingCorruptionType.DST,
    num_negatives: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes relation-aware contrastive similarity using an external batch of negative examples.

    This function extends contrastive learning beyond the current batch by using a separate,
    larger collection of potential negative examples. This approach often leads to better
    embedding quality since it provides more diverse and challenging negative examples.

    **Key Difference from In-Batch Sampling:**
    While `in_batch_relationwise_contrastive_similarity` creates negatives by reshuffling
    examples within the current batch, this function uses a completely separate set of
    pre-computed embeddings as negative candidates. This allows for:
    - More diverse negative examples
    - Better control over negative example quality
    - Ability to use hard negatives (examples that are difficult to distinguish from positives)
    - Larger pools of negatives without increasing batch size

    **Use Cases:**
    - When you have a large corpus of pre-computed embeddings to use as negatives
    - When implementing sophisticated negative sampling strategies (e.g., hard negatives)
    - When memory constraints limit your batch size but you want many negative examples
    - When training with cached embeddings from previous epochs

    Args:
        positive_src_embeddings (torch.Tensor): [B, D] Source node embeddings for the positive examples.
            These are the "query" embeddings that we want to find good matches for. B is the
            number of positive examples, D is the embedding dimension.

        positive_dst_embeddings (torch.Tensor): [B, D] Destination node embeddings for the positive examples.
            These are the true "target" embeddings that should be similar to their corresponding
            source embeddings. Each row pairs with the corresponding row in positive_src_embeddings.

        positive_condensed_edge_types (torch.Tensor): [B] Relation type IDs for the positive examples.
            Integer identifiers specifying which relation type connects each source-destination
            pair. This ensures that negative examples are only selected from the same relation type.

        negative_batch_src_embeddings (torch.Tensor): [N, D] Source node embeddings from an external batch
            that will serve as potential negative examples. N is typically much larger than B,
            providing a rich pool of negative candidates. These embeddings might come from:
            - A different batch from the same dataset
            - Pre-computed embeddings from earlier training steps
            - A carefully curated set of hard negative examples

        negative_batch_dst_embeddings (torch.Tensor): [N, D] Destination node embeddings from the external
            batch. These correspond to the source embeddings and will be used when corrupting
            the destination side of triples.

        negative_batch_condensed_edge_types (torch.Tensor): [N] Relation type IDs for the external batch.
            Used to ensure that negative examples maintain relation-type consistency. Only
            embeddings with matching relation types will be considered as valid negatives.

        temperature (float, optional): Temperature parameter for similarity scaling. Controls
            the "sharpness" of the resulting probability distribution:
            - Lower values (< 1.0): Make the model more confident, sharper distinctions
            - Higher values (> 1.0): Make the model less confident, smoother distributions
            - 1.0: No scaling applied
            Defaults to 1.0.

        scoring_function (SimilarityType, optional): Method for computing embedding similarity:
            - COSINE: Normalized dot product, measures angle between vectors (most common)
            - DOT: Raw dot product, sensitive to vector magnitudes
            - EUCLIDEAN: Negative squared L2 distance, measures geometric distance
            The choice affects how the model learns to represent relationships.
            Defaults to SimilarityType.COSINE.

        corrupt_side (NegativeSamplingCorruptionType, optional): Specifies which part of the
            knowledge graph triple to replace when creating negative examples:
            - DST: Replace destination nodes (e.g., (Albert_Einstein, "born_in", wrong_city))
            - SRC: Replace source nodes (e.g., (wrong_person, "born_in", Germany))
            - BOTH: Randomly choose to replace either source or destination for each example
            Different corruption strategies can lead to different learned representations.
            Defaults to NegativeSamplingCorruptionType.DST.

        num_negatives (Optional[int], optional): Maximum number of negative examples to use per
            positive example. Controls the computational/memory trade-off:
            - None: Use all valid negatives from the external batch (can be expensive)
            - Small number (5-20): Fast training, fewer negatives
            - Large number (100+): Slower but potentially better quality learning
            Defaults to None.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing:

        logits (torch.Tensor): [B, 1 + K] Similarity scores matrix where:
            - First column: Similarities between true positive pairs (src_i, dst_i)
            - Remaining K columns: Similarities between each positive and its K negative examples
            - Higher values indicate higher predicted similarity
            - Used as input to contrastive loss functions

        labels (torch.Tensor): [B, 1 + K] Binary label matrix corresponding to logits:
            - First column: All 1s (indicating true positive pairs)
            - Remaining K columns: All 0s (indicating negative pairs)
            - Used as targets for contrastive loss computation
            - Shape matches logits for element-wise loss calculation

    Example:
        >>> # 2 positive examples, 5 external candidates for negatives
        >>> pos_src = torch.randn(2, 128)  # 2 positive source embeddings
        >>> pos_dst = torch.randn(2, 128)  # 2 positive destination embeddings
        >>> pos_rels = torch.tensor([0, 1])  # Different relation types
        >>>
        >>> neg_src = torch.randn(5, 128)  # 5 potential negative sources
        >>> neg_dst = torch.randn(5, 128)  # 5 potential negative destinations
        >>> neg_rels = torch.tensor([0, 0, 1, 1, 2])  # Mixed relation types
        >>>
        >>> logits, labels = against_batch_relationwise_contrastive_similarity(
        ...     pos_src, pos_dst, pos_rels,
        ...     neg_src, neg_dst, neg_rels,
        ...     num_negatives=2  # Sample 2 negatives per positive
        ... )
        >>> # logits.shape: [2, 3] (1 positive + 2 negatives per example)
        >>> # labels.shape: [2, 3] (first column 1s, others 0s)
        >>> # Only relation-type-matching negatives are selected

    Note:
        This function is particularly useful in advanced training scenarios where you want
        fine-grained control over negative sampling, such as curriculum learning, hard
        negative mining, or when working with very large knowledge graphs where in-batch
        sampling provides insufficient diversity.
    """

    B, D = positive_src_embeddings.shape
    N = negative_batch_src_embeddings.shape[0]

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
    rel_mask = (
        positive_condensed_edge_types[:, None] == batch_condensed_edge_types[None, :]
    )

    # Positive similarity
    pos_logits = (
        sim_fn(positive_src_embeddings, positive_dst_embeddings).diagonal().unsqueeze(1)
    )  # [B, 1]

    # Negative similarity matrix (B x N), depends on corruption side
    if corrupt_side == NegativeSamplingCorruptionType.SRC:
        neg_sim_matrix = sim_fn(
            negative_batch_src_embeddings, positive_dst_embeddings
        )  # [N, B]
        neg_sim_matrix = neg_sim_matrix.T  # [B, N]
    elif corrupt_side == NegativeSamplingCorruptionType.DST:
        neg_sim_matrix = sim_fn(
            positive_src_embeddings, negative_batch_dst_embeddings
        )  # [B, N]
    elif corrupt_side == NegativeSamplingCorruptionType.BOTH:
        is_src_corruption = torch.rand_like(
            positive_condensed_edge_types, dtype=torch.float32
        ).bool()  # [B] Mask for src corruption
        neg_sim_src = sim_fn(
            negative_batch_src_embeddings, positive_dst_embeddings
        ).T  # [B, N]
        neg_sim_dst = sim_fn(
            positive_src_embeddings, negative_batch_dst_embeddings
        )  # [B, N]
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
            1,
            torch.zeros_like(positive_condensed_edge_types, dtype=torch.long).view(
                -1, 1
            ),
            1,
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
        1,
        torch.zeros_like(positive_condensed_edge_types, dtype=torch.long).view(-1, 1),
        1,
    )  # Set positive labels to 1 (first column)

    return logits, labels
