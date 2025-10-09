from typing import Optional

import torch
import torch.nn as nn


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

    def _calculate_batch_retrieval_loss(
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
        repeated_candidate_scores: torch.Tensor,
        candidate_ids: torch.Tensor,
        repeated_query_ids: torch.Tensor,
        device: torch.device,
        candidate_sampling_probability: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            repeated_candidate_scores (torch.Tensor): The prediction scores between each repeated query users and each candidates. In this case, `repeated` means
                that we repeat each query user based on the number of positive labels they have.
                Tensor shape: [num_positives, num_positives + num_hard_negatives + num_random_negatives]
            candidate_ids (torch.Tensor): Concatenated Ids of the candidates. Tensor shape: [num_positives + num_hard_negatives + num_random_negatives]
            repeated_query_ids (torch.Tensor): Repeated query user IDs. Tensor shape: [num_positives]
            candidate_sampling_probability (Optional[torch.Tensor]): Optional tensor of candidate sampling probabilities.
                When given will be used to correct the logits to reflect the sampling probability of negative candidates.
                Tensor shape: [num_positives + num_hard_negatives + num_random_negatives]
        """
        loss = self._calculate_batch_retrieval_loss(
            scores=repeated_candidate_scores,
            candidate_sampling_probability=candidate_sampling_probability,
            query_ids=repeated_query_ids,
            candidate_ids=candidate_ids,
            device=device,
        )
        return loss
