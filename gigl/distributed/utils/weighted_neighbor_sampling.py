"""Helpers for one-hop neighbor sampling that returns edge weights.

These helpers intentionally mirror GLT's one-hop sampling RPC shape. The only
behavioral difference is that the worker that performs ``sample_one_hop`` also
looks up weights for the returned edge ids and includes them in the response.
"""

from dataclasses import dataclass
from typing import Optional, cast

import torch
from graphlearn_torch.distributed.rpc import RpcCalleeBase
from graphlearn_torch.sampler import NeighborSampler
from graphlearn_torch.typing import EdgeType
from graphlearn_torch.utils import ensure_device


@dataclass(frozen=True)
class SortedEdgeWeightTable:
    """Local edge-weight table keyed by sampled global edge id."""

    edge_ids: torch.Tensor
    weights: torch.Tensor

    @classmethod
    def from_graph_topology(cls, graph: object) -> "SortedEdgeWeightTable":
        """Build a sorted edge-id table from a GLT Graph's local topology.

        Args:
            graph: A ``graphlearn_torch.data.Graph`` instance.

        Returns:
            Sorted edge ids and correspondingly permuted weights.

        Raises:
            ValueError: If the graph has no edge weights or has malformed topology.
        """
        topology = getattr(graph, "topo")
        edge_ids = topology.edge_ids
        weights = topology.edge_weights
        if weights is None:
            raise ValueError(
                "Weighted sampling requires local graph edge weights, but none were found."
            )
        if edge_ids is None:
            edge_ids = torch.arange(weights.numel(), dtype=torch.long)
        if edge_ids.numel() != weights.numel():
            raise ValueError(
                "Local graph edge_ids and edge_weights must have the same length "
                f"(got {edge_ids.numel()} ids and {weights.numel()} weights)."
            )

        sorted_edge_ids, order = torch.sort(edge_ids.to(torch.long).cpu())
        sorted_weights = weights.cpu()[order]
        return cls(edge_ids=sorted_edge_ids, weights=sorted_weights)

    def lookup(self, sampled_edge_ids: torch.Tensor) -> torch.Tensor:
        """Return local weights for sampled global edge ids.

        Args:
            sampled_edge_ids: 1-D tensor of sampled edge ids returned by GLT.

        Returns:
            A tensor of edge weights parallel to ``sampled_edge_ids``.

        Raises:
            KeyError: If a sampled edge id is not present in the local topology.
        """
        if sampled_edge_ids.numel() == 0:
            return self.weights.new_empty((0,)).to(sampled_edge_ids.device)
        if self.edge_ids.numel() == 0:
            raise KeyError("Cannot look up sampled edge ids in an empty edge table.")

        query = sampled_edge_ids.to(dtype=torch.long, device=self.edge_ids.device)
        positions = torch.searchsorted(self.edge_ids, query)
        in_range = positions < self.edge_ids.numel()
        matched = torch.zeros_like(in_range, dtype=torch.bool)
        if bool(in_range.any()):
            matched[in_range] = self.edge_ids[positions[in_range]] == query[in_range]
        if not bool(matched.all()):
            missing_ids = query[~matched][:10].tolist()
            raise KeyError(
                f"Sampled edge ids not found in local weighted graph: {missing_ids}"
            )

        return self.weights[positions].to(sampled_edge_ids.device)


class LocalEdgeWeightLookup:
    """Local worker lookup for edge weights keyed by sampled edge ids."""

    def __init__(self, local_graph: object):
        self._heterogeneous_graphs: Optional[dict[EdgeType, object]] = None
        self._heterogeneous_tables: Optional[dict[EdgeType, SortedEdgeWeightTable]] = (
            None
        )
        self._homogeneous_table: Optional[SortedEdgeWeightTable] = None
        if isinstance(local_graph, dict):
            self._heterogeneous_graphs = cast(dict[EdgeType, object], local_graph)
            self._heterogeneous_tables = {}
        else:
            self._homogeneous_table = SortedEdgeWeightTable.from_graph_topology(
                local_graph
            )

    def lookup(
        self,
        sampled_edge_ids: torch.Tensor,
        edge_type: Optional[EdgeType],
    ) -> torch.Tensor:
        """Return local edge weights for sampled edge ids."""
        if self._heterogeneous_tables is not None:
            if edge_type is None:
                raise ValueError(
                    "edge_type is required for heterogeneous weight lookup."
                )
            if edge_type not in self._heterogeneous_tables:
                if self._heterogeneous_graphs is None:
                    raise RuntimeError(
                        "heterogeneous graph table is not available for lookup."
                    )
                self._heterogeneous_tables[edge_type] = (
                    SortedEdgeWeightTable.from_graph_topology(
                        self._heterogeneous_graphs[edge_type]
                    )
                )
            return self._heterogeneous_tables[edge_type].lookup(sampled_edge_ids)
        if self._homogeneous_table is None:
            raise RuntimeError("homogeneous graph table is not available for lookup.")
        return self._homogeneous_table.lookup(sampled_edge_ids)


@dataclass
class WeightedNeighborOutput:
    """One-hop sampled neighbors plus sampled edge weights."""

    nbr: torch.Tensor
    nbr_num: torch.Tensor
    edge: torch.Tensor
    weight: torch.Tensor

    def to(self, device: torch.device) -> "WeightedNeighborOutput":
        """Move all output tensors to ``device``."""
        return WeightedNeighborOutput(
            nbr=self.nbr.to(device),
            nbr_num=self.nbr_num.to(device),
            edge=self.edge.to(device),
            weight=self.weight.to(device),
        )


@dataclass
class PartialWeightedNeighborOutput:
    """Weighted one-hop result for a subset of the requested source rows."""

    index: torch.Tensor
    output: WeightedNeighborOutput


def empty_weighted_neighbor_output(
    srcs: torch.Tensor,
    device: torch.device,
) -> WeightedNeighborOutput:
    """Create an empty weighted one-hop output for ``srcs``."""
    return WeightedNeighborOutput(
        nbr=torch.empty(0, dtype=torch.long, device=device),
        nbr_num=torch.zeros(srcs.size(0), dtype=torch.long, device=device),
        edge=torch.empty(0, dtype=torch.long, device=device),
        weight=torch.empty(0, dtype=torch.float32, device=device),
    )


def sample_local_one_hop_with_edge_weights(
    sampler: NeighborSampler,
    edge_weight_lookup: LocalEdgeWeightLookup,
    device: torch.device,
    srcs: torch.Tensor,
    num_nbr: int,
    edge_type: Optional[EdgeType],
) -> WeightedNeighborOutput:
    """Sample one hop locally and attach local edge weights in the same call.

    This is the local analogue of GLT's direct ``sampler.sample_one_hop`` call.
    The only added step is ``edge_weight_lookup.lookup(output.edge, edge_type)``;
    neighbor ids, neighbor counts, and sampled edge ids are otherwise returned
    exactly as GLT produced them.
    """
    output = sampler.sample_one_hop(srcs, num_nbr, edge_type)
    if output is None:
        return empty_weighted_neighbor_output(srcs=srcs, device=device)
    if output.edge is None:
        raise RuntimeError("Weighted sampling requires GLT to return edge ids.")
    weights = edge_weight_lookup.lookup(output.edge, edge_type)
    return WeightedNeighborOutput(
        nbr=output.nbr,
        nbr_num=output.nbr_num,
        edge=output.edge,
        weight=weights,
    )


class WeightedSamplingCallee(RpcCalleeBase):
    """RPC callee that samples one hop and returns edge weights with the result.

    This mirrors GLT's ``RpcSamplingCallee``. The routing, device handling, and
    remote ``sample_one_hop`` call are the same; the delta is that the callee
    converts ``NeighborOutput`` into ``WeightedNeighborOutput`` by attaching
    local edge weights before moving the result back to CPU for the RPC reply.
    """

    def __init__(
        self,
        sampler: NeighborSampler,
        edge_weight_lookup: LocalEdgeWeightLookup,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.sampler = sampler
        self.edge_weight_lookup = edge_weight_lookup
        self.device = device

    def call(self, *args, **kwargs) -> WeightedNeighborOutput:
        """Sample one hop on this worker and attach local edge weights."""
        ensure_device(self.device)
        output = sample_local_one_hop_with_edge_weights(
            sampler=self.sampler,
            edge_weight_lookup=self.edge_weight_lookup,
            device=self.device,
            srcs=args[0].to(self.device),
            num_nbr=args[1],
            edge_type=args[2],
        )
        return output.to(torch.device("cpu"))
