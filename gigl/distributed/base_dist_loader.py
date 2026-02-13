"""
GiGL's mode-agnostic distributed data loader.

Replaces GLT's DistLoader. Delegates all sampling lifecycle to a SamplingEngine,
and provides a composable ``_base_collate`` method for converting SampleMessages
into PyG Data/HeteroData objects.
"""

from typing import List, Optional, Union

import torch
from graphlearn_torch.channel import SampleMessage
from graphlearn_torch.loader import to_data, to_hetero_data
from graphlearn_torch.sampler import (
    HeteroSamplerOutput,
    SamplerOutput,
    SamplingConfig,
    SamplingType,
)
from graphlearn_torch.typing import EdgeType, NodeType, as_str, reverse_edge_type
from graphlearn_torch.utils import ensure_device, python_exit_status
from torch_geometric.data import Data, HeteroData

from gigl.distributed.sampling_engine import SamplingEngine


class BaseDistLoader(object):
    """GiGL's mode-agnostic distributed data loader.

    Replaces GLT's DistLoader. Delegates all sampling lifecycle to a
    :class:`SamplingEngine` instance.

    Subclasses override ``_collate_fn`` and use ``_base_collate`` for the core
    SampleMessage-to-PyG conversion. This composable pattern allows each
    subclass to control the collation pipeline explicitly without relying on
    ``super()._collate_fn()``.

    Args:
        engine: A :class:`SamplingEngine` that handles epoch start, sample
            retrieval, and shutdown.
        sampling_config: Configuration for sampling (batch size, neighbors, etc.).
        to_device: Target device for collated results.
        input_type: The node type of the input seeds (for heterogeneous graphs).
        node_types: List of node types in the graph.
        edge_types: List of edge types in the graph.
    """

    def __init__(
        self,
        engine: SamplingEngine,
        sampling_config: SamplingConfig,
        to_device: torch.device,
        input_type: Optional[Union[str, NodeType]] = None,
        node_types: Optional[List[NodeType]] = None,
        edge_types: Optional[List[EdgeType]] = None,
    ):
        self._engine = engine
        self.sampling_config = sampling_config
        # Unpack commonly used fields for _base_collate compatibility
        self.sampling_type = sampling_config.sampling_type
        self.batch_size = sampling_config.batch_size
        self.edge_dir = sampling_config.edge_dir
        self.to_device = to_device
        self._input_type = input_type
        self._epoch = 0
        self._num_recv = 0
        self._shutdowned = False

        self._set_ntypes_and_etypes(node_types, edge_types)

    def __del__(self):
        if python_exit_status is True or python_exit_status is None:
            return
        self.shutdown()

    def shutdown(self):
        """Release all resources held by the sampling engine."""
        if self._shutdowned:
            return
        self._engine.shutdown()
        self._shutdowned = True

    def __iter__(self):
        self._num_recv = 0
        self._engine.start_epoch(self._epoch)
        self._epoch += 1
        return self

    def __next__(self):
        if self._num_recv == self._engine.num_expected:
            raise StopIteration

        msg = self._engine.get_sample()
        if msg is None:
            raise StopIteration  # Graph store mode: server signals end of epoch

        result = self._collate_fn(msg)
        self._num_recv += 1
        return result

    def _set_ntypes_and_etypes(
        self,
        node_types: Optional[List[NodeType]],
        edge_types: Optional[List[EdgeType]],
    ):
        """Set node/edge type metadata used by ``_base_collate``.

        Ported from GLT DistLoader._set_ntypes_and_etypes.
        """
        self._node_types = node_types or []
        self._edge_types = edge_types
        self._reversed_edge_types: List[EdgeType] = []
        self._etype_str_to_rev: dict[str, EdgeType] = {}
        if self._edge_types is not None:
            for etype in self._edge_types:
                rev_etype = reverse_edge_type(etype)
                if self.edge_dir == "out":
                    self._reversed_edge_types.append(rev_etype)
                    self._etype_str_to_rev[as_str(etype)] = rev_etype
                elif self.edge_dir == "in":
                    self._reversed_edge_types.append(etype)
                    self._etype_str_to_rev[as_str(rev_etype)] = etype

    def _base_collate(self, msg: SampleMessage) -> Union[Data, HeteroData]:
        """Core collation: converts a SampleMessage into PyG Data/HeteroData.

        Ported verbatim from GLT DistLoader._collate_fn. This is a standalone
        method so subclasses can compose collation steps explicitly::

            def _collate_fn(self, msg):
                data = self._base_collate(msg)
                data = my_custom_transform(data)
                return data
        """
        ensure_device(self.to_device)
        is_hetero = bool(msg["#IS_HETERO"])

        # Extract metadata
        _metadata_dict: dict[str, torch.Tensor] = {}
        for k in msg.keys():
            if k.startswith("#META."):
                meta_key = str(k[6:])
                _metadata_dict[meta_key] = msg[k].to(self.to_device)
        metadata: Optional[dict[str, torch.Tensor]] = (
            _metadata_dict if _metadata_dict else None
        )

        # Heterogeneous sampling results
        if is_hetero:
            node_dict, row_dict, col_dict, edge_dict = {}, {}, {}, {}
            nfeat_dict, efeat_dict = {}, {}
            num_sampled_nodes_dict, num_sampled_edges_dict = {}, {}

            for ntype in self._node_types:
                ids_key = f"{as_str(ntype)}.ids"
                if ids_key in msg:
                    node_dict[ntype] = msg[ids_key].to(self.to_device)
                nfeat_key = f"{as_str(ntype)}.nfeats"
                if nfeat_key in msg:
                    nfeat_dict[ntype] = msg[nfeat_key].to(self.to_device)
                num_sampled_nodes_key = f"{as_str(ntype)}.num_sampled_nodes"
                if num_sampled_nodes_key in msg:
                    num_sampled_nodes_dict[ntype] = msg[num_sampled_nodes_key]

            for etype_str, rev_etype in self._etype_str_to_rev.items():
                rows_key = f"{etype_str}.rows"
                cols_key = f"{etype_str}.cols"
                if rows_key in msg:
                    # The edge index should be reversed.
                    row_dict[rev_etype] = msg[cols_key].to(self.to_device)
                    col_dict[rev_etype] = msg[rows_key].to(self.to_device)
                eids_key = f"{etype_str}.eids"
                if eids_key in msg:
                    edge_dict[rev_etype] = msg[eids_key].to(self.to_device)
                num_sampled_edges_key = f"{etype_str}.num_sampled_edges"
                if num_sampled_edges_key in msg:
                    num_sampled_edges_dict[rev_etype] = msg[num_sampled_edges_key]
                efeat_key = f"{etype_str}.efeats"
                if efeat_key in msg:
                    efeat_dict[rev_etype] = msg[efeat_key].to(self.to_device)

            nfeat_dict_or_none = nfeat_dict if len(nfeat_dict) > 0 else None
            efeat_dict_or_none = efeat_dict if len(efeat_dict) > 0 else None

            if self.sampling_config.sampling_type in [
                SamplingType.NODE,
                SamplingType.SUBGRAPH,
            ]:
                batch_key = f"{self._input_type}.batch"
                if msg.get(batch_key) is not None:
                    batch_dict = {
                        self._input_type: msg[f"{self._input_type}.batch"].to(
                            self.to_device
                        )
                    }
                else:
                    batch_dict = {
                        self._input_type: node_dict[self._input_type][: self.batch_size]
                    }
                batch_labels_key = f"{self._input_type}.nlabels"
                if batch_labels_key in msg:
                    batch_labels = msg[batch_labels_key].to(self.to_device)
                else:
                    batch_labels = None
                batch_label_dict = {self._input_type: batch_labels}
            else:
                batch_dict = {}
                batch_label_dict = {}

            output = HeteroSamplerOutput(
                node_dict,
                row_dict,
                col_dict,
                edge_dict if len(edge_dict) else None,
                batch_dict,
                num_sampled_nodes=num_sampled_nodes_dict,
                num_sampled_edges=num_sampled_edges_dict,
                edge_types=self._reversed_edge_types,
                input_type=self._input_type,
                device=self.to_device,
                metadata=metadata,
            )
            res_data = to_hetero_data(
                output,
                batch_label_dict,
                nfeat_dict_or_none,
                efeat_dict_or_none,
                self.edge_dir,
            )

        # Homogeneous sampling results
        else:
            ids = msg["ids"].to(self.to_device)
            rows = msg["rows"].to(self.to_device)
            cols = msg["cols"].to(self.to_device)
            eids = msg["eids"].to(self.to_device) if "eids" in msg else None
            num_sampled_nodes = (
                msg["num_sampled_nodes"] if "num_sampled_nodes" in msg else None
            )
            num_sampled_edges = (
                msg["num_sampled_edges"] if "num_sampled_edges" in msg else None
            )

            nfeats = msg["nfeats"].to(self.to_device) if "nfeats" in msg else None
            efeats = msg["efeats"].to(self.to_device) if "efeats" in msg else None

            if self.sampling_config.sampling_type in [
                SamplingType.NODE,
                SamplingType.SUBGRAPH,
            ]:
                if msg.get("batch") is not None:
                    batch = msg["batch"].to(self.to_device)
                else:
                    batch = ids[: self.batch_size]
                batch_labels = (
                    msg["nlabels"].to(self.to_device) if "nlabels" in msg else None
                )
            else:
                batch = None
                batch_labels = None

            # The edge index should be reversed.
            output = SamplerOutput(
                ids,
                cols,
                rows,
                eids,
                batch,
                num_sampled_nodes,
                num_sampled_edges,
                device=self.to_device,
                metadata=metadata,
            )
            res_data = to_data(output, batch_labels, nfeats, efeats)

        return res_data

    def _collate_fn(self, msg: SampleMessage) -> Union[Data, HeteroData]:
        """Default collation. Subclasses override this to add post-processing.

        The default implementation simply calls ``_base_collate``.
        """
        return self._base_collate(msg)
