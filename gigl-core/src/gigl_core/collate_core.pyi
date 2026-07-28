import torch

NodeTypeStr = str
EdgeTypeTuple = tuple[str, str, str]

class CollateHeteroResult:
    @property
    def node(self) -> dict[NodeTypeStr, torch.Tensor]: ...
    @property
    def edge_index(self) -> dict[EdgeTypeTuple, torch.Tensor]: ...
    @property
    def edge(self) -> dict[EdgeTypeTuple, torch.Tensor]: ...
    @property
    def x(self) -> dict[NodeTypeStr, torch.Tensor]: ...
    @property
    def edge_attr(self) -> dict[EdgeTypeTuple, torch.Tensor]: ...
    @property
    def batch(self) -> dict[NodeTypeStr, torch.Tensor]: ...
    @property
    def num_sampled_nodes(self) -> dict[NodeTypeStr, torch.Tensor]: ...
    @property
    def num_sampled_edges(self) -> dict[EdgeTypeTuple, torch.Tensor]: ...

def collate_homogeneous(
    ids: torch.Tensor,
    rows: torch.Tensor,
    cols: torch.Tensor,
    eids: torch.Tensor | None,
    nfeats: torch.Tensor | None,
    efeats: torch.Tensor | None,
    batch: torch.Tensor | None,
    num_sampled_nodes: torch.Tensor | None,
    num_sampled_edges: torch.Tensor | None,
) -> dict[str, torch.Tensor | None]: ...
def collate_heterogeneous(
    msg: dict[str, torch.Tensor],
    node_types: list[str],
    edge_type_str_to_rev: dict[str, EdgeTypeTuple],
    reversed_edge_types: list[EdgeTypeTuple],
    input_type: str,
    has_batch: bool,
    batch_size: int,
) -> CollateHeteroResult: ...
