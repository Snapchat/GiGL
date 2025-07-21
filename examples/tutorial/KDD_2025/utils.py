from torch_geometric.nn import HGTConv
from torch_geometric.typing import Metadata


def init_model(
    metadata: Metadata = (  # Hard code metadata for toy graph example.
        ["user", "story"],  # node types
        [("user", "to", "story"), ("story", "to", "user")],  # edge types
    )
) -> HGTConv:
    return HGTConv(
        in_channels=-1,  # Will be inferred after first fowrard pass
        out_channels=16,  # This is the embedding size - just an example can be changed.
        metadata=metadata,
    )
