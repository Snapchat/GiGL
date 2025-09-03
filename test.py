from gigl.common import Uri, UriFactory
from gigl.distributed.sampler import RemoteUriSamplerInput, RemoteNodeInfo
from gigl.src.common.utils.file_loader import FileLoader
import tempfile
import torch
from gigl.types.graph import FeatureInfo
import pydantic


# class RNI(pydantic.BaseModel):
#     node_type: str
#     edge_types: list[tuple[str, str, str]]
#     node_tensor_uri: str
#     num_partitions: int
#     node_feature_info: FeatureInfo
#     edge_dir: str

#     @pydantic.field_serializer('node_feature_info')
#     def serialize_node_feature_info(self, node_feature_info: FeatureInfo) -> dict:
#         return {"dim": node_feature_info.dim, "dtype": str(node_feature_info.dtype)}


uri = UriFactory.create_uri("gs://gigl-test/test.pt")
print(uri)

rni = RemoteNodeInfo(
    node_type="user",
    edge_types=[("user", "to", "item")],
    node_tensor_uri=uri.uri,
    num_partitions=1,
    node_feature_info=FeatureInfo(dim=1, dtype=torch.float32),
    edge_feature_info={("user", "to", "item"): FeatureInfo(dim=2, dtype=torch.float32)},
    edge_dir="out",
)
print(f"{rni=}")
dumped = rni.dump()
print(f"{dumped=}")
with tempfile.NamedTemporaryFile("r+t") as temp_file:
    temp_file.write(dumped)
    temp_file.flush()
    temp_file.seek(0)
    print(f"{temp_file.name=}")
    #print(f"{temp_file.read()=}")
    u = UriFactory.create_uri(temp_file.name)
    print(f"{u=}, {type(u)=}")
    loaded = RemoteNodeInfo.load(u)
    print(f"{loaded=}")
