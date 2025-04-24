from apache_beam import coders

from gigl.src.common.types.pb_wrappers.graph_data_types import (
    EdgePbWrapper,
    GraphPbWrapper,
    NodePbWrapper,
)
from snapchat.research.gbml import graph_schema_pb2

"""
In dataflow, we use wrapper object as key, value beam DoFn outputs and also for shuffle. We only 
need to serialize the proto itself and not the wrapper. The proto objects also do not contain Map, 
therefore can be deterministic. Which is specially important when shuffling with proto wrapper 
objects as key. 
"""


class GraphPbWrapperCoder(coders.Coder):
    def __init__(self) -> None:
        super().__init__()
        self.proto_coder = coders.ProtoCoder(graph_schema_pb2.Graph)

    def encode(self, o: GraphPbWrapper) -> bytes:
        return self.proto_coder.encode(o.pb)

    def decode(self, s: bytes):
        return GraphPbWrapper(pb=self.proto_coder.decode(s))

    def is_deterministic(self):
        return True


class NodePbWrapperCoder(coders.Coder):
    def __init__(self) -> None:
        super().__init__()
        self.proto_coder = coders.ProtoCoder(graph_schema_pb2.Node)

    def encode(self, o: NodePbWrapper) -> bytes:
        return self.proto_coder.encode(o.pb)

    def decode(self, s: bytes):
        return NodePbWrapper(pb=self.proto_coder.decode(s))

    def is_deterministic(self):
        return True


class EdgePbWrapperCoder(coders.Coder):
    def __init__(self) -> None:
        super().__init__()
        self.proto_coder = coders.ProtoCoder(graph_schema_pb2.Edge)

    def encode(self, o: EdgePbWrapper) -> bytes:
        return self.proto_coder.encode(o.pb)

    def decode(self, s: bytes):
        return EdgePbWrapper(pb=self.proto_coder.decode(s))

    def is_deterministic(self):
        return True
