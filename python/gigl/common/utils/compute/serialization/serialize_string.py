from gigl.common.utils.compute.serialization.coder import CoderProtocol


class StringCoder(CoderProtocol[str]):
    def __init__(self, encoding: str = "utf-8"):
        self.encoding = encoding

    def encode(self, obj: str) -> bytes:
        return obj.encode(encoding=self.encoding)

    def decode(self, byte_str: bytes) -> str:
        return byte_str.decode(encoding=self.encoding)
