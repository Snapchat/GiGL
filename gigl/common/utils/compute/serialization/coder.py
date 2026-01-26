from typing import Generic, Protocol, TypeVar

T = TypeVar("T")


class CoderProtocol(Protocol, Generic[T]):
    def encode(self, obj: T) -> bytes:
        ...

    def decode(self, byte_str: bytes) -> T:
        ...
