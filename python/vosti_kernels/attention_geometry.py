"""Immutable model-static geometry, independent of runtime token/batch sizes."""

# @kernel-bridge-begin vosti_kernels::attention_geometry
from dataclasses import dataclass


@dataclass(frozen=True)
class AttentionGeometry:
    query_heads: int
    kv_heads: int
    head_dim: int

    def __post_init__(self) -> None:
        for value in (self.query_heads, self.kv_heads, self.head_dim):
            if type(value) is not int or value <= 0:
                raise ValueError("attention geometry requires positive integers")
        if self.query_heads % self.kv_heads:
            raise ValueError("query heads must be divisible by KV heads")
        if self.head_dim % 2:
            raise ValueError("rotary attention requires an even head dimension")

    @property
    def query_width(self) -> int:
        return self.query_heads * self.head_dim

    @property
    def kv_width(self) -> int:
        return self.kv_heads * self.head_dim

    @property
    def kv_tail_shape(self) -> tuple[int, int]:
        return self.kv_heads, self.head_dim
# @kernel-bridge-end vosti_kernels::attention_geometry
