"""Fixed topology for the standalone two-T4 MiniMax-H3 runtime."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExactH3T4Topology:
    world_size: int = 2
    ulysses_degree: int = 2
    ring_degree: int = 1
    cfg_degree: int = 1
    dp_degree: int = 1
    fsdp: bool = True
    fsdp_cpu_offload: bool = False
    attention: str = "TORCH_EFFICIENT"

    def as_worker_config(self) -> dict[str, object]:
        return {
            "world_size": self.world_size,
            "ulysses_degree": self.ulysses_degree,
            "ring_degree": self.ring_degree,
            "cfg_degree": self.cfg_degree,
            "dp_degree": self.dp_degree,
            "fsdp": self.fsdp,
            "fsdp_cpu_offload": self.fsdp_cpu_offload,
            "attention": self.attention,
        }
