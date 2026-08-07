"""Actor-group ownership and bounded resource release."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .topology import ExactH3T4Topology


def remote(actor: Any, method: str, *args: Any, **kwargs: Any) -> Any:
    target = getattr(actor, method)
    remote_method = getattr(target, "remote", None)
    return remote_method(*args, **kwargs) if remote_method is not None else target(*args, **kwargs)


def resolve(ray_module: Any, value: Any) -> Any:
    getter = getattr(ray_module, "get", None)
    return getter(value) if getter is not None else value


@dataclass
class H3T4ActorGroup:
    workers: list[Any]
    ray_module: Any
    topology: ExactH3T4Topology = field(default_factory=ExactH3T4Topology)
    checkpoint: str | None = None
    alive: bool = True

    def require_alive(self) -> None:
        if not self.alive:
            raise RuntimeError("MiniMax-H3 two-T4 actor group is closed")

    def close(self, timeout_seconds: float = 30.0) -> None:
        if not self.alive:
            return
        self.alive = False
        refs: list[Any] = []
        for worker in self.workers:
            try:
                refs.append(remote(worker, "shutdown"))
            except Exception:
                continue
        if refs:
            try:
                self.ray_module.wait(refs, num_returns=len(refs), timeout=timeout_seconds)
            except Exception:
                pass
        for worker in self.workers:
            try:
                self.ray_module.kill(worker, no_restart=True)
            except Exception:
                pass
        try:
            self.ray_module.shutdown()
        except Exception:
            pass
