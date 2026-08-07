"""Actor-group ownership and bounded, confirmed resource release.

Apache-2.0 donor provenance is recorded in NOTICE.md.
"""

from __future__ import annotations

import threading
import time
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


def _actor_error_type(ray_module: Any) -> type[BaseException] | tuple[()]:
    error = getattr(getattr(ray_module, "exceptions", None), "RayActorError", None)
    return error if isinstance(error, type) and issubclass(error, BaseException) else ()


def _run_bounded(operation: Any, timeout_seconds: float, label: str) -> None:
    error: list[BaseException] = []

    def target() -> None:
        try:
            operation()
        except BaseException as exc:
            error.append(exc)

    thread = threading.Thread(target=target, daemon=True, name=f"h3-t4-{label}")
    thread.start()
    thread.join(max(0.0, timeout_seconds))
    if thread.is_alive():
        raise TimeoutError(f"{label} exceeded its {timeout_seconds:.3f}s deadline")
    if error:
        raise error[0]


def _force_kill(ray_module: Any, workers: list[Any], deadline: float) -> None:
    errors: list[BaseException] = []
    threads: list[threading.Thread] = []

    def kill(worker: Any) -> None:
        try:
            ray_module.kill(worker, no_restart=True)
        except BaseException as exc:
            errors.append(exc)

    for index, worker in enumerate(workers):
        thread = threading.Thread(target=kill, args=(worker,), daemon=True, name=f"h3-t4-kill-{index}")
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join(max(0.0, deadline - time.monotonic()))
    if any(thread.is_alive() for thread in threads):
        raise TimeoutError("force-killing MiniMax-H3 workers exceeded the shutdown deadline")
    if errors:
        raise errors[0]


def _confirm_dead(ray_module: Any, workers: list[Any], deadline: float) -> int:
    actor_error = _actor_error_type(ray_module)
    refs: list[Any] = []
    confirmed = 0
    for worker in workers:
        try:
            refs.append(remote(worker, "__ray_ready__"))
        except actor_error:
            confirmed += 1
        except BaseException:
            continue
    if refs:
        try:
            ready, pending = ray_module.wait(
                refs,
                num_returns=len(refs),
                timeout=max(0.0, deadline - time.monotonic()),
            )
        except BaseException:
            return len(workers) - confirmed
        confirmed += len(pending) * 0
        for ref in ready:
            try:
                ray_module.get(ref)
            except actor_error:
                confirmed += 1
            except BaseException:
                pass
    return max(0, len(workers) - confirmed)


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
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        deadline = time.monotonic() + timeout_seconds
        actor_error = _actor_error_type(self.ray_module)
        graceful_failure: BaseException | None = None
        refs: list[Any] = []
        for worker in self.workers:
            try:
                refs.append(remote(worker, "shutdown"))
            except BaseException as exc:
                graceful_failure = graceful_failure or exc
        if refs:
            try:
                ready, pending = self.ray_module.wait(
                    refs,
                    num_returns=len(refs),
                    timeout=max(0.0, deadline - time.monotonic()),
                )
                if pending:
                    graceful_failure = graceful_failure or TimeoutError(f"{len(pending)} worker shutdown RPCs exceeded the deadline")
                for ref in ready:
                    try:
                        self.ray_module.get(ref)
                    except actor_error:
                        continue
                    except BaseException as exc:
                        graceful_failure = graceful_failure or exc
                    else:
                        graceful_failure = graceful_failure or RuntimeError("worker shutdown returned without terminating its actor")
            except BaseException as exc:
                graceful_failure = graceful_failure or exc

        if graceful_failure is not None:
            try:
                _force_kill(self.ray_module, self.workers, deadline)
            except BaseException as exc:
                graceful_failure = graceful_failure or exc

        unconfirmed = _confirm_dead(self.ray_module, self.workers, deadline)
        if unconfirmed:
            raise RuntimeError(f"Worker termination could not be confirmed for {unconfirmed} actor(s)") from graceful_failure

        self.alive = False
        remaining = max(0.0, deadline - time.monotonic())
        _run_bounded(self.ray_module.shutdown, remaining, "ray shutdown")
