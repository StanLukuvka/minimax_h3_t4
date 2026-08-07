"""Fail-closed rank-synchronized Spectrum integration for worker-local H3 calls."""

from __future__ import annotations

from collections.abc import Callable

import torch

from .runtime import SpectrumRuntime, StepDecision


def _distributed_bool(value: bool, *, reduce: str) -> bool:
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() <= 1:
        return bool(value)
    from xfuser.core.distributed import get_sp_group

    flag = torch.tensor(int(bool(value)), device=torch.device("cuda", torch.cuda.current_device()), dtype=torch.int32)
    operation = dist.ReduceOp.MAX if reduce == "any" else dist.ReduceOp.MIN
    get_sp_group().all_reduce(flag, op=operation)
    return bool(flag.item())


def any_sequence_rank(value: bool) -> bool:
    return _distributed_bool(value, reduce="any")


def all_sequence_ranks(value: bool) -> bool:
    return _distributed_bool(value, reduce="all")


def sanitize_prediction_bounded(
    prediction: torch.Tensor,
    *,
    expected_shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
    chunk_bytes: int = 4 * 1024 * 1024,
) -> torch.Tensor | None:
    if not isinstance(prediction, torch.Tensor) or tuple(prediction.shape) != tuple(expected_shape):
        return None
    if not prediction.is_floating_point() or chunk_bytes <= 0:
        return None

    source = prediction.detach().contiguous().view(-1)
    element_size = max(1, source.element_size())
    chunk_elements = max(1, int(chunk_bytes) // element_size)
    output = torch.empty(expected_shape, device=device, dtype=dtype)
    target = output.view(-1)

    for start in range(0, source.numel(), chunk_elements):
        stop = min(source.numel(), start + chunk_elements)
        chunk = source[start:stop]
        if not bool(torch.isfinite(chunk).all().item()):
            return None
        target[start:stop].copy_(chunk.to(device=device, dtype=dtype))
    return output


class SpectrumStepAdapter:
    def __init__(
        self,
        runtime: SpectrumRuntime,
        *,
        sync_mode: Callable[[bool], bool] = any_sequence_rank,
        sync_all: Callable[[bool], bool] = all_sequence_ranks,
    ) -> None:
        self.runtime = runtime
        self.sync_mode = sync_mode
        self.sync_all = sync_all

    def begin_step(self, timestep: torch.Tensor | float, *, topology: tuple[object, ...]) -> StepDecision:
        topology_valid = False
        try:
            topology_valid = self.runtime.check_topology(topology)
        except (RuntimeError, ValueError, TypeError):
            topology_valid = False
        if not self.sync_all(topology_valid):
            self.runtime.disable_for_run("packed H3 topology was invalid or changed on one or more ranks")

        decision: StepDecision | None = None
        decision_valid = False
        try:
            decision = self.runtime.begin_step(timestep)
            decision_valid = True
        except (RuntimeError, ValueError, TypeError):
            decision_valid = False
        if not self.sync_all(decision_valid):
            decision = self.runtime.fail_closed_step(
                timestep,
                "Spectrum step initialization failed on one or more ranks",
            )
        if decision is None:  # pragma: no cover - defensive invariant
            raise RuntimeError("Spectrum runtime did not produce a step decision")
        any_actual = self.sync_mode(decision.actual)
        if any_actual and not decision.actual:
            self.runtime.force_actual("another sequence-parallel rank requires exact execution")
        current = self.runtime.current_step
        if current is None:  # pragma: no cover - defensive invariant
            raise RuntimeError("Spectrum runtime lost its active step")
        return current

    def try_forecast(
        self,
        *,
        expected_shape: tuple[int, ...],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        step = self.runtime.current_step
        if step is None:
            raise RuntimeError("forecast requested without an active Spectrum step")
        if step.actual:
            return None
        prediction: torch.Tensor | None
        try:
            raw_prediction = self.runtime.predict(device=device, dtype=dtype)
            prediction = sanitize_prediction_bounded(
                raw_prediction,
                expected_shape=expected_shape,
                device=device,
                dtype=dtype,
            )
        except (RuntimeError, ValueError, TypeError):
            prediction = None
        if not self.sync_all(prediction is not None):
            self.runtime.force_actual("forecast was missing, malformed, or nonfinite on one or more ranks")
            return None
        return prediction

    def observe_actual(self, feature: torch.Tensor, *, topology: tuple[object, ...]) -> None:
        valid = True
        try:
            self.runtime.observe_actual(feature, topology=topology)
        except (RuntimeError, ValueError, TypeError):
            valid = False
        if not self.sync_all(valid):
            self.runtime.force_actual("actual Spectrum history was invalid on one or more ranks")

    def finish_step(self) -> None:
        valid = True
        try:
            self.runtime.finish_step()
        except (RuntimeError, ValueError, TypeError):
            valid = False
        if not self.sync_all(valid):
            self.runtime.abort_run("Spectrum step completion failed on one or more ranks")
            raise RuntimeError("Spectrum step completion failed on one or more ranks")
