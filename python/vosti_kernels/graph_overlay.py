"""Per-engine CUDA-graph overlay for the canonical model forward.

This module deliberately does not implement a second forward path.  A graph
entry can only be created by surrounding the Rust/Verus ``model_forward``
call with ``capture_begin`` / ``capture_end_and_replay``.  Ready entries then
replay that captured launch sequence after copying current step values into
the persistent captured input tensors.

CUDA capture/replay fidelity is trusted.  The Rust boundary gives replay the
same contract as the eager ``model_forward``; this module supplies the runtime
protocol and keeps graph state local to one engine-owned overlay object.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Final

import torch


# @kernel-bridge-begin vosti_kernels::cuda_graph_overlay
_INPUT_NAMES: Final = (
    "input_ids",
    "positions",
    "slot_mapping",
    "cu_seqlens_q",
    "cu_seqlens_k",
    "block_table",
)
# Correctness supports every StepMode, but the default performance policy
# captures only Decode (1).  Prefill and Mixed signatures are exact-only and
# high-churn under dynamic arrivals, so their synchronous capture cost does
# not reliably amortize.  Decode is the mode with the checked padded-cover
# replay optimization.
_CAPTURE_MODES: Final = frozenset({1})
_DECODE_MODE: Final = 1
_NO_WRITE_SLOT: Final = -1


@dataclass(frozen=True, order=True)
class GraphSignature:
    """Every host-side value currently known to affect launch topology.

    The deployed paged-attention adapter uses ``max_seqlen_k`` only for a
    host-side coverage guard; it is neither a kernel argument nor a grid
    input.  ``block_table_width`` captures that guard's relevant bucket and
    therefore permits decode reuse while the exact KV length grows within a
    page.  Any adapter change that makes max-K launch-relevant must extend this
    signature before it can be source-attested.
    """

    mode: int
    num_tokens: int
    num_seqs: int
    max_seqlen_q: int
    block_table_width: int

    def as_tuple(self) -> tuple[int, ...]:
        return (
            self.mode,
            self.num_tokens,
            self.num_seqs,
            self.max_seqlen_q,
            self.block_table_width,
        )


class CudaGraphOverlay:
    """Graph cache owned by one engine/runtime wrapper.

    The object is intentionally not a module-global singleton.  Captured
    launches contain fixed model-weight, KV-cache, and buffer addresses, so a
    caller must create one overlay per engine and must not pass it to another
    engine. Pure-decode replay may pad into a larger captured signature. Pad
    rows use the verified ``-1`` no-write KV slot and their logits are sliced
    away, so covering replay neither needs nor mutates a scratch cache page.
    The service/example wrapper is responsible for maintaining the engine
    pairing; cross-engine reuse is outside the trusted replay premise.
    """

    EAGER = 0
    REPLAY = 1
    CAPTURE = 2

    def __init__(self) -> None:
        self._graphs: dict[GraphSignature, dict] = {}
        self._warmed: set[GraphSignature] = set()
        self._pool = None
        self._active: dict | None = None
        self._poisoned_reason: str | None = None
        self._eager_count = 0
        self._capture_count = 0
        self._replay_count = 0
        self._cover_replay_count = 0

    @staticmethod
    def _signature(
        mode: int,
        num_tokens: int,
        num_seqs: int,
        max_seqlen_q: int,
        block_table_width: int,
    ) -> GraphSignature:
        values = tuple(
            int(value)
            for value in (
                mode,
                num_tokens,
                num_seqs,
                max_seqlen_q,
                block_table_width,
            )
        )
        if any(value < 0 for value in values):
            raise ValueError(f"negative CUDA-graph signature component: {values}")
        return GraphSignature(*values)

    def _require_healthy(self) -> None:
        if self._poisoned_reason is not None:
            raise RuntimeError(
                f"CUDA-graph overlay is poisoned: {self._poisoned_reason}"
            )

    @staticmethod
    def _is_pure_decode(signature: GraphSignature) -> bool:
        return (
            signature.mode == _DECODE_MODE
            and signature.max_seqlen_q == 1
            and signature.num_tokens == signature.num_seqs
            and signature.num_tokens > 0
        )

    def _covering_signature(
        self,
        signature: GraphSignature,
        allow_decode_cover: bool = True,
    ) -> GraphSignature | None:
        """Return the smallest admitted capture covering ``signature``.

        Exact matches are valid for every capture mode. Shape covering is
        intentionally limited to pure decode: every real and padding segment
        then has one query row, while mixed/prefill retain exact semantics.
        """

        if signature in self._graphs:
            return signature
        if not allow_decode_cover:
            return None
        if not self._is_pure_decode(signature):
            return None
        candidates = (
            captured
            for captured in self._graphs
            if self._is_pure_decode(captured)
            # Non-exact covering replay is proved and materialized by adding
            # inert sequence rows.  A same-batch graph with only a wider block
            # table has no padding rows and is therefore not an admitted
            # cover; let that exact signature warm and capture independently.
            and captured.num_tokens > signature.num_tokens
            and captured.block_table_width >= signature.block_table_width
        )
        return min(
            candidates,
            key=lambda captured: (
                captured.num_tokens,
                captured.block_table_width,
            ),
            default=None,
        )

    def _poison(self, reason: str) -> None:
        if self._poisoned_reason is None:
            self._poisoned_reason = reason

    def probe(
        self,
        mode: int,
        num_tokens: int,
        num_seqs: int,
        max_seqlen_q: int,
        block_table_width: int,
        allow_decode_cover: bool = True,
    ) -> int:
        """Choose eager, capture, or replay for one exact launch signature."""

        self._require_healthy()
        signature = self._signature(
            mode,
            num_tokens,
            num_seqs,
            max_seqlen_q,
            block_table_width,
        )
        # Empty schedules do not provide a useful graph and may not launch all
        # of the kernels present in a nonempty forward.
        if signature.num_tokens == 0 or signature.num_seqs == 0:
            self._eager_count += 1
            return self.EAGER
        if signature.mode not in _CAPTURE_MODES:
            self._eager_count += 1
            return self.EAGER
        if self._covering_signature(signature, allow_decode_cover) is not None:
            return self.REPLAY
        # The first occurrence warms Triton/JIT state outside stream capture.
        if signature not in self._warmed:
            self._warmed.add(signature)
            self._eager_count += 1
            return self.EAGER
        return self.CAPTURE

    def capture_begin(
        self,
        mode: int,
        num_tokens: int,
        num_seqs: int,
        max_seqlen_q: int,
        block_table_width: int,
        capture_tensor: torch.Tensor,
    ) -> None:
        """Begin recording the immediately following canonical model forward."""

        self._require_healthy()
        if self._active is not None:
            self._poison("nested capture")
            raise RuntimeError("nested CUDA-graph capture")
        signature = self._signature(
            mode,
            num_tokens,
            num_seqs,
            max_seqlen_q,
            block_table_width,
        )
        if signature in self._graphs:
            self._poison("capture requested for a ready signature")
            raise RuntimeError(f"CUDA graph already exists for {signature}")
        if capture_tensor.device.type != "cuda":
            self._poison(f"capture tensor is not CUDA: {capture_tensor.device}")
            raise RuntimeError("CUDA-graph capture requires a CUDA input tensor")
        device_context = torch.cuda.device(capture_tensor.device)
        device_context.__enter__()
        try:
            if self._pool is None:
                self._pool = torch.cuda.graph_pool_handle()
            # Capture is rare and requires all warmup work to have completed.
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            context = torch.cuda.graph(graph, pool=self._pool)
            context.__enter__()
            self._active = {
                "signature": signature,
                "graph": graph,
                "context": context,
                "device": capture_tensor.device,
                "device_context": device_context,
                "origin": "model_forward",
            }
        except Exception as error:
            self._active = None
            self._poison(f"capture_begin failed: {error!r}")
            device_context.__exit__(type(error), error, error.__traceback__)
            raise

    def capture_end_and_replay(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        slot_mapping: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        block_table: torch.Tensor,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        """Finish a model-forward capture and execute it once for real output."""

        self._require_healthy()
        if self._active is None:
            self._poison("capture_end without capture_begin")
            raise RuntimeError("capture_end without active CUDA-graph capture")
        entry = self._active
        self._active = None
        try:
            entry["context"].__exit__(None, None, None)
            del entry["context"]
            entry["inputs"] = dict(
                zip(
                    _INPUT_NAMES,
                    (
                        input_ids,
                        positions,
                        slot_mapping,
                        cu_seqlens_q,
                        cu_seqlens_k,
                        block_table,
                    ),
                )
            )
            entry["logits"] = logits
            signature = entry["signature"]
            if self._is_pure_decode(signature):
                entry["decode_cu_q_template"] = torch.arange(
                    signature.num_seqs + 1,
                    dtype=cu_seqlens_q.dtype,
                    device=cu_seqlens_q.device,
                )
                entry["decode_pad_offsets"] = torch.arange(
                    1,
                    signature.num_seqs + 1,
                    dtype=cu_seqlens_k.dtype,
                    device=cu_seqlens_k.device,
                )
                entry["cover_fill_state"] = None
            self._graphs[signature] = entry
            # Stream capture records the canonical forward.  Replay once so
            # this step observes a physically materialized result and KV state.
            entry["graph"].replay()
            self._capture_count += 1
            self._replay_count += 1
            return logits
        except Exception as error:
            self._graphs.pop(entry.get("signature"), None)
            self._poison(f"capture finalization/replay failed: {error!r}")
            raise
        finally:
            entry.pop("device_context").__exit__(None, None, None)

    def replay(
        self,
        mode: int,
        max_seqlen_q: int,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        slot_mapping: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        block_table: torch.Tensor,
        allow_decode_cover: bool = True,
    ) -> torch.Tensor:
        """Replay an exact or pure-decode covering graph."""

        self._require_healthy()
        num_tokens = int(input_ids.shape[0])
        num_seqs = int(cu_seqlens_q.shape[0]) - 1
        block_table_width = int(block_table.shape[1])
        signature = self._signature(
            mode,
            num_tokens,
            num_seqs,
            max_seqlen_q,
            block_table_width,
        )
        captured_signature = self._covering_signature(
            signature, allow_decode_cover
        )
        if captured_signature is None:
            self._poison(f"replay without captured model_forward: {signature}")
            raise RuntimeError(f"no CUDA graph for {signature}")
        entry = self._graphs[captured_signature]
        if entry.get("origin") != "model_forward":
            self._poison("graph entry has invalid capture provenance")
            raise RuntimeError("CUDA graph was not captured from model_forward")
        try:
            current = {
                "input_ids": input_ids,
                "positions": positions,
                "slot_mapping": slot_mapping,
                "cu_seqlens_q": cu_seqlens_q,
                "cu_seqlens_k": cu_seqlens_k,
                "block_table": block_table,
            }
            with torch.cuda.device(entry["device"]):
                if captured_signature == signature:
                    self._install_exact_inputs(entry, current)
                else:
                    self._install_decode_cover_inputs(
                        entry, current, signature, captured_signature
                    )
                entry["graph"].replay()
            self._replay_count += 1
            if captured_signature != signature:
                self._cover_replay_count += 1
                return entry["logits"][:num_seqs]
            return entry["logits"]
        except Exception as error:
            # Once replay has been attempted, the KV state may be partially
            # mutated.  Never fall back to eager execution in this engine.
            self._poison(f"replay failed: {error!r}")
            raise

    @staticmethod
    def _require_same_tensor_role(
        name: str, captured: torch.Tensor, current: torch.Tensor
    ) -> None:
        if captured.device != current.device:
            raise RuntimeError(
                f"graph input device drift for {name}: "
                f"{captured.device} != {current.device}"
            )
        if captured.dtype != current.dtype:
            raise RuntimeError(
                f"graph input dtype drift for {name}: "
                f"{captured.dtype} != {current.dtype}"
            )

    @classmethod
    def _install_exact_inputs(cls, entry: dict, current: dict) -> None:
        captured = entry["inputs"]
        for name in _INPUT_NAMES:
            cls._require_same_tensor_role(name, captured[name], current[name])
            if captured[name].shape != current[name].shape:
                raise RuntimeError(
                    f"exact graph input shape drift for {name}: "
                    f"{tuple(captured[name].shape)} != {tuple(current[name].shape)}"
                )
            captured[name].copy_(current[name], non_blocking=True)
        # An exact replay overwrites every row, including rows that a prior
        # smaller covering replay had made inert.  Invalidate the shape cache
        # so the next cover reinstalls -1 no-write slots and zero block-table
        # padding even when its (batch, width) matches that prior cover.
        if "cover_fill_state" in entry:
            entry["cover_fill_state"] = None

    @classmethod
    def _install_decode_cover_inputs(
        cls,
        entry: dict,
        current: dict,
        actual: GraphSignature,
        captured_signature: GraphSignature,
    ) -> None:
        """Install a smaller decode step into a covering capture's buffers."""

        if not cls._is_pure_decode(actual) or not cls._is_pure_decode(
            captured_signature
        ):
            raise RuntimeError("covering replay is restricted to pure decode")
        batch = actual.num_seqs
        capacity = captured_signature.num_seqs
        width = actual.block_table_width
        captured_width = captured_signature.block_table_width
        if batch >= capacity or width > captured_width:
            raise RuntimeError(
                f"invalid decode cover {actual} with {captured_signature}"
            )

        buffers = entry["inputs"]
        expected_shapes = {
            "input_ids": (batch,),
            "positions": (batch,),
            "slot_mapping": (batch,),
            "cu_seqlens_q": (batch + 1,),
            "cu_seqlens_k": (batch + 1,),
            "block_table": (batch, width),
        }
        captured_shapes = {
            "input_ids": (capacity,),
            "positions": (capacity,),
            "slot_mapping": (capacity,),
            "cu_seqlens_q": (capacity + 1,),
            "cu_seqlens_k": (capacity + 1,),
            "block_table": (capacity, captured_width),
        }
        for name in _INPUT_NAMES:
            cls._require_same_tensor_role(name, buffers[name], current[name])
            if tuple(current[name].shape) != expected_shapes[name]:
                raise RuntimeError(
                    f"decode cover current shape drift for {name}: "
                    f"{tuple(current[name].shape)} != {expected_shapes[name]}"
                )
            if tuple(buffers[name].shape) != captured_shapes[name]:
                raise RuntimeError(
                    f"decode cover captured shape drift for {name}: "
                    f"{tuple(buffers[name].shape)} != {captured_shapes[name]}"
                )

        buffers["input_ids"][:batch].copy_(
            current["input_ids"], non_blocking=True
        )
        buffers["positions"][:batch].copy_(
            current["positions"], non_blocking=True
        )
        buffers["slot_mapping"][:batch].copy_(
            current["slot_mapping"], non_blocking=True
        )
        buffers["cu_seqlens_q"][: batch + 1].copy_(
            current["cu_seqlens_q"], non_blocking=True
        )
        buffers["cu_seqlens_k"][: batch + 1].copy_(
            current["cu_seqlens_k"], non_blocking=True
        )
        buffers["block_table"][:batch, :width].copy_(
            current["block_table"], non_blocking=True
        )

        # Static padding fields only need refilling when the covered shape
        # changes. The cumulative K tail is dynamic because real context
        # lengths advance on every decode step.
        fill_state = (batch, width)
        if entry.get("cover_fill_state") != fill_state:
            buffers["input_ids"][batch:].zero_()
            buffers["positions"][batch:].zero_()
            buffers["slot_mapping"][batch:].fill_(_NO_WRITE_SLOT)
            buffers["cu_seqlens_q"][batch + 1 :].copy_(
                entry["decode_cu_q_template"][batch + 1 :],
                non_blocking=True,
            )
            if width < captured_width:
                buffers["block_table"][:batch, width:].zero_()
            buffers["block_table"][batch:].zero_()
            entry["cover_fill_state"] = fill_state

        pads = capacity - batch
        buffers["cu_seqlens_k"][batch + 1 :].copy_(
            buffers["cu_seqlens_k"][batch]
            + entry["decode_pad_offsets"][:pads],
            non_blocking=True,
        )

    def stats_json(self) -> str:
        payload = {
            "capture_count": self._capture_count,
            "cover_replay_count": self._cover_replay_count,
            "eager_count": self._eager_count,
            "graph_count": len(self._graphs),
            "poisoned_reason": self._poisoned_reason,
            "replay_count": self._replay_count,
            "signatures": [
                list(signature.as_tuple()) for signature in sorted(self._graphs)
            ],
            "warmed_count": len(self._warmed),
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))
# @kernel-bridge-end vosti_kernels::cuda_graph_overlay
