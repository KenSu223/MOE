"""Layer-streaming weight loader.

Reads safetensors shards lazily (memory-mapped), stacks the per-expert matrices of one decoder layer into fused
tensors, and hands the layer to the GPU. A background thread prefetches layer l+1 into pinned host buffers while
layer l is being computed, so a pass over the checkpoint is bounded by disk/page-cache bandwidth.

ext14 (Phase 6): dense families load their MLP as expert 0 (gate_up [1, 2*inter, hidden], down [1, hidden, inter]) and
have router = None; q/k/v biases (bq, bk, bv) and the Qwen2-MoE shared expert (sh_gate_up [2*shared_inter, hidden],
sh_down [hidden, shared_inter], sh_router [1, hidden]) are optional LayerWeights fields (None for the original families).
"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from typing import Optional

import torch
from safetensors import safe_open

from .arch import ArchSpec


@dataclass
class LayerWeights:
    ln1: torch.Tensor
    ln2: torch.Tensor
    wq: torch.Tensor          # (n_heads*head_dim, hidden)
    wk: torch.Tensor          # (n_kv*head_dim, hidden)
    wv: torch.Tensor
    wo: torch.Tensor          # (hidden, n_heads*head_dim)
    q_norm: Optional[torch.Tensor]
    k_norm: Optional[torch.Tensor]
    router: torch.Tensor      # (n_experts, hidden)
    gate_up: torch.Tensor     # (n_experts, 2*inter, hidden): rows [0:inter] = gate, [inter:2*inter] = up
    down: torch.Tensor        # (n_experts, hidden, inter)
    # ext14 (Phase 6) optional parts (None for qwen3_moe / mixtral / olmoe)
    bq: Optional[torch.Tensor] = None         # q/k/v projection biases (Qwen2, Qwen2-MoE)
    bk: Optional[torch.Tensor] = None
    bv: Optional[torch.Tensor] = None
    sh_gate_up: Optional[torch.Tensor] = None  # shared expert (2*shared_inter, hidden): gate rows, then up rows
    sh_down: Optional[torch.Tensor] = None     # shared expert (hidden, shared_inter)
    sh_router: Optional[torch.Tensor] = None   # shared expert gate (1, hidden)


class CheckpointStore:
    def __init__(self, snapshot: str, spec: ArchSpec, device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
        self.dir, self.spec, self.device, self.dtype = snapshot, spec, device, dtype
        idx = os.path.join(snapshot, "model.safetensors.index.json")
        if os.path.exists(idx):
            with open(idx) as f:
                self.weight_map = json.load(f)["weight_map"]
        else:
            self.weight_map = None
            self._single = os.path.join(snapshot, "model.safetensors")
        self._handles: dict[str, object] = {}
        self._lock = threading.Lock()

    # -- raw access -------------------------------------------------------------------------------------------
    def _handle(self, shard: str):
        with self._lock:
            h = self._handles.get(shard)
            if h is None:
                h = safe_open(os.path.join(self.dir, shard), framework="pt", device="cpu")
                self._handles[shard] = h
            return h

    def has(self, name: str) -> bool:
        return name in self.weight_map if self.weight_map is not None else name in self._handle(os.path.basename(self._single)).keys()

    def cpu_tensor(self, name: str) -> torch.Tensor:
        shard = self.weight_map[name] if self.weight_map is not None else os.path.basename(self._single)
        return self._handle(shard).get_tensor(name)

    def gpu_tensor(self, name: str) -> torch.Tensor:
        return self.cpu_tensor(name).to(self.device, self.dtype)

    # -- globals ----------------------------------------------------------------------------------------------
    def load_globals(self) -> dict[str, torch.Tensor]:
        s = self.spec
        embed = self.gpu_tensor(s.k_embed)
        head = embed if (s.tie_embeddings or not self.has(s.k_head)) else self.gpu_tensor(s.k_head)
        return {"embed": embed, "norm": self.gpu_tensor(s.k_norm), "head": head}

    # -- one layer --------------------------------------------------------------------------------------------
    def _fill_layer_host(self, l: int, buf: "HostLayerBuffers") -> None:
        s = self.spec
        for e in range(s.n_experts):
            buf.gate_up[e, : s.inter].copy_(self.cpu_tensor(s.k_gate.format(l=l, e=e)))
            buf.gate_up[e, s.inter :].copy_(self.cpu_tensor(s.k_up.format(l=l, e=e)))
            buf.down[e].copy_(self.cpu_tensor(s.k_down.format(l=l, e=e)))
        buf.small = {
            "ln1": self.cpu_tensor(s.k_ln1.format(l=l)), "ln2": self.cpu_tensor(s.k_ln2.format(l=l)),
            "wq": self.cpu_tensor(s.k_q.format(l=l)), "wk": self.cpu_tensor(s.k_k.format(l=l)),
            "wv": self.cpu_tensor(s.k_v.format(l=l)), "wo": self.cpu_tensor(s.k_o.format(l=l)),
            "router": self.cpu_tensor(s.k_router.format(l=l)) if s.k_router else None,  # ext14: None for dense families
            "q_norm": self.cpu_tensor(s.k_qn.format(l=l)) if s.k_qn else None,
            "k_norm": self.cpu_tensor(s.k_kn.format(l=l)) if s.k_kn else None,
        }
        if s.k_qb:  # ext14: q/k/v biases
            buf.small.update(bq=self.cpu_tensor(s.k_qb.format(l=l)), bk=self.cpu_tensor(s.k_kb.format(l=l)),
                             bv=self.cpu_tensor(s.k_vb.format(l=l)))
        if s.shared_inter:  # ext14: Qwen2-MoE shared expert
            buf.sh_gate_up[: s.shared_inter].copy_(self.cpu_tensor(s.k_sh_gate.format(l=l)))
            buf.sh_gate_up[s.shared_inter :].copy_(self.cpu_tensor(s.k_sh_up.format(l=l)))
            buf.sh_down.copy_(self.cpu_tensor(s.k_sh_down.format(l=l)))
            buf.small["sh_router"] = self.cpu_tensor(s.k_sh_router.format(l=l))
        buf.layer = l

    def _to_device(self, buf: "HostLayerBuffers") -> LayerWeights:
        d, dt = self.device, self.dtype
        cv = lambda t: None if t is None else t.to(d, dt, non_blocking=False)
        sm = buf.small
        lw = LayerWeights(
            ln1=cv(sm["ln1"]), ln2=cv(sm["ln2"]), wq=cv(sm["wq"]), wk=cv(sm["wk"]), wv=cv(sm["wv"]), wo=cv(sm["wo"]),
            q_norm=cv(sm["q_norm"]), k_norm=cv(sm["k_norm"]), router=cv(sm["router"]),
            gate_up=buf.gate_up.to(d, dt, non_blocking=True), down=buf.down.to(d, dt, non_blocking=True),
        )
        if "bq" in sm:  # ext14
            lw.bq, lw.bk, lw.bv = cv(sm["bq"]), cv(sm["bk"]), cv(sm["bv"])
        if buf.sh_gate_up is not None:  # ext14
            lw.sh_gate_up = buf.sh_gate_up.to(d, dt, non_blocking=True)
            lw.sh_down = buf.sh_down.to(d, dt, non_blocking=True)
            lw.sh_router = cv(sm["sh_router"])
        return lw

    def load_layer(self, l: int) -> LayerWeights:
        """Synchronous single-layer load (used by tests and the reference checks)."""
        buf = HostLayerBuffers(self.spec, self.dtype, pinned=False)
        self._fill_layer_host(l, buf)
        w = self._to_device(buf)
        torch.cuda.synchronize()
        return w

    # -- ext14: raw expert matrices (CPU, checkpoint dtype) -----------------------------------------------------
    def expert_tensors(self, l: int, e: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(W_gate [inter, hidden], W_up [inter, hidden], W_down [hidden, inter]) of expert e at layer l, CPU tensors in
        the checkpoint's dtype. Dense families: e = 0 is the MLP. Qwen2-MoE: e = n_experts is the shared expert."""
        s = self.spec
        if s.shared_inter and e == s.n_experts:
            return (self.cpu_tensor(s.k_sh_gate.format(l=l)), self.cpu_tensor(s.k_sh_up.format(l=l)),
                    self.cpu_tensor(s.k_sh_down.format(l=l)))
        assert 0 <= e < s.n_experts, f"expert {e} out of range"
        return (self.cpu_tensor(s.k_gate.format(l=l, e=e)), self.cpu_tensor(s.k_up.format(l=l, e=e)),
                self.cpu_tensor(s.k_down.format(l=l, e=e)))


class HostLayerBuffers:
    def __init__(self, spec: ArchSpec, dtype: torch.dtype, pinned: bool = True):
        self.gate_up = torch.empty(spec.n_experts, 2 * spec.inter, spec.hidden, dtype=dtype, pin_memory=pinned)
        self.down = torch.empty(spec.n_experts, spec.hidden, spec.inter, dtype=dtype, pin_memory=pinned)
        self.sh_gate_up = self.sh_down = None  # ext14: Qwen2-MoE shared expert
        if getattr(spec, "shared_inter", 0):
            self.sh_gate_up = torch.empty(2 * spec.shared_inter, spec.hidden, dtype=dtype, pin_memory=pinned)
            self.sh_down = torch.empty(spec.hidden, spec.shared_inter, dtype=dtype, pin_memory=pinned)
        self.small: dict = {}
        self.layer: int = -1


class LayerStreamer:
    """Iterates layers 0..L-1, yielding LayerWeights on the GPU with one-layer-ahead host prefetch.

    Two pinned host buffer slots alternate: while the GPU consumes layer l (already copied to device), the prefetch
    thread fills the other slot with layer l+1. The device copy of l+1 is issued right before it is yielded, so the
    device holds at most two layers of weights at a time (the previous one is freed when the caller drops it).
    """

    def __init__(self, store: CheckpointStore, layers: Optional[range] = None):
        self.store, self.spec = store, store.spec
        self.layers = list(layers if layers is not None else range(self.spec.n_layers))
        self.slots = [HostLayerBuffers(self.spec, store.dtype), HostLayerBuffers(self.spec, store.dtype)]

    def __iter__(self):
        if not self.layers:
            return
        pending: dict[int, threading.Thread] = {}

        def start(i: int):
            slot = self.slots[i % 2]
            t = threading.Thread(target=self.store._fill_layer_host, args=(self.layers[i], slot), daemon=True)
            t.start()
            pending[i] = t

        start(0)
        for i, l in enumerate(self.layers):
            pending.pop(i).join()
            slot = self.slots[i % 2]
            assert slot.layer == l
            w = self.store._to_device(slot)
            torch.cuda.current_stream().synchronize()   # host buffers may be refilled after this point
            if i + 1 < len(self.layers):
                start(i + 1)
            yield l, w
            del w
