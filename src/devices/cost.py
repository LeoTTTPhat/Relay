"""Handoff cost model.

latency = serialize (measured CPU time x source cpu_factor)
        + transfer (bytes / bandwidth + RTT)
        + target re-prefill (min(tokens, target ctx) / target prefill rate)

energy has two terms and the first version of this model reported only one:
  radio   = tx_power x transfer_time + tail_power x tail_time
  compute = target SoC inference power x prefill_time
Reporting radio alone understates a handoff's cost by the term that scales
with the thing the policy actually controls -- how many tokens the target has
to prefill -- and makes the differences between policies look far smaller
than they are.  Both are reported separately, and ``energy_j`` keeps its
original radio-only meaning so earlier numbers remain comparable.
"""
from dataclasses import dataclass

from devices.constants import DEVICES, LINKS


def usable_ctx(dst: str) -> int:
    """Target context minus the generation headroom every policy is
    charged (see policies.policies.headroom)."""
    ctx = DEVICES[dst]["ctx"]
    return ctx - max(256, ctx // 8)


@dataclass
class HandoffCost:
    serialize_s: float
    transfer_s: float
    prefill_s: float
    latency_s: float
    energy_j: float          # radio only (transfer + tail)
    compute_energy_j: float  # target-side prefill
    total_energy_j: float
    bytes_wire: int
    tokens_shipped: int
    tokens_prefilled: int   # after target ctx cap
    truncated_at_target: bool


def handoff_cost(bytes_wire: int, tokens: int, src: str, dst: str,
                 link: str, serialize_cpu_s: float) -> HandoffCost:
    d_src, d_dst, l = DEVICES[src], DEVICES[dst], LINKS[link]
    ser = serialize_cpu_s * d_src["cpu_factor"]
    tr = bytes_wire * 8 / (l["bw_mbps"] * 1e6) + l["rtt_s"]
    tok_pref = min(tokens, usable_ctx(dst))
    pre = tok_pref / d_dst["prefill_tps"]
    energy = (l["tx_mw"] / 1000.0) * tr + (l["tail_mw"] / 1000.0) * l["tail_s"]
    compute = (d_dst.get("infer_mw", 0.0) / 1000.0) * pre
    return HandoffCost(
        serialize_s=ser, transfer_s=tr, prefill_s=pre,
        latency_s=ser + tr + pre, energy_j=energy,
        compute_energy_j=compute, total_energy_j=energy + compute,
        bytes_wire=bytes_wire, tokens_shipped=tokens,
        tokens_prefilled=tok_pref,
        truncated_at_target=tokens > usable_ctx(dst))
