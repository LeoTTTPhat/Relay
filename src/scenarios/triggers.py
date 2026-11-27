"""Mobility scenarios and handoff trigger policies (E5).

Scenarios (SPEC): leave-desk (laptop->phone), enter-kitchen (phone->speaker),
go-run (phone->watch), return-home (watch->laptop); the src/dst/link map is
`devices.constants.SCENARIOS`.

Trigger policies model the user-perceived stall at a device switch:
  immediate   -- distill+ship at the switch: stall = serialize + transfer +
                 target prefill of the block.
  defer       -- wait for the in-flight step to complete (0.5 x step tokens /
                 source decode rate in expectation), then as immediate, with
                 a consistent step boundary.
  predictive  -- pre-ship the block distilled DELTA_STEPS before the switch
                 while the source link is idle; at the switch, ship only the
                 raw delta (steps since pre-ship), capped at the Relay
                 budget. Stall = delta transfer + delta prefill. Wasted
                 bytes = P(wrong prediction) x pre-shipped bytes; predictor
                 accuracy is a modeled parameter (reported at 0.9/0.7/0.5).
"""
import gzip

from devices.constants import DEVICES, SCENARIOS  # noqa: F401  (re-export)
from devices.cost import handoff_cost
from traces.parser import est_tokens
from policies.policies import p4_relay, relay_budget

DELTA_STEPS = 2


def immediate_stall(traj, k, src, dst, link, distiller):
    blk = p4_relay(traj, k, dst, distiller=distiller)
    c = handoff_cost(blk.bytes_wire, blk.tokens_resume, src, dst, link,
                     blk.serialize_cpu_s)
    return c.latency_s, 0.0, blk


def defer_stall(traj, k, src, dst, link, distiller):
    lat, _, blk = immediate_stall(traj, k, src, dst, link, distiller)
    wait = 0.5 * traj.steps[k].tokens / DEVICES[src]["decode_tps"]
    return wait + lat, 0.0, blk


def predictive_stall(traj, k, src, dst, link, distiller):
    k0 = max(1, k - DELTA_STEPS)
    pre = p4_relay(traj, k0, dst, distiller=distiller)
    delta_txt = "\n".join(s.content for s in traj.steps[k0 + 1:k + 1])
    delta_txt = delta_txt[:relay_budget(dst) * 4]
    if delta_txt:
        dbytes = len(gzip.compress(delta_txt.encode(), 6))
        dtok = min(est_tokens(delta_txt), DEVICES[dst]["ctx"])
    else:
        dbytes = dtok = 0
    c = handoff_cost(dbytes, dtok, src, dst, link, 0.0005)
    return c.latency_s, float(pre.bytes_wire), pre
