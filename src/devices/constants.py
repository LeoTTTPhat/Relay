"""Capability classes and link profiles with sourced constants.

These are MODELED constants; each carries a source comment. The evaluation
is trace-driven with modeled links -- no live multi-device deployment is
claimed (stated in the paper's limitations).

Context windows are deliberate design points for on-device assistants of
each class (small models with small KV budgets on wearables; larger on
laptops). Prefill/decode rates are order-of-magnitude figures for the model
class each device would run, following the on-device LLM measurement
literature:
  * MELTing point (Laskaridis et al., arXiv:2403.12844) measures prefill
    throughput on phone-class SoCs in the tens-to-hundreds of tokens/s for
    sub-2B models and shows an order-of-magnitude spread across device
    tiers; we adopt watch 15 tok/s (0.5B-class NPU-less), speaker 40,
    phone 90, laptop 300 (M-series).
  * ``infer_mw`` is sustained SoC power while the target prefills. The same
    on-device measurement literature puts phone-class LLM inference in the
    several-watt range and reports that inference is thermally limited on
    sustained load; wearables run far below that envelope and laptops far
    above it. We adopt watch 0.9 W, speaker 2.5 W, phone 4.5 W, laptop
    20 W. These are the *least* well-sourced constants in the model, which
    is why E16 sweeps them over an order of magnitude and reports the
    conclusion only where it survives the sweep. Omitting this term was the
    larger error: the first version of this system reported radio energy
    alone, and radio energy is not where a handoff spends its joules.
Link profiles:
  * BLE: application-layer GATT throughput commonly measured at
    0.1--0.2 Mbps (Gomez et al., Sensors 2012); ~50 ms connection-event
    latency; ~30 mW active radio.
  * WiFi: tens of Mbps uplink on 802.11n/ac in homes; ~15 ms RTT;
    ~400 mW active; ~240 ms tail (Huang et al., MobiSys'12 report WiFi
    tail energy far below cellular).
  * LTE: ~5 Mbps uplink; ~70 ms RTT; ~1200 mW active transfer power and a
    ~11.5 s high-power tail at ~1060 mW after transfer (Huang et al.,
    MobiSys'12, "A Close Examination of Performance and Power
    Characteristics of 4G LTE Networks").
"""

DEVICES = {
    # ctx: max resumption-context tokens the class can hold
    # prefill_tps / decode_tps: tokens per second for the class's model
    # cpu_factor: serialization CPU slowdown relative to this workstation
    # infer_mw: sustained SoC power while prefilling (see module docstring)
    "watch":   {"ctx": 2048,  "prefill_tps": 15.0,  "decode_tps": 6.0,
                "cpu_factor": 8.0, "infer_mw": 900.0},
    "speaker": {"ctx": 4096,  "prefill_tps": 40.0,  "decode_tps": 10.0,
                "cpu_factor": 4.0, "infer_mw": 2500.0},
    "phone":   {"ctx": 8192,  "prefill_tps": 90.0,  "decode_tps": 18.0,
                "cpu_factor": 2.0, "infer_mw": 4500.0},
    "laptop":  {"ctx": 32768, "prefill_tps": 300.0, "decode_tps": 35.0,
                "cpu_factor": 1.0, "infer_mw": 20000.0},
}

LINKS = {
    # bw_mbps: application-layer throughput; rtt_s: round-trip latency;
    # tx_mw: active radio power while transferring; tail_s/tail_mw: post-
    # transfer high-power tail (dominant for LTE per Huang et al.)
    "ble":  {"bw_mbps": 0.15, "rtt_s": 0.050, "tx_mw": 30.0,
             "tail_s": 0.0,  "tail_mw": 0.0},
    "wifi": {"bw_mbps": 25.0, "rtt_s": 0.015, "tx_mw": 400.0,
             "tail_s": 0.24, "tail_mw": 250.0},
    "lte":  {"bw_mbps": 5.0,  "rtt_s": 0.070, "tx_mw": 1200.0,
             "tail_s": 11.5, "tail_mw": 1060.0},
}

# Mobility scenarios (SPEC): source -> target with the link used in E1/E3/E5
SCENARIOS = {
    "leave-desk":    {"src": "laptop", "dst": "phone",   "link": "wifi"},
    "enter-kitchen": {"src": "phone",  "dst": "speaker", "link": "wifi"},
    "go-run":        {"src": "phone",  "dst": "watch",   "link": "ble"},
    "return-home":   {"src": "watch",  "dst": "laptop",  "link": "ble"},
}
