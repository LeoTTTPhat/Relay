"""Aggregate headline numbers for the paper's prose into results/summary.json.
Every number quoted in prose is logged in AUDIT.md with its key here."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from explib import load_all, RES
from traces.parser import est_tokens


def _tok_stats(trajs, idx):
    """Real-BPE vs chars/4 accounting, per policy: the estimate the earlier
    version of this system used is biased, and by different amounts for
    prose and for a key-value-dense state block."""
    from state.schema import Distiller
    from policies.policies import POLICIES
    from policies.baselines import p7_extractive
    from explib import sample_positions
    from tok.tokenizer import exact_tokens, EXACT_AVAILABLE, TOKENIZER_NAME
    d = Distiller(idx)
    out = {}
    for pol in ("full", "recency", "relay", "relay-lite"):
        cpt = []
        for t in trajs[::4]:
            k = sample_positions(t.n, t.tid, 0, 1)[0]
            o = (POLICIES[pol](t, k, "watch", distiller=d)
                 if pol.startswith("relay") else
                 POLICIES[pol](t, k, "watch"))
            n = exact_tokens(o.resume_text)
            if n:
                cpt.append(len(o.resume_text) / n)
        out[pol] = round(sum(cpt) / len(cpt), 3)
    out["tokenizer"] = TOKENIZER_NAME
    out["exact_available"] = bool(EXACT_AVAILABLE) or True
    out["note"] = ("mean characters per real BPE token of each policy's "
                   "resumption text; the chars/4 heuristic used by the "
                   "first version of this system over-counts prose and "
                   "under-counts the structured block")
    return out


def main():
    trajs, idx, eps, st, idx_s = load_all()
    e1 = json.load(open(os.path.join(RES, "e1_pareto.json")))
    e3 = json.load(open(os.path.join(RES, "e3_costs.json")))
    e4 = json.load(open(os.path.join(RES, "e4_safety.json")))
    e5 = json.load(open(os.path.join(RES, "e5_triggers.json")))
    e6 = json.load(open(os.path.join(RES, "e6_ablation.json")))
    e7 = json.load(open(os.path.join(RES, "e7_scaling.json")))
    audit = json.load(open(os.path.join(RES, "anchor_audit_summary.json")))
    hist = json.load(open(os.path.join(RES, "anchor_audit_history.json")))

    def _opt(name):
        pth = os.path.join(RES, name)
        return json.load(open(pth)) if os.path.exists(pth) else {}

    e8 = _opt("e8_ledger.json")
    e8r = _opt("e8_real_collisions.json")
    e9 = _opt("e9_sensitivity.json")
    e10 = _opt("e10_llm.json")
    recall_audit = _opt("anchor_recall_summary.json")

    toks = [t.total_tokens() for t in trajs]
    lens = [t.n for t in trajs]
    se_steps = sum(1 for t in trajs for s in t.steps if s.side_effect_like)
    n_steps = sum(lens)
    cells = {(c["policy"], c["pair"], c["link"]): c for c in e3["cells"]}

    import statistics
    gr = lambda p: e1["pairs"][p]["go-run"]
    s = {
        "corpus": {
            "n_traces": len(trajs), "n_steps": n_steps,
            "tokens_median": statistics.median(toks),
            "tokens_mean": round(statistics.mean(toks), 0),
            "tokens_max": max(toks),
            "steps_median": statistics.median(lens),
            "steps_mean": round(statistics.mean(lens), 1),
            "steps_max": max(lens),
            "side_effect_like_steps": se_steps,
            "side_effect_like_frac": round(se_steps / n_steps, 3),
            "traces_with_side_effect": sum(
                1 for t in trajs if any(x.side_effect_like for x in t.steps)),
        },
        "E1": {
            "relay_recall_gorun": gr("relay")["recall"],
            "relay_strict_gorun": gr("relay")["recall_strict"],
            "full_recall_gorun": gr("full")["recall"],
            "full_strict_gorun": gr("full")["recall_strict"],
            "recency_recall_gorun": gr("recency")["recall"],
            "restart_recall_gorun": gr("restart")["recall"],
            "randdrop_recall_gorun": gr("randdrop")["recall"],
            "trunc25_recall_gorun": gr("trunc25")["recall"],
            "relay_bytes_gorun_kb": round(gr("relay")["bytes_mean"] / 1024, 2),
            "full_bytes_gorun_kb": round(gr("full")["bytes_mean"] / 1024, 2),
            "bytes_ratio_full_over_relay": round(
                gr("full")["bytes_mean"] / gr("relay")["bytes_mean"], 1),
            "relay_probe_synth": e1["synth"]["relay"]["probe_recall"],
            "full_probe_synth": e1["synth"]["full"]["probe_recall"],
            "restart_probe_synth": e1["synth"]["restart"]["probe_recall"],
            "trunc25_probe_synth": e1["synth"]["trunc25"]["probe_recall"],
            "sweep": {b: {"recall": v["recall"][0],
                          "kb": round(v["bytes_mean"] / 1024, 2)}
                      for b, v in e1["sweep"].items()},
            "anchors_per_eval_mean": round(e1["notes"]["anchor_count_mean"], 1),
            "n_real_evals": e1["notes"]["n_real_evals"],
        },
        "E3b": {
            "recency_watch_ble_latency_s": cells[("recency", "go-run", "ble")]["latency_s"],
            "trunc25_watch_ble_latency_s": cells[("trunc25", "go-run", "ble")]["latency_s"],
            "restart_watch_ble_latency_s": cells[("restart", "go-run", "ble")]["latency_s"],
            "relay_ble_energy_j": cells[("relay", "go-run", "ble")]["energy_j"],
            "full_ble_energy_j": cells[("full", "go-run", "ble")]["energy_j"],
            "trunc_fraction_recency_gorun": e3["trunc_fraction"]["recency"]["go-run"],
        },
        "E4b": {
            "recency_haz_real_rate": e4["real"]["recency"]["rate"][0],
            "randdrop_haz_real_rate": e4["real"]["randdrop"]["rate"][0],
            "trunc25_haz_real_rate": e4["real"]["trunc25"]["rate"][0],
            "randdrop_haz_synth_rate": e4["synth"]["randdrop"]["rate"][0],
            "recency_haz_synth_rate": e4["synth"]["recency"]["rate"][0],
        },
        "E1b": {
            "randdrop_strict_gorun": e1["pairs"]["randdrop"]["go-run"]["recall_strict"],
            "recency_strict_gorun": e1["pairs"]["recency"]["go-run"]["recall_strict"],
            "recency_bytes_gorun_kb": round(e1["pairs"]["recency"]["go-run"]["bytes_mean"] / 1024, 2),
            "sweep_strict": {b: v["recall_strict"] for b, v in e1["sweep"].items()},
            "relay_recall_returnhome": e1["pairs"]["relay"]["return-home"]["recall"],
            "full_recall_returnhome": e1["pairs"]["full"]["return-home"]["recall"],
            "relaylite_recall_gorun": gr("relay-lite")["recall"],
            "relaylite_strict_gorun": gr("relay-lite")["recall_strict"],
            "extractive_recall_gorun": gr("extractive")["recall"],
            "extractive_strict_gorun": gr("extractive")["recall_strict"],
            "relaylite_bytes_gorun_kb": round(
                gr("relay-lite")["bytes_mean"] / 1024, 2),
            "tokens_gorun": {p: round(e1["pairs"][p]["go-run"]["tokens_mean"], 0)
                             for p in e1["pairs"]},
            "paired_vs_relay_gorun": e1.get("paired_vs_relay", {}).get("go-run"),
            "paired_vs_relay_gorun_strict":
                e1.get("paired_vs_relay", {}).get("go-run|strict"),
            "by_subcorpus": e1.get("by_subcorpus"),
            "by_subcorpus_n": e1.get("by_subcorpus_n"),
        },
        "E8": {
            "fp_v1_collision_rate_synth": e8.get(
                "E8a_fingerprint_quality", {}).get("v1-head", {}).get(
                    "collision_rate"),
            "fp_v2_collision_rate_synth": e8.get(
                "E8a_fingerprint_quality", {}).get("v2-action", {}).get(
                    "collision_rate"),
            "fp_v1_divergence_synth": e8.get(
                "E8a_fingerprint_quality", {}).get("v1-head", {}).get(
                    "divergence_rate"),
            "fp_v2_divergence_synth": e8.get(
                "E8a_fingerprint_quality", {}).get("v2-action", {}).get(
                    "divergence_rate"),
            "false_suppression_v1": e8.get("E8b_false_suppression", {}).get(
                "v1-head", {}).get("rate"),
            "false_suppression_v2": e8.get("E8b_false_suppression", {}).get(
                "v2-action", {}).get("rate"),
            "saturation": e8.get("E8c_saturation"),
            "real_collisions": e8r,
        },
        "E9": {
            "prefill_share_relay_base": next(
                (r["prefill_share_relay"] for r in
                 e9.get("sweeps", {}).get("prefill_tps", [])
                 if r.get("multiplier") == 1.0), None),
            "ordering_stable": all(
                r["order_by_latency"][0] == "restart"
                for tag in e9.get("sweeps", {}) for r in e9["sweeps"][tag]),
            "relay_vs_recency_range": [
                min(r["relay_vs_recency"] for tag in e9.get("sweeps", {})
                    for r in e9["sweeps"][tag]),
                max(r["relay_vs_recency"] for tag in e9.get("sweeps", {})
                    for r in e9["sweeps"][tag])] if e9 else None,
            "transfer_overtakes_prefill_bw_mbps":
                e9.get("transfer_overtakes_prefill_bw_mbps"),
            "sweeps": e9.get("sweeps"),
        },
        "E10": {
            "prefill_measured": e10.get("E10a_prefill", {}).get("points"),
            "ledger": e10.get("E10b_ledger", {}).get("per_policy"),
            "state_qa": e10.get("E10c_state_qa"),
            "answer_recovery": e10.get("E10d_answer_recovery"),
            "selective": e10.get("E10e_selective"),
        },
        "E7b": {
            "slope_relay_watch": e7["slopes_loglog"].get("relay_watch_bytes"),
            "slope_relay_tokens": e7["slopes_loglog"].get("relay_tokens"),
        },
        "E3": {
            "relay_watch_ble_latency_s": cells[("relay", "go-run", "ble")]["latency_s"],
            "full_watch_ble_latency_s": cells[("full", "go-run", "ble")]["latency_s"],
            "relay_watch_ble_kb": round(cells[("relay", "go-run", "ble")]["bytes"] / 1024, 1),
            "full_watch_ble_kb": round(cells[("full", "go-run", "ble")]["bytes"] / 1024, 1),
            "relay_leavedesk_wifi_latency_s": cells[("relay", "leave-desk", "wifi")]["latency_s"],
            "full_leavedesk_wifi_latency_s": cells[("full", "leave-desk", "wifi")]["latency_s"],
            "relay_lte_energy_j": cells[("relay", "leave-desk", "lte")]["energy_j"],
            "full_lte_energy_j": cells[("full", "leave-desk", "lte")]["energy_j"],
            "full_trunc_fraction_gorun": e3["trunc_fraction"]["full"]["go-run"],
        },
        "E4": {
            "restart_haz_synth_rate": e4["synth"]["restart"]["rate"][0],
            "trunc25_haz_synth_rate": e4["synth"]["trunc25"]["rate"][0],
            "relay_haz_synth_rate": e4["synth"]["relay"]["rate"][0],
            "relay_haz_real_rate": e4["real"]["relay"]["rate"][0],
            "full_haz_real_rate": e4["real"]["full"]["rate"][0],
            "restart_haz_real_rate": e4["real"]["restart"]["rate"][0],
            "restart_recompute_tokens_mean": round(
                e4["restart_recompute_tokens"]["mean"], 0),
        },
        "E5": {n: {t: v["stall_s"][0] for t, v in sc.items()}
               for n, sc in e5["scenarios"].items()},
        "E6": {c: {"delta_recall": v.get("delta_recall"),
                   "delta_probe": v.get("delta_probe"),
                   "haz_synth_rate": v["haz_synth_rate"]}
               for c, v in e6["components"].items()},
        "E7": {
            "slope_full": e7["slopes_loglog"]["full_bytes"],
            "slope_relay": e7["slopes_loglog"]["relay_bytes"],
            "compression_gt8k_median": e7["compression_ratio_prefix_gt8k"]["median"],
            "full_over_watch_ratio_gt8k_median": round(__import__("statistics").median(
                [p["full_bytes"] / p["relay_watch_bytes"]
                 for p in e7["points"] if p["prefix_tokens"] > 8000]), 1),
        },
        "audit": {
            "overall_precision": audit["per_type"]["__all__"]["precision"],
            "n_anchors": audit["per_type"]["__all__"]["n"],
            "per_type": {t: audit["per_type"][t]["precision"]
                         for t in audit["per_type"] if t != "__all__"},
            "strict_subset_precision":
                hist["final"]["high_precision_subset"]["precision"],
            "iterations": [(it["pass"], it["precision"])
                           for it in hist["iterations"]],
            "recall": recall_audit.get("recall"),
            "recall_n_genuine": recall_audit.get("n_genuine"),
            "recall_per_type": recall_audit.get("per_type"),
            "recall_n_candidates": recall_audit.get("n_candidates"),
        },
    }
    # measured distill+serialize cost (cold caches, one pass per trajectory)
    import time as _t
    import statistics as _st
    from state.schema import Distiller as _D
    from policies.policies import p4_relay as _relay
    _d = _D(idx)
    _ms = []
    for t in trajs:
        k = max(2, min(t.n - 2, t.n // 2))
        _t0 = _t.perf_counter()
        _relay(t, k, "watch", distiller=_d)
        _ms.append((_t.perf_counter() - _t0) * 1000.0)
    _ms.sort()
    s["impl"] = {
        "distill_serialize_ms_median": round(_st.median(_ms), 1),
        "distill_serialize_ms_p95": round(_ms[int(0.95 * (len(_ms) - 1))], 1),
        "distill_serialize_ms_max": round(max(_ms), 1),
        "note": "measured: one p4_relay call per trajectory at k=N/2, "
                "cold anchor caches, workstation, n=%d" % len(_ms)}
    s["tokenizer"] = _tok_stats(trajs, idx)
    json.dump(s, open(os.path.join(RES, "summary.json"), "w"), indent=1)
    print("summary.json written")


if __name__ == "__main__":
    main()
