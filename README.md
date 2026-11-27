# Relay

Reproduction package for capability-aware migration of LLM-agent state across devices, including duplicate suppression, durable action ledgers, crash recovery, delegation and continuation experiments.

## Setup and quick verification

Use Python 3.9 or 3.10 on macOS or Linux. The pinned environment was checked with Python 3.9.6 on Apple Silicon. All **184 Who&When trajectories**, the synthetic generators, cached model outputs and the exact Qwen2.5 tokenizer are included.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/verify_artifact.py
bash scripts/run_all.sh --check
```

The quick checks load the included traces, verify exact token accounting, exercise bounded protocol exploration, check action schema versions, and test delegation merging. Run `python scripts/check_invariants.py` separately for the complete corpus-wide budget and ledger invariants.

## Recompute results

```bash
# Recompute summaries, paired analyses and the latest controls from cached runs.
bash scripts/run_all.sh --analysis

# Rerun the model-free experiments, audits and real-process crash study.
bash scripts/run_all.sh --no-llm
```

`--analysis` writes summaries under `results/`, the equal-ledger-access and prototype-timing analyses under `results_rev2026-11/`, and plots under `figures/`. `--no-llm` additionally runs E1–E9, E11–E16 where model-free, E18–E22 and the cached-input scoring for E24/E25/E27. Exact BPE counting uses the bundled tokenizer, so it does not depend on a model cache or network access. E20 starts local HTTP processes and injects real process crashes; run it on a host that permits loopback sockets and subprocesses.

Run in a working copy when collecting a new measurement set: experiment scripts update their matching files in `results/`. The integrity manifest describes the recorded release before reruns. Timing and process scheduling measurements vary between runs.

## Experiment map

| Study | Entry points |
| --- | --- |
| E1–E4 fidelity, position, link cost and exposure | `run_core.py` |
| E4b–E9 identity loss, triggers, ablations, scale, ledger and sensitivity | `run_e4b_identity.py`, `run_e5_triggers.py` through `run_e9_sensitivity.py` |
| E10 local-model proxy and prefill timing | `run_e10_llm.py`, `measure_prefill.py` |
| E11 label audits | `audit_extractor.py`, `audit_recall.py`, `audit_sidefx.py` |
| E12–E16 identity attacks, protocol, independence, capabilities and energy | `run_e12_adversarial.py` through `run_e16_energy.py`, plus `run_e14a_stronger.py` |
| E17 compression baselines | `run_e17_compression.py` |
| E18–E22 capacity, actionability, durable crashes, schema versions and delegation | `run_e18_ledger_scale.py` through `run_e22_delegation.py` |
| E23/E23b closed-loop and binding-budget continuation | `run_e23_continuation.py`, `run_e23b_binding.py`, `analyze_e23_paired.py`, `analyze_e23b_paired.py` |
| E24/E25 translated traces and rewritten corrections | `gen_e24_translate.py`, `run_e24_multilingual.py`, `gen_e25_paraphrases.py`, `run_e25_paraphrase_family.py` |
| E26 two-process handoff prototype | `run_e26_prototype.py`, `analyze_e26.py` |
| E27/E28 held-out rewrites and ledger comprehension | `gen_e27_heldout_paraphrases.py`, `run_e27_conservative.py`, `run_e28_ledger_comprehension.py` |
| Equal ledger access and stage-separated timing | `rev2611_fair_ledger_access.py`, `rev2611_prototype_timing.py` |
| Protocol exploration and determinism | `check_protocol_exhaustive.py`, `check_protocol_prefix_replay.py`, `probe_determinism.py` |

All entry points above are under `scripts/`. Each script declares its inputs, seed, settings and outputs. Per-episode model decisions, generated rewrites, translations and judged audit inputs are retained under `results/`.

## Fresh model-dependent runs

The local inference code uses PyTorch/Transformers and cached Hugging Face checkpoints. Prepare the following models before offline execution, using the IDs selected inside the corresponding scripts:

```bash
python scripts/download_models.py
python scripts/run_e10_llm.py
python scripts/run_e14a_stronger.py
python scripts/run_e17_compression.py
RELAY_E23_MODEL=Qwen/Qwen2.5-3B-Instruct RELAY_E23_N=200 \
  python scripts/run_e23_continuation.py
python scripts/run_e23b_binding.py
python scripts/analyze_e23b_paired.py
python scripts/run_e26_prototype.py
python scripts/run_e28_ledger_comprehension.py
```

Qwen2.5 checkpoints from 0.5B to 7B serve the proxy, continuation, compression and prototype studies. E24 regeneration uses Helsinki-NLP OPUS-MT English-to-German/Spanish/French models. E25 and E27 fresh rewrite generation use a local [Ollama](https://ollama.com/) server with `gemma3:27b` and `mistral-small3.1:24b`. Scoring the included generation caches works without those servers. The model-download helper lists the exact IDs used by the source code.

E23b sharding and merge settings are `RELAY_E23B_SHARD` and `RELAY_E23B_MERGE`; other sample limits and model choices are declared near the top of each runner. Match the settings in the retained per-run metadata when reproducing a specific subset. Model-dependent steps can take several hours and exact responses depend on hardware and inference versions.

## Data and layout

| Path | Contents |
| --- | --- |
| `src/` | Trace parser, synthetic generators, distiller, policies, capability models, dispatcher and durable protocol |
| `scripts/` | Runners, audits, analyses and invariant checks |
| `data/who_and_when/Who&When/` | 58 hand-crafted and 126 algorithm-generated source trajectories |
| `data/tokenizer/` | Pinned Qwen2.5 tokenizer with its source revision |
| `results/` | Recorded measurements, raw decisions, cached generations and judged audit inputs |
| `results_rev2026-11/` | Equal-access and timing controls |
| `environment.json` | Validated interpreter and package versions |
| `MANIFEST.sha256` | Release-file integrity checks |

The trace source is [Kevin355/Who_and_When](https://huggingface.co/datasets/Kevin355/Who_and_When). `bash scripts/download_data.sh` can acquire it again; the packaged JSON traces are ready to load at their local paths. Original data and model terms apply to their respective assets.

Fidelity uses heuristic future-relevance anchors. Real-trace side-effect labels represent exposure over side-effect-like steps. Controlled synthetic episodes support the action-safety studies. Link/device constants are modeled, and E26 is a same-host process emulation. These distinctions are retained in the output metadata.
