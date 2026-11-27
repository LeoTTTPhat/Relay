#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"
export MPLBACKEND=Agg

analysis() {
  for script in analyze_e23_paired analyze_e23_advisory analyze_e23b_paired \
    analyze_e23b_blocks e23b_padding_stats analyze_e26 analyze_e27_real_units \
    rev2611_fair_ledger_access rev2611_prototype_timing make_summary \
    make_figures make_framework_fig make_protocol_fig rev2611_make_status_fig; do
    "$PY" "scripts/$script.py"
  done
}

case "${1:---analysis}" in
  --check)
    "$PY" scripts/verify_artifact.py
    RELAY_EXH_SINGLE=3 RELAY_EXH_DELEG=3 \
      RELAY_EXH_OUT=protocol_exhaustive_quick.json \
      "$PY" scripts/check_protocol_exhaustive.py
    "$PY" scripts/run_e21_schema_version.py
    "$PY" scripts/run_e22_delegation.py
    ;;
  --analysis) analysis ;;
  --no-llm)
    for script in audit_extractor audit_recall audit_sidefx; do
      "$PY" "scripts/$script.py" score
    done
    for script in check_invariants run_core run_e4b_identity run_e5_triggers \
      run_e6_ablation run_e8_ledger run_e9_sensitivity run_e7_scaling \
      run_e12_adversarial run_e13_protocol run_e15_capability run_e16_energy \
      run_e14_independence run_e18_ledger_scale run_e19_actionability \
      run_e20_durable_crash run_e21_schema_version run_e22_delegation count_tcb \
      run_e24_multilingual run_e25_paraphrase_family run_e27_conservative; do
      "$PY" "scripts/$script.py"
    done
    analysis
    ;;
  *) echo "Usage: bash scripts/run_all.sh [--check|--analysis|--no-llm]" >&2; exit 2 ;;
esac
