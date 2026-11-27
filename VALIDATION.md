# Validation

Checked on 27 November 2026 with the interpreter and dependencies in `environment.json`.

- Release-file integrity, all 184 source trajectories and the bundled exact tokenizer verified.
- Bounded protocol exploration completed at event depth 3 for runtime and delegation sequences.
- Schema-version and delegation checks completed.
- `bash scripts/run_all.sh --analysis` completed in an isolated copy, including paired analyses and plots.
- Equal-ledger-access outputs matched the recorded JSON values except for elapsed runtime.
- Prototype-timing outputs matched exactly.
- Summary values matched except for freshly measured distillation timing and the tokenizer's local source path.

Fresh model inference and the complete model-free rerun, including the real-process crash study, were not executed during this packaging check. Original recorded measurements remain intact in this release.
