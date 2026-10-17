# Value-aware HOLD evidence (IBM z17 VM, type 9175, `nnpa`)

| File | What it is |
|---|---|
| `value_hold_comparison.md` | Old vs new on the test days (scored once) and the validation days (used for the choice) |
| `value_hold_eval.json` | Full output of `scripts/value_hold_eval.py`: validation tuning, validation-only sensitivity, per-scheme results, and a check that the old policy reproduces the deployed `metrics.json` exactly |
| `run_vm.log` | Console log of that run, plus `uname -m`, `/proc/sysinfo` type/model and the `nnpa` flag |
| `thresholds_value_hold.json` | The thresholds now deployed as `~/shared/scamshield/models/thresholds.json`. The previous file is kept on the VM as `thresholds.plain-backup-20261017.json` |
| `deploy_vm.txt` | The api's start-up line after redeploying: it loaded `hold_strict_threshold=0.852`, `hold_min_usd=5000` and runs the anomaly model on `torch_nnpa` |
| `replay_30k_after_deploy_vm.json` | A 30k live replay (5% injected) against the redeployed api: server p50 4.2 ms, p99 54 ms. **Its injected-payment counts are not comparable** with `../replay/replay_30k_inject5pct_vm.json`. The api's store already held the earlier replays (18 HTTP 409 duplicates), and they are replayed into the rolling state on restart. Because the seed is the same, the injected "new" payees had already been seen, which is why more injections were allowed. The policy change alone cannot cause that: it only turns holds into warnings. |
| `pytest_vm.txt` | The test suite in the app image on the VM: 353 passed. The 2 failures are existing tests that assume NNPA is absent |
