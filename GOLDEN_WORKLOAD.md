# Golden Workload Local Scoring

This local golden suite is for ranking Docker Compose experiments on your rented GPU.
It is synthetic and is not the official BTC workload.

## One-command score

From the repository root:

```bash
bash scripts/run_golden_score.sh docker-compose-260725-101045-rtx4090-mimic-h200mig.yaml
```

For another experiment:

```bash
bash scripts/run_golden_score.sh docker-compose-260725-101041-e01-bf16-bf16-align-balanced.yaml
```

The script will:

1. Start the selected Docker Compose file.
2. Apply local H200-MIG-like CPU/RAM limits.
3. Mount `./llm-serving-baseline/model:/model:ro`.
4. Wait for `http://127.0.0.1:8000/health`.
5. Run all golden traces.
6. Save per-trace benchmark results.
7. Compute final weighted `golden_score`.
8. Stop the Docker Compose project.

## Inputs

Golden suite:

```text
configs/golden_suite.json
```

Trace files:

```text
configs/golden_traces/golden_a_sanity.json
configs/golden_traces/golden_b_balanced.json
configs/golden_traces/golden_c_high_concurrency.json
configs/golden_traces/golden_d_long_prefix.json
configs/golden_traces/golden_e_memory_pressure.json
configs/golden_traces/golden_f_decode_heavy.json
```

## Scoring

The final score is a weighted mean of ERS:

```text
GoldenScore =
  0.05 * ERS(golden_a_sanity)
+ 0.35 * ERS(golden_b_balanced)
+ 0.20 * ERS(golden_c_high_concurrency)
+ 0.20 * ERS(golden_d_long_prefix)
+ 0.10 * ERS(golden_e_memory_pressure)
+ 0.10 * ERS(golden_f_decode_heavy)
```

If a trace fails or has no `aggregate.json`, that trace contributes `0` for its weight.

## Outputs

Results are saved under:

```text
results/golden_runs/<run_id>/
```

Important files:

```text
golden_summary.json
golden_summary.csv
<run_id>_<trace_id>/requests.json
<run_id>_<trace_id>/requests.csv
<run_id>_<trace_id>/aggregate.json
<run_id>_<trace_id>/aggregate.csv
```

Use `golden_summary.csv` to compare many compose files.

## Practical Ranking Rule

Prefer configs with:

```text
highest golden_score
0 timeout_requests
0 zero_token_responses
0 failed_requests
stable startup
reasonable p95 TTFT and p95 TPOT
```

A high score with errors or OOM risk should not be used as the final candidate.
