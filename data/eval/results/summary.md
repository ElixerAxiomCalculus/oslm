# OSlm evaluation summary

Setups: ft. Identical decoding (temperature 0.2, top_p 0.9, seed 42). Correctness and grounding are judged by qwen3:8b with a fixed rubric (oslm/eval/judge.py).

| Metric | ft |
|---|---|
| Correctness (0–2) | — |
|   · concept | — |
|   · code_write | — |
|   · bug_find | — |
| Grounding (claims supported) | — |
| Citation validity | 0% |
| Code compiles | — |
| Code runs | — |
| TSan clean | — |
| Helgrind clean | — |
| Bug precision | — |
| Bug recall | — |
| Correct declines (out of scope) | 100% |
| False declines (in scope) | — |
| Retrieval hit@4 | — |
| Retrieval MRR | — |
| TTFT p50 (ms) | 470.00 |
| TTFT p95 (ms) | 39331.70 |
| Total p50 (ms) | 4264.50 |
| Probe pass rate | 92% |

## Judge agreement

Pending: fill in the `manual_correctness` column of `manual_scores.csv` (20% sample), then re-run `oslm eval`.

## Findings

No judged results yet.

Plots: `comparison.png`, `probes.png`
