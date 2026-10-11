# OSlm evaluation summary

Setups: base, base_rag, ft, ft_rag. Identical decoding (temperature 0.2, top_p 0.9, seed 42). Correctness and grounding are judged by qwen3:8b with a fixed rubric (oslm/eval/judge.py).

| Metric | base | base_rag | ft | ft_rag |
|---|---|---|---|---|
| Correctness (0–2) | 1.39 | 1.50 | 0.96 | 1.03 |
|   · concept | 1.80 | 1.95 | 1.60 | 1.77 |
|   · code_write | 1.05 | 1.32 | 0.45 | 0.35 |
|   · bug_find | 1.33 | 1.21 | 0.82 | 0.95 |
| Grounding (claims supported) | — | 87% | — | 72% |
| Citation validity | 0% | 86% | 0% | 72% |
| Code compiles | 60% | 45% | 20% | 22% |
| Code runs | 20% | 20% | 12% | 10% |
| TSan clean | 15% | 12% | 15% | 8% |
| Helgrind clean | 15% | 10% | 12% | 10% |
| Bug precision | 79% | 80% | 76% | 76% |
| Bug recall | 76% | 69% | 76% | 66% |
| Correct declines (out of scope) | 100% | 75% | 100% | 100% |
| False declines (in scope) | 2% | 5% | 0% | 2% |
| Retrieval hit@4 | — | 95% | — | 92% |
| Retrieval MRR | — | 0.90 | — | 0.88 |
| TTFT p50 (ms) | 401.50 | 791.50 | 376.00 | 768.00 |
| TTFT p95 (ms) | 999.00 | 1824.70 | 909.10 | 1623.20 |
| Total p50 (ms) | 10980.00 | 7894.50 | 3836.00 | 4169.00 |
| Probe pass rate | 68% | 74% | 61% | 66% |

## Judge agreement

Pending: fill in the `manual_correctness` column of `manual_scores.csv` (20% sample), then re-run `oslm eval`.

## Findings

The best setup on judged correctness is **base_rag** (1.50/2). Retrieval moves the base model from 1.39 to 1.50. Fine-tuning alone moves it from 1.39 to 0.96. With retrieval, fine-tuning changes correctness from 1.50 to 1.03 and grounding from 87% to 72%. base_rag bug-finding F1 is 0.74. ft_rag is not the top setup here; see the per-category rows and probe failures for where it loses.

Plots: `comparison.png`, `probes.png`
