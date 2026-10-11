# Analysis of the four-setup evaluation

Evaluated 2026-10-11 on 119 test items (40 concept, 40 code-write, 39 find-the-bug) and 38 examiner-style probes,
with identical decoding for every setup (temperature 0.2, top_p 0.9, seed 42). Correctness (0–2), grounding and
probe pass/fail were scored by one local judge (`qwen3:8b`, thinking on) with the rubric in `oslm/eval/judge.py`.
Code metrics come from the WSL sandbox (gcc + ThreadSanitizer + Helgrind), not from the judge.

## Headline

| | base | base_rag | ft | ft_rag |
|---|---|---|---|---|
| Correctness (0–2) | 1.39 | **1.50** | 0.96 | 1.03 |
| · concept | 1.80 | **1.95** | 1.60 | 1.77 |
| · code-write | 1.05 | **1.32** | 0.45 | 0.35 |
| · find-the-bug | **1.33** | 1.21 | 0.82 | 0.95 |
| Grounding (claims supported by cited notes) | — | **87%** | — | 72% |
| Concept answers that cite the notes | — | **65%** | — | 10% |
| Bug-finding F1 | **0.77** | 0.74 | 0.76 | 0.70 |
| Probe pass rate | 68% | **74%** | 61% | 66% |
| Correct out-of-scope declines | **100%** | 75% | **100%** | **100%** |
| False declines (in scope) | 2% | 5% | **0%** | 2% |
| Time to first token, p50 | 0.40 s | 0.79 s | **0.38 s** | 0.77 s |
| Total time, p50 | 11.0 s | 7.9 s | **3.8 s** | 4.2 s |

**Retrieval helps; fine-tuning, as trained here, hurts.** RAG lifts the base model from 1.39 to 1.50 and gives
well-grounded, cited answers (87% of claims supported). The fine-tuned model is faster and declines more reliably,
but its answers score lower in every content category, so **base_rag is the best setup** and the project's
hypothesis that ft_rag would be best is not supported by this run.

## Why the fine-tuned model loses

The model learned the *style of its training data* very faithfully, and that style was the problem:

1. **Code boilerplate copied from the training CSV.** 47 of the 49 code answers in the training data begin with
   `#define _XOPEN_SOURCE 700`, a dozen includes and a custom `ck()` error-checking helper. The fine-tuned model
   reproduces this in 40/40 code answers and calls `ck()` in 38/40, often without defining it. That yields
   "undeclared identifier" compile errors (compile rate 22% vs 45% for base_rag), and the extra length pushes
   11/40 programs past the 2,000-token limit, cut off before `main()`.
2. **Terse answers.** The training answers are short, and so are the model's: median 114 tokens for concept answers
   (base_rag: 390) and 282 for bug reviews (base_rag: 1,114). The rubric rewards complete answers (an interleaving
   table, a minimal diff and the reasoning), so short answers lose points even when the verdict is right.
3. **Overconfident bug verdicts.** Shorter reasoning comes with more wrong verdicts. ft_rag invented a deadlock in
   a correct bounded buffer (`correct-1`) and missed the data race in `inj-2` (it did **not** leak its instructions;
   the failure is the missed race).
4. **Few citations.** Only about 60 of the 385 training examples carried `[n]` citations (the external answers were
   not written against the notes, and citations were added only where a sentence clearly matched a chunk). The
   model learned to rarely cite: 10% of ft_rag concept answers cite vs 65% for base_rag.
5. **Small dataset.** 347 training examples for 2 epochs. Validation loss fell steadily (2.38 → 1.88) and
   plateaued, so the model fit the data well; the data just taught the wrong habits for this rubric.

## What fine-tuning did improve

- **No refusals of code requests.** With notes attached, the base model refused 18/40 code-write requests ("the
  provided context does not contain sufficient information"), over-applying the system prompt's "facts only from
  the notes" rule. The fine-tuned model never refused.
- **Declines and focus.** 100% correct out-of-scope declines (base_rag 75%) and the fewest false declines (0–2%).
- **Speed.** 2–3× faster end-to-end because answers are shorter.

## Other observations

- All four setups fail the two non-English probes (Hindi, Tamil) and the very long rambling-question probe.
- Code-write is weak for every setup (≤ 20% of programs both compile and run cleanly). The best base programs often
  have real synchronization bugs; gcc 15 also rejects incompatible-pointer and implicit-declaration code that older
  compilers only warned about.
- The fine-tuned model sometimes emits `\r\n` inside code although the training text contains none. The GGUF
  tokenizer metadata was checked and matches the base model exactly, so this comes from the weights. The serving
  pipeline now canonicalises newlines, and stored answers were re-verified before scoring.

## Recommendations

1. **For the demo, default to `base_rag`**, which gives the best answers. Keep the setup switches so the examiner can
   still compare all four live.
2. **Retrain with cleaned data:** strip the `_XOPEN_SOURCE`/`ck()` boilerplate from code answers (plain pthread
   calls), keep full-length explanations and bug reviews (table + diff + reasoning), and add more examples with
   citations to the attached notes. The import/verify/format pipeline (`oslm sft --step import …`) makes this a data
   change only.
3. **Fix the base model's code refusals** with one line in the `code_write` instruction ("write the program even if
   the notes don't show this exact problem"). This was not changed for this run, because it would break the
   "training prompt = serving prompt" rule for the fine-tuned model mid-comparison.
4. **Judge agreement:** a human should score the 20% sample in `manual_scores.csv`; `oslm eval` then reports
   exact-match and Cohen's κ between the judge and the humans.
