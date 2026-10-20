from __future__ import annotations

import json
from pathlib import Path

import pytest

from oslm.checker.report import build_report
from oslm.rag.prompts import SYSTEM_PROMPT, ContextChunk, build_user_message
from oslm.sft.format import dedup_key, to_messages
from oslm.sft.generate import flagged, verified_correct
from oslm.sft.llm import extract_json
from oslm.sft.verify import rule_problems

FIXTURES = Path(__file__).parent / "fixtures"
CTX = [ContextChunk("OSTEP", "Locks", "3", "A lock ..."), ContextChunk("Wikipedia", "Race", "", "A race ...")]


def item(answer: str, **kw) -> dict:
    base = {
        "id": "x",
        "category": "concept",
        "mode": "concept",
        "variant": "standard",
        "question": "What is a lock?",
        "chunk_ids": ["a", "b"],
        "answer": answer,
        "note": "",
    }
    base.update(kw)
    base["user_message"] = build_user_message(
        base["question"], base["mode"], CTX if base["chunk_ids"] is not None else None
    )
    return base


class TestRules:
    def test_good_concept_passes(self) -> None:
        answer = (
            "A lock is like a bathroom key: one holder enters while everyone else waits [1]. "
            "Precisely, it provides mutual exclusion for a critical section that accesses shared state [1]. "
            "For example, both the read and write parts of counter++ must remain under the same lock [2]. "
            "A common misconception is that merely declaring a mutex protects nearby data; every participating "
            "thread must follow the locking protocol. Which statements form the critical section in a protected increment?"
        )
        assert rule_problems(item(answer)) == []

    def test_out_of_range_citation(self) -> None:
        assert any("citation" in p for p in rule_problems(item("Locks [3] matter a great deal.")))

    def test_citation_in_no_notes_answer(self) -> None:
        assert "citations in a no-notes answer" in rule_problems(item("Locks [1] matter a lot here.", chunk_ids=None))

    def test_distractor_must_not_be_cited(self) -> None:
        it = item(
            "Locks [2] matter a lot here.", variant="distractor", note="Context chunk [2] is unrelated to the question"
        )
        assert "cited the distractor chunk" in rule_problems(it)

    def test_decline_required(self) -> None:
        it = item("RCU is a scheme where readers ...", variant="irrelevant_context")
        assert "should decline but does not" in rule_problems(it)
        ok = item(
            "The notes don't cover RCU. I can help with locks, semaphores or deadlock.", variant="irrelevant_context"
        )
        assert rule_problems(ok) == []

    def test_bug_verdicts(self) -> None:
        bug = item(
            """Verdict: bug found
The counter is unprotected, so both threads can read the same value and lose an increment.
| Step | Thread 1 | Thread 2 | Shared value |
|---|---|---|---|
| 1 | reads 0 | | 0 |
| 2 | | reads 0 | 0 |
```diff
-    ++count;
+    pthread_mutex_lock(&lock); ++count; pthread_mutex_unlock(&lock);
```""",
            category="bug_find",
            mode="bug_find",
            has_bug=True,
        )
        assert rule_problems(bug) == []
        wrong = item("Verdict: no synchronization bug\nLooks fine.", category="bug_find", mode="bug_find", has_bug=True)
        assert "missing 'Verdict: bug found'" in rule_problems(wrong)

    def test_trace_needs_table(self) -> None:
        assert "trace without an interleaving table" in rule_problems(
            item("Step 1 happens, then step 2 happens.", mode="trace")
        )

    def test_filler_rejected(self) -> None:
        assert "filler opening" in rule_problems(item("Great question! A lock is a key [1]."))


class TestLabels:
    def test_flagged_on_real_racy_output(self) -> None:
        raw = json.loads((FIXTURES / "checker" / "racy_counter.json").read_text(encoding="utf-8"))
        assert flagged(build_report(raw))

    def test_clean_report_is_verified_correct(self) -> None:
        raw = {
            "status": "completed",
            "steps": {
                k: {"exit_code": 0, "elapsed_ms": 5, "output": "", "truncated": False}
                for k in ("compile_plain", "compile_tsan", "run", "tsan", "helgrind")
            },
        }
        report = build_report(raw)
        assert verified_correct(report) and not flagged(report)

    def test_extract_json_handles_fences(self) -> None:
        assert extract_json('Here:\n```json\n{"a": [1]}\n```') == {"a": [1]}


class TestFormat:
    def test_messages_match_inference_layout(self) -> None:
        it = item("Answer [1].")
        it["history"] = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
        msgs = to_messages(it)
        assert msgs[0] == {"role": "system", "content": SYSTEM_PROMPT}
        assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user", "assistant"]
        assert "<context>" in msgs[3]["content"] and dedup_key(it).startswith("hi ||")

    @pytest.mark.integration
    def test_renders_through_qwen_chat_template(self) -> None:
        from oslm.index.embed import get_tokenizer

        msgs = to_messages(item("A lock is a key [1]."))
        text = get_tokenizer().apply_chat_template(msgs, tokenize=False)
        assert "<|im_start|>assistant" in text and "<think>" not in text and SYSTEM_PROMPT[:40] in text

    @pytest.mark.integration
    def test_token_length_counts_tokens_not_dict_keys(self) -> None:
        from oslm.sft.format import token_length

        n = token_length(to_messages(item("A lock is a key [1]. " * 20)))
        assert n > 300  # system prompt alone is several hundred tokens

    @pytest.mark.integration
    def test_leakage_and_duplicates_detected(self) -> None:
        from oslm.sft.format import near_duplicates

        keys = ["What is a race condition?", "what is a race condition", "Explain the sleeping barber problem"]
        dup, leak = near_duplicates(keys, ["Define a race condition."], 0.92)
        assert 1 in dup and 0 in leak and 2 not in leak

    def test_empty_dataset_has_no_duplicates(self) -> None:
        from oslm.sft.format import near_duplicates

        assert near_duplicates([], ["What is a race condition?"], 0.92) == (set(), set())


@pytest.mark.integration
class TestVerifierLive:
    async def test_racy_vs_correct_fixtures(self) -> None:
        from oslm.checker.wsl_bridge import CheckerBridge

        bridge = CheckerBridge()
        racy = ["racy_counter", "lock_order", "unlock_unheld", "deadlock"]
        clean = ["correct_counter", "crlf_unicode"]
        reports = await bridge.check_many([(FIXTURES / "c" / f"{n}.c").read_bytes().decode() for n in racy + clean])
        assert all(flagged(r) for r in reports[: len(racy)])
        assert all(verified_correct(r) and not flagged(r) for r in reports[len(racy) :])
