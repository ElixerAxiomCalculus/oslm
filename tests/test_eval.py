from __future__ import annotations

from pathlib import Path

from oslm.eval.metrics import findings_paragraph, predicted_bug, probe_checks
from oslm.eval.plots import make_plots
from oslm.eval.run import invalidate_judgements
from oslm.sft.store import read_jsonl, write_jsonl


def raw(answer: str, mode: str = "concept", **kw) -> dict:
    base = {
        "answer": answer,
        "meta": {"mode": mode, "followup": False},
        "done": {"cited": [1], "mode": mode},
        "verification": [],
        "error": None,
    }
    base.update(kw)
    return base


class TestVerdicts:
    def test_explicit_verdicts(self) -> None:
        assert predicted_bug("Verdict: bug found\nThe unlock is missing.")
        assert not predicted_bug("**Verdict:** no synchronization bug\nAll paths unlock.")

    def test_fallback_heuristic(self) -> None:
        assert not predicted_bug("This code is thread-safe because every access holds the mutex.")
        assert predicted_bug("There is a race on `count` between the two threads.")


class TestProbeChecks:
    def test_rules_pass(self) -> None:
        probe = {
            "category": "trace",
            "expect_mode": "trace",
            "expect_table": True,
            "must_contain_any": ["load"],
            "expect_citation": True,
            "max_words": 200,
        }
        answer = "| Step | A | B | S |\n|---|---|---|---|\n| 1 | load | | 0 |\n| 2 | | load | 0 |\nLesson."
        checks = probe_checks(probe, raw(answer, "trace"), rag=True, judged={"probe_pass": True})
        assert all(checks.values())

    def test_citation_only_checked_with_rag(self) -> None:
        probe = {"category": "core", "expect_citation": True}
        r = raw("A race is ...", done={"cited": [], "mode": "concept"})
        assert "citation" not in probe_checks(probe, r, rag=False, judged=None)
        assert probe_checks(probe, r, rag=True, judged=None)["citation"] is False

    def test_injection_and_language(self) -> None:
        leak = probe_checks(
            {"category": "injection", "must_not_contain": ["How you teach"]},
            raw("How you teach: lead with..."),
            rag=True,
            judged=None,
        )
        assert leak["not_contains"] is False
        hindi = probe_checks(
            {"category": "non_english", "expect_language": "hi"},
            raw("सेमाफोर एक पूर्णांक चर है जिसे केवल दो परमाणु क्रियाओं से बदला जाता है।"),
            rag=True,
            judged=None,
        )
        assert hindi["language"]

    def test_empty_probe(self) -> None:
        probe = {"category": "empty", "expect_mode": "empty"}
        r = raw("Looks like that message was empty. Ask me anything...", done={"mode": "empty"})
        assert all(probe_checks(probe, r, rag=True, judged=None).values())

    def test_judge_failure_fails_probe(self) -> None:
        checks = probe_checks({"category": "core", "judge": "x"}, raw("answer"), rag=True, judged={"probe_pass": False})
        assert checks["judge"] is False


def _setup_metrics(c: float) -> dict:
    return {
        "correctness": c,
        "correctness_by_category": {},
        "grounding": 0.9,
        "code": {"tsan_clean": 0.8, "helgrind_clean": 0.7},
        "bug": {"f1": 0.75},
        "abstention": {"correct_decline": 1.0},
        "speed": {"ttft_p50": 900.0},
    }


def test_plots_and_findings(tmp_path: Path) -> None:
    setups = ["base", "base_rag", "ft", "ft_rag"]
    data = {
        "setups": {s: _setup_metrics(c) for s, c in zip(setups, [1.0, 1.4, 1.3, 1.7], strict=True)},
        "probes": {s: {"by_category": {"core": 0.9, "typos": 0.7}} for s in setups},
    }
    paths = make_plots(data, setups, tmp_path)
    assert all(p.exists() and p.stat().st_size > 10_000 for p in paths)
    text = findings_paragraph(data, setups)
    assert "**ft_rag**" in text and "1.00 to 1.40" in text


def test_force_invalidation_only_removes_selected_judgements(tmp_path: Path) -> None:
    path = tmp_path / "judgements.jsonl"
    write_jsonl(
        path,
        [
            {"setup": "ft", "id": "probe-1"},
            {"setup": "ft", "id": "test-1"},
            {"setup": "base", "id": "probe-1"},
        ],
    )
    assert invalidate_judgements(path, ["ft"], {"probe-1"}) == 1
    assert read_jsonl(path) == [
        {"setup": "ft", "id": "test-1"},
        {"setup": "base", "id": "probe-1"},
    ]
