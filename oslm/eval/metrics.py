from __future__ import annotations

import csv
import json
import random
import re
from collections import defaultdict
from statistics import mean
from typing import Any

import numpy as np

from oslm.config import load_config
from oslm.sft.store import read_jsonl
from oslm.sft.verify import DECLINE, VERDICT_BUG, VERDICT_OK

TABLE_ROW = re.compile(r"^\s*\|", re.M)
NO_BUG_HINT = re.compile(r"\bno (synchronization )?bugs?\b|\bis (thread[- ]safe|correct)\b", re.I)


def predicted_bug(answer: str) -> bool:
    if VERDICT_OK.search(answer):
        return False
    if VERDICT_BUG.search(answer):
        return True
    return not NO_BUG_HINT.search(answer[:400])


def _pct(values: list[float], q: float) -> float | None:
    return float(np.percentile(values, q)) if values else None


def _rate(flags: list[bool]) -> float | None:
    return sum(flags) / len(flags) if flags else None


def _language_ok(answer: str, lang: str) -> bool:
    letters = re.findall(r"\w", answer)
    script = {"hi": r"[ऀ-ॿ]", "ta": r"[஀-௿]"}[lang]
    native = len(re.findall(script, answer))
    return (letters and native / len(letters) > 0.3) or bool(re.search(r"\bEnglish\b", answer))


def probe_checks(
    probe: dict[str, Any], raw: dict[str, Any], rag: bool, judged: dict[str, Any] | None
) -> dict[str, bool]:
    a = raw["answer"]
    low = a.lower()
    meta, done = raw.get("meta", {}), raw.get("done", {})
    checks: dict[str, bool] = {"no_error": raw.get("error") is None}
    if probe.get("expect_mode"):
        mode = done.get("mode") if probe["expect_mode"] == "empty" else meta.get("mode")
        # A not-in-notes switch still counts for concept probes only if it was the intended outcome.
        checks["mode"] = mode == probe["expect_mode"]
    if probe.get("must_contain_any"):
        checks["contains_any"] = any(s.lower() in low for s in probe["must_contain_any"])
    if probe.get("must_contain_all"):
        checks["contains_all"] = all(s.lower() in low for s in probe["must_contain_all"])
    if probe.get("must_not_contain"):
        checks["not_contains"] = not any(s.lower() in low for s in probe["must_not_contain"])
    if probe.get("expect_citation") and rag:
        checks["citation"] = bool(done.get("cited"))
    if probe.get("max_words"):
        checks["length"] = len(a.split()) <= probe["max_words"]
    if probe.get("expect_table"):
        checks["table"] = len(TABLE_ROW.findall(a)) >= 4
    if probe.get("expect_verdict"):
        checks["verdict"] = predicted_bug(a) == (probe["expect_verdict"] == "bug")
    if probe.get("expect_program_clean"):
        checks["program_clean"] = any(v.get("status") == "clean" for v in raw.get("verification", []))
    if probe.get("expect_followup"):
        checks["followup"] = bool(meta.get("followup"))
    if probe.get("expect_language"):
        checks["language"] = _language_ok(a, probe["expect_language"])
    if probe.get("category") == "empty":
        checks["friendly"] = len(a) > 20
    if probe.get("judge") and judged is not None and judged.get("probe_pass") is not None:
        checks["judge"] = bool(judged["probe_pass"])
    return checks


def compute(setups: list[str]) -> dict[str, Any]:
    from oslm.eval.run import load_suite

    cfg = load_config()
    results = cfg.path("eval") / "results"
    items = {it["id"]: it for it in load_suite("all")}
    judgements = {(j["setup"], j["id"]): j for j in read_jsonl(results / "judgements.jsonl")}
    out: dict[str, Any] = {"setups": {}, "probes": {}}

    for setup in setups:
        rag = setup.endswith("_rag")
        raws = [r for r in read_jsonl(results / f"raw_{setup}.jsonl") if r["id"] in items]
        test = [r for r in raws if r["suite"] == "testset"]
        probes = [r for r in raws if r["suite"] == "probes"]
        corr: dict[str, list[float]] = defaultdict(list)
        grounding: list[float] = []
        for r in test:
            j = judgements.get((setup, r["id"]))
            if j and isinstance(j.get("correctness"), (int, float)):
                corr[r["category"]].append(float(j["correctness"]))
            # Grounding only exists when notes were attached; the judge sometimes scores it anyway for no-RAG setups.
            if rag and j and isinstance(j.get("grounding"), (int, float)):
                grounding.append(float(j["grounding"]))
        all_corr = [v for vs in corr.values() for v in vs]

        valid = sum(r["done"].get("citations", {}).get("valid", 0) for r in raws)
        invalid = sum(r["done"].get("citations", {}).get("invalid", 0) for r in raws)

        code_rows = [r for r in test if r["category"] == "code_write"]
        reports = [r["verification"][0] if r["verification"] else None for r in code_rows]
        code = {
            "compiles": _rate([bool(v) and v["status"] != "compile_error" for v in reports]),
            "runs": _rate(
                [
                    bool(v)
                    and v["status"] in ("clean", "issues")
                    and not v["timed_out"]
                    and not any(f["kind"] in ("crash", "resource_limit") for f in v["findings"])
                    for v in reports
                ]
            ),
            "tsan_clean": _rate([bool(v) and bool(v.get("tsan_clean")) for v in reports]),
            "helgrind_clean": _rate([bool(v) and bool(v.get("helgrind_clean")) for v in reports]),
        }

        tp = fp = fn = tn = 0
        for r in test:
            if r["category"] != "bug_find":
                continue
            truth, pred = items[r["id"]]["has_bug"], predicted_bug(r["answer"])
            tp += truth and pred
            fp += (not truth) and pred
            fn += truth and not pred
            tn += (not truth) and not pred
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        f1 = 2 * precision * recall / (precision + recall) if precision and recall else None

        concept = [r for r in test if r["category"] == "concept"]
        false_decline = _rate([bool(DECLINE.search(r["answer"][:300])) for r in concept])
        oos = [r for r in probes if items[r["id"]]["category"] in ("os_out_of_scope", "off_topic")]
        correct_decline = _rate(
            [all(probe_checks(items[r["id"]], r, rag, judgements.get((setup, r["id"]))).values()) for r in oos]
        )

        ttft = [r["done"]["timings"]["ttft_ms"] for r in raws if r["done"].get("timings", {}).get("ttft_ms")]
        total = [r["done"]["timings"]["total_ms"] for r in raws if r["done"].get("timings", {}).get("total_ms")]

        hits, rr = [], []
        if rag:
            for r in concept:
                gold = items[r["id"]].get("gold_urls", [])
                urls = [s["url"] for s in r["sources"]][:4]
                rank = next((i for i, u in enumerate(urls, start=1) if u in gold), None)
                hits.append(rank is not None)
                rr.append(1 / rank if rank else 0.0)

        out["setups"][setup] = {
            "n_testset": len(test),
            "n_probes": len(probes),
            "correctness": mean(all_corr) if all_corr else None,
            "correctness_by_category": {k: mean(v) for k, v in corr.items()},
            "grounding": mean(grounding) if grounding else None,
            "citation_validity": valid / (valid + invalid) if valid + invalid else None,
            "citation_rate": _rate([bool(r["done"].get("cited")) for r in concept]) if rag else None,
            "code": code,
            "bug": {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn},
            "abstention": {"correct_decline": correct_decline, "false_decline": false_decline},
            "speed": {
                "ttft_p50": _pct(ttft, 50),
                "ttft_p95": _pct(ttft, 95),
                "total_p50": _pct(total, 50),
                "total_p95": _pct(total, 95),
            },
            "retrieval": {"hit@4": _rate(hits), "mrr": mean(rr) if rr else None} if rag else None,
        }

        by_cat: dict[str, list[bool]] = defaultdict(list)
        failures = []
        for r in probes:
            probe = items[r["id"]]
            checks = probe_checks(probe, r, rag, judgements.get((setup, r["id"])))
            ok = all(checks.values())
            by_cat[probe["category"]].append(ok)
            if not ok:
                failures.append({"id": r["id"], "failed": [k for k, v in checks.items() if not v]})
        out["probes"][setup] = {
            "by_category": {k: sum(v) / len(v) for k, v in by_cat.items()},
            "overall": _rate([x for v in by_cat.values() for x in v]),
            "failures": failures,
        }
    return out


def _fmt(v: Any, pct: bool = False, digits: int = 2) -> str:
    if v is None:
        return "—"
    return f"{v * 100:.0f}%" if pct else f"{v:.{digits}f}"


def write_manual_sheet(setups: list[str], frac: float = 0.2) -> None:
    cfg = load_config()
    results = cfg.path("eval") / "results"
    path = results / "manual_scores.csv"
    if path.exists():
        return
    judgements = {(j["setup"], j["id"]): j for j in read_jsonl(results / "judgements.jsonl")}
    rows = [(s, r) for s in setups for r in read_jsonl(results / f"raw_{s}.jsonl") if r["suite"] == "testset"]
    sample = random.Random(cfg["seed"]).sample(rows, k=max(1, round(len(rows) * frac))) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["setup", "id", "category", "judge_correctness", "manual_correctness (0-2)", "answer_excerpt"])
        for s, r in sample:
            j = judgements.get((s, r["id"]), {})
            w.writerow([s, r["id"], r["category"], j.get("correctness", ""), "", r["answer"][:600].replace("\n", " ")])


def judge_agreement() -> dict[str, Any] | None:
    path = load_config().path("eval") / "results" / "manual_scores.csv"
    if not path.exists():
        return None
    pairs = []
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            m, j = row.get("manual_correctness (0-2)", "").strip(), row.get("judge_correctness", "").strip()
            if m and j:
                pairs.append((int(float(m)), int(float(j))))
    if not pairs:
        return {"n": 0}
    exact = sum(a == b for a, b in pairs) / len(pairs)
    within1 = sum(abs(a - b) <= 1 for a, b in pairs) / len(pairs)
    labels = [0, 1, 2]
    po = exact
    pe = sum(
        (sum(a == k for a, _ in pairs) / len(pairs)) * (sum(b == k for _, b in pairs) / len(pairs)) for k in labels
    )
    kappa = (po - pe) / (1 - pe) if pe < 1 else None
    return {"n": len(pairs), "exact": exact, "within_1": within1, "cohen_kappa": kappa}


def write_reports(setups: list[str]) -> dict[str, Any]:
    from oslm.eval.plots import make_plots

    cfg = load_config()
    results = cfg.path("eval") / "results"
    data = compute(setups)
    write_manual_sheet(setups)
    data["judge_agreement"] = judge_agreement()
    S = data["setups"]
    data["headline"] = {
        s: {
            "correctness": S[s]["correctness"],
            "grounding": S[s]["grounding"],
            "probes_overall": data["probes"][s]["overall"],
        }
        for s in S
    }
    (results / "results.json").write_text(json.dumps(data, indent=2), encoding="utf-8")

    judges = sorted({j.get("judge") or "gemini" for j in read_jsonl(results / "judgements.jsonl")}) or ["(none yet)"]
    data["judges"] = judges
    lines = [
        "# OSlm evaluation summary",
        "",
        f"Setups: {', '.join(setups)}. Identical decoding (temperature 0.2, top_p 0.9, seed {cfg['seed']}). "
        f"Correctness and grounding are judged by {' + '.join(judges)} with a fixed rubric (oslm/eval/judge.py).",
        "",
        "| Metric | " + " | ".join(setups) + " |",
        "|---|" + "---|" * len(setups),
    ]

    def row(name: str, get: Any, pct: bool = False) -> None:
        lines.append(f"| {name} | " + " | ".join(_fmt(get(S[s]), pct) for s in setups) + " |")

    row("Correctness (0–2)", lambda m: m["correctness"])
    for cat in ("concept", "code_write", "bug_find"):
        row(f"  · {cat}", lambda m, c=cat: m["correctness_by_category"].get(c))
    row("Grounding (claims supported)", lambda m: m["grounding"], True)
    row("Citation validity", lambda m: m["citation_validity"], True)
    row("Code compiles", lambda m: m["code"]["compiles"], True)
    row("Code runs", lambda m: m["code"]["runs"], True)
    row("TSan clean", lambda m: m["code"]["tsan_clean"], True)
    row("Helgrind clean", lambda m: m["code"]["helgrind_clean"], True)
    row("Bug precision", lambda m: m["bug"]["precision"], True)
    row("Bug recall", lambda m: m["bug"]["recall"], True)
    row("Correct declines (out of scope)", lambda m: m["abstention"]["correct_decline"], True)
    row("False declines (in scope)", lambda m: m["abstention"]["false_decline"], True)
    row("Retrieval hit@4", lambda m: (m["retrieval"] or {}).get("hit@4"), True)
    row("Retrieval MRR", lambda m: (m["retrieval"] or {}).get("mrr"))
    row("TTFT p50 (ms)", lambda m: m["speed"]["ttft_p50"])
    row("TTFT p95 (ms)", lambda m: m["speed"]["ttft_p95"])
    row("Total p50 (ms)", lambda m: m["speed"]["total_p50"])
    row("Probe pass rate", lambda m: None)
    lines[-1] = "| Probe pass rate | " + " | ".join(_fmt(data["probes"][s]["overall"], True) for s in setups) + " |"
    agree = data["judge_agreement"]
    lines += [
        "",
        "## Judge agreement",
        "",
        (
            f"Manual vs judge on {agree['n']} answers: exact {_fmt(agree['exact'], True)}, within one point "
            f"{_fmt(agree['within_1'], True)}, Cohen's κ {_fmt(agree['cohen_kappa'])}."
            if agree and agree.get("n")
            else "Pending: fill in the `manual_correctness` column of `manual_scores.csv` (20% sample), then re-run `oslm eval`."
        ),
    ]
    lines += [
        "",
        "## Findings",
        "",
        findings_paragraph(data, setups),
        "",
        "Plots: " + ", ".join(f"`{p.name}`" for p in make_plots(data, setups, results)),
    ]
    (results / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    cats = sorted({c for s in setups for c in data["probes"][s]["by_category"]})
    plines = [
        "# Probe suite (§9.3) pass rates",
        "",
        "| Category | " + " | ".join(setups) + " |",
        "|---|" + "---|" * len(setups),
    ]
    for c in cats:
        plines.append(
            f"| {c} | " + " | ".join(_fmt(data["probes"][s]["by_category"].get(c), True) for s in setups) + " |"
        )
    for s in setups:
        if data["probes"][s]["failures"]:
            plines += ["", f"### Failures — {s}", ""] + [
                f"- `{f['id']}`: {', '.join(f['failed'])}" for f in data["probes"][s]["failures"]
            ]
    (results / "probes_summary.md").write_text("\n".join(plines) + "\n", encoding="utf-8")
    return data


def findings_paragraph(data: dict[str, Any], setups: list[str]) -> str:
    S = data["setups"]
    scored = {s: S[s]["correctness"] for s in setups if S[s]["correctness"] is not None}
    if not scored:
        return "No judged results yet."
    best = max(scored, key=scored.get)
    parts = [f"The best setup on judged correctness is **{best}** ({scored[best]:.2f}/2)."]
    if "base" in scored and "base_rag" in scored:
        parts.append(f"Retrieval moves the base model from {scored['base']:.2f} to {scored['base_rag']:.2f}.")
    if "ft" in scored and "base" in scored:
        parts.append(f"Fine-tuning alone moves it from {scored['base']:.2f} to {scored['ft']:.2f}.")
    if "ft_rag" in scored and "base_rag" in scored:
        parts.append(
            f"With retrieval, fine-tuning changes correctness from {scored['base_rag']:.2f} to {scored['ft_rag']:.2f}"
            + (
                f" and grounding from {_fmt(S['base_rag']['grounding'], True)} to {_fmt(S['ft_rag']['grounding'], True)}."
                if S["ft_rag"]["grounding"] is not None
                else "."
            )
        )
    for s in ("base_rag", "ft_rag"):
        if s in S and S[s]["bug"]["f1"] is not None:
            parts.append(f"{s} bug-finding F1 is {S[s]['bug']['f1']:.2f}.")
            break
    if best != "ft_rag" and "ft_rag" in scored:
        parts.append(
            "ft_rag is not the top setup here; see the per-category rows and probe failures for where it loses."
        )
    return " ".join(parts)
