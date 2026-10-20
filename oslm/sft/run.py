from __future__ import annotations

import asyncio
import json
import logging
import time

from oslm.config import load_config
from oslm.sft.store import read_jsonl, write_jsonl

log = logging.getLogger(__name__)


def _paths() -> dict[str, object]:
    sft = load_config().path("sft")
    raw = sft / "raw"
    return {
        "dir": sft,
        "questions": raw / "questions.jsonl",
        "programs": raw / "programs.jsonl",
        "items": raw / "items.jsonl",
        "mt_items": raw / "multiturn_items.jsonl",
        "answers": raw / "answers.jsonl",
        "verified": raw / "verified.jsonl",
        "review": sft / "review.html",
        "stats": sft / "stats.json",
    }


def _row_counts() -> int:
    p = _paths()
    return sum(len(read_jsonl(p[k])) for k in ("questions", "programs", "items", "mt_items", "answers"))  # type: ignore[arg-type]


def _free_gpu() -> None:
    """Drops the embedder and reranker so a local generator model (qwen3:8b) fits in 8 GB of VRAM."""
    import gc

    from oslm.index.embed import get_embedder
    from oslm.rag.rerank import get_reranker

    get_embedder.cache_clear()
    get_reranker.cache_clear()
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        log.debug("torch not available to release VRAM")


def _seconds_until_quota_reset() -> float:
    """Free-tier daily quotas reset at midnight US Pacific time."""
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo

    now = datetime.now(ZoneInfo("America/Los_Angeles"))
    reset = (now + timedelta(days=1)).replace(hour=0, minute=10, second=0, microsecond=0)
    return (reset - now).total_seconds()


async def _generate(limit: int | None) -> bool:
    """One generation pass; returns True if it stopped because the free daily quota ran out."""
    from oslm.checker.wsl_bridge import CheckerBridge
    from oslm.sft import generate as g
    from oslm.sft.llm import make_llm

    cfg = load_config()
    seed = cfg["seed"]
    p = _paths()
    plan = g.scaled_plan()
    if limit:
        plan = {k: min(v, limit) for k, v in plan.items()}
    lite = make_llm(lite=True)
    main = make_llm()

    log.info("step 1/5: questions %s", plan)
    await g.generate_questions(lite, p["questions"], plan, seed)
    log.info("step 2/5: bug programs (verified in the WSL sandbox)")
    bridge = CheckerBridge(concurrency=3)
    await bridge.warmup()
    await g.generate_programs(main, bridge, p["programs"], plan, seed)

    log.info("step 3/5: items with retrieval context")
    g.build_items(read_jsonl(p["questions"]), read_jsonl(p["programs"]), p["items"], plan, seed)
    _free_gpu()
    items = read_jsonl(p["items"])
    if limit:
        items = items[: limit * 4]
    log.info("step 4/5: answers for %d items", len(items))
    await g.generate_answers(main, items, p["answers"])

    log.info("step 5/5: multi-turn follow-ups")
    await g.build_multiturn(
        [a for a in read_jsonl(p["answers"]) if a["category"] != "multiturn"], p["mt_items"], plan, seed
    )
    _free_gpu()
    await g.generate_answers(main, read_jsonl(p["mt_items"]), p["answers"])
    log.info("Gemini calls per model: main=%s lite=%s", main.calls, lite.calls)
    return main.all_exhausted or lite.all_exhausted


def run_sft(
    step: str = "all",
    force: bool = False,
    limit: int | None = None,
    until_done: bool = False,
    csv_path: str | None = None,
) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "google_genai", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    cfg = load_config()
    p = _paths()
    # Formatting and review consume the verifier output, so --force must not erase it.
    # Only steps that actually recreate verified.jsonl may remove the old file.
    if force and step in ("verify", "all"):
        for key in ("verified",):
            p[key].unlink(missing_ok=True)  # type: ignore[attr-defined]

    if step in ("generate", "all"):
        while True:
            before = _row_counts()
            exhausted = asyncio.run(_generate(limit))
            added = _row_counts() - before
            log.info("generation pass added %d rows (quota exhausted: %s)", added, exhausted)
            if not until_done or (not exhausted and added == 0):
                break
            if exhausted:
                wait = _seconds_until_quota_reset()
                log.info("free daily quota used up; sleeping %.1f h until it resets", wait / 3600)
                time.sleep(wait)

    if step == "import":
        from pathlib import Path

        from oslm.sft.importer import import_csv

        if not csv_path:
            raise SystemExit("--step import needs --csv <file>")
        print(json.dumps(import_csv(Path(csv_path), replace=force), indent=2))
        return 0

    if step in ("verify", "all"):
        from oslm.sft.refresh import refresh_item
        from oslm.sft.verify import run_verify

        answers = {a["id"]: refresh_item(a) for a in read_jsonl(p["answers"])}  # last write wins
        verified = run_verify(list(answers.values()))
        write_jsonl(p["verified"], verified)
        write_jsonl(
            p["dir"] / "rejected.jsonl",
            [  # type: ignore[operator]
                {"id": v["id"], "category": v["category"], "variant": v.get("variant"), "problems": v["problems"]}
                for v in verified
                if not v["ok"]
            ],
        )

    if step in ("format", "all"):
        from oslm.sft.format import format_dataset

        stats = format_dataset(read_jsonl(p["verified"]), p["dir"], cfg["seed"])
        p["stats"].write_text(json.dumps(stats, indent=2), encoding="utf-8")  # type: ignore[attr-defined]
        print(json.dumps(stats, indent=2))

    if step in ("review", "all"):
        from oslm.sft.review import render_review

        n = render_review(read_jsonl(p["dir"] / "train.jsonl"), p["review"])  # type: ignore[operator]
        print(f"review sheet with {n} samples -> {p['review']}")
    return 0
