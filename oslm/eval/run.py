from __future__ import annotations

import asyncio
import json
import logging
import subprocess
from typing import Any

from oslm.config import load_config
from oslm.sft.store import append_jsonl, read_jsonl

log = logging.getLogger(__name__)

SETUPS = ("base", "base_rag", "ft", "ft_rag")


def load_suite(suite: str) -> list[dict[str, Any]]:
    eval_dir = load_config().path("eval")
    items: list[dict[str, Any]] = []
    if suite in ("testset", "all"):
        items += [{**r, "suite": "testset"} for r in read_jsonl(eval_dir / "testset.jsonl")]
    if suite in ("probes", "all"):
        items += [{**r, "suite": "probes", "question": r["message"]} for r in read_jsonl(eval_dir / "probes.jsonl")]
    return items


def _ollama_has(tag: str) -> bool:
    out = subprocess.run(["ollama", "list"], capture_output=True, text=True, encoding="utf-8").stdout
    return any(line.split()[0].split(":")[0] == tag.split(":")[0] for line in out.splitlines()[1:] if line.strip())


async def reverify(out_path: Any) -> int:
    """Canonicalises stored answers' newlines and sandbox-checks complete C programs that were never checked.

    Answers collected before newline canonicalisation (the fine-tuned GGUF emits "\\r\\n" in code) hid their programs
    from the detector; this repairs those rows in place so code metrics reflect every program. Idempotent.
    """
    from oslm.checker.wsl_bridge import CheckerBridge
    from oslm.rag.pipeline import complete_c_programs
    from oslm.sft.store import write_jsonl

    rows = read_jsonl(out_path)
    todo = []
    for row in rows:
        text = row["answer"].replace("\r\n", "\n")
        changed = text != row["answer"]
        row["answer"] = text
        programs = complete_c_programs(text)
        if programs and (changed or not row.get("verification")):
            todo.append((row, programs))
    if todo:
        checker = CheckerBridge()
        await checker.warmup()
        for row, programs in todo:
            row["verification"] = [r.to_dict() for r in await checker.check_many(programs)]
    write_jsonl(out_path, rows)
    return len(todo)


async def collect(setup: str, items: list[dict[str, Any]], out_path: Any) -> None:
    from oslm.checker.wsl_bridge import CheckerBridge
    from oslm.rag.pipeline import ChatRequest, Pipeline, answer_text

    done = {r["id"] for r in read_jsonl(out_path)}
    todo = [it for it in items if it["id"] not in done]
    if not todo:
        return
    checker = CheckerBridge()
    await checker.warmup()
    pipe = Pipeline(checker=checker)
    try:
        for i, it in enumerate(todo, start=1):
            text, info = await answer_text(
                pipe, ChatRequest(it["question"], setup=setup, history=it.get("history", []))
            )
            row = {
                "id": it["id"],
                "suite": it["suite"],
                "category": it["category"],
                "setup": setup,
                "answer": text,
                "meta": info.get("meta", {}),
                "sources": info.get("sources", {}).get("sources", []),
                "done": info.get("done", {}),
                "error": info.get("error"),
                "verification": info["verification"],
            }
            append_jsonl(out_path, [row])
            if i % 10 == 0:
                log.info("%s: %d/%d", setup, i, len(todo))
    finally:
        await pipe.client.aclose()


def run_eval(setups: list[str], suite: str = "all", force: bool = False, limit: int | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "sentence_transformers", "google_genai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    cfg = load_config()
    results = cfg.path("eval") / "results"
    results.mkdir(parents=True, exist_ok=True)
    items = load_suite(suite)
    if limit:
        items = items[:limit]
    ran = []
    for setup in setups:
        tag = cfg.ollama["ft_tag"] if setup.startswith("ft") else cfg.ollama["base_tag"]
        if not _ollama_has(tag):
            log.warning("skipping %s: Ollama model %s not available", setup, tag)
            continue
        out = results / f"raw_{setup}.jsonl"
        if force:
            out.unlink(missing_ok=True)
        log.info("collecting %s on %d items", setup, len(items))
        asyncio.run(collect(setup, items, out))
        n = asyncio.run(reverify(out))
        if n:
            log.info("%s: re-verified %d answers' programs", setup, n)
        ran.append(setup)

    from oslm.eval.judge import run_judge
    from oslm.eval.metrics import write_reports

    run_judge(ran)
    summary = write_reports(ran)
    print(json.dumps(summary["headline"], indent=2))
    return 0
