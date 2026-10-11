from __future__ import annotations

import json
from dataclasses import dataclass

from oslm.config import load_config


@dataclass
class SmokeResult:
    hit_at_k: float
    mrr: float
    k: int
    misses: list[str]


def _is_gold(url: str, gold: list[str]) -> bool:
    return any(g.lower() in url.lower() for g in gold)


def score(ranked_urls: list[list[str]], golds: list[list[str]], questions: list[str], k: int) -> SmokeResult:
    hits, rr, misses = 0, 0.0, []
    for urls, gold, q in zip(ranked_urls, golds, questions, strict=True):
        rank = next((i for i, u in enumerate(urls[:k], start=1) if _is_gold(u, gold)), None)
        if rank:
            hits += 1
            rr += 1.0 / rank
        else:
            misses.append(q)
    n = max(len(golds), 1)
    return SmokeResult(hits / n, rr / n, k, misses)


NEAR_MISS_QUESTIONS = [
    "How do futexes work inside the Linux kernel?",
    "Explain read-copy-update (RCU) and grace periods",
    "What is a seqlock and when is it better than a rwlock?",
    "How does hardware transactional memory (Intel TSX) work?",
    "Explain Go's select statement over channels",
    "How does the Raft consensus algorithm elect a leader?",
    "What are memory_order_acquire and memory_order_release in C++ atomics?",
    "Explain the MESI cache coherence protocol",
    "How do GPU warps synchronize with __syncthreads?",
    "What is the ABA problem in lock-free stacks and how do hazard pointers solve it?",
    "Explain round-robin scheduling with a time quantum",
    "How does demand paging handle a page fault?",
    "Explain RAID 5 parity",
    "What is a B-tree?",
    "How does TCP congestion control work?",
]


def calibrate_threshold() -> dict[str, float]:
    """Top rerank scores for covered vs uncovered questions; suggests a not-in-notes threshold."""
    from oslm.rag.rerank import rerank
    from oslm.rag.retrieve import dense_search

    cfg = load_config()
    path = cfg.path("eval") / "retrieval_smoke.jsonl"
    positives = [json.loads(line)["question"] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def top_score(q: str) -> float:
        top = rerank(q, dense_search(q), k=1)
        return top[0].rerank_score if top else 0.0

    pos = sorted(top_score(q) for q in positives)
    neg = sorted((top_score(q), q) for q in NEAR_MISS_QUESTIONS)
    print(f"covered:   min={pos[0]:.3f} p10={pos[len(pos) // 10]:.3f} median={pos[len(pos) // 2]:.3f}")
    for s, q in neg:
        print(f"  uncovered {s:.3f}  {q}")
    # Favour answering: a false decline on a covered question costs more than a weak answer to an uncovered one.
    threshold = round(min(pos[0] * 0.5, max(s for s, _ in neg) + 0.01), 3)
    print(f"suggested relevance_threshold = {threshold}")
    return {"threshold": threshold, "pos_min": pos[0], "neg_max": neg[-1][0]}


def run_smoke(k: int = 5) -> int:
    from oslm.rag.rerank import rerank
    from oslm.rag.retrieve import dense_search

    cfg = load_config()
    path = cfg.path("eval") / "retrieval_smoke.jsonl"
    items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    questions = [it["question"] for it in items]
    golds = [it["gold_url_contains"] for it in items]

    dense_urls, rerank_urls, top_scores = [], [], []
    for q in questions:
        candidates = dense_search(q)
        dense_urls.append([c.metadata.get("url", "") for c in candidates])
        top = rerank(q, list(candidates), k=k)
        rerank_urls.append([c.metadata.get("url", "") for c in top])
        top_scores.append(top[0].rerank_score if top else 0.0)

    dense = score(dense_urls, golds, questions, k)
    reranked = score(rerank_urls, golds, questions, k)
    print(f"dense    hit@{k}={dense.hit_at_k:.3f}  MRR={dense.mrr:.3f}")
    print(f"reranked hit@{k}={reranked.hit_at_k:.3f}  MRR={reranked.mrr:.3f}")
    print(f"top rerank score: min={min(top_scores):.3f} median={sorted(top_scores)[len(top_scores) // 2]:.3f}")
    for q in reranked.misses:
        print(f"  miss: {q}")
    out = cfg.path("eval") / "results" / "retrieval_smoke.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "dense": dense.__dict__,
                "reranked": reranked.__dict__,
                "top_scores": top_scores,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0 if reranked.hit_at_k >= 0.85 else 1
