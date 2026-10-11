from __future__ import annotations

import html
import random
from pathlib import Path
from typing import Any

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>OSlm SFT review</title>
<style>
:root {{ --bg:#fafaf7; --ink:#111; --grey:#474747; --rule:#d9d9d4; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#121212; --ink:#ededed; --grey:#b8b8b8; --rule:#2a2a2a; }} }}
body {{ background:var(--bg); color:var(--ink); font:16px/1.55 system-ui, sans-serif; margin:0 auto; max-width:980px; padding:24px 16px; }}
h1 {{ font-size:24px; }} .meta {{ color:var(--grey); font-size:13px; }}
article {{ border:1px solid var(--rule); border-radius:8px; padding:14px 18px; margin:18px 0; }}
pre {{ white-space:pre-wrap; word-break:break-word; font:13px/1.5 ui-monospace, Consolas, monospace; background:rgba(127,127,127,.08); padding:10px; border-radius:6px; }}
details summary {{ cursor:pointer; color:var(--grey); }}
label {{ margin-right:14px; }} textarea {{ width:100%; min-height:48px; }}
</style></head><body>
<h1>OSlm fine-tuning data — review sample</h1>
<p class="meta">{n} of {total} training examples (random {pct}% sample). For each: is the answer correct, in teacher style, grounded in the context, and right about bugs? Note any problem below it.</p>
{articles}
</body></html>"""


def render_review(rows: list[dict[str, Any]], out: Path, frac: float = 0.1, seed: int = 42) -> int:
    rng = random.Random(seed)
    sample = rng.sample(rows, k=max(1, round(len(rows) * frac))) if rows else []
    articles = []
    for i, row in enumerate(sample, start=1):
        msgs = row["messages"]
        user = next(m for m in reversed(msgs) if m["role"] == "user")["content"]
        history = [m for m in msgs[1:-2]]
        hist_html = "".join(f"<pre><b>{h['role']}:</b> {html.escape(h['content'][:1500])}</pre>" for h in history)
        articles.append(
            f"<article><div class='meta'>#{i} · {html.escape(row['category'])} · {html.escape(row.get('variant', ''))} · "
            f"mode={html.escape(row['mode'])} · {row['n_tokens']} tokens · id {html.escape(row['id'])}</div>"
            + (f"<details><summary>earlier conversation</summary>{hist_html}</details>" if history else "")
            + f"<details><summary>user message (with context)</summary><pre>{html.escape(user)}</pre></details>"
            f"<h3>Answer</h3><pre>{html.escape(msgs[-1]['content'])}</pre>"
            f"<label><input type='radio' name='v{i}'> good</label><label><input type='radio' name='v{i}'> fix</label>"
            f"<label><input type='radio' name='v{i}'> drop</label><textarea placeholder='notes'></textarea></article>"
        )
    out.write_text(
        PAGE.format(n=len(sample), total=len(rows), pct=round(frac * 100), articles="\n".join(articles)),
        encoding="utf-8",
    )
    return len(sample)
