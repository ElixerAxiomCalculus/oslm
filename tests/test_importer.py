from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from oslm.sft import importer

TEMPLATE = Path(__file__).resolve().parent.parent / "data" / "sft" / "external_template.csv"


def test_template_parses_with_multiline_cells() -> None:
    rows, problems = importer.read_rows(TEMPLATE)
    assert problems == []
    assert [r["category"] for r in rows] == ["concept", "bug_find", "os_out_of_scope"]
    assert "```c" in rows[1]["question"] and rows[1]["has_bug"] is True
    assert rows[0]["has_bug"] is None


def test_bad_rows_reported(tmp_path: Path) -> None:
    p = tmp_path / "x.csv"
    p.write_text(
        'category,question,answer\nconcept,Q?,\nweird,Q2?,A2\nBug,"Q3?","Verdict: bug found"\n', encoding="utf-8"
    )
    rows, problems = importer.read_rows(p)
    assert [r["category"] for r in rows] == ["bug_find"]
    assert len(problems) == 2 and "unknown category" in problems[1]


def test_missing_columns(tmp_path: Path) -> None:
    p = tmp_path / "x.csv"
    p.write_text("q,a\nx,y\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        importer.read_rows(p)


def test_add_citations_marks_supported_sentences(monkeypatch: pytest.MonkeyPatch) -> None:
    import oslm.index.embed as embed

    def fake_embed(texts: list[str]) -> np.ndarray:
        # Sentences mentioning "lock" match chunk 2; everything else matches nothing.
        out = []
        for t in texts:
            if t.startswith("CHUNK2") or "lock" in t:
                out.append([0.0, 1.0])
            elif t.startswith("CHUNK1"):
                out.append([1.0, 0.0])
            else:
                out.append([0.6, -0.8])
        return np.array(out)

    monkeypatch.setattr(embed, "embed_passages", fake_embed)
    answer = "A mutex lock gives one thread at a time access to data. Pizza is tasty and quite round indeed.\n```c\nint lock;\n```\nDone?"
    out = importer.add_citations(answer, [{"text": "CHUNK1"}, {"text": "CHUNK2"}])
    assert "access to data [2]." in out
    assert "round indeed." in out and "indeed [" not in out
    assert "int lock;" in out and "lock; [" not in out


def test_existing_citations_left_alone() -> None:
    assert importer.add_citations("Locks help [1].", [{"text": "x"}]) == "Locks help [1]."


def test_parse_followup() -> None:
    q = "Earlier — Student: How does a semaphore buffer work? | Tutor: empty reserves space; full announces items.\nNow: what if there are 3 producers?"
    history, msg = importer.parse_followup(q)
    assert history == [
        {"role": "user", "content": "How does a semaphore buffer work?"},
        {"role": "assistant", "content": "empty reserves space; full announces items."},
    ]
    assert msg == "what if there are 3 producers?"
    assert importer.parse_followup("just a question") is None


def test_code_answer_cleanup_removes_memorised_scaffold() -> None:
    answer = """Plan.\n```c
#define _XOPEN_SOURCE 700
#include <pthread.h>
#include <semaphore.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
static void ck(int e) { if (e) exit(1); }
#define SO(x) do { if ((x)==-1) exit(1); } while (0)
#define P(s) do { if (sem_wait(&(s))==-1) exit(1); } while (0)
#define V(s) SO(sem_post(&(s)))
sem_t ready;
int main(void) { ck(pthread_mutex_lock(0)); P(ready); V(ready); return 0; }
```"""
    cleaned = importer.clean_external_answer(answer, "code_write")
    assert "_XOPEN_SOURCE" not in cleaned and "static void ck" not in cleaned
    assert "#define P(" not in cleaned and "#define V(" not in cleaned and "#define SO(" not in cleaned
    assert "assert((pthread_mutex_lock(0)) == 0)" in cleaned
    assert "assert(sem_wait(&(ready)) == 0)" in cleaned
    assert "assert(sem_post(&(ready)) == 0)" in cleaned
    assert "semaphore.h" in cleaned and "stdio.h" not in cleaned
    assert "pthread.h" in cleaned and "assert.h" in cleaned


def test_non_code_answers_are_not_mechanically_rewritten() -> None:
    answer = "A ck() helper can centralize error handling."
    assert importer.clean_external_answer(answer, "concept") == answer


def test_code_answer_cleanup_handles_pointer_pv_functions() -> None:
    answer = """```c
#include <semaphore.h>
#include <assert.h>
static void P(sem_t *s) {
    while (sem_wait(s) == -1) {}
}
static void V(sem_t *s) { sem_post(s); }
static void keep_me(void) { sem_init(&ready, 0, 0); }
sem_t ready;
int main(void) { keep_me(); P(&ready); V(&ready); return 0; }
```"""
    cleaned = importer.clean_external_answer(answer, "code_write")
    assert "static void P" not in cleaned and "static void V" not in cleaned
    assert "assert(sem_wait(&ready) == 0)" in cleaned
    assert "assert(sem_post(&ready) == 0)" in cleaned
    assert "&(&ready)" not in cleaned
    assert "static void keep_me" in cleaned
