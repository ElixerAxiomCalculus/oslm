from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

from oslm.config import ROOT, load_config
from oslm.ingest.crawler import Crawler, FetchResult, Source, read_sources
from oslm.ingest.loaders import LOCAL_SUFFIXES, html_to_text, load_local, load_pdf
from oslm.ingest.manifest import Document, Manifest, ManifestEntry, Segment, content_hash, make_doc_id

log = logging.getLogger(__name__)

COURSE_LICENSE = "Course material (private, not redistributed)"


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def document_from_fetch(result: FetchResult, source: Source) -> Document | None:
    is_pdf = source.type == "pdf" or "pdf" in result.content_type or result.body[:5] == b"%PDF-"
    if is_pdf:
        title, segments = load_pdf(result.body)
        doc_type = "pdf"
    else:
        title, text = html_to_text(result.body)
        segments = [Segment(text=text)]
        doc_type = "html"
    segments = [s for s in segments if s.text.strip()]
    if not segments:
        return None
    title = title.strip() or Path(result.final_url).stem
    meta = ManifestEntry(
        doc_id=make_doc_id(source.source_name, title, result.url),
        source_name=source.source_name,
        url_or_path=result.url,
        type=doc_type,
        license=source.license,
        trust_tier=source.trust_tier,
        title=title,
        retrieved_at=_now(),
        sha256=content_hash(segments),
        topic_tags=list(source.topic_tags),
    )
    return Document(meta, segments)


def ingest_web(manifest: Manifest, force: bool = False) -> tuple[int, int]:
    cfg = load_config()
    sources = read_sources(cfg.path("sources_csv"))
    crawler = Crawler(
        cache_dir=cfg.path("cache") / "http",
        user_agent=cfg.crawler["user_agent"],
        allowed_domains={s.domain for s in sources},
        delay_s=cfg.crawler["delay_s"],
        retries=cfg.crawler["retries"],
        timeout_s=cfg.crawler["timeout_s"],
    )
    added = skipped = 0
    for source in sources:
        if not force and manifest.by_location(source.url) and source.max_depth == 0:
            skipped += 1
            continue
        # --force re-extracts documents; the HTTP cache still applies (delete data/cache/http to re-download).
        for result in crawler.crawl(source):
            doc = document_from_fetch(result, source)
            if doc is None:
                log.warning("no text extracted from %s", result.url)
                continue
            if manifest.add(doc, replace=force):
                added += 1
                log.info("added %s (%d segments)", doc.meta.doc_id, len(doc.segments))
            else:
                skipped += 1
    log.info("web: %d added, %d unchanged, %d network requests", added, skipped, crawler.network_requests)
    return added, skipped


def ingest_course(manifest: Manifest, course_dir: Path, force: bool = False) -> tuple[int, int]:
    added = skipped = 0
    files = sorted(p for p in course_dir.rglob("*") if p.is_file() and p.suffix.lower() in LOCAL_SUFFIXES)
    for path in files:
        rel = path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)
        existing = manifest.by_location(rel)
        if existing and not force and manifest.doc_path(existing.doc_id).exists():
            skipped += 1
            continue
        try:
            doc_type, title, segments = load_local(path)
        except Exception as exc:
            log.warning("failed to load %s: %s", path.name, exc)
            continue
        segments = [s for s in segments if s.text.strip()]
        if not segments:
            log.warning("no text in %s (scanned PDF?)", path.name)
            continue
        meta = ManifestEntry(
            doc_id=make_doc_id("course", title, rel),
            source_name=f"Course: {path.parent.name}" if path.parent != course_dir else "Course notes",
            url_or_path=rel,
            type=doc_type,
            license=COURSE_LICENSE,
            trust_tier=1,
            title=title,
            retrieved_at=_now(),
            sha256=content_hash(segments),
        )
        if manifest.add(Document(meta, segments), replace=force):
            added += 1
        else:
            skipped += 1
    log.info("course: %d added, %d unchanged (%d files)", added, skipped, len(files))
    return added, skipped


def run_ingest(force: bool = False, web: bool = True, course: bool = True) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config()
    manifest = Manifest(cfg.path("raw"))
    if web:
        ingest_web(manifest, force=force)
    if course:
        ingest_course(manifest, cfg.path("course"), force=force)
    print(f"manifest: {len(manifest.entries)} documents -> {manifest.path}")
    return 0
