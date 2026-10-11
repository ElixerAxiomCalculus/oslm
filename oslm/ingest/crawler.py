from __future__ import annotations

import csv
import hashlib
import json
import logging
import time
import urllib.robotparser
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urldefrag, urljoin, urlparse

import requests
from bs4 import BeautifulSoup

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Source:
    url: str
    source_name: str
    type: str
    license: str
    trust_tier: int
    max_depth: int
    path_prefix: str
    topic_tags: tuple[str, ...]

    @property
    def domain(self) -> str:
        return urlparse(self.url).netloc


def read_sources(path: Path) -> list[Source]:
    with path.open(encoding="utf-8", newline="") as f:
        return [
            Source(
                url=row["url"].strip(),
                source_name=row["source_name"].strip(),
                type=row["type"].strip(),
                license=row["license"].strip(),
                trust_tier=int(row["trust_tier"]),
                max_depth=int(row["max_depth"]),
                path_prefix=row["path_prefix"].strip(),
                topic_tags=tuple(t for t in row["topic_tags"].split(";") if t),
            )
            for row in csv.DictReader(f)
        ]


@dataclass
class FetchResult:
    url: str
    final_url: str
    status: int
    content_type: str
    body: bytes
    from_cache: bool


class Crawler:
    def __init__(
        self,
        cache_dir: Path,
        user_agent: str,
        allowed_domains: set[str],
        delay_s: float = 1.0,
        retries: int = 3,
        timeout_s: float = 20.0,
        session: requests.Session | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.user_agent = user_agent
        self.allowed_domains = allowed_domains
        self.delay_s = delay_s
        self.retries = retries
        self.timeout_s = timeout_s
        self.session = session or requests.Session()
        self.session.headers["User-Agent"] = user_agent
        self._sleep = sleep
        self._clock = clock
        self._robots: dict[str, urllib.robotparser.RobotFileParser] = {}
        self._last_request: dict[str, float] = {}
        self.network_requests = 0

    def _cache_paths(self, url: str) -> tuple[Path, Path]:
        key = hashlib.sha256(url.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{key}.bin", self.cache_dir / f"{key}.json"

    def _throttle(self, domain: str) -> None:
        robots = self._robots.get(domain)
        crawl_delay = robots.crawl_delay(self.user_agent) if robots else None
        delay = max(self.delay_s, float(crawl_delay or 0))
        last = self._last_request.get(domain)
        if last is not None:
            wait = delay - (self._clock() - last)
            if wait > 0:
                self._sleep(wait)
        self._last_request[domain] = self._clock()

    def _get(self, url: str) -> requests.Response:
        domain = urlparse(url).netloc
        last_exc: Exception | None = None
        for attempt in range(self.retries + 1):
            self._throttle(domain)
            try:
                self.network_requests += 1
                resp = self.session.get(url, timeout=self.timeout_s)
                if resp.status_code < 500 and resp.status_code != 429:
                    return resp
                last_exc = requests.HTTPError(f"HTTP {resp.status_code}")
            except requests.RequestException as exc:
                last_exc = exc
            if attempt < self.retries:
                self._sleep(2**attempt)
        raise RuntimeError(f"giving up on {url}: {last_exc}")

    def robots_allows(self, url: str) -> bool:
        parsed = urlparse(url)
        domain = parsed.netloc
        if domain not in self._robots:
            parser = urllib.robotparser.RobotFileParser()
            robots_url = f"{parsed.scheme}://{domain}/robots.txt"
            try:
                resp = self._get(robots_url)
                if resp.status_code >= 400:
                    parser.parse([])
                else:
                    parser.parse(resp.text.splitlines())
            except RuntimeError:
                log.warning("robots.txt unreachable for %s; treating as disallowed", domain)
                parser.parse(["User-agent: *", "Disallow: /"])
            self._robots[domain] = parser
        return self._robots[domain].can_fetch(self.user_agent, url)

    def fetch(self, url: str, force: bool = False) -> FetchResult | None:
        body_path, meta_path = self._cache_paths(url)
        if not force and body_path.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            return FetchResult(
                url, meta["final_url"], meta["status"], meta["content_type"], body_path.read_bytes(), True
            )
        if urlparse(url).netloc not in self.allowed_domains:
            log.info("skip (domain not whitelisted): %s", url)
            return None
        if not self.robots_allows(url):
            log.info("skip (robots.txt): %s", url)
            return None
        resp = self._get(url)
        result = FetchResult(url, resp.url, resp.status_code, resp.headers.get("Content-Type", ""), resp.content, False)
        if resp.status_code == 200:
            body_path.write_bytes(result.body)
            meta_path.write_text(
                json.dumps(
                    {"final_url": result.final_url, "status": result.status, "content_type": result.content_type}
                ),
                encoding="utf-8",
            )
        return result

    def in_scope(self, url: str, source: Source) -> bool:
        parsed = urlparse(url)
        return (
            parsed.scheme in ("http", "https")
            and parsed.netloc == source.domain
            and parsed.netloc in self.allowed_domains
            and parsed.path.startswith(source.path_prefix)
        )

    def crawl(self, source: Source, force: bool = False) -> list[FetchResult]:
        """Breadth-first crawl from the source URL, staying under its path prefix and max depth."""
        results: list[FetchResult] = []
        seen = {source.url}
        queue: deque[tuple[str, int]] = deque([(source.url, 0)])
        while queue:
            url, depth = queue.popleft()
            try:
                result = self.fetch(url, force=force)
            except RuntimeError as exc:
                log.warning("%s", exc)
                continue
            if result is None or result.status != 200:
                continue
            results.append(result)
            if depth >= source.max_depth or "html" not in result.content_type:
                continue
            for link in BeautifulSoup(result.body, "lxml").find_all("a", href=True):
                target = urldefrag(urljoin(result.final_url, link["href"]))[0]
                if target not in seen and self.in_scope(target, source):
                    seen.add(target)
                    queue.append((target, depth + 1))
        return results
