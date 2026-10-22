"""Serial rate-limited requests per outlet, with an append-only request log."""

import gzip
import json
import time
from datetime import datetime, timezone
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import requests


class Client:
    def __init__(self, domain, config, log_dir):
        self.domain = domain
        self.config = config
        self.log_dir = log_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers["User-Agent"] = config["user_agent"]
        self.session.headers.update({"Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                                     "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.8"})
        self.last_request = 0
        self.robots = None

    def allowed_host(self, url):
        parts = urlsplit(url)
        return parts.scheme in {"http", "https"} and parts.hostname in {self.domain, "www." + self.domain}

    def get(self, url, *, robots=False):
        if not self.allowed_host(url):
            raise ValueError(f"Unexpected source host: {url}")
        if not robots and self.robots and not self.robots.can_fetch(self.config["user_agent"], url):
            raise ValueError(f"robots.txt disallows: {url}")
        for attempt in range(self.config["retries"] + 1):
            delay = self.config["request_delay_seconds"]
            if self.robots:
                delay = max(delay, self.robots.crawl_delay(self.config["user_agent"]) or self.robots.crawl_delay("*") or 0)
            time.sleep(max(0, delay - (time.monotonic() - self.last_request)))
            started = datetime.now(timezone.utc).isoformat()
            result = {"url": url, "attempt": attempt + 1, "requested_at": started}
            response = None
            try:
                self.last_request = time.monotonic()
                # Validate each redirect before requesting its destination.
                target = url
                for _ in range(6):
                    if not self.allowed_host(target):
                        raise ValueError(f"Cross-host redirect refused: {target}")
                    if not robots and self.robots and not self.robots.can_fetch(self.config["user_agent"], target):
                        raise ValueError(f"Redirect disallowed by robots: {target}")
                    response = self.session.get(target, timeout=self.config["timeout_seconds"], allow_redirects=False)
                    if response.is_redirect:
                        from urllib.parse import urljoin
                        target = urljoin(target, response.headers["Location"])
                        time.sleep(delay)
                        self.last_request = time.monotonic()
                    else:
                        break
                else:
                    raise ValueError("Too many redirects")
                result.update(status=response.status_code, final_url=response.url)
                response.raise_for_status()
                body = response.content
                if body.startswith(b"\x1f\x8b"):
                    body = gzip.decompress(body)
                return body, response.url
            except (requests.RequestException, ValueError, OSError) as exc:
                result["error"] = str(exc)
                # Dân Trí's Varnish rate limit answers 428 instead of 429.
                retryable = response is None or response.status_code in {428, 429} or response.status_code >= 500
                if attempt >= self.config["retries"] or not retryable:
                    raise
                wait = 2 ** attempt
                if response is not None:
                    retry_after = response.headers.get("Retry-After", "")
                    if response.status_code in {428, 429}:
                        result["retry_after"] = retry_after or None
                    # Dân Trí blocks for ~95 s (median) once its window is exceeded.
                    if response.status_code == 428:
                        wait = max(wait, self.config.get("rate_limit_cooldown_seconds", 0))
                    if retry_after.isdigit():
                        wait = max(wait, int(retry_after))
                time.sleep(wait)
            finally:
                with (self.log_dir / "requests.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(result, ensure_ascii=False) + "\n")

    def load_robots(self):
        url = "https://" + self.domain + "/robots.txt"
        try:
            body, _ = self.get(url, robots=True)
        except requests.HTTPError as exc:
            if exc.response.status_code != 404:
                raise
            body = b"User-agent: *\nAllow: /\n"
        (self.log_dir / "robots.txt").write_bytes(body)
        self.robots = RobotFileParser(url)
        self.robots.parse(body.decode("utf-8-sig").splitlines())
