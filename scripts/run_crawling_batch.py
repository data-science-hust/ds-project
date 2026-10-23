"""Run the outlet crawl to quota, several outlets in parallel.

Writes batch status and per-source console logs to data/crawl_work/_batches/<run_id>/.
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "data/crawl_work"


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--sources", nargs="+", default=["all"])
    parser.add_argument("--workers", type=int, default=2, help="Independent outlets; each outlet keeps its own request delay")
    parser.add_argument("--reuse-discovery-cache", action="store_true", help="Reuse URL metadata/XML only; never reuse article selection across runs")
    parser.add_argument("--stop", action="store_true", help="Request a running batch to stop after its active request")
    args = parser.parse_args()
    if not args.run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in args.run_id):
        parser.error("Invalid run-id")
    if not 1 <= args.workers <= 12:
        parser.error("workers must be between 1 and 12")
    directory = WORK / "_batches" / args.run_id
    if args.stop:
        if not (directory / ".running").exists():
            parser.error("No active batch lock found")
        (directory / "stop_requested").write_text(now(), encoding="utf-8")
        print("Stop requested; see batch status.json for completion.")
        return 0
    with (ROOT / "configs/crawling/sources.csv").open(encoding="utf-8-sig") as stream:
        sources = [row["source"] for row in csv.DictReader(stream)]
    names = (["yan", "saostar"] + [source for source in sources if source not in {"yan", "saostar"}]
             if args.sources == ["all"] else list(dict.fromkeys(args.sources)))
    if set(names) - set(sources):
        parser.error("Unknown source")
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory / ".running"
    try:
        with lock.open("x") as stream:
            stream.write(str(os.getpid()))
    except FileExistsError:
        parser.error("Batch is locked; verify the recorded process before resuming")
    status_path = directory / "status.json"
    snapshot = {name: (ROOT / "configs/crawling" / name).read_text(encoding="utf-8")
                for name in ("run.json", "profiles.json", "sources.csv")}
    mutex = threading.Lock()
    try:
        status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {
            "run_id": args.run_id, "created_at": now(), "config_snapshot": snapshot, "sources": {}}
        if status["config_snapshot"] != snapshot:
            parser.error("Batch config changed; use a new run-id")
        if (directory / "stop_requested").exists():
            parser.error("Previous stop request exists; use a new run-id or remove it after reviewing the stopped batch")
        status.update(state="running", pid=os.getpid(), workers=args.workers)

        def update(source=None, **values):
            with mutex:
                target = status if source is None else status["sources"].setdefault(source, {})
                target.update(values)
                status["updated_at"] = now()
                temporary = status_path.with_suffix(".json.tmp")
                temporary.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
                temporary.replace(status_path)

        def worker(source):
            if status["sources"].get(source, {}).get("state") == "complete":
                return
            if (directory / "stop_requested").exists():
                update(source, state="stopped")
                return
            destination = WORK / source / args.run_id
            if args.reuse_discovery_cache and not (destination / "run_manifest.json").exists():
                cache = destination / "discovery_cache"
                cache.mkdir(parents=True, exist_ok=True)
                provenance = []
                prior = sorted((WORK / source).glob("*/discovery_cache/*"), key=lambda p: p.stat().st_mtime)
                for path in prior:
                    if path.parent == cache or path.suffix not in {".json", ".xml"}:
                        continue
                    shutil.copy2(path, cache / path.name)
                    provenance.append(str(path.relative_to(ROOT)))
                (cache.parent / "cache_reuse.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
            update(source, state="running", started_at=now())
            command = [sys.executable, "-u", str(ROOT / "scripts/crawl_data.py"),
                       "--sources", source, "--run-id", args.run_id]
            with (directory / f"{source}.console.log").open("a", encoding="utf-8") as output:
                process = subprocess.Popen(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT)
                update(source, child_pid=process.pid)
                stopped = False
                while process.poll() is None:
                    if (directory / "stop_requested").exists():
                        stopped = True
                        process.terminate()
                        try:
                            process.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                        # The child is now confirmed stopped; clear only its run lock.
                        child_lock = destination / ".running"
                        if child_lock.exists():
                            child_lock.unlink()
                        break
                    time.sleep(1)
                code = process.returncode
                report = destination / "crawl_report.json"
                quota = json.loads(report.read_text(encoding="utf-8")) if report.exists() else None
                update(source, state="stopped" if stopped else "complete" if code == 0 else "needs_review",
                       finished_at=now(), crawl_exit_code=code, quota=quota)

        update()
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            tasks = {pool.submit(worker, source): source for source in names}
            for future in as_completed(tasks):
                try:
                    future.result()
                except Exception as exc:
                    update(tasks[future], state="needs_review", error=str(exc), finished_at=now())
        stopped = (directory / "stop_requested").exists()
        complete = all(status["sources"].get(source, {}).get("state") == "complete" for source in names)
        update(state="stopped" if stopped else "complete" if complete else "needs_review", finished_at=now())
        return 0 if complete else 1
    finally:
        lock.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
