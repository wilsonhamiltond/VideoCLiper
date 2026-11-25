"""Check criteria: does an account's moment scoring pick what really gets views in its niche?

For the account's own YouTube Shorts and, optionally, a reference channel of the same niche, the most- and
least-viewed shorts are downloaded (360p) and analyzed like any video (speech, loudness, motion). The page then
scores each short with every moment style and compares the scores with the real views (see "Check criteria"
in the account form), and the AI can write the account's "what goes viral here" from what the top ones share.

The newest shorts are left out: they have few views only because they are new.
One check runs at a time; its downloads are deleted as soon as each short is analyzed, and the analysis is kept in
data/criteria_cache/ (a few KB per short) so checking again, with other settings or criteria, takes seconds.
"""
import json
import shutil
import statistics
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException

import channels
import fetch
import publish

LIST_MAX = 300     # shorts listed per channel
SKIP_NEWEST = 10   # newest shorts left out (not enough time to get views)
MAX_PER_GROUP = 10
CACHE = Path(__file__).parent / "data" / "criteria_cache"  # analyzed shorts by video id, so re-checks are quick
router = APIRouter()
_jobs: dict[str, dict] = {}
_analyze = None    # server.analyze_path, set by start()
_lock = threading.Lock()


def start(analyze_fn):
    global _analyze
    _analyze = analyze_fn


def shorts_url(v: str) -> str:
    """'@handle', a channel ID or link -> the channel's Shorts tab."""
    return channels.channel_url(v).rsplit("/", 1)[0] + "/shorts"


def _run(cmd: list[str], timeout: int) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout,
                          creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


def list_shorts(url: str) -> tuple[str, list[dict]]:
    """(channel name, [{id, title, views}]) newest first."""
    cmd = [a for a in fetch.ytdlp_base() if a != "--no-playlist"]
    cmd += ["--flat-playlist", "--playlist-end", str(LIST_MAX), "-J", url]
    r = _run(cmd, 300)
    if r.returncode != 0:
        raise RuntimeError(fetch.ytdlp_error(r.stderr))
    meta = json.loads(r.stdout)
    out = [{"id": e["id"], "title": e.get("title") or "", "views": int(e["view_count"])}
           for e in meta.get("entries") or [] if e and e.get("id") and e.get("view_count") is not None]
    return meta.get("channel") or meta.get("uploader") or meta.get("title") or url, out


def pick(items: list[dict], n: int) -> list[dict]:
    """The n most- and n least-viewed, after leaving out the newest (marked group "top" / "low")."""
    pool = items[SKIP_NEWEST:] if len(items) > SKIP_NEWEST + 4 else items
    ranked = sorted(pool, key=lambda x: x["views"], reverse=True)
    n = min(n, len(ranked) // 2)
    return [{**x, "group": "top"} for x in ranked[:n]] + [{**x, "group": "low"} for x in ranked[len(ranked) - n:]]


def download(vid: str, folder: Path) -> Path:
    cmd = fetch.ytdlp_base() + ["-q", "-f", "18/b[height<=480]/bv*[height<=480]+ba/w", "--merge-output-format", "mp4",
                                "-o", str(folder / "%(id)s.%(ext)s"), f"https://www.youtube.com/shorts/{vid}"]
    r = _run(cmd, 600)
    files = list(folder.glob(f"{vid}.*"))
    if r.returncode != 0 or not files:
        raise RuntimeError(fetch.ytdlp_error(r.stderr) if r.returncode else "download failed")
    return files[0]


def _slim(tr: dict) -> dict:
    """What the page needs to score a short."""
    segs = [{"start": s.get("start"), "end": s.get("end"), "text": s.get("text"), "words": s.get("words") or []}
            for s in tr.get("segments") or []]
    return {"hop": tr.get("hop"), "loudness": tr.get("loudness") or [], "motion": tr.get("motion") or [],
            "segments": segs, "language": tr.get("language")}


def run(job: dict, sources: list[dict], n: int):
    tmp = Path(tempfile.mkdtemp(prefix="clipstudio_check_"))
    try:
        plan = []
        for src in sources:
            job["step"] = f"Listing the shorts of {src['label']}…"
            name, items = list_shorts(src["url"])
            if len(items) < 4:
                raise RuntimeError(f"{name} has too few Shorts with view counts to compare ({len(items)}).")
            src.update(name=name, count=len(items), median=int(statistics.median(x["views"] for x in items)))
            src["shorts"] = []
            plan += [(src, x) for x in pick(items, n)]
        for i, (src, x) in enumerate(plan):
            if job.get("cancel"):
                raise RuntimeError("cancelled")
            job.update(progress=round(i / len(plan), 3), step=f"Short {i + 1} of {len(plan)} ({src['name']}): downloading…")
            folder = tmp / x["id"]
            folder.mkdir()
            cached = CACHE / f"{x['id']}.json"
            try:
                if cached.exists():
                    tr = json.loads(cached.read_text("utf-8"))
                else:
                    path = download(x["id"], folder)
                    job["step"] = f"Short {i + 1} of {len(plan)} ({src['name']}): analyzing…"
                    tr = _slim(_analyze(path, cancelled=lambda: bool(job.get("cancel"))))
                    CACHE.mkdir(parents=True, exist_ok=True)
                    cached.write_text(json.dumps(tr), "utf-8")
                src["shorts"].append({**x, "rel": round(x["views"] / max(1, src["median"]), 3), "tr": tr})
            except Exception as e:  # one short that won't download (private, removed…) doesn't stop the check
                if job.get("cancel"):
                    raise
                job.setdefault("skipped", []).append(f"{x['id']}: {str(e)[:120]}")
            finally:
                shutil.rmtree(folder, ignore_errors=True)
        job.update(status="done", progress=1.0, step="Done",
                   result={"sources": [{k: s.get(k) for k in ("role", "label", "name", "url", "count", "median", "shorts")}
                                       for s in sources]})
    except Exception as e:
        job.update(status="error", error="Cancelled." if str(e) == "cancelled" else str(e))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@router.post("/api/accounts/{acct_id}/check-criteria")
def check_criteria(acct_id: str, body: dict):
    """{"reference"?: channel of the same niche, "n"?: shorts per group}. Starts the check; poll GET below."""
    acct = next((a for a in publish.load() if a["id"] == acct_id), None)
    if not acct:
        raise HTTPException(404, "That account was removed.")
    with _lock:
        if any(j["status"] == "running" for j in _jobs.values()):
            raise HTTPException(409, "A check is already running. Wait for it or cancel it.")
        sources = []
        own = acct.get("yt_handle") or acct.get("yt_channel") or acct.get("yt_channel_id")
        if acct["platform"] == "youtube" and own:
            sources.append({"role": "own", "label": acct.get("label") or "this account", "url": shorts_url(own)})
        ref = str(body.get("reference") or "").strip()
        if ref:
            sources.append({"role": "reference", "label": ref, "url": shorts_url(ref)})
        if not sources:
            raise HTTPException(400, "This account has no YouTube channel to read. Add a reference channel of the same niche.")
        n = max(3, min(MAX_PER_GROUP, int(body.get("n") or 6)))
        jid = uuid.uuid4().hex
        _jobs[jid] = {"status": "running", "progress": 0.0, "step": "Starting…", "account": acct_id, "time": time.time()}
    threading.Thread(target=run, args=(_jobs[jid], sources, n), daemon=True).start()
    return {"job": jid}


@router.get("/api/criteria-check/{jid}")
def check_status(jid: str):
    job = _jobs.get(jid)
    if not job:
        raise HTTPException(404, "That check is gone. Run it again.")
    if job["status"] != "running":
        _jobs.pop(jid, None)  # results are handed out once
    return job


@router.post("/api/criteria-check/{jid}/cancel")
def check_cancel(jid: str):
    job = _jobs.get(jid)
    if job:
        job["cancel"] = True
    return {"ok": True}
