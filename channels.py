"""Passive mode: YouTube channels watched per account. When a channel uploads a new video, it is imported in the
background and its 2 most viral moments are found (the same pipeline as "Import from YouTube").

Saved in data/channels.json as [{"id", "account_id", "name", "url", "created", "checked", "error",
"videos": [{"id", "title", "url", "status": "running"|"done"|"error", "nid", "result", "error", "time"}]}]
(newest video first, a few kept per channel).

Only videos published in the last CHANNEL_MAX_AGE_H hours (default 24) are picked up; older uploads are ignored.

The watcher checks every channel once at startup and then every CHANNEL_POLL_MIN minutes (default 60), one
channel at a time, waiting for each import to finish so downloads and transcriptions never pile up.
"""
import json
import os
import re
import subprocess
import threading
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException

import fetch
import notify
import publish

FILE = Path(__file__).parent / "data" / "channels.json"
POLL_MIN = max(5.0, float(os.environ.get("CHANNEL_POLL_MIN") or 60))
CLIPS = 2         # viral clips found per new video
KEEP_VIDEOS = 5   # videos remembered per channel
MAX_AGE_H = max(1.0, float(os.environ.get("CHANNEL_MAX_AGE_H") or 24))  # only videos newer than this
_lock = threading.Lock()
_wake = threading.Event()
_force: set[str] = set()  # channel ids to re-process even if their latest video was done already
_import = None            # server.import_moments, set by start()
router = APIRouter()


def load() -> list[dict]:
    try:
        items = json.loads(FILE.read_text("utf-8"))
        return items if isinstance(items, list) else []
    except (OSError, ValueError):
        return []


def _write(items: list[dict]):
    FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(items, indent=2, ensure_ascii=False), "utf-8")
    tmp.replace(FILE)


def _edit(cid: str, fn):
    """Change one channel under the lock (it may have been removed meanwhile: then nothing happens)."""
    with _lock:
        items = load()
        ch = next((c for c in items if c["id"] == cid), None)
        if ch:
            fn(ch)
            _write(items)
        return ch


def channel_url(v: str) -> str:
    """'@handle', 'youtube.com/@handle', '.../channel/UC…' or '.../c/name' -> the channel's Videos tab."""
    v = (v or "").strip()
    if re.fullmatch(r"@[\w.\-]{2,}", v):
        v = "https://www.youtube.com/" + v
    elif re.fullmatch(r"UC[\w\-]{22}", v):
        v = "https://www.youtube.com/channel/" + v
    elif not re.match(r"^https?://", v) and re.match(r"^(www\.|m\.)?youtube\.com/", v):
        v = "https://" + v
    m = re.match(r"^https?://(?:www\.|m\.)?youtube\.com/(@[\w.\-]+|channel/UC[\w\-]{22}|c/[\w.\-]+|user/[\w.\-]+)", v)
    if not m:
        raise HTTPException(400, "Paste a YouTube channel: @handle or a link like https://www.youtube.com/@name")
    return f"https://www.youtube.com/{m.group(1)}/videos"


def latest_video(url: str) -> dict | None:
    """The newest finished upload on the channel's Videos tab published in the last MAX_AGE_H hours
    (live and upcoming streams are skipped). None when the channel uploaded nothing that recent."""
    cmd = [a for a in fetch.ytdlp_base() if a != "--no-playlist"]
    # approximate_date turns the tab's "5 hours ago" into a timestamp, so no per-video request is needed.
    cmd += ["--flat-playlist", "--playlist-end", "5", "--extractor-args", "youtubetab:approximate_date", "-J", url]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if r.returncode != 0:
        raise RuntimeError(fetch.ytdlp_error(r.stderr))
    meta = json.loads(r.stdout)
    cutoff = time.time() - MAX_AGE_H * 3600
    for e in meta.get("entries") or []:
        if not e or not e.get("id") or e.get("live_status") in ("is_live", "is_upcoming", "post_live"):
            continue
        ts = e.get("timestamp") or e.get("release_timestamp")
        if ts and ts < cutoff:
            return None  # the tab is newest first, so everything after this is older still
        return {"id": e["id"], "title": e.get("title") or "Video", "published": ts,
                "url": f"https://www.youtube.com/watch?v={e['id']}",
                "channel": meta.get("channel") or meta.get("uploader") or ""}
    return None


def process(cid: str):
    """Look for a new video on one channel and, if there is one, import it and wait for its 2 best moments."""
    ch = next((c for c in load() if c["id"] == cid), None)
    if not ch or not _import:
        return
    try:
        v = latest_video(ch["url"])
    except Exception as e:
        _edit(cid, lambda c: c.update(checked=time.time(), error=str(e)[:400]))
        return
    _edit(cid, lambda c: c.update(checked=time.time(), error=None))
    if not v:
        return
    known = next((x for x in ch.get("videos") or [] if x["id"] == v["id"]), None)
    if known and known["status"] in ("done", "running", "deleted") and cid not in _force:
        return
    _force.discard(cid)

    done = threading.Event()
    entry = {"id": v["id"], "title": v["title"], "url": v["url"], "published": v.get("published"),
             "status": "running", "time": time.time()}

    def finished(result, error):
        try:
            def put(c):
                for x in c.get("videos") or []:
                    if x["id"] == v["id"]:
                        x.update(status="done" if result else "error", result=result, error=error, time=time.time())
            _edit(cid, put)
        finally:
            done.set()

    try:
        # The account's viral video criteria pick the moments (its clip length here; its style and criteria in the page).
        acct = next((a for a in publish.load() if a["id"] == ch["account_id"]), {})
        started = _import({"url": v["url"], "count": CLIPS, "len": acct.get("viral_len") or 60, "account": ch["account_id"]},
                          on_done=finished,
                          title=f"{ch['name']}: {v['title']}")
    except Exception as e:
        entry.update(status="error", error=getattr(e, "detail", None) or str(e))
        started = None
    else:
        entry["nid"] = started["notification"]

    def add(c):
        c["videos"] = ([entry] + [x for x in c.get("videos") or [] if x["id"] != v["id"]])[:KEEP_VIDEOS]
        if v.get("channel") and not c.get("channel_title"):
            c["channel_title"] = v["channel"]
    _edit(cid, add)
    if started:
        done.wait(timeout=6 * 3600)


def _watch():
    while True:
        for c in load():
            try:
                process(c["id"])
            except Exception as e:  # never let one channel stop the watcher
                print("channel watcher:", c.get("name"), e, flush=True)
        _wake.wait(POLL_MIN * 60)
        _wake.clear()


def start(import_fn):
    """Called once by server.py with its import_moments(body, on_done, title)."""
    global _import
    _import = import_fn
    # Imports that were running when the server stopped will never report back.
    with _lock:
        items, changed = load(), False
        for c in items:
            for x in c.get("videos") or []:
                if x["status"] == "running":
                    x.update(status="error", error="The server stopped before this finished.")
                    changed = True
        if changed:
            _write(items)
    threading.Thread(target=_watch, daemon=True, name="channel-watcher").start()


def request_for(nid: str) -> dict | None:
    """The import a channel started for this notification, so the bell can retry it."""
    for c in load():
        for x in c.get("videos") or []:
            if x.get("nid") == nid:
                return {"url": x["url"], "count": CLIPS, "len": 60}
    return None


def _public(c: dict) -> dict:
    out = {**c, "videos": []}
    for x in c.get("videos") or []:
        x = dict(x)
        if x["status"] == "running" and x.get("nid") in notify.live:
            x["live"] = notify.live[x["nid"]]  # {"progress", "phase", "step"}
        out["videos"].append(x)
    return out


# ---------- API ----------

@router.get("/api/channels")
def list_channels(account: str = ""):
    items = [c for c in load() if not account or c["account_id"] == account]
    return {"channels": [_public(c) for c in sorted(items, key=lambda c: c.get("created", 0))],
            "poll_min": POLL_MIN, "max_age_h": MAX_AGE_H}


@router.post("/api/channels")
def add_channel(body: dict):
    """{"account_id", "name", "channel"}: watch a YouTube channel for this account."""
    acct_id = str(body.get("account_id") or "")
    if not any(a["id"] == acct_id for a in publish.load()):
        raise HTTPException(404, "That account is gone.")
    name = str(body.get("name") or "").strip()[:120]
    if not name:
        raise HTTPException(400, "Give the channel a name.")
    url = channel_url(str(body.get("channel") or ""))
    with _lock:
        items = load()
        if any(c["account_id"] == acct_id and c["url"].lower() == url.lower() for c in items):
            raise HTTPException(400, "This account already follows that channel.")
        item = {"id": uuid.uuid4().hex, "account_id": acct_id, "name": name, "url": url,
                "created": time.time(), "checked": None, "error": None, "videos": []}
        items.append(item)
        _write(items)
    _wake.set()  # look at its latest video right away
    return item


@router.post("/api/channels/check")
def check_all():
    """Wake the watcher: every channel is checked for a new video now."""
    _wake.set()
    return {"ok": True}


@router.post("/api/channels/{cid}/check")
def check_channel(cid: str, body: dict | None = None):
    """Check now; with {"again": true} the latest video is processed again even if it was done."""
    if not any(c["id"] == cid for c in load()):
        raise HTTPException(404, "That channel is gone.")
    if (body or {}).get("again"):
        _force.add(cid)
    _wake.set()
    return {"ok": True}


@router.post("/api/channels/{cid}/videos/{vid}/delete")
def delete_video(cid: str, vid: str):
    """Remove a video's clips: its download and analysis are deleted (unless a saved project uses them), and it is
    kept as "deleted" so the watcher doesn't import it again."""
    ch = next((c for c in load() if c["id"] == cid), None)
    v = next((x for x in (ch or {}).get("videos") or [] if x["id"] == vid), None)
    if not v:
        raise HTTPException(404, "That video is gone.")
    if v["status"] == "running":
        raise HTTPException(400, "It's still importing. Cancel it from the notifications first.")
    media = (v.get("result") or {}).get("media", "")
    if re.fullmatch(r"[0-9a-f]{32}", media):
        projects_dir = Path(__file__).parent / "data" / "projects"
        used = False
        for f in projects_dir.glob("*.json") if projects_dir.exists() else []:
            try:
                used = used or json.loads(f.read_text("utf-8")).get("media") == media
            except (OSError, ValueError):
                pass
        if not used:
            fetch.delete_media(media)
            (Path(__file__).parent / "data" / "analysis" / f"{media}.json").unlink(missing_ok=True)

    def mark(c):
        for x in c.get("videos") or []:
            if x["id"] == vid:
                x.update(status="deleted", result=None, error=None, time=time.time())
    _edit(cid, mark)
    return {"ok": True}


@router.post("/api/channels/{cid}/delete")  # POST, so the X-Clip-Studio header check applies
def delete_channel(cid: str):
    with _lock:
        items = load()
        left = [c for c in items if c["id"] != cid]
        if len(left) == len(items):
            raise HTTPException(404, "That channel is gone.")
        _write(left)
    return {"ok": True}
