"""Notifications: background jobs (videos from a guide) report here when they start, progress and finish.

Kept in data/notifications.json so a finished video is still announced after the page was closed or
reloaded. Progress of running jobs lives in memory (see `live`); the page polls /api/notifications.
"""

import json
import threading
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException

ROOT = Path(__file__).parent
FILE = ROOT / "data" / "notifications.json"
KEEP = 60  # newest notifications kept

router = APIRouter()
_lock = threading.Lock()
live: dict[str, dict] = {}  # notification id -> {"progress", "step", "status"} while its job runs


def _load() -> list[dict]:
    try:
        return json.loads(FILE.read_text("utf-8"))
    except (OSError, ValueError):
        return []


def _save(items: list[dict]):
    FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(items[:KEEP], ensure_ascii=False), "utf-8")
    tmp.replace(FILE)


def add(kind: str, title: str, body: str = "", status: str = "info", **extra) -> str:
    nid = uuid.uuid4().hex
    item = {"id": nid, "kind": kind, "title": title, "body": body, "status": status, "time": time.time(),
            "read": status == "running", **extra}
    with _lock:
        _save([item] + _load())
    return nid


def update(nid: str, **fields):
    item = None
    with _lock:
        items = _load()
        for it in items:
            if it["id"] == nid:
                it.update(fields)
                if fields.get("status") in ("done", "error", "cancelled"):
                    it["time_done"] = time.time()
                item = it
                break
        _save(items)
    if fields.get("status") in ("done", "error", "cancelled"):
        live.pop(nid, None)
    if item and fields.get("status") in ("done", "error"):
        _push(item)


def _push(it: dict):
    """Phone / browser push for a finished job (push.py)."""
    import push
    moments, ok = it.get("kind") == "moments", it["status"] == "done"
    head = ("Clips ready" if moments else "Video ready") if ok else ("Import failed" if moments else "Video failed")
    if moments and ok and not it.get("result"):
        head = "Recording ended"
    push.send(f"{head} · {it.get('title', '')}"[:120], it.get("body", ""), it["id"])


def _interrupted():
    """Jobs that were running when the server stopped will never finish: say so once at startup."""
    with _lock:
        items, changed = _load(), False
        for it in items:
            if it.get("status") == "running":
                it.update(status="error", read=False, body="The server stopped before this finished. Start it again.")
                changed = True
        if changed:
            _save(items)


_interrupted()


@router.get("/api/notifications")
def list_notifications():
    items = _load()
    for it in items:
        if it.get("status") == "running" and it["id"] in live:
            it.update(live[it["id"]])
    return {"items": items, "unread": sum(not it.get("read") for it in items),
            "running": sum(it.get("status") == "running" for it in items)}


@router.post("/api/notifications/read")
def mark_read(body: dict | None = None):
    ids = set((body or {}).get("ids") or [])
    with _lock:
        items = _load()
        for it in items:
            if (not ids or it["id"] in ids) and it.get("status") != "running":
                it["read"] = True
        _save(items)
    return {"ok": True}


@router.post("/api/notifications/clear")
def clear():
    """Removes finished notifications; running jobs stay."""
    with _lock:
        _save([it for it in _load() if it.get("status") == "running"])
    return {"ok": True}


@router.post("/api/notifications/{nid}/delete")
def delete(nid: str):
    with _lock:
        items = _load()
        it = next((x for x in items if x["id"] == nid), None)
        if not it:
            raise HTTPException(404, "unknown notification")
        if it.get("status") == "running":
            raise HTTPException(400, "Cancel the job first.")
        _save([x for x in items if x["id"] != nid])
    media = (it.get("result") or {}).get("media") or ""
    return {"ok": True, "media": media if media and delete_video(media) else ""}


def delete_video(media: str) -> bool:
    """Removing a finished notification removes its video too: the saved video in the studio
    (data/projects), its analysis (data/analysis) and every file of it in downloads/."""
    import fetch
    import projects
    if not fetch.MEDIA_ID.match(media):
        return False
    for path in projects.PROJECTS.glob("*.json") if projects.PROJECTS.exists() else []:
        try:
            if json.loads(path.read_text("utf-8")).get("media") == media:
                path.unlink(missing_ok=True)
        except (OSError, ValueError):
            continue
    (ROOT / "data" / "analysis" / f"{media}.json").unlink(missing_ok=True)
    for path in fetch.DOWNLOADS.glob(f"{media}.*"):  # .mp4, .srt, .body.mp4, .words.json…
        try:
            path.unlink(missing_ok=True)
        except OSError as e:  # still open by a player or ffmpeg (Windows)
            print(f"couldn't delete {path.name}: {e}")
    return True
