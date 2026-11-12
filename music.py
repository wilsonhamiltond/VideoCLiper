"""Background music library: audio files you upload once and reuse on any clip.

Tracks are kept in data/music/ on this machine. The page plays them under the clip in the preview
and mixes them into the export, looped to the clip length, at the volume set per clip.
"""

import json
import re
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

ROOT = Path(__file__).parent
DIR = ROOT / "data" / "music"
INDEX = DIR / "index.json"
TRACK_ID = re.compile(r"^[0-9a-f]{32}$")
TYPES = {".mp3": "audio/mpeg", ".m4a": "audio/mp4", ".aac": "audio/aac", ".wav": "audio/wav",
         ".ogg": "audio/ogg", ".oga": "audio/ogg", ".opus": "audio/ogg", ".flac": "audio/flac", ".webm": "audio/webm"}
MAX_BYTES = 60 * 1024 * 1024

router = APIRouter()


def load() -> list[dict]:
    try:
        return json.loads(INDEX.read_text("utf-8"))
    except (OSError, ValueError):
        return []


def store(tracks: list[dict]):
    DIR.mkdir(parents=True, exist_ok=True)
    INDEX.write_text(json.dumps(tracks, ensure_ascii=False), "utf-8")


def track_file(tid: str) -> Path:
    t = next((t for t in load() if t["id"] == tid), None) if TRACK_ID.match(tid) else None
    path = DIR / (tid + t["ext"]) if t else None
    if not path or not path.exists():
        raise HTTPException(404, "That music track is gone. Add it again.")
    return path


@router.get("/api/music")
def list_music():
    return {"tracks": [{k: t[k] for k in ("id", "name", "added")} for t in load()]}


@router.post("/api/music")
async def add_music(request: Request, name: str = ""):
    """The body is the raw audio file."""
    ext = Path(name).suffix.lower()
    if ext not in TYPES:
        raise HTTPException(400, "Use an audio file: MP3, M4A, WAV, OGG, OPUS or FLAC.")
    DIR.mkdir(parents=True, exist_ok=True)
    tid = uuid.uuid4().hex
    dst, size = DIR / (tid + ext), 0
    try:
        with dst.open("wb") as f:
            async for chunk in request.stream():
                size += len(chunk)
                if size > MAX_BYTES:
                    raise HTTPException(413, "That file is over 60 MB. Use a shorter or compressed track.")
                f.write(chunk)
    except BaseException:
        dst.unlink(missing_ok=True)
        raise
    if not size:
        dst.unlink(missing_ok=True)
        raise HTTPException(400, "empty upload")
    title = re.sub(r"[\x00-\x1f]+", " ", Path(name).stem).strip()[:80] or "Track"
    track = {"id": tid, "name": title, "ext": ext, "added": int(time.time())}
    store([track] + load())
    return {k: track[k] for k in ("id", "name", "added")}


@router.get("/api/music/{tid}")
def get_music(tid: str):
    path = track_file(tid)
    return FileResponse(path, media_type=TYPES.get(path.suffix, "application/octet-stream"))


@router.post("/api/music/{tid}/delete")
def delete_music(tid: str):
    tracks = load()
    t = next((t for t in tracks if t["id"] == tid), None)
    if not t:
        raise HTTPException(404, "unknown track")
    (DIR / (tid + t["ext"])).unlink(missing_ok=True)
    store([x for x in tracks if x["id"] != tid])
    return {"ok": True}
