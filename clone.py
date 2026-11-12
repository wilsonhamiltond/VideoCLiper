"""Cloned voices: record a sample of your own voice, then have the reader speak in it.

Samples are stored in voices/ (24 kHz mono WAV + voices/index.json). Speech is generated locally by
voice_server.py, which runs in .venv-voice and is started here on first use.
"""
import json
import os
import re
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

router = APIRouter()
ROOT = Path(__file__).parent
VOICES = ROOT / "voices"
INDEX = VOICES / "index.json"
WORKER_PY = ROOT / ".venv-voice" / "Scripts" / "python.exe"
WORKER_URL = "http://127.0.0.1:8765"
_worker: subprocess.Popen | None = None
_worker_lock = threading.Lock()


def load_index() -> list[dict]:
    try:
        return json.loads(INDEX.read_text(encoding="utf-8"))
    except Exception:
        return []


def save_index(items: list[dict]):
    VOICES.mkdir(exist_ok=True)
    INDEX.write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")


def worker_call(path: str, payload: dict | None = None, timeout: float = 3):
    req = urllib.request.Request(WORKER_URL + path, data=json.dumps(payload).encode() if payload is not None else None,
                                 headers={"Content-Type": "application/json", "X-Clip-Studio": "1"},
                                 method="POST" if payload is not None else "GET")
    return urllib.request.urlopen(req, timeout=timeout)


def worker_health() -> dict | None:
    try:
        with worker_call("/health", timeout=1) as r:
            return json.loads(r.read())
    except Exception:
        return None


def ensure_worker() -> dict:
    """Start voice_server.py if it isn't running, and ask it to load the model."""
    global _worker
    if not WORKER_PY.exists():
        raise HTTPException(503, "Voice cloning isn't installed (missing .venv-voice). See README → Clone your voice.")
    with _worker_lock:
        health = worker_health()
        if health is None:
            if _worker is None or _worker.poll() is not None:
                log = open(VOICES.parent / "voice_server.log", "ab")
                env = {**os.environ, "TQDM_DISABLE": "1"}   # no per-token progress bars in the log
                _worker = subprocess.Popen([str(WORKER_PY), str(ROOT / "voice_server.py")], cwd=ROOT, env=env,
                                           stdout=log, stderr=subprocess.STDOUT,
                                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            for _ in range(120):          # importing torch takes a while the first time
                time.sleep(0.5)
                if (health := worker_health()) is not None:
                    break
                if _worker.poll() is not None:
                    raise HTTPException(502, "The voice-cloning worker exited. See voice_server.log.")
            else:
                raise HTTPException(504, "The voice-cloning worker didn't start in time.")
        if health.get("state") in ("idle", "error"):
            with worker_call("/load", {}) as r:
                health = json.loads(r.read())
        return health


@router.get("/api/clone/status")
def status():
    return {"installed": WORKER_PY.exists(), "worker": worker_health(), "voices": len(load_index())}


@router.post("/api/clone/start")
def start():
    return ensure_worker()


@router.get("/api/clone/voices")
def voices():
    return {"voices": load_index()}


@router.post("/api/clone/voices")
async def add_voice(request: Request, name: str = "My voice", language: str = "en"):
    """The body is a recording (any format ffmpeg reads). Keeps up to 30 s, trims leading silence, evens loudness."""
    raw = await request.body()
    if len(raw) < 1000:
        raise HTTPException(400, "The recording is empty.")
    VOICES.mkdir(exist_ok=True)
    vid = uuid.uuid4().hex[:10]
    dst = VOICES / f"{vid}.wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f:
        f.write(raw)
    try:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", f.name, "-ac", "1", "-ar", "24000",
                        "-af", "silenceremove=start_periods=1:start_threshold=-45dB,loudnorm=I=-18:TP=-2",
                        "-t", "30", str(dst)], check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        raise HTTPException(400, "Couldn't read that recording: " + e.stderr.decode(errors="replace")[-300:])
    finally:
        Path(f.name).unlink(missing_ok=True)
    seconds = (dst.stat().st_size - 44) / (24000 * 2)
    if seconds < 5:
        dst.unlink(missing_ok=True)
        raise HTTPException(400, f"Only {seconds:.1f} s of voice: record at least 8–10 seconds of talking.")
    items = load_index()
    item = {"id": vid, "name": re.sub(r"\s+", " ", name).strip()[:40] or "My voice",
            "language": language[:2].lower(), "seconds": round(seconds, 1)}
    items.append(item)
    save_index(items)
    return item


@router.delete("/api/clone/voices/{vid}")
def delete_voice(vid: str):
    items = load_index()
    if not any(v["id"] == vid for v in items):
        raise HTTPException(404, "Unknown voice.")
    save_index([v for v in items if v["id"] != vid])
    (VOICES / f"{Path(vid).name}.wav").unlink(missing_ok=True)
    return {"ok": True}


@router.get("/api/clone/sample/{vid}")
def sample(vid: str):
    from fastapi.responses import FileResponse

    path = VOICES / f"{Path(vid).name}.wav"
    if not path.is_file():
        raise HTTPException(404, "Unknown voice.")
    return FileResponse(path, media_type="audio/wav")


@router.get("/api/clone/speak")
def speak(text: str, voice: str, language: str = "", emotion: float = 0.5):
    item = next((v for v in load_index() if v["id"] == voice), None)
    if not item:
        raise HTTPException(404, "Unknown cloned voice.")
    health = ensure_worker()
    for _ in range(1200):                # first use downloads (~3 GB) and loads the model
        if health.get("state") == "ready":
            break
        if health.get("state") == "error":
            raise HTTPException(502, "Voice model failed to load: " + health.get("error", ""))
        time.sleep(0.5)
        health = worker_health() or {}
    try:
        upstream = worker_call("/speak", {"text": text, "sample": str(VOICES / f"{voice}.wav"),
                                          "language": language or item["language"], "sample_language": item["language"],
                                          "emotion": emotion}, timeout=300)
    except urllib.error.HTTPError as e:
        raise HTTPException(e.code, json.loads(e.read() or b"{}").get("detail", "voice worker error"))

    def relay():
        with upstream:
            while chunk := upstream.read1(16384):
                yield chunk

    return StreamingResponse(relay(), media_type="audio/mpeg", headers={"Cache-Control": "no-store"})
