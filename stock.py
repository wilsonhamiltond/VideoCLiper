"""Make a short from stock footage: search Pexels for a topic ("motivation"), download a few clips,
crop each to 9:16 and merge them with crossfades into one vertical video.

The result lands in downloads/ like an imported link, so the page opens it the same way and the
voice-over, captions, background music and export all work on it as usual.

Needs a free Pexels API key (https://www.pexels.com/api/), saved in data/stock.json or set as
PEXELS_API_KEY. Pexels videos are free to use and modify; credit is appreciated but not required.
"""

import json
import os
import random
import shutil
import subprocess
import threading
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException

import fetch

ROOT = Path(__file__).parent
CONFIG = ROOT / "data" / "stock.json"
UA = {"User-Agent": "ClipStudio/1.0"}
W, H, FPS = 1080, 1920, 30
FADE = 0.5  # crossfade between clips, seconds

router = APIRouter()
stock_jobs: dict[str, dict] = {}


def load_config() -> dict:
    try:
        return json.loads(CONFIG.read_text("utf-8"))
    except (OSError, ValueError):
        return {}


def pexels_key() -> str:
    return os.environ.get("PEXELS_API_KEY") or load_config().get("pexels_key", "")


def pexels(path: str, params: dict, key: str) -> dict:
    url = "https://api.pexels.com" + path + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={**UA, "Authorization": key})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            raise RuntimeError("Pexels refused the API key. Check it at pexels.com/api and save it again.")
        if e.code == 429:
            raise RuntimeError("Pexels rate limit reached (200 searches an hour). Try again later.")
        raise RuntimeError(f"Pexels search failed: HTTP {e.code}")
    except OSError as e:
        raise RuntimeError(f"Couldn't reach Pexels: {e}")


def best_file(video: dict) -> dict | None:
    """The smallest MP4 that still fills a 1080x1920 frame, or the largest one there is."""
    files = [f for f in video.get("video_files") or [] if f.get("file_type") == "video/mp4" and f.get("width") and f.get("height")]
    if not files:
        return None
    big = [f for f in files if min(f["width"], f["height"]) >= W]
    return min(big, key=lambda f: f["width"] * f["height"]) if big else max(files, key=lambda f: f["width"] * f["height"])


def download(url: str, dst: Path, job: dict):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r, open(dst, "wb") as out:
        while chunk := r.read(1 << 20):
            if job.get("cancel"):
                raise RuntimeError("cancelled")
            out.write(chunk)


def ffmpeg(args: list[str]):
    p = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError("ffmpeg failed: " + (p.stderr.strip().splitlines() or ["unknown error"])[-1])


def run_stock(job_id: str, query: str, seconds: float, count: int, key: str):
    job = stock_jobs[job_id]
    media = uuid.uuid4().hex
    work = fetch.DOWNLOADS / f"stock-parts-{media}"
    try:
        fetch.DOWNLOADS.mkdir(exist_ok=True)
        work.mkdir()
        # Each clip runs a bit longer than its share so the crossfades don't eat into the total.
        seg = (seconds + (count - 1) * FADE) / count
        job.update(status="searching", progress=0.03)
        found = []
        for orientation in ("portrait", "landscape"):  # portrait first; landscape is center-cropped if needed
            res = pexels("/videos/search", {"query": query, "orientation": orientation, "per_page": 60, "size": "medium"}, key)
            found += [v for v in res.get("videos") or [] if (v.get("duration") or 0) >= seg + 0.5 and best_file(v)]
            if len(found) >= count * 2:
                break
        if not found:
            raise RuntimeError(f"No stock videos found for “{query}”. Try another word, or English keywords.")
        # A random pick from the results, at most one clip per creator when possible, so each run looks different.
        random.shuffle(found)
        picks, users = [], set()
        for v in found:
            if (v.get("user") or {}).get("id") not in users:
                picks.append(v)
                users.add((v.get("user") or {}).get("id"))
            if len(picks) == count:
                break
        picks += [v for v in found if v not in picks][: count - len(picks)]
        n = len(picks)
        seg = (seconds + (n - 1) * FADE) / n

        parts = []
        for i, v in enumerate(picks):
            if job.get("cancel"):
                raise RuntimeError("cancelled")
            job.update(status="downloading", progress=0.05 + 0.65 * i / n, step=f"{i + 1} of {n}")
            raw = work / f"raw{i}.mp4"
            download(best_file(v)["link"], raw, job)
            job.update(status="cutting", progress=0.05 + 0.65 * (i + 0.8) / n)
            # Skip the first second (often a slow fade-in), then a random stretch of the rest.
            room = max(0.0, float(v["duration"]) - seg - 0.2)
            start = round(random.uniform(min(1.0, room), room), 2) if room > 0 else 0
            out = work / f"seg{i}.mp4"
            ffmpeg(["-ss", str(start), "-i", str(raw), "-t", f"{seg:.3f}", "-an",
                    "-vf", f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},setsar=1,fps={FPS},format=yuv420p",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "19", str(out)])
            raw.unlink(missing_ok=True)
            parts.append(out)

        job.update(status="merging", progress=0.75)
        total = n * seg - (n - 1) * FADE
        inputs = [a for p in parts for a in ("-i", str(p))]
        chain, last = [], "0:v"
        for k in range(1, n):
            label = f"v{k}"
            chain.append(f"[{last}][{k}:v]xfade=transition=fade:duration={FADE}:offset={k * (seg - FADE):.3f}[{label}]")
            last = label
        chain.append(f"[{last}]fade=t=in:st=0:d=0.3,fade=t=out:st={total - 0.6:.3f}:d=0.6[vout]")
        out = fetch.DOWNLOADS / f"{media}.mp4"
        # A silent sound track: the page's export mixes narration and music into the video's audio.
        ffmpeg([*inputs, "-f", "lavfi", "-t", f"{total:.3f}", "-i", "anullsrc=r=48000:cl=stereo",
                "-filter_complex", ";".join(chain), "-map", "[vout]", "-map", f"{n}:a",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "19", "-c:a", "aac", "-b:a", "128k",
                "-t", f"{total:.3f}", "-movflags", "+faststart", str(out)])

        credits = [{"by": (v.get("user") or {}).get("name", ""), "url": v.get("url", "")} for v in picks]
        title = f"{query.strip().capitalize()} short"
        job.update(status="done", progress=1.0, result={
            "media": media, "name": fetch.safe_name(title), "title": title, "size": out.stat().st_size,
            "width": W, "height": H, "duration": round(total, 2), "clips": n, "query": query, "credits": credits})
    except Exception as e:
        (fetch.DOWNLOADS / f"{media}.mp4").unlink(missing_ok=True)
        job.update(status="error", error="Cancelled." if str(e) == "cancelled" else str(e), cancelled=str(e) == "cancelled")
    finally:
        shutil.rmtree(work, ignore_errors=True)


@router.get("/api/stock/config")
def stock_config():
    return {"has_key": bool(pexels_key()), "from_env": bool(os.environ.get("PEXELS_API_KEY"))}


@router.post("/api/stock/key")
def save_key(body: dict):
    key = str(body.get("key") or "").strip()
    if key:
        pexels("/videos/search", {"query": "nature", "per_page": 1}, key)  # raises with a clear message if it's wrong
    CONFIG.parent.mkdir(exist_ok=True)
    CONFIG.write_text(json.dumps({**load_config(), "pexels_key": key}), "utf-8")
    return {"has_key": bool(key)}


@router.post("/api/stock")
def start_stock(body: dict):
    query = str(body.get("query") or "").strip()[:80]
    if not query:
        raise HTTPException(400, "Type what the video is about, e.g. motivation.")
    key = pexels_key()
    if not key:
        raise HTTPException(400, "Add your free Pexels API key first.")
    try:
        seconds = max(8.0, min(90.0, float(body.get("seconds") or 30)))
        count = max(2, min(12, int(body.get("clips") or 0) or round(seconds / 5)))
    except ValueError:
        raise HTTPException(400, "seconds and clips must be numbers")
    job_id = uuid.uuid4().hex
    stock_jobs[job_id] = {"status": "searching", "progress": 0.0}
    threading.Thread(target=run_stock, args=(job_id, query, seconds, count, key), daemon=True).start()
    return {"job": job_id}


@router.post("/api/stock/{job_id}/stop")
def stop_stock(job_id: str):
    job = stock_jobs.get(job_id)
    if not job or job["status"] in ("done", "error"):
        raise HTTPException(404, "Nothing to stop.")
    job["cancel"] = True
    return {"ok": True}


@router.get("/api/stock/{job_id}")
def stock_status(job_id: str):
    job = stock_jobs.get(job_id)
    if not job:
        raise HTTPException(404, "unknown job")
    if job["status"] in ("done", "error"):
        stock_jobs.pop(job_id)
    return job
