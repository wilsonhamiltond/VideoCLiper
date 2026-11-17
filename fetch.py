"""Import a video from a link (YouTube, Twitch, Kick, X and anything else yt-dlp supports).

- Videos and past streams: downloaded whole, or just a time range for long VODs.
- Live streams: recorded for a set number of minutes from now.

Downloads land in downloads/ and are served back to the page at /api/media/<id>, so the browser
streams them like any other video (with seeking) instead of loading them into memory.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

ROOT = Path(__file__).parent
DOWNLOADS = ROOT / "downloads"
MEDIA_ID = re.compile(r"^[0-9a-f]{32}$")
TIME = re.compile(r"^(\d+:){0,2}\d+(\.\d+)?$")  # 90, 1:30, 1:02:30
MAX_RECORD_MIN = 180

# Default quality cap when the page doesn't pick one: 0 = the highest resolution the video has (8K included).
# 8K matters for vertical clips: a 9:16 crop of 4K is only 1215x2160, of 8K it's 2430x4320 (true 4K exports).
DEFAULT_MAX_HEIGHT = int(os.environ.get("IMPORT_MAX_HEIGHT", "0"))

# YouTube login cookies for yt-dlp: a Netscape-format cookies.txt (exported from a browser), or a browser
# name to read them from directly (e.g. "firefox"; Chrome/Edge on Windows lock theirs while open).
COOKIES_FILE = Path(os.environ.get("YTDLP_COOKIES") or ROOT / "cookies.txt")
COOKIES_FROM_BROWSER = os.environ.get("YTDLP_COOKIES_FROM_BROWSER", "").strip()


def format_args(max_height: int) -> list[str]:
    """yt-dlp format choice: highest resolution first, never traded away for codec or HDR.

    Within the same resolution: higher fps, then SDR over HDR (HDR looks washed out in browsers and
    exports), then H.264 (fastest to decode and export), then the next-easiest codec. So 1080p and below
    stay H.264, 4K is VP9, and 8K (YouTube offers it only as AV1) is AV1.
    """
    res = f"res:{max_height}" if max_height else "res"
    return ["-f", "bv*+ba/b", "-S", f"{res},fps,hdr:sdr,vcodec:h264,acodec:m4a"]


def probe_video(path: Path) -> dict:
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,codec_name,color_transfer",
           "-of", "json", str(path)]
    try:
        s = json.loads(subprocess.run(cmd, capture_output=True, text=True).stdout)["streams"][0]
    except (ValueError, KeyError, IndexError):
        return {}
    # PQ (smpte2084) and HLG are the HDR transfer curves.
    return {"width": s.get("width"), "height": s.get("height"), "codec": s.get("codec_name"),
            "hdr": s.get("color_transfer") in ("smpte2084", "arib-std-b67")}

router = APIRouter()
fetch_jobs: dict[str, dict] = {}
procs: dict[str, subprocess.Popen] = {}  # running yt-dlp / ffmpeg per job, for Stop


def ytdlp_base() -> list[str]:
    cmd = [sys.executable, "-m", "yt_dlp", "--no-playlist", "--no-warnings", "--encoding", "utf-8"]
    # YouTube needs a JavaScript runtime to unlock most formats. yt-dlp looks for Deno by default;
    # Node is far more common on Windows, so offer it too.
    runtimes = [r for r in ("deno", "node") if shutil.which(r)]
    if runtimes:
        cmd += ["--js-runtimes", ",".join(runtimes)]
    # Signed-in cookies get past YouTube's "confirm you're not a bot" wall.
    if COOKIES_FILE.exists():
        cmd += ["--cookies", str(COOKIES_FILE)]
    elif COOKIES_FROM_BROWSER:
        cmd += ["--cookies-from-browser", COOKIES_FROM_BROWSER]
    return cmd


def cookies_problem() -> str | None:
    """Why cookies.txt can't sign in to YouTube, or None if it looks like a signed-in export."""
    if not COOKIES_FILE.exists():
        return None
    names = set()
    for line in COOKIES_FILE.read_text(errors="replace").splitlines():
        parts = line.removeprefix("#HttpOnly_").split("\t")
        if len(parts) >= 7 and "youtube.com" in parts[0]:
            names.add(parts[5])
    if not names:
        return "cookies.txt has no youtube.com cookies"
    # A signed-in youtube.com session always has LOGIN_INFO and the first-party SID cookies.
    missing = [n for n in ("LOGIN_INFO", "SAPISID", "__Secure-1PSID") if n not in names]
    if missing and "__Secure-3PSID" in names:
        # Only the third-party copies came along: exported from a Google tab, not a youtube.com tab.
        return ("cookies.txt was exported from a Google page rather than youtube.com (it has only YouTube's "
                f"third-party cookies and lacks {', '.join(missing)}); export it while the youtube.com tab is open")
    if missing:
        return f"cookies.txt isn't a signed-in YouTube session (it lacks {', '.join(missing)})"
    return None


COOKIE_HOWTO = ("Export it again while signed in to youtube.com: in a private window, sign in, open youtube.com, "
                "export the cookies with a cookies.txt extension, then close that window right away so they aren't "
                "rotated. (Chrome and Edge block yt-dlp from reading their cookies, which loses the login ones.)")


def ytdlp_error(stderr: str) -> str:
    lines = [l for l in stderr.splitlines() if l.startswith("ERROR:")]
    msg = (lines[-1][6:] if lines else stderr.strip()[-300:]).strip() or "yt-dlp failed"
    # yt-dlp's own text is a wall of flags and URLs; say what it means for the user instead.
    if "confirm your age" in msg or "age-restricted" in msg or "inappropriate for some users" in msg:
        problem = cookies_problem()
        if problem:
            return f"This video is age-restricted, so YouTube needs a signed-in account, but {problem}. {COOKIE_HOWTO}"
        if COOKIES_FILE.exists() or COOKIES_FROM_BROWSER:
            return ("This video is age-restricted and YouTube didn't accept your cookies. They may have expired, or the "
                    "account isn't age-verified (check by opening the video on youtube.com). " + COOKIE_HOWTO)
        return ("This video is age-restricted, so YouTube needs a signed-in, age-verified account. Put that account's "
                "cookies in cookies.txt in the app folder. " + COOKIE_HOWTO + " Or download the video yourself and drop the file in.")
    if "Sign in to confirm" in msg or "not a bot" in msg:
        problem = cookies_problem()
        if problem:
            return f"YouTube wants this computer to sign in to confirm it's not a bot, but {problem}. {COOKIE_HOWTO}"
        if COOKIES_FILE.exists() or COOKIES_FROM_BROWSER:
            return ("YouTube still asks to confirm it's not a bot even with your cookies. They may have expired: "
                    "export a fresh cookies.txt (or sign in again in the browser) and retry.")
        return ("YouTube wants this computer to sign in to confirm it's not a bot. Fix: export your YouTube cookies "
                "to cookies.txt in the app folder (or set YTDLP_COOKIES_FROM_BROWSER=firefox) and restart the app. "
                "It also often clears by itself within an hour, or download the video yourself and drop the file in.")
    return msg


def to_seconds(t: str) -> float:
    s = 0.0
    for part in t.split(":"):
        s = s * 60 + float(part)
    return s


def safe_name(title: str) -> str:
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', " ", title).strip(" .")
    return (re.sub(r"\s+", " ", name)[:80] or "video") + ".mp4"


class Cancelled(Exception):
    pass


def kill_tree(proc: subprocess.Popen):
    """Kill a process and its children (yt-dlp runs ffmpeg for merging and live streams)."""
    if proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True)
    else:
        proc.kill()


def live_formats(meta: dict, max_height: int) -> list[dict]:
    """Streams to record: the best picture (up to max_height, 0 = no cap), plus a separate sound stream if the picture has none.

    YouTube live serves picture and sound as separate HLS streams; Twitch and Kick serve them together.
    """
    fmts = [f for f in meta.get("formats") or [] if f.get("url") and f.get("protocol", "").startswith(("m3u8", "http"))]
    rank = lambda f: ((f.get("height") or 0), f.get("fps") or 0, f.get("tbr") or 0)
    video = max((f for f in fmts if f.get("vcodec") != "none" and f.get("height") and (not max_height or f["height"] <= max_height)),
                key=rank, default=None)
    if not video:
        return []
    if video.get("acodec") not in (None, "none"):
        return [video]
    audio = max((f for f in fmts if f.get("vcodec") == "none" and f.get("acodec") != "none"),
                key=lambda f: (f.get("abr") or f.get("tbr") or 0), default=None)
    return [video, audio] if audio else [video]


def record_live(job_id: str, meta: dict, out: Path, limit: float, max_height: int) -> bool:
    """Record a live stream with our own ffmpeg, so Stop can end it early and still save a playable file.

    Returns False if no recordable stream was found; yt-dlp then records it instead.
    """
    streams = live_formats(meta, max_height)
    if not streams:
        return False
    job = fetch_jobs[job_id]
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-nostats"]
    for f in streams:
        headers = {**(meta.get("http_headers") or {}), **(f.get("http_headers") or {})}
        if headers:
            cmd += ["-headers", "".join(f"{k}: {v}\r\n" for k, v in headers.items())]
        cmd += ["-i", f["url"]]
    audio_map = ["-map", "1:a:0"] if len(streams) == 2 else ["-map", "0:a:0?"]
    cmd += ["-t", f"{limit:g}", "-map", "0:v:0", *audio_map, "-c", "copy", "-movflags", "+faststart", str(out)]
    job["clock"] = (time.time(), limit)
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    job["recorder"] = True
    procs[job_id] = proc
    if job.get("stopping"):  # Stop pressed while the link was still being read
        stop_recorder(proc)
    err = proc.stderr.read().decode("utf-8", "replace")
    proc.wait()
    job.pop("clock", None)
    if not out.exists() or out.stat().st_size < 1024:
        if job.get("stopping"):
            raise Cancelled()
        raise RuntimeError("ffmpeg couldn't record the stream: " + (err.strip()[-300:] or f"exit code {proc.returncode}"))
    return True


def stop_recorder(proc: subprocess.Popen):
    # "q" is ffmpeg's own quit key: it stops reading and finishes the MP4 properly, unlike a kill.
    try:
        proc.stdin.write(b"q")
        proc.stdin.flush()
    except OSError:
        pass


def finish(job: dict, media: str, out: Path, title: str, live: bool):
    job.update(status="done", progress=1.0, result={
        "media": media, "name": safe_name(title), "size": out.stat().st_size, "title": title, "live": live, **probe_video(out),
    })


def run_fetch(job_id: str, url: str, start: str, end: str, record_min: float, max_height: int):
    job = fetch_jobs[job_id]
    media = uuid.uuid4().hex
    out = DOWNLOADS / f"{media}.mp4"
    try:
        DOWNLOADS.mkdir(exist_ok=True)
        info = subprocess.run(ytdlp_base() + ["-J", "--", url], capture_output=True, text=True, encoding="utf-8", errors="replace")
        if job.get("stopping"):
            raise Cancelled()
        if info.returncode != 0:
            raise RuntimeError(ytdlp_error(info.stderr))
        meta = json.loads(info.stdout)
        if meta.get("_type") == "playlist":
            raise RuntimeError("That's a playlist or channel link. Paste the link of one video.")
        live = bool(meta.get("is_live"))
        title = meta.get("title") or "video"
        job.update(title=title, live=live)

        if live:
            job.update(status="recording", progress=0.0)
            if record_live(job_id, meta, out, record_min * 60, max_height):
                return finish(job, media, out, title, True)

        # YouTube throttles ffmpeg's ranged reads to ~1.5x real time, while its own downloader pulls the
        # whole video at ~20x. So for a YouTube range, fetch everything and cut the range out locally.
        cut_locally = (start or end) and not live and meta.get("extractor_key") == "Youtube"
        dest = DOWNLOADS / f"{media}.full.mp4" if cut_locally else out
        cmd = ytdlp_base() + format_args(max_height) + [
            "--merge-output-format", "mp4", "-o", str(dest), "-N", "4",
            "--no-part", "--newline", "--progress",
            "--progress-template", "download:PROGRESS %(progress.downloaded_bytes)s %(progress.total_bytes)s %(progress.total_bytes_estimate)s",
        ]
        limit = 0.0
        if live:
            limit = record_min * 60
            # Record from now; ffmpeg stops by itself after `limit` seconds and closes the file cleanly.
            cmd += ["--downloader", "ffmpeg", "--downloader-args", f"ffmpeg_i:-t {limit:g}"]
        elif start or end:
            dur = meta.get("duration") or 0
            limit = (to_seconds(end) if end else dur) - (to_seconds(start) if start else 0)
            if dur and to_seconds(start or "0") >= dur:
                raise RuntimeError(f"The video is only {int(dur // 60)}:{int(dur % 60):02d} long; “From” is past the end.")
            if not cut_locally:
                cmd += ["--download-sections", f"*{start or '0'}-{end or 'inf'}"]
        cmd += ["--", url]

        job.update(status="recording" if live else "downloading", progress=0.0)
        if limit and not cut_locally:
            # ffmpeg (live and time-range downloads) reports no byte progress; time it instead.
            job["clock"] = (time.time(), limit)
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
        procs[job_id] = proc
        if job.get("stopping"):
            kill_tree(proc)
        err_lines: list[str] = []
        threading.Thread(target=lambda: err_lines.extend(proc.stderr), daemon=True).start()
        first, last_total = True, 0
        for line in proc.stdout:
            if not line.startswith("PROGRESS "):
                continue
            done, total, est = (int(float(v)) if v not in ("NA", "None") else 0 for v in line.split()[1:4])
            total = total or est
            if not total:
                continue
            if last_total and total != last_total and done < total * 0.05:
                first = False  # video finished, audio started (they're downloaded separately)
            last_total = total
            job.pop("clock", None)
            # Video is ~90% of the bytes; give the audio pass the last stretch of the bar.
            frac = done / total
            job["progress"] = round(min(0.99, 0.9 * frac if first else 0.9 + 0.1 * frac), 3)
        proc.wait()
        job.pop("clock", None)
        if job.get("stopping"):
            raise Cancelled()
        if proc.returncode != 0 or not dest.exists() or dest.stat().st_size == 0:
            raise RuntimeError(ytdlp_error("".join(err_lines)))
        if cut_locally:
            job.update(status="cutting", progress=0.99)
            # Stream copy: instant, starts on the keyframe at or just before `start`.
            cut = ["ffmpeg", "-y", "-loglevel", "error"] + (["-ss", start] if start else []) + ["-i", str(dest)]
            cut += (["-t", f"{limit:g}"] if end else []) + ["-c", "copy", "-avoid_negative_ts", "make_zero", "-movflags", "+faststart", str(out)]
            res = subprocess.run(cut, capture_output=True, text=True, encoding="utf-8", errors="replace")
            dest.unlink(missing_ok=True)
            if res.returncode != 0:
                raise RuntimeError("ffmpeg couldn't cut the range: " + res.stderr.strip()[-300:])
        finish(job, media, out, title, live)
    except Exception as e:
        proc = procs.get(job_id)
        if proc:
            kill_tree(proc)
            proc.wait()  # Windows can't delete files a dying ffmpeg still holds
        out.unlink(missing_ok=True)
        (DOWNLOADS / f"{media}.full.mp4").unlink(missing_ok=True)
        job.update(status="error", error="Cancelled." if isinstance(e, Cancelled) else str(e), cancelled=isinstance(e, Cancelled))
    finally:
        procs.pop(job_id, None)


@router.post("/api/fetch")
def start_fetch(body: dict):
    url = str(body.get("url", "")).strip()
    if not re.match(r"^https?://\S+$", url):
        raise HTTPException(400, "Paste a full link starting with https://")
    if len(re.findall(r"https?://", url)) > 1:
        raise HTTPException(400, "That box has more than one link pasted together. Clear it and paste one link.")
    start, end = str(body.get("start") or "").strip(), str(body.get("end") or "").strip()
    for t in (start, end):
        if t and not TIME.match(t):
            raise HTTPException(400, f"'{t}' isn't a time. Use seconds or h:mm:ss, like 1:02:30.")
    if start and end and to_seconds(end) <= to_seconds(start):
        raise HTTPException(400, "The end time must be after the start time.")
    try:
        record_min = float(body.get("record_min") or 10)
    except ValueError:
        raise HTTPException(400, "record_min must be a number")
    record_min = max(0.5, min(MAX_RECORD_MIN, record_min))
    try:
        max_height = int(body.get("max_height", DEFAULT_MAX_HEIGHT) or 0)  # 0 = best available
    except ValueError:
        raise HTTPException(400, "max_height must be a number")
    job_id = uuid.uuid4().hex
    fetch_jobs[job_id] = {"status": "reading link", "progress": 0.0}
    threading.Thread(target=run_fetch, args=(job_id, url, start, end, record_min, max(0, max_height)), daemon=True).start()
    return {"job": job_id}


@router.post("/api/fetch/{job_id}/stop")
def stop_fetch(job_id: str):
    """Live recording: stop now and keep what's recorded. Download: cancel and delete the partial file."""
    job = fetch_jobs.get(job_id)
    if not job or job["status"] in ("done", "error"):
        raise HTTPException(404, "Nothing to stop.")
    job["stopping"] = True
    proc = procs.get(job_id)
    if proc and job.get("recorder"):
        job["status"] = "saving"
        stop_recorder(proc)
    elif proc:
        kill_tree(proc)
    return {"ok": True}


@router.get("/api/fetch/{job_id}")
def fetch_status(job_id: str):
    job = fetch_jobs.get(job_id)
    if not job:
        raise HTTPException(404, "unknown job")
    if job["status"] in ("done", "error"):
        fetch_jobs.pop(job_id)
    out = {k: v for k, v in job.items() if k != "clock"}
    clock = job.get("clock")  # read once: the download thread may drop it any moment
    if clock:
        t0, limit = clock
        out["progress"] = round(min(0.99, (time.time() - t0) / limit), 3)
        out["elapsed"] = round(time.time() - t0)
        out["limit"] = round(limit)
    return out


def find_media(media: str) -> Path | None:
    """downloads/<id>.mp4 for imported links; uploads keep their own extension (.mov, .webm…)."""
    if not MEDIA_ID.match(media or ""):
        return None
    path = DOWNLOADS / f"{media}.mp4"
    if path.exists():
        return path
    return next((p for p in DOWNLOADS.glob(f"{media}.*") if p.stem == media and p.suffix != ".part"), None)


def media_exists(media: str) -> bool:
    return find_media(media) is not None


def media_path(media: str) -> Path:
    path = find_media(media)
    if not path:
        raise HTTPException(404, "That video is gone from the downloads folder. Import or upload it again.")
    return path


def delete_media(media: str):
    path = find_media(media)
    if path:
        path.unlink(missing_ok=True)
        path.with_suffix(".srt").unlink(missing_ok=True)  # subtitles of a video made from a guide


MEDIA_TYPES = {".mp4": "video/mp4", ".m4v": "video/mp4", ".mov": "video/mp4", ".webm": "video/webm",
               ".mkv": "video/x-matroska", ".mp3": "audio/mpeg", ".m4a": "audio/mp4", ".wav": "audio/wav"}


@router.get("/api/media/{media}")
def get_media(media: str):
    path = media_path(media)
    return FileResponse(path, media_type=MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream"))
