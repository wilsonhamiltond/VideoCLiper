"""Video from a guide: turns a written video guide (sections with "Visual:" and "Narración:") into one
finished, narrated video.

    1. Apertura (0:00 - 0:45)
    * Visual: concert crowds, luxury cars... [clip title](https://studio.youtube.com/video/VIDEO_ID)
    * Narración: "Text the voice reads... (Pausa breve mientras corre el clip) ...more text."

For each section:
  - the narration is read by an Edge neural voice (es-DO by default),
  - the B-roll under the voice is stock footage from Pexels, searched from the "Visual:" line (the local
    Ollama model turns it into English search words when it's running), mixed with muted shots of any
    YouTube clip linked in the section,
  - a parenthesis in the narration, like "(Inserto de su testimonio)", plays the linked clip with its own
    sound. Put a time range in it to pick the moment: "(Inserto 0:12-0:31)". Without one, the first
    `insert` seconds of the clip play.
  - sections whose heading has a dash ("Primer caso: Paramba — El peso...") get a title on screen.

The guide's time ranges are only estimates; the voice sets the real length. The result lands in
downloads/ like an imported link (plus an .srt of the narration), so the page opens it as usual.

Also runs from the command line:  .venv\\Scripts\\python guide.py my-guide.txt [--format 9:16]
"""

import asyncio
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import threading
import time
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

import fetch
import music
import narrate
import notify
import stock

ROOT = Path(__file__).parent
CACHE = fetch.DOWNLOADS / "guide-cache"  # downloaded YouTube clips and stock footage, reused on the next run
FPS = 30
FONT = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "arialbd.ttf"
DEFAULT_VOICE = "es-DO-EmilioNeural"
OLLAMA = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")

router = APIRouter()
guide_jobs: dict[str, dict] = {}

# ---------- the Guides list: every video made from a guide, kept in data/guides.json ----------
# One entry per run: its title, the guide text and options (to make it again), status, and the finished video.
GUIDES_FILE = ROOT / "data" / "guides.json"
_guides_lock = threading.Lock()


def _guides_load() -> list[dict]:
    try:
        return json.loads(GUIDES_FILE.read_text("utf-8"))
    except (OSError, ValueError):
        return []


def _guides_save(items: list[dict]):
    GUIDES_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = GUIDES_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(items, ensure_ascii=False), "utf-8")
    tmp.replace(GUIDES_FILE)


def guides_add(entry: dict):
    with _guides_lock:
        _guides_save([entry] + _guides_load())


def guides_update(gid: str, **fields):
    with _guides_lock:
        items = _guides_load()
        for it in items:
            if it["id"] == gid:
                it.update(fields)
                break
        _guides_save(items)


def _guides_seed():
    """First run: the guide videos already announced in the notifications become the start of the list."""
    if GUIDES_FILE.exists():
        return
    items = []
    for n in notify._load():
        if n.get("kind") != "guide":
            continue
        r = n.get("result") or {}
        items.append({"id": n.get("job") or n["id"], "title": n.get("title") or r.get("title") or "Video",
                      "created": n.get("time"), "finished": n.get("time_done"), "status": n.get("status") or "done",
                      "format": n.get("format"), "sections": n.get("sections") or len(r.get("sections") or []),
                      "error": n.get("body") if n.get("status") == "error" else "", "result": r or None, "nid": n["id"]})
    _guides_save(items)

YT_ID = re.compile(r"(?:studio\.youtube\.com/video/|youtu\.be/|youtube\.com/(?:watch\?(?:.*&)?v=|shorts/|live/|embed/))([\w-]{11})")
MD_LINK = re.compile(r"\[([^\]]*)\]\(((?:https?://|video:)[^)\s]+)\)")
UPLOAD_ID = re.compile(r"^video:([0-9a-f]{32})$")
VIDEO_EXT = {".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi", ".mts", ".m2ts", ".3gp", ".wmv"}
URL = re.compile(r"https?://\S+")
HEADING = re.compile(r"^\s*(?:#+\s*)?(\d{1,2})[.)]\s+(.+?)\s*$")
TIME_RANGE = re.compile(r"\(?\s*(\d{1,2}:\d{2}(?::\d{2})?)\s*[-–—]\s*(\d{1,2}:\d{2}(?::\d{2})?)\s*\)?")
FIELD = re.compile(r"^\s*[*\-•]?\s*(visual|visuales|narraci[oó]n|narration|voz|texto)\s*:\s*(.*)$", re.I)


# ---------- parsing ----------

def to_sec(t: str) -> float:
    s = 0.0
    for p in t.split(":"):
        s = s * 60 + float(p)
    return s


def parse_guide(text: str) -> dict:
    title, sections, cur, field = "", [], None, None
    for raw in text.replace("\r", "").split("\n"):
        line = raw.strip()
        if not line:
            continue
        m = re.match(r"^(?:t[ií]tulo(?: de trabajo)?|title)\s*:\s*(.+)$", line, re.I)
        if m and not sections:
            title = m.group(1).strip()
            continue
        if re.match(r"^(?:duraci[oó]n|duration)", line, re.I) and not sections:
            continue
        h = HEADING.match(line)
        if h and not FIELD.match(line):
            head = h.group(2)
            tr = TIME_RANGE.search(head)
            if tr:
                head = (head[: tr.start()] + head[tr.end():]).strip()
            cur = {"n": int(h.group(1)), "heading": head.strip(" :"), "visual": "", "narration": "",
                   "planned": [to_sec(tr.group(1)), to_sec(tr.group(2))] if tr else None}
            sections.append(cur)
            field = None
            continue
        f = FIELD.match(line)
        if f and cur is not None:
            field = "visual" if f.group(1).lower().startswith("vis") else "narration"
            cur[field] = (cur[field] + " " + f.group(2)).strip()
        elif cur is not None and field:
            cur[field] = (cur[field] + " " + line.lstrip("*-• ")).strip()
    if not sections:
        raise ValueError("No sections found. Number each section (\"1. Apertura\") and give it a "
                         "\"Visual:\" and a \"Narración:\" line.")
    for s in sections:
        links = [(t, u) for t, u in MD_LINK.findall(s["visual"])] or [("", u) for u in URL.findall(s["visual"])]
        s["clips"] = []
        for label, url in links:
            # Your own uploads ([name](video:<id>)), YouTube (studio links included), or any other page yt-dlp can
            # download from (TikTok, Instagram, Facebook, X...).
            up, yt = UPLOAD_ID.match(url), YT_ID.search(url)
            if up:
                clip = {"id": "up-" + up.group(1), "kind": "upload", "label": label.strip(), "url": url}
            elif yt:
                clip = {"id": "yt-" + yt.group(1), "kind": "link", "label": label.strip(), "url": f"https://www.youtube.com/watch?v={yt.group(1)}"}
            elif url.startswith("http"):
                clip = {"id": "web-" + hashlib.sha1(url.encode()).hexdigest()[:16], "kind": "link", "label": label.strip(), "url": url}
            else:
                continue
            if clip["id"] not in [c["id"] for c in s["clips"]]:
                s["clips"].append(clip)
        s["visual_text"] = re.sub(r"\s+", " ", URL.sub("", MD_LINK.sub(lambda m: m.group(1), s["visual"]))).strip()
        s["pieces"] = split_narration(s["narration"])
        # "Primer caso: Paramba — El peso de no olvidar" -> on-screen title "Paramba — El peso de no olvidar"
        s["overlay"] = re.sub(r"^[^:]*caso\s*:\s*", "", s["heading"], flags=re.I) if re.search(r"[—–]| - ", s["heading"]) else ""
        if not any(p["type"] == "say" for p in s["pieces"]):
            raise ValueError(f"Section {s['n']} ({s['heading']}) has no narration.")
    return {"title": title or sections[0]["heading"], "sections": sections}


def split_narration(text: str) -> list[dict]:
    text = text.strip().strip('"“”«»').strip()
    out, pos = [], 0
    for m in re.finditer(r"\(([^()]*)\)", text):
        before = clean_say(text[pos: m.start()])
        if before:
            out.append({"type": "say", "text": before})
        tr = TIME_RANGE.search(m.group(1))
        # Which clip of the section: "(Video 2 0:05-0:20)" / "(clip 2)"; otherwise they're used in order.
        which = re.search(r"\b(?:video|clip|inserto|insert)\s*#?(\d)\b(?!:)", m.group(1), re.I)
        out.append({"type": "insert", "note": m.group(1).strip(),
                    "range": [to_sec(tr.group(1)), to_sec(tr.group(2))] if tr else None,
                    "clip": int(which.group(1)) - 1 if which else None,
                    "whole": bool(re.search(r"complet|entero|whole|full", m.group(1), re.I))})
        pos = m.end()
    rest = clean_say(text[pos:])
    if rest:
        out.append({"type": "say", "text": rest})
    return out


def clean_say(t: str) -> str:
    return re.sub(r"\s+", " ", t).strip().strip('"“”«»').strip()


# ---------- helpers ----------

def ffmpeg(args: list[str], cwd: Path | None = None):
    p = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", cwd=cwd)
    if p.returncode:
        raise RuntimeError("ffmpeg failed: " + (p.stderr.strip().splitlines() or ["unknown error"])[-1])


def duration(path: Path) -> float:
    p = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                       capture_output=True, text=True)
    try:
        return float(p.stdout.strip())
    except ValueError:
        return 0.0


def has_audio(path: Path) -> bool:
    p = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index", "-of", "csv=p=0", str(path)],
                       capture_output=True, text=True)
    return bool(p.stdout.strip())


def check(job: dict):
    if job.get("cancel"):
        raise RuntimeError("cancelled")


def speak(text: str, voice: str, rate: int, dst: Path) -> list[dict]:
    """Writes the speech to dst (mp3) and returns word timings."""
    import edge_tts

    async def run():
        audio, words = bytearray(), []
        com = edge_tts.Communicate(text, voice, rate=f"{rate:+d}%", boundary="WordBoundary")
        async for chunk in com.stream():
            if chunk["type"] == "audio":
                audio += chunk["data"]
            elif chunk["type"] == "WordBoundary":
                s = chunk["offset"] / 1e7
                words.append({"w": chunk["text"], "s": s, "e": s + chunk["duration"] / 1e7})
        return bytes(audio), words

    last = None
    for attempt in range(4):
        try:
            audio, words = asyncio.run(run())
            if audio:
                dst.write_bytes(audio)
                return narrate.attach_punctuation(text, words)
        except Exception as e:  # the service drops requests in bursts; the same text works a moment later
            last = e
        import time
        time.sleep(1 + 2 * attempt)
    raise RuntimeError(f"Text-to-speech failed (needs internet; try another voice): {last}")


def ollama_list(prompt: str, temperature: float = 0.4, n: int = 6) -> list[str]:
    """A list of short strings from the local Ollama model; [] if it isn't running."""
    try:
        with urllib.request.urlopen(OLLAMA + "/api/tags", timeout=3) as r:
            names = [m["name"] for m in json.load(r)["models"]]
    except Exception:
        return []
    model = os.environ.get("OLLAMA_MODEL") or next((n for p in ("qwen", "gemma", "llama", "mistral") for n in names if p in n), names[0] if names else "")
    if not model:
        return []
    schema = {"type": "object", "properties": {"queries": {"type": "array", "items": {"type": "string"}}}, "required": ["queries"]}
    body = json.dumps({"model": model, "stream": False, "format": schema, "options": {"temperature": temperature},
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    try:
        req = urllib.request.Request(OLLAMA + "/api/chat", body, {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=180) as r:
            qs = json.loads(json.load(r)["message"]["content"]).get("queries") or []
    except Exception:
        return []
    out = []
    for q in qs:
        q = re.sub(r"\s+", " ", str(q)).strip(" \"'")[:80]
        if q and q.lower() not in [x.lower() for x in out]:
            out.append(q)
    return out[:n]


def ollama_queries(section: dict, orientation: str) -> list[str]:
    """English stock-footage search words for a section."""
    return ollama_list(
        "You pick stock footage for a documentary video. Write 5 different Pexels search queries in English, "
        "2-4 words each, for generic B-roll that fits this scene. No people's names, no brands, nothing that "
        "needs a specific person. Mood and places are good (e.g. 'empty recording studio', 'city highway night').\n\n"
        f"Scene visual: {section['visual_text'][:600]}\nNarration: {section['narration'][:900]}")


def youtube_queries(section: dict, title: str) -> list[str]:
    """YouTube searches that find real footage of what a section talks about (interviews, shows, the people named)."""
    qs = ollama_list(
        "You research footage for a documentary video on YouTube. Write 4 different YouTube search queries, "
        "3-7 words each, in the same language as the scene, that would find real videos of what this scene "
        "shows: the people named (full artist names), the specific interview, show, radio moment or event "
        "described, or performances. Mix specific queries (person + moment, e.g. 'Paramba entrevista llorando "
        "infancia') with broader ones (person + 'en vivo' / 'entrevista'). No hashtags, no quotes.\n\n"
        f"Video: {title[:200]}\nSection: {section['heading'][:200]}\nScene visual: {section['visual_text'][:600]}\n"
        f"Narration: {section['narration'][:900]}", temperature=0.5, n=4)
    if qs:
        return qs
    # Without the AI: the section heading (names usually live there) and the start of the Visual line.
    head = re.sub(r"^[^:]*caso\s*:\s*", "", section["heading"], flags=re.I)
    head = re.split(r"\s+[—–-]\s+", head)[0]
    vis = " ".join(re.sub(r"[^\w\s]", " ", section["visual_text"]).split()[:7])
    return [q for q in dict.fromkeys([f"{head} entrevista", f"{head} en vivo", vis]) if q.strip()]


def youtube_search(query: str, n: int = 8) -> list[dict]:
    """YouTube results for a query through yt-dlp (no API key): id, title, channel, duration, views."""
    cmd = [*fetch.ytdlp_base(), "--flat-playlist", "--dump-json", f"ytsearch{n}:{query}"]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
    except subprocess.TimeoutExpired:
        return []
    out = []
    for line in p.stdout.splitlines():
        try:
            v = json.loads(line)
        except ValueError:
            continue
        vid = v.get("id")
        if not vid or len(vid) != 11:
            continue
        url = v.get("url") or ""
        out.append({"id": vid, "title": v.get("title") or "", "channel": v.get("channel") or v.get("uploader") or "",
                    "duration": v.get("duration") or 0, "views": v.get("view_count") or 0,
                    "short": "/shorts/" in url, "url": f"https://www.youtube.com/watch?v={vid}",
                    "thumb": f"https://i.ytimg.com/vi/{vid}/mqdefault.jpg", "query": query})
    return out


def fallback_queries(section: dict) -> list[str]:
    parts = [p.strip(" .") for p in re.split(r"[,;.]| y | o ", section["visual_text"]) if len(p.strip()) > 3]
    return [" ".join(p.split()[:5]) for p in parts[:5]] or ["city night", "concert crowd"]


# ---------- the job ----------

class Builder:
    def __init__(self, job: dict, guide: dict, opts: dict):
        self.job, self.guide, self.o = job, guide, opts
        self.W, self.H = (1080, 1920) if opts["format"] == "9:16" else (1920, 1080)
        self.orientation = "portrait" if opts["format"] == "9:16" else "landscape"
        self.work = fetch.DOWNLOADS / f"guide-parts-{uuid.uuid4().hex}"
        self.used_stock: set = set()
        self.credits: list = []
        self.shot_no = 0

    # --- progress ---
    def step(self, status: str, progress: float, detail: str = ""):
        check(self.job)
        self.job.update(status=status, progress=round(progress, 3), step=detail)
        if self.job.get("nid"):
            notify.live[self.job["nid"]] = {"progress": round(progress, 3), "step": detail, "phase": status}

    # --- video filters ---
    def fill_vf(self) -> str:
        return f"scale={self.W}:{self.H}:force_original_aspect_ratio=increase,crop={self.W}:{self.H},setsar=1,fps={FPS},format=yuv420p"

    def fit_fc(self, inp: str = "0:v", out: str = "v") -> str:
        """Any shape into the frame without cropping: the clip itself centered over a blurred copy of it."""
        W, H = self.W, self.H
        return (f"[{inp}]fps={FPS},split=2[bga][fga];[bga]scale={W // 4}:{H // 4}:force_original_aspect_ratio=increase,"
                f"crop={W // 4}:{H // 4},boxblur=8:2,scale={W}:{H},eq=brightness=-0.08[bg];"
                f"[fga]scale={W}:{H}:force_original_aspect_ratio=decrease[fg];[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1,format=yuv420p[{out}]")

    def enc(self) -> list[str]:
        return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-r", str(FPS)]

    # --- sources ---
    def source(self, clip: dict) -> Path:
        if clip["kind"] == "upload":
            f = next(CACHE.glob(clip["id"] + ".*"), None)
            if not f:
                raise RuntimeError(f"The video “{clip['label'] or clip['url']}” isn't on this computer any more. Add it again.")
            return f
        return self.download_link(clip)

    def download_link(self, clip: dict) -> Path:
        dst = CACHE / f"{clip['id']}.mp4"
        if dst.exists() and dst.stat().st_size > 0:
            return dst
        tmp = CACHE / f"{clip['id']}.part.mp4"
        cmd = [*fetch.ytdlp_base(), *fetch.format_args(1080), "--merge-output-format", "mp4", "--force-overwrites",
               "-o", str(tmp), clip["url"]]
        p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if p.returncode or not tmp.exists():
            raise RuntimeError(f"Couldn't download {clip['label'] or clip['url']}: {fetch.ytdlp_error(p.stderr)}")
        tmp.replace(dst)
        return dst

    def stock_pool(self, queries: list[str], spanish: bool) -> list[dict]:
        key = stock.pexels_key()
        if not key:
            raise RuntimeError("Add your free Pexels API key first (Stock video dialog), or set PEXELS_API_KEY.")
        pools = []
        for q in queries:
            params = {"query": q, "orientation": self.orientation, "per_page": 15, "size": "medium"}
            if spanish:
                params["locale"] = "es-ES"
            try:
                res = stock.pexels("/videos/search", params, key)
            except RuntimeError as e:
                if "API key" in str(e) or "rate limit" in str(e):
                    raise
                continue
            vids = [v for v in res.get("videos") or [] if stock.best_file(v) and (v.get("duration") or 0) >= 4]
            random.shuffle(vids)
            pools.append(vids)
        # Round-robin across queries so a section mixes its themes.
        out, seen = [], set()
        for i in range(max((len(p) for p in pools), default=0)):
            for p in pools:
                if i < len(p) and p[i]["id"] not in seen:
                    seen.add(p[i]["id"])
                    out.append(p[i])
        return out

    def stock_file(self, v: dict) -> Path:
        dst = CACHE / f"pexels-{v['id']}.mp4"
        if not (dst.exists() and dst.stat().st_size > 0):
            tmp = CACHE / f"pexels-{v['id']}.part"
            stock.download(stock.best_file(v)["link"], tmp, self.job)
            tmp.replace(dst)
        return dst

    # --- shots ---
    def render_stock_shot(self, v: dict, dur: float) -> Path:
        src = self.stock_file(v)
        room = max(0.0, float(v.get("duration") or duration(src)) - dur - 0.3)
        start = round(random.uniform(min(0.8, room), room), 2) if room > 0 else 0
        out = self.work / f"shot{self.shot_no:04d}.mp4"
        self.shot_no += 1
        loop = ["-stream_loop", "-1"] if room <= 0 else []
        ffmpeg([*loop, "-ss", str(start), "-i", str(src), "-t", f"{dur:.3f}", "-an", "-vf", self.fill_vf(), *self.enc(), str(out)])
        by = (v.get("user") or {}).get("name", "")
        if by and by not in [c["by"] for c in self.credits]:
            self.credits.append({"by": by, "url": v.get("url", "")})
        return out

    def render_clip_shot(self, src: Path, start: float, dur: float) -> Path:
        out = self.work / f"shot{self.shot_no:04d}.mp4"
        self.shot_no += 1
        ffmpeg(["-ss", f"{start:.3f}", "-i", str(src), "-t", f"{dur:.3f}", "-an", "-filter_complex", self.fit_fc(),
                "-map", "[v]", *self.enc(), str(out)])
        return out

    def broll(self, dur: float, sec: dict, pool: list, clip_srcs: list, shot_len: float) -> list[Path]:
        """Shots that fill `dur` seconds: stock footage, alternating with muted shots of the section's clips."""
        n = max(1, round(dur / shot_len))
        lens = [dur / n] * n
        shots = []
        for i, d in enumerate(lens):
            long_enough = [c for c in clip_srcs if c[1] >= d + 0.8]
            if long_enough and (i % 2 == 1 or not pool):
                src, length, avoid = long_enough[(i // 2) % len(long_enough)]
                # A stretch of the clip away from the part played with sound, so it isn't seen twice in a row.
                spots = [t for t in frange(0.3, length - d - 0.3, d) if not (avoid and avoid[0] - d < t < avoid[1])]
                spots = spots or list(frange(0.3, length - d - 0.3, d)) or [0.0]
                shots.append(self.render_clip_shot(src, random.choice(spots), d))
            else:
                if not pool:
                    pool.extend(self.stock_pool(["city night", "people walking", "sunset"], False))
                fresh = [v for v in pool if v["id"] not in self.used_stock] or pool
                v = fresh[0]
                self.used_stock.add(v["id"])
                pool.remove(v)
                pool.append(v)  # reused only once every clip in the pool has been seen
                shots.append(self.render_stock_shot(v, d))
        return shots

    # --- pieces: one narration line or one clip insert, each a finished video with sound ---
    def concat(self, files: list[Path], out: Path, copy: bool = True):
        lst = out.with_suffix(".txt")
        lst.write_text("".join(f"file '{f.name}'\n" for f in files), "utf-8")
        ffmpeg(["-f", "concat", "-safe", "0", "-i", lst.name, *(["-c", "copy"] if copy else self.enc()), out.name], cwd=self.work)

    def say_piece(self, idx: int, audio: Path, dur: float, shots: list[Path], overlay: str) -> Path:
        vid = self.work / f"p{idx:03d}-v.mp4"
        self.concat(shots, vid)
        out = self.work / f"p{idx:03d}.mp4"
        vf = []
        if overlay:
            (self.work / f"t{idx:03d}.txt").write_text(overlay, "utf-8")
            size = round(self.H * (0.05 if self.W > self.H else 0.032))
            y = "h*0.80" if self.W > self.H else "h*0.70"
            vf.append(f"drawtext=fontfile=font.ttf:textfile=t{idx:03d}.txt:fontsize={size}:fontcolor=white:"
                      f"box=1:boxcolor=black@0.55:boxborderw={size // 2}:x=(w-text_w)/2:y={y}:"
                      f"alpha='if(lt(t,0.4),t/0.4,if(lt(t,4.2),1,max(0,(4.8-t)/0.6)))'")
        ffmpeg(["-i", vid.name, "-i", audio.name, "-filter_complex",
                f"[1:a]aresample=48000,aformat=channel_layouts=stereo,apad[a]" + (f";[0:v]{vf[0]}[v]" if vf else ""),
                "-map", "[v]" if vf else "0:v", "-map", "[a]", "-t", f"{dur:.3f}",
                *(self.enc() if vf else ["-c:v", "copy"]), "-c:a", "aac", "-b:a", "192k", "-ar", "48000", out.name], cwd=self.work)
        return out

    def insert_piece(self, idx: int, src: Path, start: float, dur: float) -> Path:
        out = self.work / f"p{idx:03d}.mp4"
        a = "[0:a]aresample=48000,aformat=channel_layouts=stereo,afade=t=in:d=0.15,afade=t=out:st={0}:d=0.4,apad[a]".format(max(0, dur - 0.4))
        if not has_audio(src):
            a = "anullsrc=r=48000:cl=stereo[a]"
        ffmpeg(["-ss", f"{start:.3f}", "-i", str(src), "-t", f"{dur:.3f}", "-filter_complex", self.fit_fc() + ";" + a,
                "-map", "[v]", "-map", "[a]", "-t", f"{dur:.3f}", *self.enc(), "-c:a", "aac", "-b:a", "192k", "-ar", "48000", str(out)])
        return out

    # --- the whole run ---
    def run(self) -> dict:
        o, secs = self.o, self.guide["sections"]
        CACHE.mkdir(parents=True, exist_ok=True)
        self.work.mkdir(parents=True)
        if FONT.exists():
            shutil.copy(FONT, self.work / "font.ttf")

        # 1. Voice for every narration line.
        says = [(si, pi, p) for si, s in enumerate(secs) for pi, p in enumerate(s["pieces"]) if p["type"] == "say"]
        for k, (si, pi, p) in enumerate(says):
            self.step("voice", 0.02 + 0.18 * k / len(says), f"{k + 1} of {len(says)}")
            mp3 = self.work / f"say-{si}-{pi}.mp3"
            p["words"] = speak(p["text"], o["voice"], o["rate"], mp3)
            p["audio"], p["dur"] = mp3, duration(mp3) + 0.45  # a breath before the next line

        # 2. Clips linked in the guide.
        clip_ids = list(dict.fromkeys(c["id"] for s in secs for c in s["clips"]))
        files = {}
        for k, cid in enumerate(clip_ids):
            c = next(c for s in secs for c in s["clips"] if c["id"] == cid)
            self.step("clips", 0.20 + 0.10 * k / max(1, len(clip_ids)), c["label"] or c["url"])
            if c["kind"] == "upload":
                files[cid] = self.source(c)
                continue
            files[cid] = self.source(c)

        # 3. Stock search words per section.
        for k, s in enumerate(secs):
            self.step("planning", 0.30 + 0.05 * k / len(secs), s["heading"])
            q = ollama_queries(s, self.orientation) if o.get("ai", True) else []
            s["queries"], s["spanish"] = (q, False) if q else (fallback_queries(s), True)

        # 4. Render each section, piece by piece.
        pieces, timeline, srt, t = [], [], [], 0.0
        total = sum(len(s["pieces"]) for s in secs)
        done = 0
        for si, s in enumerate(secs):
            pool = self.stock_pool(s["queries"], s["spanish"])
            if not pool and not s["clips"]:
                pool = self.stock_pool(["city night", "people walking", "sunset"], False)
            srcs = []
            for c in s["clips"]:
                f = files[c["id"]]
                srcs.append([f, duration(f), None])
            shot_len = o["shot"] * (0.5 if si == 0 else 1.0)  # quick cuts for the opening
            sec_start, overlay_left = t, s["overlay"]
            ins_n = 0
            for pi, p in enumerate(s["pieces"]):
                self.step("rendering", 0.35 + 0.55 * done / total, f"{s['heading']} · part {pi + 1} of {len(s['pieces'])}")
                if p["type"] == "say":
                    shots = self.broll(p["dur"], s, pool, srcs, shot_len)
                    path = self.say_piece(len(pieces), p["audio"], p["dur"], shots, overlay_left)
                    overlay_left = ""
                    for w in p["words"]:
                        srt.append((t + w["s"], t + w["e"], w["w"]))
                    t += p["dur"]
                elif srcs:
                    k = p["clip"] if p["clip"] is not None and p["clip"] < len(srcs) else ins_n % len(srcs)
                    f, length, _ = srcs[k]
                    ins_n += 1
                    a, b = p["range"] or ((0.0, length) if p["whole"] else (0.0, min(length, o["insert"])))
                    a, b = max(0.0, min(a, length - 1)), max(a + 1, min(b, length))
                    srcs[k][2] = (a, b)
                    path = self.insert_piece(len(pieces), f, a, b - a)
                    t += b - a
                else:  # a stage direction with no clip linked: a short pause on B-roll
                    sil = self.work / f"sil{len(pieces)}.wav"
                    ffmpeg(["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", "-t", "1.2", str(sil)])
                    path = self.say_piece(len(pieces), sil, 1.2, self.broll(1.2, s, pool, srcs, shot_len), "")
                    t += 1.2
                pieces.append(path)
                done += 1
            timeline.append({"n": s["n"], "title": s["heading"], "start": round(sec_start, 2), "end": round(t, 2),
                             "planned": s["planned"]})

        # 5. Join, and lay the music under it.
        self.step("mixing", 0.92, "Joining sections" + (" and adding music" if o.get("music") else ""))
        body = self.work / "body.mp4"
        self.concat(pieces, body)
        media = uuid.uuid4().hex
        out = fetch.DOWNLOADS / f"{media}.mp4"
        total_s = duration(body) or t
        track = None
        if o.get("music"):
            try:
                track = music.track_file(o["music"])
            except Exception:
                track = None
        fade = f"afade=t=out:st={max(0, total_s - 3):.3f}:d=3"
        if track:
            vol = o["music_volume"]
            ffmpeg(["-i", str(body), "-stream_loop", "-1", "-i", str(track), "-filter_complex",
                    f"[1:a]aresample=48000,aformat=channel_layouts=stereo,volume={vol},afade=t=in:d=2,{fade}[m];"
                    f"[0:a][m]amix=inputs=2:duration=first:normalize=0,alimiter=limit=0.95[a]",
                    "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-t", f"{total_s:.3f}",
                    "-movflags", "+faststart", str(out)])
        else:
            ffmpeg(["-i", str(body), "-af", fade, "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(out)])

        (fetch.DOWNLOADS / f"{media}.srt").write_text(make_srt(srt), "utf-8")
        title = self.guide["title"]
        return {"media": media, "name": fetch.safe_name(title), "title": title, "size": out.stat().st_size,
                "width": self.W, "height": self.H, "duration": round(duration(out), 2), "sections": timeline,
                "credits": self.credits, "srt": f"/api/guide/srt/{media}", "file": str(out)}


def frange(a: float, b: float, step: float):
    step = max(step, 0.5)
    while a <= b:
        yield round(a, 2)
        a += step


def make_srt(words: list[tuple]) -> str:
    """Caption lines of up to ~7 words, broken at sentence ends."""
    def ts(x):
        ms = int(round(x * 1000))
        return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"
    lines, cur = [], []
    for w in words:
        cur.append(w)
        if len(cur) >= 7 or re.search(r"[.!?…:;]$", w[2]) or (len(cur) >= 4 and w[2].endswith(",")):
            lines.append(cur)
            cur = []
    if cur:
        lines.append(cur)
    return "".join(f"{i}\n{ts(l[0][0])} --> {ts(l[-1][1])}\n{' '.join(w[2] for w in l)}\n\n" for i, l in enumerate(lines, 1))


def options(body: dict) -> dict:
    fmt = "9:16" if str(body.get("format")) == "9:16" else "16:9"
    try:
        return {"format": fmt, "voice": str(body.get("voice") or DEFAULT_VOICE)[:80],
                "rate": max(-30, min(30, int(body.get("rate") or 0))),
                "shot": max(2.0, min(10.0, float(body.get("shot") or 4.5))),
                "insert": max(4.0, min(120.0, float(body.get("insert") or 20))),
                "music": str(body.get("music") or ""), "music_volume": max(0.0, min(1.0, float(body.get("music_volume") or 0.10))),
                "ai": body.get("ai", True) is not False}
    except (TypeError, ValueError):
        raise HTTPException(400, "rate, shot, insert and music_volume must be numbers")


def run_job(job_id: str, guide: dict, opts: dict):
    job = guide_jobs[job_id]
    b = Builder(job, guide, opts)
    nid = job.get("nid")
    try:
        r = b.run()
        job.update(status="done", progress=1.0, result=r)
        guides_update(job_id, status="done", finished=time.time(), result=r, prev_result=None)
        if nid:
            notify.update(nid, status="done", read=False, progress=1.0, result=r,
                          body=f"{fmt_len(r['duration'])} · {r['width']}×{r['height']} · {len(r['sections'])} sections. Ready to open.")
    except Exception as e:
        cancelled = str(e) == "cancelled"
        job.update(status="error", error="Cancelled." if cancelled else str(e), cancelled=cancelled)
        guides_update(job_id, status="cancelled" if cancelled else "error", finished=time.time(),
                      error="" if cancelled else str(e))
        if nid:
            notify.update(nid, status="cancelled" if cancelled else "error", read=cancelled,
                          body="Cancelled." if cancelled else str(e))
    finally:
        shutil.rmtree(b.work, ignore_errors=True)


def fmt_len(sec: float) -> str:
    sec = int(round(sec))
    return f"{sec // 3600}:{sec // 60 % 60:02d}:{sec % 60:02d}" if sec >= 3600 else f"{sec // 60}:{sec % 60:02d}"


def summary(guide: dict) -> dict:
    """What the page shows before rendering: the sections it found, the clips and the inserts."""
    words = sum(len(p["text"].split()) for s in guide["sections"] for p in s["pieces"] if p["type"] == "say")
    return {"title": guide["title"], "words": words, "est_seconds": round(words / 2.5),
            "sections": [{"n": s["n"], "heading": s["heading"], "overlay": s["overlay"], "planned": s["planned"],
                          "clips": [{"id": c["id"], "label": c["label"], "kind": c["kind"],
                                     "missing": c["kind"] == "upload" and not any(CACHE.glob(c["id"] + ".*"))} for c in s["clips"]],
                          "inserts": sum(p["type"] == "insert" for p in s["pieces"]),
                          "words": sum(len(p["text"].split()) for p in s["pieces"] if p["type"] == "say")}
                         for s in guide["sections"]]}


@router.post("/api/guide/parse")
def parse(body: dict):
    try:
        return summary(parse_guide(str(body.get("text") or "")))
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.post("/api/guide")
def start_guide(body: dict):
    try:
        guide = parse_guide(str(body.get("text") or "")[:60000])
    except ValueError as e:
        raise HTTPException(400, str(e))
    if not stock.pexels_key():
        raise HTTPException(400, "Add your free Pexels API key first (Stock video dialog).")
    job_id, opts = uuid.uuid4().hex, options(body)
    # A notification follows the job: progress while it runs, then "ready" (or the error) even if the page was closed.
    nid = notify.add("guide", guide["title"], "Starting…", status="running", job=job_id, progress=0.0,
                     format=opts["format"], sections=len(guide["sections"]))
    guide_jobs[job_id] = {"status": "voice", "progress": 0.0, "nid": nid}
    _guides_seed()
    entry = {"id": job_id, "title": guide["title"], "created": time.time(), "status": "running", "format": opts["format"],
             "sections": len(guide["sections"]), "text": str(body.get("text") or "")[:60000], "options": opts, "nid": nid}
    replaces = str(body.get("replaces") or "")
    with _guides_lock:
        items = _guides_load()
        old = next((i for i, g in enumerate(items) if g["id"] == replaces), None)
        if old is None:
            items.insert(0, entry)
        else:  # an edited guide: remade in place, its last video stays playable until the new one is ready
            prev = items[old].get("result") or items[old].get("prev_result")
            items[old] = {**entry, "created": items[old].get("created") or entry["created"], "edited": time.time(),
                          **({"prev_result": prev} if prev else {})}
        _guides_save(items)
    threading.Thread(target=run_job, args=(job_id, guide, opts), daemon=True).start()
    return {"job": job_id, "notification": nid, **summary(guide)}


@router.post("/api/guide/video")
async def add_video(request: Request, name: str = ""):
    """Your own footage for a guide. The body is the raw video file; the page links it as [name](video:<id>)."""
    ext = Path(name).suffix.lower()
    if ext not in VIDEO_EXT:
        raise HTTPException(400, "Use a video file: MP4, MOV, MKV, WEBM, AVI…")
    CACHE.mkdir(parents=True, exist_ok=True)
    vid = uuid.uuid4().hex
    dst, size = CACHE / f"up-{vid}{ext}", 0
    try:
        with dst.open("wb") as f:
            async for chunk in request.stream():
                size += len(chunk)
                f.write(chunk)
    except BaseException:
        dst.unlink(missing_ok=True)
        raise
    secs = duration(dst) if size else 0
    if not secs and size:
        # Recordings from browsers and some phones/screen recorders (WebM) have no length in the header, which
        # also makes seeking slow. Re-save them once as a normal MP4.
        tmp, fixed = CACHE / f"fix-{vid}.mp4", CACHE / f"up-{vid}.mp4"
        try:
            await asyncio.to_thread(ffmpeg, ["-i", str(dst), "-c:v", "libx264", "-preset", "veryfast", "-crf", "19",
                                             "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", str(tmp)])
            dst.unlink(missing_ok=True)
            tmp.replace(fixed)
            dst, secs = fixed, duration(fixed)
        except RuntimeError:
            tmp.unlink(missing_ok=True)
    if not secs:
        dst.unlink(missing_ok=True)
        raise HTTPException(400, "That file isn't a video ffmpeg can read.")
    label = re.sub(r"[\[\]()\x00-\x1f]+", " ", Path(name).stem).strip()[:60] or "Mi video"
    return {"id": vid, "name": label, "duration": round(secs, 2), "link": f"[{label}](video:{vid})"}


@router.post("/api/guide/find")
def find_clips(body: dict):
    """Finds YouTube videos for one section: the AI writes searches from the section, yt-dlp runs them.
    Send `query` to run your own search instead."""
    try:
        guide = parse_guide(str(body.get("text") or ""))
    except ValueError as e:
        raise HTTPException(400, str(e))
    n = int(body.get("n") or 0)
    sec = next((s for s in guide["sections"] if s["n"] == n), None)
    if not sec:
        raise HTTPException(404, f"No section {n} in the guide.")
    own = str(body.get("query") or "").strip()[:120]
    queries = [own] if own else youtube_queries(sec, guide["title"])
    have = {c["id"].removeprefix("yt-") for c in sec["clips"]}
    results, seen = [], set(have)
    with ThreadPoolExecutor(4) as pool:
        batches = list(pool.map(lambda q: youtube_search(q, 10 if own else 6), queries))
    # Interleave the searches so the top of the list mixes them.
    for i in range(max((len(b) for b in batches), default=0)):
        for b in batches:
            if i < len(b) and b[i]["id"] not in seen:
                seen.add(b[i]["id"])
                results.append(b[i])
    return {"queries": queries, "results": results[:24]}


@router.post("/api/guide/link")
def add_link(body: dict):
    """Checks a video link (YouTube, TikTok, Instagram…) and returns it as [title](url) for a section's Visual line."""
    url = str(body.get("url") or "").strip()
    if not re.match(r"^https?://\S+$", url):
        raise HTTPException(400, "Paste a full link, starting with https://")
    yt = YT_ID.search(url)
    if yt:  # studio.youtube.com and shorts links -> a normal watch link yt-dlp can always read
        url = f"https://www.youtube.com/watch?v={yt.group(1)}"
    cmd = [*fetch.ytdlp_base(), "--skip-download", "--print", "%(title)s", "--print", "%(duration)s", url]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
    except subprocess.TimeoutExpired:
        raise HTTPException(504, "The site took too long to answer. Try again.")
    if p.returncode:
        raise HTTPException(400, fetch.ytdlp_error(p.stderr))
    out = p.stdout.strip().splitlines()
    title = re.sub(r"[\[\]()\x00-\x1f]+", " ", out[0] if out else "").strip()[:70] or "clip"
    try:
        secs = float(out[1])
    except (IndexError, ValueError):
        secs = 0
    return {"name": title, "duration": round(secs, 1), "url": url, "link": f"[{title}]({url})"}


@router.get("/api/guides")
def list_guides():
    """The Guides page: newest first, with live progress for the ones still being made."""
    _guides_seed()
    out = []
    for g in _guides_load():
        g = {**{k: v for k, v in g.items() if k != "text"}, "has_text": bool(g.get("text"))}  # the text comes with /api/guides/{id}
        if g.get("status") == "running":
            job = guide_jobs.get(g["id"])
            if job is None:  # the server restarted while it ran
                g.update(status="error", error="Interrupted: the server stopped while it was being made.")
            else:
                lv = notify.live.get(g.get("nid") or "", {})
                g.update(progress=lv.get("progress", job.get("progress", 0)), step=lv.get("step", ""), phase=lv.get("phase", job.get("status")))
        r = g.get("result") or {}
        g["available"] = bool(r.get("media")) and fetch.media_exists(r["media"])
        pr = g.get("prev_result") or {}
        if not pr.get("media") or not fetch.media_exists(pr["media"]):
            g.pop("prev_result", None)
        out.append(g)
    return {"guides": out}


@router.get("/api/guides/{gid}")
def get_guide(gid: str):
    for g in _guides_load():
        if g["id"] == gid:
            return g
    raise HTTPException(404, "No such guide.")


@router.post("/api/guides/{gid}/retry")
def retry_guide(gid: str):
    """Makes a guide's video again with the same text and settings, in its place in the list."""
    g = next((x for x in _guides_load() if x["id"] == gid), None)
    if not g:
        raise HTTPException(404, "No such guide.")
    if g.get("status") == "running" and gid in guide_jobs:
        raise HTTPException(409, "It's still being made.")
    if not g.get("text"):
        raise HTTPException(400, "This guide's text wasn't saved, so it can't be made again. Create it again from the guide.")
    return start_guide({**(g.get("options") or {}), "text": g["text"], "replaces": gid})


@router.post("/api/guides/{gid}/delete")
def delete_guide(gid: str):
    """Removes it from the list (the video file stays in downloads/)."""
    with _guides_lock:
        items = _guides_load()
        left = [g for g in items if g["id"] != gid]
        if len(left) == len(items):
            raise HTTPException(404, "No such guide.")
        _guides_save(left)
    return {"ok": True}


@router.post("/api/guide/{job_id}/stop")
def stop_guide(job_id: str):
    job = guide_jobs.get(job_id)
    if not job or job["status"] in ("done", "error"):
        raise HTTPException(404, "Nothing to stop.")
    job["cancel"] = True
    return {"ok": True}


@router.get("/api/guide/srt/{media}")
def guide_srt(media: str):
    f = fetch.DOWNLOADS / f"{media}.srt"
    if not re.fullmatch(r"[0-9a-f]{32}", media) or not f.exists():
        raise HTTPException(404, "No subtitles for this video.")
    return FileResponse(f, media_type="application/x-subrip", filename="subtitles.srt")


@router.get("/api/guide/{job_id}")
def guide_status(job_id: str):
    job = guide_jobs.get(job_id)
    if not job:
        raise HTTPException(404, "unknown job")
    if job["status"] in ("done", "error"):
        guide_jobs.pop(job_id)
    return job


if __name__ == "__main__":
    import argparse
    import time

    ap = argparse.ArgumentParser(description="Make a narrated video from a written guide.")
    ap.add_argument("guide", help="text file with the guide")
    ap.add_argument("--format", default="16:9", choices=["16:9", "9:16"])
    ap.add_argument("--voice", default=DEFAULT_VOICE)
    ap.add_argument("--rate", type=int, default=0, help="voice speed, -30..30 percent")
    ap.add_argument("--shot", type=float, default=4.5, help="seconds per B-roll shot")
    ap.add_argument("--insert", type=float, default=20, help="seconds of a clip insert without a time range")
    ap.add_argument("--music", default="", help="track id from data/music/index.json")
    ap.add_argument("--music-volume", type=float, default=0.10)
    ap.add_argument("--no-ai", action="store_true", help="don't ask Ollama for stock search words")
    ap.add_argument("--parse-only", action="store_true")
    a = ap.parse_args()
    g = parse_guide(Path(a.guide).read_text("utf-8"))
    print(json.dumps(summary(g), ensure_ascii=False, indent=2))
    if not a.parse_only:
        jid = "cli"
        guide_jobs[jid] = {"status": "voice", "progress": 0.0}
        th = threading.Thread(target=run_job, args=(jid, g, options({"format": a.format, "voice": a.voice, "rate": a.rate, "shot": a.shot,
                              "insert": a.insert, "music": a.music, "music_volume": a.music_volume, "ai": not a.no_ai})))
        th.start()
        last = None
        while th.is_alive():
            j = guide_jobs[jid]
            now = (j["status"], j.get("step"))
            if now != last:
                print(f"{j['progress'] * 100:5.1f}%  {j['status']:<10} {j.get('step', '')}", flush=True)
                last = now
            time.sleep(0.5)
        j = guide_jobs[jid]
        print(json.dumps(j.get("result") or j, ensure_ascii=False, indent=2))
