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
import tempfile
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
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
PHOTO_SECONDS = 6  # an uploaded photo becomes a clip this long, with a slow zoom
URL = re.compile(r"https?://\S+")
_RNG = r"(?:\s*\(\s*(\d{1,2}:\d{2}(?::\d{2})?)\s*[-–—]\s*(\d{1,2}:\d{2}(?::\d{2})?)\s*\))?"
# "[frases 2-3]" after a clip: it shows while the voice says sentences 2 to 3 of the section's narration.
_SAY = r"(?:\s*\[\s*(?:[Ff]rases?|[Ss]entences?)\s*(\d{1,2})(?:\s*[-–—]\s*(\d{1,2}))?\s*\])?"
# "[encuadre 20,10,45,40]" after a clip: only that area of its picture is used (left, top, width, height in % of it).
_N = r"\s*(-?\d{1,3}(?:\.\d+)?)\s*"
_CROP = r"(?:\s*\[\s*(?:[Ee]ncuadre|[Cc]rop)" + _N + "," + _N + "," + _N + "," + _N + r"\])?"
LINK_RANGE = re.compile(r"\[([^\]]*)\]\(((?:https?://|video:)[^)\s]+)\)" + _RNG + _SAY + _CROP)
BARE_RANGE = re.compile(r"(?<![(\w])(https?://[^\s()]+)" + _RNG + _SAY + _CROP)
RANGE_AFTER = re.compile(r"\s*\(\s*\d{1,2}:\d{2}(?::\d{2})?\s*[-–—]\s*\d{1,2}:\d{2}(?::\d{2})?\s*\)"
                         r"|\s*\[\s*(?:frases?|sentences?)\s*\d{1,2}(?:\s*[-–—]\s*\d{1,2})?\s*\]"
                         r"|\s*\[\s*(?:encuadre|crop)[-\d.,\s]*\]", re.I)


def parse_crop(vals) -> list[float] | None:
    """[left, top, width, height] as fractions of the picture, or None when missing or not a real area. The box
    may reach past the picture (zoomed out): that part is filled with a blurred copy of the clip."""
    if not vals or not all(vals):
        return None
    x, y, w, h = (max(-500.0, min(500.0, float(v))) / 100 for v in vals)
    if w < 0.05 or h < 0.05 or x >= 1 or y >= 1 or x + w <= 0 or y + h <= 0:
        return None
    if abs(x) < 0.005 and abs(y) < 0.005 and abs(w - 1) < 0.005 and abs(h - 1) < 0.005:
        return None
    return [round(x, 4), round(y, 4), round(w, 4), round(h, 4)]
# A section heading: "1. Apertura (0:00-0:45)", "## 2) Caso", or "[0:50–1:30] 1. MONCHY CADENA" / "[0:00–0:25] INTRO".
# Sections are numbered in order of appearance (the number in the heading is ignored), matching index.html.
_T = r"\d{1,2}:\d{2}(?::\d{2})?"
HEADING = re.compile(rf"^\s*(?:#+\s*)?(?:\[\s*({_T})\s*[-–—]\s*({_T})\s*\]\s*(?:\d{{1,2}}[.)]\s+)?|\d{{1,2}}[.)]\s+)(\S.*?)\s*$")
TIME_RANGE = re.compile(r"\(?\s*(\d{1,2}:\d{2}(?::\d{2})?)\s*[-–—]\s*(\d{1,2}:\d{2}(?::\d{2})?)\s*\)?")
FIELD = re.compile(r"^\s*[*\-•]?\s*(visual|visuales|narraci[oó]n|narration|voz)\s*:\s*(.*)$", re.I)
# "* Efecto: Punch zoom + Whoosh": a visual effect and a sound effect where the section starts.
FX_LINE = re.compile(r"^\s*[*\-•]?\s*(?:efectos?|effects?|fx)\s*:\s*(.*)$", re.I)
FX_VIS = {"punch": "Punch zoom", "shake": "Shake", "flash": "Flash", "glitch": "Glitch", "push": "Slow push-in"}
FX_SND = {"whoosh": "Whoosh", "boom": "Boom", "hit": "Hit", "riser": "Riser", "ding": "Ding", "pop": "Pop", "glitch": "Glitch"}
FX_ALIAS = {"zoom": "punch", "golpe": "punch", "sacudida": "shake", "temblor": "shake", "destello": "flash",
            "acercamiento": "push", "push-in": "push", "push in": "push", "swoosh": "whoosh", "bum": "boom", "golpe seco": "hit",
            "subida": "riser", "campana": "ding"}


def fx_key(text: str, table: dict) -> str:
    t = text.strip().lower()
    if not t or t in ("none", "ninguno", "nada", "-"):
        return ""
    for k, label in table.items():
        if t in (k, label.lower()):
            return k
    a = FX_ALIAS.get(t, "")
    return a if a in table else ""


def parse_fx(text: str) -> dict:
    """"Punch zoom + Whoosh" -> {"v": "punch", "s": "whoosh"}; one name alone is the visual if it is one, else the sound."""
    # "… cada 3s" / "… every 3 s": repeated through the section's narration instead of once at the start.
    m = re.search(r"\b(?:cada|every)\s*(\d+(?:[.,]\d+)?)\s*(?:s|seg|segundos?|seconds?)?\b", text, re.I)
    every =max(1.0, min(30.0, float(m.group(1).replace(",", ".")))) if m else 0
    if m:
        text = text[:m.start()] + text[m.end():]
    parts = [p for p in re.split(r"\s*[+/|,]\s*", text.strip()) if p]
    if len(parts) >= 2:
        return {"v": fx_key(parts[0], FX_VIS), "s": fx_key(parts[1], FX_SND), "every": every}
    one = parts[0] if parts else ""
    v = fx_key(one, FX_VIS)
    return {"v": v, "s": "" if v else fx_key(one, FX_SND), "every": every}


# Production notes that are not read aloud ("Texto: …" on screen, "Momento emotivo: …", "Música: …").
NOTE = re.compile(r"^\s*[*\-•]?\s*(texto(?: en pantalla)?|text|momento\b[^:]{0,20}|m[uú]sica|music|nota|note|edici[oó]n|sfx|efecto[^:]{0,15})\s*:", re.I)
QUOTE_START = re.compile(r"^\s*[*\-•]?\s*[\"“«]")


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
            title = m.group(1).strip().strip('"“”«»').strip()
            continue
        if re.match(r"^(?:duraci[oó]n|duration)", line, re.I) and not sections:
            continue
        h = HEADING.match(line)
        if h and not FIELD.match(line):
            head = h.group(3)
            planned = [to_sec(h.group(1)), to_sec(h.group(2))] if h.group(1) else None
            tr = TIME_RANGE.search(head)
            if tr:
                head = (head[: tr.start()] + head[tr.end():]).strip()
                planned = planned or [to_sec(tr.group(1)), to_sec(tr.group(2))]
            cur = {"n": len(sections) + 1, "heading": head.strip(" :"), "visual": "", "narration": "",
                   "planned": planned, "fx": {"v": "", "s": ""}}
            sections.append(cur)
            field = None
            continue
        if cur is None:
            continue
        f = FIELD.match(line)
        fx = FX_LINE.match(line)
        if f:
            field = "visual" if f.group(1).lower().startswith("vis") else "narration"
            cur[field] = (cur[field] + " " + f.group(2)).strip()
        elif fx:
            cur["fx"] = parse_fx(fx.group(1))
            field = None
        elif NOTE.match(line):
            field = None
        elif field != "narration" and QUOTE_START.match(line):
            # A quoted paragraph with no "Narración:" label is the narration.
            field = "narration"
            cur[field] = (cur[field] + " " + line.lstrip("*-• ")).strip()
        elif field:
            cur[field] = (cur[field] + " " + line.lstrip("*-• ")).strip()
    if not sections:
        raise ValueError("No sections found. Number each section (\"1. Apertura\") and give it a "
                         "\"Visual:\" and a \"Narración:\" line.")
    for s in sections:
        # "[title](link) (0:12-0:31)": the part of that clip to use (the time range is optional).
        links = [(m.group(1), m.group(2), m.group(3), m.group(4), m.group(5), m.group(6), m.group(7, 8, 9, 10))
                 for m in LINK_RANGE.finditer(s["visual"])]
        links = links or [("", m.group(1), m.group(2), m.group(3), m.group(4), m.group(5), m.group(6, 7, 8, 9))
                          for m in BARE_RANGE.finditer(s["visual"])]
        s["clips"] = []
        for label, url, ra, rb, sa, sb, crop in links:
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
            rng = [to_sec(ra), to_sec(rb)] if ra and rb and to_sec(rb) > to_sec(ra) else None
            say = sorted([int(sa), int(sb or sa)]) if sa else None
            same = next((c for c in s["clips"] if c["id"] == clip["id"]), None)
            if same is None:
                clip["raws"] = [url]  # the link as written in the guide, so the page can remove it
                clip["range"], clip["say"], clip["crop"] = rng, say, parse_crop(crop)
                s["clips"].append(clip)
            else:
                if url not in same["raws"]:
                    same["raws"].append(url)
                same["range"] = same["range"] or rng
                same["say"] = same["say"] or say
                same["crop"] = same["crop"] or parse_crop(crop)
        vis = RANGE_AFTER.sub("", s["visual"])
        s["visual_text"] = re.sub(r"\s+", " ", URL.sub("", MD_LINK.sub(lambda m: m.group(1), vis))).strip()
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
    t = re.sub(r"\[[^\]]*\]\s*([.,;:]?)", "", t)  # "[Agrega aquí: …]." placeholders are notes, not text to read
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


def _metadata_series(cmd: list[str], key: str) -> list[tuple[float, float]]:
    """(time, value) pairs from an ffmpeg run that prints frame metadata (metadata=print:file=-)."""
    p = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", *cmd, "-f", "null", "-"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    out, t = [], None
    for line in p.stdout.splitlines():
        m = re.search(r"pts_time:(-?[\d.]+)", line)
        if m:
            t = float(m.group(1))
        elif t is not None and line.startswith(key + "="):
            try:
                out.append((t, float(line.split("=", 1)[1])))
            except ValueError:
                pass
    return out


def clip_scores(f: Path) -> list[float]:
    """How lively each second of a clip is (0..1): movement in the picture plus how loud it is. Cached next to the
    clip. Used to pick the best stretches of clips that have no part chosen."""
    st = f.stat()
    cache, key = f.with_name(f.name + ".scores.json"), f"{st.st_size}-{int(st.st_mtime)}"
    try:
        c = json.loads(cache.read_text("utf-8"))
        if c.get("key") == key:
            return c["scores"]
    except (OSError, ValueError, KeyError):
        pass
    n = max(1, int(duration(f)) + 1)
    motion, loud = [[] for _ in range(n)], [[] for _ in range(n)]
    for t, v in _metadata_series(["-i", str(f), "-an", "-vf", "fps=4,scale=160:-2,select='gte(scene\\,0)',metadata=print:file=-"],
                                 "lavfi.scene_score"):
        if 0 <= int(t) < n:
            motion[int(t)].append(v)
    if has_audio(f):
        for t, v in _metadata_series(["-i", str(f), "-vn", "-af", "ebur128=metadata=1,ametadata=print:key=lavfi.r128.M:file=-"],
                                     "lavfi.r128.M"):
            if 0 <= int(t) < n:
                loud[int(t)].append(v)
    mv = [sum(x) / len(x) if x else 0.0 for x in motion]
    # A hard cut is a single spike; cap it so steady movement counts more than one cut.
    top = sorted(mv)[int(len(mv) * 0.9)] or 1e-6
    mv = [min(1.0, v / top) for v in mv]
    ld = [min(1.0, max(0.0, (sum(x) / len(x) + 45) / 30)) if x else 0.0 for x in loud]  # -45 LUFS..-15 LUFS -> 0..1
    scores = [round(0.6 * a + 0.4 * b, 4) for a, b in zip(mv, ld)]
    try:
        cache.write_text(json.dumps({"key": key, "scores": scores}), "utf-8")
    except OSError:
        pass
    return scores


FACE_MODEL = ROOT / "data" / "models" / "face_detection_yunet.onnx"
_faces = {}


def _face_detector():
    """A function small_bgr_frame -> [(x, y, w, h), …], or None when no face model is available: OpenCV's YuNet
    (FACE_MODEL, if present) or its old Haar cascade (OpenCV 4)."""
    import cv2
    if "fn" in _faces:
        return _faces["fn"]
    fn = None
    if FACE_MODEL.exists() and hasattr(cv2, "FaceDetectorYN"):
        det = cv2.FaceDetectorYN.create(str(FACE_MODEL), "", (320, 320), 0.7)
        def fn(img, det=det):
            det.setInputSize((img.shape[1], img.shape[0]))
            _, found = det.detect(img)
            return [tuple(r[:4]) for r in found] if found is not None else []
    elif hasattr(cv2, "CascadeClassifier"):
        cas = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        if not cas.empty():
            def fn(img, cas=cas):
                return list(cas.detectMultiScale(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 1.1, 5, minSize=(16, 16)))
    _faces["fn"] = fn
    return fn


def track_subject(f: Path, start: float, dur: float, out_ratio: float, area: list | None = None):
    """Where to put a crop window of the frame's shape (out_ratio = width / height) so it follows the action in
    f from `start` for `dur` seconds: faces first (the biggest, staying with the one already followed), else where
    the picture moves. With `area` (a chosen ⛶ area, fractions of the picture) the window has that area's size and
    only its position follows; a side bigger than the picture stays where it was (it can't move that way).
    Returns (crop_w, crop_h, [(t, x, y), …]) in source pixels (x/y negative when bigger than the picture), smoothed,
    or None."""
    import cv2
    import numpy as np
    detect = _face_detector()
    cap = cv2.VideoCapture(str(f))
    if not cap.isOpened():
        return None
    W0, H0 = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    if not W0 or not H0:
        return None
    if area:
        cw, ch = int(area[2] * W0) // 2 * 2, int(area[3] * H0) // 2 * 2
        fixed_x = int(area[0] * W0) // 2 * 2 if cw >= W0 else None
        fixed_y = int(area[1] * H0) // 2 * 2 if ch >= H0 else None
        if fixed_x is not None and fixed_y is not None:
            return None  # bigger than the picture both ways: nothing to move
    else:
        if abs(W0 / H0 - out_ratio) < 0.02:
            return None  # same shape: nothing to follow
        cw, ch = (H0 * out_ratio, H0) if W0 / H0 > out_ratio else (W0, W0 / out_ratio)
        cw, ch = int(cw) // 2 * 2, int(ch) // 2 * 2
        fixed_x = fixed_y = None
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    step = max(1, round(fps / 6))  # look 6 times a second
    k, prev, pts, last = 0, None, [], (W0 / 2, H0 * 0.4)
    scale = 320 / max(W0, H0)
    while k / fps <= dur + 0.2:
        ok, frame = cap.read()
        if not ok:
            break
        if k % step == 0:
            small = cv2.resize(frame, (max(1, int(W0 * scale)), max(1, int(H0 * scale))))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            faces = detect(small) if detect else []
            target = None
            if not len(faces):
                # No face model (or no face found): the top of the people. The first rows from the top with enough
                # skin color are usually foreheads; the window is put so they sit near its top, keeping heads in.
                ycc = cv2.cvtColor(small, cv2.COLOR_BGR2YCrCb)
                skin = cv2.inRange(ycc, (40, 138, 80), (240, 175, 128))
                skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
                rows = (skin > 0).mean(axis=1)
                band = np.convolve(rows, np.ones(3) / 3, "same")
                top = next((i for i, v in enumerate(band) if v > 0.06), None)
                if top is not None:
                    hy = top / scale
                    cols = (skin[top:top + max(3, int(skin.shape[0] * 0.15))] > 0).mean(axis=0)
                    hx = (np.arange(len(cols)) * cols).sum() / cols.sum() / scale if cols.sum() else W0 / 2
                    target = (hx, hy - ch * 0.25 + ch / 2)  # the first skin ~25% below the window's top: room for hair
            if len(faces):
                # The biggest face, but stay with the one followed so far when it's still there. Faces of similar
                # size next to it are framed together when they fit (two people talking).
                cand = sorted(faces, key=lambda r: -r[2] * r[3])
                near = [r for r in cand if abs((r[0] + r[2] / 2) / scale - last[0]) < cw * 0.25]
                main = (near or cand)[0]
                group = [r for r in cand if r[2] >= main[2] * 0.5]
                x0 = min(r[0] for r in group) / scale; x1 = max(r[0] + r[2] for r in group) / scale
                y0 = min(r[1] for r in group) / scale; y1 = max(r[1] + r[3] for r in group) / scale
                if x1 - x0 > cw * 0.95 or y1 - y0 > ch * 0.95:  # they don't fit together: just the main one
                    x0, y0 = main[0] / scale, main[1] / scale
                    x1, y1 = x0 + main[2] / scale, y0 + main[3] / scale
                top = y0 - ch * 0.15                      # some room above the head
                if top + ch < y1 + ch * 0.05:             # but the chin must fit
                    top = y1 + ch * 0.05 - ch
                target = ((x0 + x1) / 2, top + ch / 2)
            elif prev is not None:
                diff = cv2.absdiff(gray, prev)
                _, mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
                m = cv2.moments(mask)
                if m["m00"] > 255 * gray.size * 0.01:  # enough of the picture moved
                    target = (m["m10"] / m["m00"] / scale, m["m01"] / m["m00"] / scale)
            prev = gray
            if target is not None:
                last = target
            pts.append((k / fps, last))
        k += 1
    cap.release()
    if not pts:
        return None
    # Smooth: a moving average, then follow it at a capped speed so the window glides instead of jumping.
    xs = np.array([p[1][0] for p in pts]); ys = np.array([p[1][1] for p in pts])
    if len(xs) >= 5:
        kern = np.ones(5) / 5
        xs = np.convolve(np.pad(xs, 2, mode="edge"), kern, "valid"); ys = np.convolve(np.pad(ys, 2, mode="edge"), kern, "valid")
    out, cx, cy = [], xs[0], ys[0]
    max_step = max(W0, H0) * 0.35 / 6  # at most 35% of the picture per second
    for (t, _), tx, ty in zip(pts, xs, ys):
        cx += max(-max_step, min(max_step, tx - cx)); cy += max(-max_step, min(max_step, ty - cy))
        x = fixed_x if fixed_x is not None else int(max(0, min(W0 - cw, cx - cw / 2))) // 2 * 2
        y = fixed_y if fixed_y is not None else int(max(0, min(H0 - ch, cy - ch / 2))) // 2 * 2
        out.append((round(t, 3), x, y))
    return cw, ch, out, int(W0), int(H0)


def best_starts(scores: list[float], lo: float, hi: float, seg: float, step: float | None = None) -> list[float]:
    """Start times of seg-long stretches inside lo..hi, best first (by the average liveliness they cover)."""
    step = step or seg
    starts = list(frange(lo, max(lo, hi - seg), step)) or [lo]
    def sc(t):
        a, b = int(t), max(int(t) + 1, int(t + seg + 0.999))
        part = scores[a:b] or [0.0]
        return sum(part) / len(part)
    return sorted(starts, key=lambda t: -sc(t))


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


_clip_locks: dict[str, threading.Lock] = {}


def download_clip(clip: dict) -> Path:
    """A linked clip (YouTube, TikTok…) downloaded once to the cache; the render and the clip picker share it."""
    dst = CACHE / f"{clip['id']}.mp4"
    with _clip_locks.setdefault(clip["id"], threading.Lock()):
        if dst.exists() and dst.stat().st_size > 0:
            return dst
        CACHE.mkdir(parents=True, exist_ok=True)
        tmp = CACHE / f"{clip['id']}.part.mp4"
        cmd = [*fetch.ytdlp_base(), *fetch.format_args(1080), "--merge-output-format", "mp4", "--force-overwrites",
               "-o", str(tmp), clip["url"]]
        p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if p.returncode or not tmp.exists():
            raise RuntimeError(f"Couldn't download {clip.get('label') or clip['url']}: {fetch.ytdlp_error(p.stderr)}")
        tmp.replace(dst)
        return dst


def clip_from_link(url: str) -> dict | None:
    """The clip a link in a Visual line stands for (same ids as parse_guide)."""
    up, yt = UPLOAD_ID.match(url), YT_ID.search(url)
    if up:
        return {"id": "up-" + up.group(1), "kind": "upload", "label": "", "url": url}
    if yt:
        return {"id": "yt-" + yt.group(1), "kind": "link", "label": "", "url": f"https://www.youtube.com/watch?v={yt.group(1)}"}
    if url.startswith("http"):
        return {"id": "web-" + hashlib.sha1(url.encode()).hexdigest()[:16], "kind": "link", "label": "", "url": url}
    return None


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

    def clip_fc(self, crop: list | None = None) -> str:
        """Your clips in the frame: "fill" crops them to the full width and height; "fit" shows all of the clip over
        a blurred copy of itself (vertical clips get blurred sides). A chosen area ("encuadre", picked in the frame's
        shape) is cut out first and fills the frame."""
        if crop:
            x, y, w, h = crop
            ix, iy = max(0.0, x), max(0.0, y)
            iw, ih = min(1.0, x + w) - ix, min(1.0, y + h) - iy  # the part of the box that is on the picture
            if ix - x < 0.002 and iy - y < 0.002 and iw > w - 0.004 and ih > h - 0.004:
                cut = f"crop=w=trunc(iw*{w}/2)*2:h=trunc(ih*{h}/2)*2:x=iw*{x}:y=ih*{y}"
                return f"[0:v]{cut},{self.fill_vf()}[v]"
            # Zoomed out past the picture: the picture part, scaled and placed where it falls in the box, over a
            # blurred copy of the clip filling the rest.
            W, H = self.W, self.H
            fw, fh = max(2, round(iw / w * W / 2) * 2), max(2, round(ih / h * H / 2) * 2)
            ox, oy = round((ix - x) / w * W), round((iy - y) / h * H)
            return (f"[0:v]fps={FPS},split=2[bga][fga];[bga]scale={W // 4}:{H // 4}:force_original_aspect_ratio=increase,"
                    f"crop={W // 4}:{H // 4},boxblur=8:2,scale={W}:{H},eq=brightness=-0.08[bg];"
                    f"[fga]crop=w=trunc(iw*{iw:.4f}/2)*2:h=trunc(ih*{ih:.4f}/2)*2:x=iw*{ix:.4f}:y=ih*{iy:.4f},"
                    f"scale={fw}:{fh}[fg];[bg][fg]overlay={ox}:{oy},setsar=1,format=yuv420p[v]")
        return f"[0:v]{self.fill_vf()}[v]" if self.o.get("fit", "fill") == "fill" else self.fit_fc()

    def shot_fc(self, src: Path, start: float, dur: float, crop: list | None) -> str:
        """The filter that puts a clip in the frame: a chosen area, or (setting "follow") a window that follows
        the faces / action, or the plain fill / fit."""
        if self.o.get("fit") == "follow":
            try:
                tr = track_subject(src, start, dur, self.W / self.H, crop)
            except Exception:
                tr = None
            if tr:
                cw, ch, pts, W0, H0 = tr
                # The part of the window on the picture (all of it, unless the chosen area is bigger than the
                # picture on one side: then that side is the whole picture and the rest is blurred fill).
                over_x, over_y = cw > W0, ch > H0
                iw, ih = (W0 if over_x else cw), (H0 if over_y else ch)
                # One position per frame, between the tracked points, sent to crop as commands.
                lines, j = [], 0
                for i in range(int(dur * FPS) + 2):
                    t = i / FPS
                    while j + 1 < len(pts) and pts[j + 1][0] <= t:
                        j += 1
                    a, b = pts[j], pts[min(j + 1, len(pts) - 1)]
                    u = 0 if b[0] <= a[0] else min(1, max(0, (t - a[0]) / (b[0] - a[0])))
                    x, y = round(a[1] + (b[1] - a[1]) * u) // 2 * 2, round(a[2] + (b[2] - a[2]) * u) // 2 * 2
                    lines.append(f"{t:.3f} crop x {0 if over_x else x}, crop y {0 if over_y else y};")
                self.cmd_no = getattr(self, "cmd_no", 0) + 1
                name = f"follow{self.cmd_no:04d}.txt"
                (self.work / name).write_text("\n".join(lines) + "\n", "utf-8")
                x0, y0 = (0 if over_x else pts[0][1]), (0 if over_y else pts[0][2])
                cut = f"sendcmd=f={name},crop=w={iw}:h={ih}:x={x0}:y={y0}"
                if not over_x and not over_y:
                    return f"[0:v]{cut},scale={self.W}:{self.H},setsar=1,fps={FPS},format=yuv420p[v]"
                W, H = self.W, self.H
                k = W / cw  # source pixels -> frame pixels
                fw, fh = max(2, round(iw * k / 2) * 2), max(2, round(ih * k / 2) * 2)
                ox, oy = (round(-pts[0][1] * k) if over_x else 0), (round(-pts[0][2] * k) if over_y else 0)
                return (f"[0:v]fps={FPS},split=2[bga][fga];[bga]scale={W // 4}:{H // 4}:force_original_aspect_ratio=increase,"
                        f"crop={W // 4}:{H // 4},boxblur=8:2,scale={W}:{H},eq=brightness=-0.08[bg];"
                        f"[fga]{cut},scale={fw}:{fh}[fg];[bg][fg]overlay={ox}:{oy},setsar=1,format=yuv420p[v]")
        if not crop and self.o.get("fit") == "follow":
            return f"[0:v]{self.fill_vf()}[v]"
        return self.clip_fc(crop)

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
        return download_clip(clip)

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

    def render_clip_shot(self, src: Path, start: float, dur: float, length: float = 0, crop: list | None = None) -> Path:
        out = self.work / f"shot{self.shot_no:04d}.mp4"
        self.shot_no += 1
        loop = ["-stream_loop", "-1"] if length and start + dur > length else []  # a clip shorter than the shot repeats
        fc = self.shot_fc(src, start, min(dur, max(0.5, (length or dur) - start)) if loop else dur, crop)
        ffmpeg([*loop, "-ss", f"{start:.3f}", "-i", str(src), "-t", f"{dur:.3f}", "-an", "-filter_complex", fc,
                "-map", "[v]", *self.enc(), str(out)], cwd=self.work)
        return out

    def broll(self, dur: float, sec: dict, pool: list, clip_srcs: list, shot_len: float) -> list[Path]:
        """Shots that fill `dur` seconds. A section with its own clips uses only those (muted, taking turns, moving
        through each clip so the same stretch isn't repeated); a section without clips uses stock footage."""
        n = max(1, round(dur / shot_len))
        lens = [dur / n] * n
        shots = []
        if clip_srcs:
            # Each clip: {"f", "len", "lo", "hi", "avoid", "k"}: only its lo..hi window (the time range set on the
            # clip, or all of it) is used, away from the part played with sound, stepping forward each shot.
            # A clip never plays past the end of its part: when its part is shorter than the shot, the next clip
            # carries on for the rest of the shot.
            for d in lens:
                left = d
                while left > 0.04:
                    c = clip_srcs[(sec.setdefault("_turn", 0)) % len(clip_srcs)]
                    sec["_turn"] += 1
                    lo, hi, avoid = c["lo"], c["hi"], c["avoid"]
                    micro = self.o.get("micro") or 0  # micro cuts: no piece of a clip longer than this
                    seg = min(left, hi - lo, micro or left)
                    if left - seg < 0.4:  # don't leave a sliver for the next clip; this one takes it (repeating)
                        seg = left if hi - lo >= left - 0.4 else seg
                    edge = 0.3 if hi - lo > seg + 0.6 else 0.0
                    spots = [t for t in frange(lo + edge, hi - seg - edge, seg) if not (avoid and avoid[0] - seg < t < avoid[1])]
                    spots = spots or list(frange(lo + edge, hi - seg - edge, seg)) or [lo]
                    if not c["picked"]:  # no part chosen: its liveliest stretches first
                        rank = self.ranked(c, lo + edge, hi - edge, seg)
                        spots = sorted(spots, key=lambda t: rank.index(t) if t in rank else len(rank))
                    start = spots[c["k"] % len(spots)]
                    c["k"] += 1
                    if start + seg > hi + 0.05:  # only when the sliver above made it longer than the part
                        shots += self.clip_run(c, seg)
                    else:
                        shots.append(self.render_clip_shot(c["f"], start, seg, c["len"], c.get("crop")))
                    left -= seg
            return shots
        for d in lens:
            if not pool:
                pool.extend(self.stock_pool(["city night", "people walking", "sunset"], False))
            fresh = [v for v in pool if v["id"] not in self.used_stock] or pool
            v = fresh[0]
            self.used_stock.add(v["id"])
            pool.remove(v)
            pool.append(v)  # reused only once every clip in the pool has been seen
            shots.append(self.render_stock_shot(v, d))
        return shots

    def clip_run(self, c: dict, length: float) -> list[Path]:
        """`length` seconds of one clip played straight from the start of its part (repeating the part if short)."""
        win, out, left = max(0.5, c["hi"] - c["lo"]), [], length
        micro = self.o.get("micro") or 0
        if micro and length > micro + 0.4 and not c["picked"] and win > micro * 2:
            # Micro cuts: the clip's liveliest micro-long moments (no part chosen), shown in the clip's order.
            n = -(-int(length * 1000) // int(micro * 1000))  # pieces needed (rounded up)
            best = self.ranked(c, c["lo"], c["hi"], micro)[:n]
            best = sorted(best) * (-(-n // max(1, len(best))))  # a short clip repeats its moments
            for i in range(n):
                d = min(micro, left)
                if left - d < 0.4:
                    d = left
                out.append(self.render_clip_shot(c["f"], best[i], d, c["len"], c.get("crop")))
                left -= d
                if left <= 0.01:
                    break
            return out
        start = c["lo"]
        if not c["picked"] and win > length + 0.5:  # no part chosen: the liveliest stretch that long
            start = self.ranked(c, c["lo"], c["hi"], length, 0.5)[0]
            win = c["hi"] - start
        while left > 0.01:
            d = min(left, win)
            out.append(self.render_clip_shot(c["f"], start, d, c["len"], c.get("crop")))
            left -= d
        return out

    def ranked(self, c: dict, lo: float, hi: float, seg: float, step: float | None = None) -> list[float]:
        """Start times of seg-long stretches of clip c between lo and hi, liveliest first."""
        if c.get("scores") is None:
            try:
                c["scores"] = clip_scores(c["f"])
            except Exception:
                c["scores"] = []
        return best_starts(c["scores"], lo, hi, seg, step) if c["scores"] else list(frange(lo, max(lo, hi - seg), step or seg)) or [lo]

    def broll_timed(self, dur: float, sec: dict, pool: list, srcs: list, shot_len: float,
                    segs: list[tuple[float, float, dict]]) -> list[Path]:
        """Shots for one narration line where some clips are tied to sentences ("frases 2-3"): each of those plays
        while its sentences are spoken; the rest of the line is filled by the other clips taking turns."""
        if not segs:
            return self.broll(dur, sec, pool, srcs, shot_len)
        free = [c for c in srcs if not c.get("say")] or srcs
        shots, t = [], 0.0
        for a, b, c in sorted(segs, key=lambda x: x[0]):
            a = max(a, t)  # two clips on the same sentence: the first one keeps it
            if b - a < 0.3:
                continue
            if a - t > 0.05:
                shots += self.broll(a - t, sec, pool, free, shot_len)
            shots += self.clip_run(c, b - a)
            t = b
        if dur - t > 0.05:
            shots += self.broll(dur - t, sec, pool, free, shot_len)
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

    def insert_piece(self, idx: int, src: Path, start: float, dur: float, crop: list | None = None) -> Path:
        out = self.work / f"p{idx:03d}.mp4"
        a = "[0:a]aresample=48000,aformat=channel_layouts=stereo,afade=t=in:d=0.15,afade=t=out:st={0}:d=0.4,apad[a]".format(max(0, dur - 0.4))
        if not has_audio(src):
            a = "anullsrc=r=48000:cl=stereo[a]"
        ffmpeg(["-ss", f"{start:.3f}", "-i", str(src), "-t", f"{dur:.3f}", "-filter_complex", self.shot_fc(src, start, dur, crop) + ";" + a,
                "-map", "[v]", "-map", "[a]", "-t", f"{dur:.3f}", *self.enc(), "-c:a", "aac", "-b:a", "192k", "-ar", "48000", str(out)],
               cwd=self.work)
        return out

    # --- effects: a visual hit and a sound at the start of a section (synthesized, no files needed) ---
    SFX = {
        "boom": "aevalsrc='0.9*sin(2*PI*(70-30*t)*t)*exp(-4*t)':d=1.2",
        "hit": "aevalsrc='(0.6*sin(2*PI*110*t)+0.45*(random(0)*2-1))*exp(-14*t)':d=0.5",
        "whoosh": "anoisesrc=d=0.7:c=pink:a=0.8,bandpass=f=900:w=1400,volume='sin(PI*t/0.7)':eval=frame",
        "riser": "aevalsrc='0.45*sin(2*PI*(200*t+350*t*t))*(t/1.3)':d=1.3",
        "ding": "aevalsrc='0.5*sin(2*PI*1320*t)*exp(-5*t)+0.2*sin(2*PI*2640*t)*exp(-7*t)':d=1.2",
        "pop": "aevalsrc='0.8*sin(2*PI*(900-3000*t)*t)*exp(-40*t)':d=0.15",
        "glitch": "aevalsrc='0.35*sin(2*PI*180*t)/abs(sin(2*PI*180*t)+0.0001)*lt(mod(t\\,0.08)\\,0.04)':d=0.4",
    }

    def fx_vf(self, v: str, every: float = 0, dur: float = 0) -> str:
        """The visual effect at the start of the piece, or every `every` seconds through it."""
        W, H = self.W, self.H
        T = f"mod(t\\,{every})" if every else "t"  # the effect's own clock: restarts every `every` seconds
        if v == "flash":
            if every:  # a white layer whose opacity jumps up every `every` s and fades (computed small, then scaled)
                return (f"color=white:s=16x16:r={FPS}:d={dur + 1:.2f},format=rgba,"
                        f"geq=r=255:g=255:b=255:a='255*max(0\\,1-mod(T\\,{every})/0.35)',scale={W}:{H}[w];"
                        f"[0:v][w]overlay=shortest=1[v0]")
            return (f"color=white:s={W}x{H}:r={FPS}:d=0.35,format=rgba,fade=t=out:st=0:d=0.35:alpha=1[w];"
                    f"[0:v][w]overlay=eof_action=pass[v0]")
        return {
            "punch": f"[0:v]scale=w='{W}*(1+0.18*max(0\\,1-{T}/0.5)^3)':h=-2:eval=frame,crop={W}:{H}[v0]",
            "push": f"[0:v]scale=w='{W}*(1+0.08*min({T}\\,3)/3)':h=-2:eval=frame,crop={W}:{H}[v0]",
            "shake": f"[0:v]scale={W * 108 // 100}:-2,crop={W}:{H}:x='(iw-ow)/2+(iw-ow)/2*sin(t*70)*max(0\\,1-{T}/0.5)'"
                     f":y='(ih-oh)/2+(ih-oh)/2*cos(t*53)*max(0\\,1-{T}/0.5)'[v0]",
            "glitch": f"[0:v]rgbashift=rh=18:bh=-18:gv=6:enable='lt({T}\\,0.4)',noise=alls=45:allf=t:enable='lt({T}\\,0.4)'[v0]",
        }[v]

    def apply_fx(self, piece: Path, fx: dict, every: float = 0) -> Path:
        """The section's effect on a piece: once at its start, or every `every` seconds through it."""
        v, s = fx.get("v") or "", fx.get("s") or ""
        if v not in FX_VIS:
            v = ""
        if s not in self.SFX:
            s = ""
        if not v and not s:
            return piece
        out = piece.with_name(piece.stem + "-fx.mp4")
        dur = duration(piece)
        fc, args = [], ["-i", piece.name]
        if v:
            fc.append(self.fx_vf(v, every, dur) + f";[v0]setsar=1,format=yuv420p[v]")
        if s:
            args += ["-f", "lavfi", "-i", self.SFX[s]]
            loop = (f",apad=whole_dur={every},atrim=0:{every},aloop=loop=-1:size={int(every * 48000)}" if every else "")
            fc.append(f"[1:a]aresample=48000,aformat=channel_layouts=stereo,volume=0.8{loop}[f];"
                      "[0:a][f]amix=inputs=2:duration=first:normalize=0,alimiter=limit=0.95[a]")
        ffmpeg([*args, "-filter_complex", ";".join(fc), "-map", "[v]" if v else "0:v", "-map", "[a]" if s else "0:a",
                *(self.enc() if v else ["-c:v", "copy"]), "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
                "-t", f"{dur:.3f}", out.name], cwd=self.work)
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

        # 3. Stock search words per section (a section with its own clips uses only those, no stock).
        for k, s in enumerate(secs):
            if s["clips"]:
                s["queries"], s["spanish"] = [], False
                continue
            self.step("planning", 0.30 + 0.05 * k / len(secs), s["heading"])
            q = ollama_queries(s, self.orientation) if o.get("ai", True) else []
            s["queries"], s["spanish"] = (q, False) if q else (fallback_queries(s), True)

        # 4. Render each section, piece by piece.
        pieces, timeline, srt, t = [], [], [], 0.0
        total = sum(len(s["pieces"]) for s in secs)
        done = 0
        for si, s in enumerate(secs):
            pool = self.stock_pool(s["queries"], s["spanish"]) if not s["clips"] else []
            if not pool and not s["clips"]:
                pool = self.stock_pool(["city night", "people walking", "sunset"], False)
            srcs = []
            for c in s["clips"]:
                f = files[c["id"]]
                length = duration(f)
                lo, hi = c.get("range") or (0.0, length)
                lo = max(0.0, min(lo, length - 1))
                hi = max(lo + 1, min(hi, length))
                srcs.append({"f": f, "len": length, "lo": lo, "hi": hi, "avoid": None, "k": 0, "say": c.get("say"),
                             "crop": c.get("crop"), "picked": bool(c.get("range")), "scores": None})
            shot_len = o["shot"] * (0.5 if si == 0 and s["n"] == 1 else 1.0)  # quick cuts for the opening
            sec_start, overlay_left = t, s["overlay"]
            ins_n = 0
            sent_n = 0  # sentences of the section's narration already spoken (for clips tied to "frases N-M")
            for pi, p in enumerate(s["pieces"]):
                self.step("rendering", 0.35 + 0.55 * done / total, f"{s['heading']} · part {pi + 1} of {len(s['pieces'])}")
                if p["type"] == "say":
                    spans = sentence_spans(p["words"], p["dur"])
                    first, last = sent_n + 1, sent_n + len(spans)
                    segs = []
                    for c in srcs:
                        if c.get("say") and c["say"][1] >= first and c["say"][0] <= last:
                            a, b = max(c["say"][0], first), min(c["say"][1], last)
                            segs.append((spans[a - first][0], spans[b - first][1], c))
                    sent_n = last
                    shots = self.broll_timed(p["dur"], s, pool, srcs, shot_len, segs)
                    path = self.say_piece(len(pieces), p["audio"], p["dur"], shots, overlay_left)
                    overlay_left = ""
                    for w in p["words"]:
                        srt.append((t + w["s"], t + w["e"], w["w"]))
                    t += p["dur"]
                elif srcs:
                    k = p["clip"] if p["clip"] is not None and p["clip"] < len(srcs) else ins_n % len(srcs)
                    c = srcs[k]
                    f, length = c["f"], c["len"]
                    ins_n += 1
                    # A range in the narration is in the clip's own time; otherwise the clip's window
                    # (all of it with "completo", else its first `insert` seconds).
                    a, b = p["range"] or ((c["lo"], c["hi"]) if p["whole"] else (c["lo"], min(c["hi"], c["lo"] + o["insert"])))
                    if not p["range"] and not p["whole"] and not c["picked"] and c["hi"] - c["lo"] > o["insert"] + 1:
                        a = self.ranked(c, c["lo"], c["hi"], o["insert"], 1.0)[0]  # no part chosen: its liveliest stretch
                        b = a + o["insert"]
                    a, b = max(0.0, min(a, length - 1)), max(a + 1, min(b, length))
                    c["avoid"] = (a, b)
                    path = self.insert_piece(len(pieces), f, a, b - a, c.get("crop"))
                    t += b - a
                else:  # a stage direction with no clip linked: a short pause on B-roll
                    sil = self.work / f"sil{len(pieces)}.wav"
                    ffmpeg(["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", "-t", "1.2", str(sil)])
                    path = self.say_piece(len(pieces), sil, 1.2, self.broll(1.2, s, pool, srcs, shot_len), "")
                    t += 1.2
                if s.get("fx") and s["fx"].get("every") and p["type"] == "say":
                    path = self.apply_fx(path, s["fx"], s["fx"]["every"])  # repeated through the narration
                elif pi == 0 and s.get("fx"):
                    path = self.apply_fx(path, s["fx"])
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
                "fit": body.get("fit") if body.get("fit") in ("fit", "follow") else "fill",
                "fitv": 2,  # saved after "follow faces" existed (older "fill" choices are read as "follow")
                # Micro cuts: no piece of one of your clips lasts longer than this (0 = off). Default 3 s.
                "micro": max(0.0, min(10.0, float(body.get("micro") if body.get("micro") is not None else 3))),
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
            "fx": {"visual": FX_VIS, "sound": FX_SND},
            "sections": [{"n": s["n"], "heading": s["heading"], "overlay": s["overlay"], "planned": s["planned"], "fx": s["fx"],
                          "clips": [{"id": c["id"], "label": c["label"], "kind": c["kind"],
                                     "missing": c["kind"] == "upload" and not any(CACHE.glob(c["id"] + ".*")),
                                     "raws": c["raws"], "range": c.get("range"), "say": c.get("say"),
                                     "crop": c.get("crop")} for c in s["clips"]],
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


# Section previews made on this run of the server, so a newer preview of a section can delete the old file.
_previews: set[str] = set()


@router.post("/api/guide/preview")
def preview_section(body: dict):
    """Makes one section on its own (voice, clips, stock, effect, music) to check it before the whole video.
    Poll /api/guide/{job}; the finished file plays from /api/media/{media}. `replace` deletes an older preview."""
    try:
        guide = parse_guide(str(body.get("text") or "")[:60000])
    except ValueError as e:
        raise HTTPException(400, str(e))
    n = int(body.get("n") or 0)
    sec = next((s for s in guide["sections"] if s["n"] == n), None)
    if not sec:
        raise HTTPException(404, f"No section {n} in the guide.")
    if not sec["clips"] and not stock.pexels_key():
        raise HTTPException(400, "Add your free Pexels API key first (Stock video dialog), or add your own clips to this section.")
    old = str(body.get("replace") or "")
    if old in _previews:
        _previews.discard(old)
        for ext in (".mp4", ".srt"):
            (fetch.DOWNLOADS / f"{old}{ext}").unlink(missing_ok=True)
    job_id = uuid.uuid4().hex
    guide_jobs[job_id] = {"status": "voice", "progress": 0.0, "preview": True}
    one = {"title": f"{guide['title']} · {n}. {sec['heading']}", "sections": [sec]}
    threading.Thread(target=run_preview, args=(job_id, one, options(body)), daemon=True).start()
    return {"job": job_id}


def run_preview(job_id: str, guide: dict, opts: dict):
    job = guide_jobs[job_id]
    b = Builder(job, guide, opts)
    try:
        r = b.run()
        _previews.add(r["media"])
        job.update(status="done", progress=1.0, result=r)
    except Exception as e:
        cancelled = str(e) == "cancelled"
        job.update(status="error", error="Cancelled." if cancelled else str(e), cancelled=cancelled)
    finally:
        shutil.rmtree(b.work, ignore_errors=True)


@router.post("/api/guides/draft")
def save_draft(body: dict):
    """Saves the guide and its settings without making the video. Send `id` to update a draft (or a guide's text)
    already in the list; a draft is made later with "Create the video" (which replaces it in place)."""
    text = str(body.get("text") or "")[:60000]
    if not text.strip():
        raise HTTPException(400, "The guide is empty.")
    try:
        g = parse_guide(text)
        title, n = g["title"], len(g["sections"])
    except ValueError:  # a draft may be unfinished
        m = re.search(r"^\s*(?:t[ií]tulo(?: de trabajo)?|title)\s*:\s*(.+)$", text, re.I | re.M)
        title, n = (m.group(1).strip().strip('"“”«»').strip() if m else "") or "Untitled draft", 0
    gid, opts, now = str(body.get("id") or ""), options(body), time.time()
    _guides_seed()
    with _guides_lock:
        items = _guides_load()
        # The guide being edited, or else the one in the list with the same title: saving overwrites it, so one
        # video has one entry. A made video stays playable (as its previous version) until the new one is made.
        it = next((x for x in items if x["id"] == gid), None) if gid else None
        if it is None:
            it = next((x for x in items if (x.get("title") or "").strip().lower() == title.strip().lower()
                       and x.get("status") != "running"), None)
        if it is not None and it.get("status") == "running":
            raise HTTPException(409, "That video is still being made. Save the draft when it's done.")
        if it is None:
            it = {"id": uuid.uuid4().hex, "created": now}
            items.insert(0, it)
        elif it.get("result"):
            it["prev_result"] = it.pop("result")
        it.pop("error", None)
        it.update(title=title, status="draft", format=opts["format"], sections=n, text=text, options=opts, edited=now)
        _guides_save(items)
    return {"id": it["id"], "title": title, "sections": n}


@router.post("/api/guide/video")
async def add_video(request: Request, name: str = ""):
    """Your own footage for a guide. The body is the raw video (or photo) file; the page links it as [name](video:<id>).
    A photo is turned into a short clip with a slow zoom, so it's used like any other clip."""
    ext = Path(name).suffix.lower()
    if ext in IMAGE_EXT:
        return await add_photo(request, name, ext)
    if ext not in VIDEO_EXT:
        raise HTTPException(400, "Use a video or photo file: MP4, MOV, MKV, WEBM, JPG, PNG, WEBP…")
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


async def add_photo(request: Request, name: str, ext: str) -> dict:
    CACHE.mkdir(parents=True, exist_ok=True)
    vid = uuid.uuid4().hex
    src, png, dst = CACHE / f"photo-{vid}{ext}", CACHE / f"photo-{vid}.png", CACHE / f"up-{vid}.mp4"
    try:
        with src.open("wb") as f:
            async for chunk in request.stream():
                f.write(chunk)
        await asyncio.to_thread(photo_clip, src, png, dst)
    except (OSError, ValueError, RuntimeError) as e:
        dst.unlink(missing_ok=True)
        raise HTTPException(400, f"Couldn't use that photo: {e}")
    finally:
        src.unlink(missing_ok=True)
        png.unlink(missing_ok=True)
    label = re.sub(r"[\[\]()\x00-\x1f]+", " ", Path(name).stem).strip()[:60] or "Mi foto"
    return {"id": vid, "name": label, "duration": round(duration(dst), 2), "link": f"[{label}](video:{vid})", "photo": True}


def photo_clip(src: Path, png: Path, dst: Path):
    """A PHOTO_SECONDS clip of a photo at its own shape (the edit fits it over a blurred copy), slowly zooming in,
    with a silent sound track so it also works as an insert."""
    from PIL import Image, ImageOps
    with Image.open(src) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")  # phone photos: apply the rotation they were taken with
        k = 1920 / max(im.size)
        w, h = (max(2, round(im.width * k / 2) * 2), max(2, round(im.height * k / 2) * 2))
        im.resize((w, h), Image.LANCZOS).save(png)
    frames = PHOTO_SECONDS * FPS
    vf = (f"scale={w * 2}:{h * 2},zoompan=z='1+0.08*on/{frames}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
          f":d={frames}:s={w}x{h}:fps={FPS},format=yuv420p")
    ffmpeg(["-i", str(png), "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo", "-vf", vf, "-frames:v", str(frames),
            "-t", str(PHOTO_SECONDS), "-c:v", "libx264", "-preset", "veryfast", "-crf", "19", "-c:a", "aac", "-shortest",
            str(dst)])


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


_voice_len: dict[tuple, tuple] = {}  # (text, voice, rate) -> (seconds with the breath, word timings)


def sentence_spans(words: list[dict], dur: float) -> list[tuple[float, float, str]]:
    """A narration line's sentences on its own clock: (start, end, text). Each runs until the next one starts
    (the first from 0, the last to the end of the line), so together they cover the whole line. The render and
    the part picker both use this, so "frases 2-3" means the same sentences in both."""
    groups, cur = [], []
    for w in words:
        cur.append(w)
        if re.search(r"[.!?…]['\"”»)]*$", w["w"]):
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)
    out = []
    for i, g in enumerate(groups):
        start = 0.0 if i == 0 else g[0]["s"]
        end = groups[i + 1][0]["s"] if i + 1 < len(groups) else dur
        out.append((start, end, " ".join(w["w"] for w in g)))
    return out


@router.post("/api/guide/voice-length")
def voice_length(body: dict):
    """How long the voice talks in section n, and its sentences with their times (same voice and speed as the
    video, measured by making the speech). Inserts are not counted in the times."""
    try:
        guide = parse_guide(str(body.get("text") or "")[:60000])
    except ValueError as e:
        raise HTTPException(400, str(e))
    n = int(body.get("n") or 0)
    sec = next((s for s in guide["sections"] if s["n"] == n), None)
    if not sec:
        raise HTTPException(404, f"No section {n} in the guide.")
    o = options(body)
    total, sentences = 0.0, []
    with tempfile.TemporaryDirectory(prefix="clipstudio_voice_") as tmp:
        for i, p in enumerate(x for x in sec["pieces"] if x["type"] == "say"):
            key = (p["text"], o["voice"], o["rate"])
            if key not in _voice_len:
                mp3 = Path(tmp) / f"v{i}.mp3"
                try:
                    words = speak(p["text"], o["voice"], o["rate"], mp3)
                except Exception as e:
                    raise HTTPException(502, f"Couldn't make the voice: {e}")
                _voice_len[key] = (duration(mp3) + 0.45, words)  # the same breath the video leaves after each line
            dur, words = _voice_len[key]
            for a, b, txt in sentence_spans(words, dur):
                sentences.append({"n": len(sentences) + 1, "text": txt, "start": round(total + a, 2),
                                  "end": round(total + b, 2), "seconds": round(b - a, 2)})
            total += dur
    return {"seconds": round(total, 2), "inserts": sum(p["type"] == "insert" for p in sec["pieces"]),
            "clips": len(sec["clips"]), "sentences": sentences}


CLIP_ID = re.compile(r"^(up-[0-9a-f]{32}|yt-[\w-]{11}|web-[0-9a-f]{16})$")


@router.post("/api/guide/clip")
def prepare_clip(body: dict):
    """Gets a section's clip ready to watch in the part picker (downloads a link once, like the render does)."""
    clip = clip_from_link(str(body.get("url") or "").strip())
    if not clip:
        raise HTTPException(400, "That isn't a clip link.")
    if clip["kind"] == "upload":
        f = next((x for x in CACHE.glob(clip["id"] + ".*") if ".part" not in x.name), None)
        if not f:
            raise HTTPException(404, "This video isn't on this computer any more. Add it again.")
    else:
        try:
            f = download_clip(clip)
        except RuntimeError as e:
            raise HTTPException(400, str(e))
    return {"id": clip["id"], "src": f"/api/guide/clip/{clip['id']}", "duration": round(duration(f), 2)}


@router.get("/api/guide/clip/{cid}")
def serve_clip(cid: str):
    if not CLIP_ID.match(cid):
        raise HTTPException(404, "No such clip.")
    f = next((x for x in CACHE.glob(cid + ".*") if ".part" not in x.name), None)
    if not f:
        raise HTTPException(404, "No such clip.")
    return FileResponse(f, media_type="video/mp4" if f.suffix.lower() in (".mp4", ".m4v") else None)


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
