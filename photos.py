"""Photo posts: pick photos (Pexels, your own uploads or AI-generated), get AI captions and post them everywhere.

How each platform gets the post (their official APIs decide this):
  - Facebook Page: a real photo post (one photo, or several in one post), uploaded straight from this machine.
  - Instagram: a real photo or carousel post. Instagram only fetches photos from a public web address, so Pexels
    photos go by their Pexels address, and your own / AI photos are hosted on your connected Facebook Page as
    unpublished photos (deleted again once Instagram has them). With neither, it gets a slideshow Reel instead.
  - TikTok and YouTube: a vertical slideshow video of the photos (TikTok's photo posts need a public domain
    verified with TikTok; YouTube has no API for photo posts), published like any clip.

Photos live in data/photos/ (index.json + one JPEG each).
"""
import base64
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

import music
import publish
import stock
import viral_image
from publish import GRAPH, PublishError, http

DIR = publish.DATA / "photos"
INDEX = DIR / "index.json"
PHOTO_ID = re.compile(r"^[0-9a-f]{32}$")
MAX_UPLOAD = 30 * 1024 * 1024
MAX_PHOTOS = 10  # Instagram carousels take up to 10
SD_URL = os.environ.get("SD_URL", "http://127.0.0.1:7860")  # Stable Diffusion WebUI (A1111 / Forge) with --api
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
UA = {"User-Agent": "ClipStudio/1.0"}

router = APIRouter()
_lock = threading.Lock()
_gen_jobs: dict[str, dict] = {}
_slides: dict[str, Path] = {}  # slideshow cache: one render serves every account it's posted to
_slides_lock = threading.Lock()


# ---------- library ----------
def _load() -> list[dict]:
    try:
        return json.loads(INDEX.read_text("utf-8"))
    except (OSError, ValueError):
        return []


def _store(items: list[dict]):
    DIR.mkdir(parents=True, exist_ok=True)
    tmp = INDEX.with_suffix(".tmp")
    tmp.write_text(json.dumps(items, ensure_ascii=False, indent=1), "utf-8")
    tmp.replace(INDEX)


def get_photo(pid: str) -> dict:
    p = next((p for p in _load() if p["id"] == pid), None) if PHOTO_ID.match(pid or "") else None
    if not p or not (DIR / p["file"]).exists():
        raise PublishError("A photo of this post was removed from the library. Pick it again.")
    return p


def photo_path(p: dict) -> Path:
    return DIR / p["file"]


def _size(path: Path) -> tuple[int, int]:
    res = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                          "-of", "csv=p=0:s=x", str(path)], capture_output=True, text=True)
    try:
        w, h = res.stdout.strip().split("x")[:2]
        return int(w), int(h)
    except ValueError:
        raise HTTPException(400, "That file isn't an image this app can read. Use JPG, PNG or WebP.")


def _to_jpeg(src: Path, dst: Path, long_side: int = 2160):
    """Any image ffmpeg can read -> sRGB JPEG, longest side at most `long_side`."""
    scale = f"scale='if(gt(iw,ih),min({long_side},iw),-2)':'if(gt(iw,ih),-2,min({long_side},ih))'"
    res = subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(src), "-frames:v", "1", "-vf", scale + ",format=yuvj420p",
                          "-q:v", "2", str(dst)], capture_output=True, text=True)
    if res.returncode != 0 or not dst.exists():
        raise HTTPException(400, "Couldn't read that image: " + (res.stderr.strip()[-200:] or "unknown format"))


def _add(src: Path, source: str, **meta) -> dict:
    """Store an image in the library and return its record."""
    DIR.mkdir(parents=True, exist_ok=True)
    pid = uuid.uuid4().hex
    dst = DIR / f"{pid}.jpg"
    _to_jpeg(src, dst)
    w, h = _size(dst)
    item = {"id": pid, "file": dst.name, "source": source, "w": w, "h": h, "added": int(time.time()),
            **{k: v for k, v in meta.items() if v}}
    with _lock:
        _store([item] + _load())
    return item


def _public(p: dict) -> dict:
    return {k: p.get(k) for k in ("id", "source", "w", "h", "added", "alt", "credit", "credit_url", "prompt")} | {
        "src": f"/api/photos/file/{p['id']}"}


@router.get("/api/photos")
def library():
    return {"photos": [_public(p) for p in _load() if (DIR / p["file"]).exists()]}


@router.get("/api/photos/file/{pid}")
def photo_file(pid: str):
    try:
        return FileResponse(photo_path(get_photo(pid)), media_type="image/jpeg")
    except PublishError as e:
        raise HTTPException(404, str(e))


@router.post("/api/photos/{pid}/delete")
def delete_photo(pid: str):
    with _lock:
        items = _load()
        p = next((x for x in items if x["id"] == pid), None)
        if not p:
            raise HTTPException(404, "unknown photo")
        (DIR / p["file"]).unlink(missing_ok=True)
        _store([x for x in items if x["id"] != pid])
    return {"ok": True}


@router.post("/api/photos/upload")
async def upload_photo(request: Request, name: str = ""):
    """The body is the raw image file."""
    work = Path(tempfile.mkdtemp(prefix="clipstudio_photo_"))
    src, size = work / "upload.bin", 0
    try:
        with src.open("wb") as f:
            async for chunk in request.stream():
                size += len(chunk)
                if size > MAX_UPLOAD:
                    raise HTTPException(413, "That image is over 30 MB.")
                f.write(chunk)
        if not size:
            raise HTTPException(400, "empty upload")
        return _public(_add(src, "upload", alt=Path(name).stem[:120] or None))
    finally:
        shutil.rmtree(work, ignore_errors=True)


# ---------- Pexels ----------
def _pexels_key() -> str:
    key = stock.pexels_key()
    if not key:
        raise HTTPException(400, "Add your free Pexels API key first (pexels.com/api).")
    return key


@router.get("/api/photos/search")
def search(q: str = "", page: int = 1, orientation: str = ""):
    q = q.strip()[:80]
    if not q:
        raise HTTPException(400, "Type what the photos should show, e.g. sunset beach.")
    params = {"query": q, "per_page": 30, "page": max(1, min(50, page))}
    if orientation in ("portrait", "landscape", "square"):
        params["orientation"] = orientation
    try:
        r = stock.pexels("/v1/search", params, _pexels_key())
    except RuntimeError as e:
        raise HTTPException(400, str(e))
    return {"total": r.get("total_results", 0), "page": params["page"], "results": [{
        "pexels_id": p["id"], "w": p["width"], "h": p["height"], "alt": p.get("alt") or "",
        "thumb": p["src"]["medium"], "original": p["src"]["original"],
        "credit": p.get("photographer"), "credit_url": p.get("photographer_url"), "url": p.get("url"),
    } for p in r.get("photos", [])]}


def _pexels_url(url: str) -> str:
    u = urllib.parse.urlparse(str(url or ""))
    if u.scheme != "https" or u.netloc != "images.pexels.com":
        raise HTTPException(400, "Not a Pexels photo.")
    return f"https://images.pexels.com{u.path}"  # without size parameters


@router.post("/api/photos/pexels")
def add_pexels(body: dict):
    """Save a Pexels search result in the library (its public address is kept for Instagram)."""
    base = _pexels_url(body.get("original"))
    existing = next((p for p in _load() if p.get("public_url") == base and (DIR / p["file"]).exists()), None)
    if existing:
        return _public(existing)
    work = Path(tempfile.mkdtemp(prefix="clipstudio_pexels_"))
    try:
        src = work / "pexels.jpg"
        req = urllib.request.Request(base + "?auto=compress&cs=tinysrgb&fm=jpg&w=2160", headers=UA)
        try:
            with urllib.request.urlopen(req, timeout=60) as r, src.open("wb") as f:
                shutil.copyfileobj(r, f)
        except (urllib.error.URLError, OSError) as e:
            raise HTTPException(502, f"Couldn't download the photo from Pexels: {e}")
        return _public(_add(src, "pexels", public_url=base, alt=str(body.get("alt") or "")[:200],
                            credit=str(body.get("credit") or "")[:80], credit_url=str(body.get("credit_url") or "")[:200]))
    finally:
        shutil.rmtree(work, ignore_errors=True)


# ---------- AI images (local only) ----------
ASPECTS = {"4:5": (896, 1120), "1:1": (1024, 1024), "9:16": (768, 1344), "16:9": (1344, 768)}


def _get_json(url: str, payload: dict | None = None, timeout: float = 5) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode() if payload is not None else None,
                                 headers={"Content-Type": "application/json"}, method="POST" if payload is not None else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b"{}")


def _ollama(path: str, payload: dict | None = None, timeout: float = 5) -> dict:
    return _get_json(OLLAMA_URL + path, payload, timeout)


def _ollama_image_model() -> str | None:
    try:
        names = [m["name"] for m in _ollama("/api/tags")["models"]]
    except Exception:
        return None
    for n in names:
        try:
            caps = _ollama("/api/show", {"model": n}).get("capabilities", [])
        except Exception:
            continue
        if "image" in caps or "image_generation" in caps:
            return n
    return None


def image_provider() -> dict | None:
    try:
        _get_json(SD_URL + "/sdapi/v1/sd-models", timeout=3)
        return {"kind": "sd", "name": "Stable Diffusion WebUI"}
    except Exception:
        pass
    model = _ollama_image_model()
    return {"kind": "ollama", "name": model} if model else None


@router.get("/api/photos/ai")
def ai_status():
    p = image_provider()
    return {"available": bool(p), "provider": p and p["name"]}


def _free_ollama_vram():
    """Unload Ollama's models from the GPU: the vision model (~6 GB) and SDXL don't both fit in 12 GB,
    and SD then crawls (about 10x slower). Ollama reloads its model on the next AI caption request."""
    try:
        for m in _ollama("/api/ps").get("models", []):
            _ollama("/api/generate", {"model": m["name"], "keep_alive": 0}, timeout=30)
    except Exception:
        pass  # Ollama not running: nothing to free


def _generate(job: dict, prompt: str, aspect: str, provider: dict):
    w, h = ASPECTS.get(aspect, ASPECTS["4:5"])
    work = Path(tempfile.mkdtemp(prefix="clipstudio_ai_"))
    try:
        if provider["kind"] == "sd":
            job["status"] = "freeing GPU memory"
            _free_ollama_vram()
        job["status"] = f"generating with {provider['name']}"
        if provider["kind"] == "sd":
            r = _get_json(SD_URL + "/sdapi/v1/txt2img", {
                "prompt": prompt + ", highly detailed, sharp focus, professional photo",
                "negative_prompt": "text, watermark, logo, blurry, deformed, low quality, extra fingers",
                "width": w, "height": h, "steps": 28, "cfg_scale": 6}, timeout=900)
            data = (r.get("images") or [""])[0]
        else:
            r = _ollama("/api/generate", {"model": provider["name"], "prompt": prompt, "stream": False,
                                                "width": w, "height": h}, timeout=900)
            data = r.get("image") or (r.get("images") or [""])[0]
        raw = base64.b64decode(str(data).split(",", 1)[-1]) if data else b""
        if not raw:
            raise RuntimeError("the image model returned no image")
        src = work / "ai.png"
        src.write_bytes(raw)
        job.update(status="done", photo=_public(_add(src, "ai", prompt=prompt[:500])))
    except HTTPException as e:
        job.update(status="error", error=e.detail)
    except Exception as e:
        job.update(status="error", error=f"Image generation failed: {e}")
    finally:
        shutil.rmtree(work, ignore_errors=True)


@router.post("/api/photos/generate")
def generate(body: dict):
    prompt = str(body.get("prompt") or "").strip()[:800]
    if not prompt:
        raise HTTPException(400, "Describe the image to generate.")
    provider = image_provider()
    if not provider:
        raise HTTPException(503, "No local image model is running. Start Stable Diffusion WebUI (Automatic1111 or Forge) "
                                 "with the --api flag on port 7860, or install an Ollama image model, then try again.")
    job_id = uuid.uuid4().hex
    job = _gen_jobs[job_id] = {"status": "starting"}
    threading.Thread(target=_generate, args=(job, prompt, str(body.get("aspect") or "4:5"), provider), daemon=True).start()
    return {"job": job_id}


@router.get("/api/photos/generate/{job_id}")
def generate_status(job_id: str):
    job = _gen_jobs.get(job_id)
    if not job:
        raise HTTPException(404, "unknown job")
    if job["status"] in ("done", "error"):
        _gen_jobs.pop(job_id)
    return job


def thumb_b64(pid: str, side: int = 768) -> str:
    """A small JPEG of a library photo, base64, for the vision model."""
    p = get_photo(pid)
    with tempfile.TemporaryDirectory(prefix="clipstudio_th_") as d:
        out = Path(d) / "t.jpg"
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(photo_path(p)), "-vf",
                        f"scale='if(gt(iw,ih),{side},-2)':'if(gt(iw,ih),-2,{side})'", "-q:v", "4", str(out)], capture_output=True)
        return base64.b64encode(out.read_bytes()).decode()


# ---------- viral image: one designed image made from 1 or more post photos ----------
@router.post("/api/photos/design")
def design(body: dict):
    """Render a viral-style image (hook text, before/after, collage, top list, meme) and add it to the library."""
    ids = [str(x) for x in body.get("photos") or []][:5]
    if not ids:
        raise HTTPException(400, "Add at least one photo to the post first.")
    layout = body.get("layout") if body.get("layout") in viral_image.LAYOUTS else "hook"
    try:
        paths = [photo_path(get_photo(pid)) for pid in ids]
    except PublishError as e:
        raise HTTPException(400, str(e))
    labels = body.get("labels") or []
    if isinstance(labels, str):
        labels = [x for x in labels.split("|")]
    work = Path(tempfile.mkdtemp(prefix="clipstudio_design_"))
    try:
        out = viral_image.render(paths, work / "design.jpg", layout=layout, aspect=str(body.get("aspect") or "4:5"),
                                 text=str(body.get("text") or ""), accent=str(body.get("accent") or ""),
                                 labels=[str(x) for x in labels], position=str(body.get("position") or "top"),
                                 handle=str(body.get("handle") or ""), vs=bool(body.get("vs")))
        return _public(_add(out, "design", alt=str(body.get("text") or "")[:200] or None, layout=layout, made_from=ids))
    except (OSError, ValueError) as e:
        raise HTTPException(400, f"Couldn't make the image: {e}")
    finally:
        shutil.rmtree(work, ignore_errors=True)


# ---------- slideshow video (TikTok, YouTube, Instagram fallback) ----------
FPS, SW, SH, FADE = 30, 1080, 1920, 0.5


def _run(cmd: list[str], what: str):
    res = subprocess.run(["ffmpeg", "-y", "-v", "error", *cmd], capture_output=True, text=True)
    if res.returncode != 0:
        raise PublishError(f"Couldn't make the {what}: " + (res.stderr.strip()[-300:] or "ffmpeg failed"))


def slideshow(photos: list[dict], secs: float, track: str | None, job: dict) -> Path:
    """A 1080x1920 video: each photo over a blurred copy of itself with a slow zoom, crossfaded, with music."""
    key = hashlib.sha1(json.dumps([[p["id"] for p in photos], secs, track]).encode()).hexdigest()
    with _slides_lock:
        hit = _slides.get(key)
        if hit and hit.exists():
            return hit
        work = Path(tempfile.mkdtemp(prefix="clipstudio_slides_"))
        frames = round(secs * FPS)
        segs = []
        for i, p in enumerate(photos):
            job["status"] = f"making the slideshow (photo {i + 1} of {len(photos)})"
            seg = work / f"seg{i}.mp4"
            zoom_in = i % 2 == 0  # alternate zooming in and out so it doesn't feel mechanical
            z = f"1+0.07*on/{frames}" if zoom_in else f"1.07-0.07*on/{frames}"
            vf = (f"[0:v]split=2[a][b];[a]scale={SW}:{SH}:force_original_aspect_ratio=increase,crop={SW}:{SH},boxblur=40:4[bg];"
                  f"[b]scale={SW}:{SH}:force_original_aspect_ratio=decrease[fg];[bg][fg]overlay=(W-w)/2:(H-h)/2,"
                  f"scale={SW * 2}:{SH * 2},zoompan=z='{z}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={frames}:s={SW}x{SH}:fps={FPS},"
                  f"format=yuv420p[v]")
            _run(["-i", str(photo_path(p)), "-filter_complex", vf, "-map", "[v]", "-frames:v", str(frames),
                  "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", str(seg)], "slideshow")
            segs.append(seg)
        total = len(segs) * secs - (len(segs) - 1) * FADE
        inputs, chain, last = [], [], "0:v"
        for s in segs:
            inputs += ["-i", str(s)]
        for i in range(1, len(segs)):
            out = f"x{i}"
            chain.append(f"[{last}][{i}:v]xfade=transition=fade:duration={FADE}:offset={i * (secs - FADE):.3f}[{out}]")
            last = out
        n = len(segs)
        if track:
            inputs += ["-stream_loop", "-1", "-i", str(music.track_file(track))]
            chain.append(f"[{n}:a]atrim=0:{total:.3f},afade=t=out:st={max(0, total - 1.5):.3f}:d=1.5,volume=0.9[a]")
        else:
            inputs += ["-f", "lavfi", "-t", f"{total:.3f}", "-i", "anullsrc=r=48000:cl=stereo"]
            chain.append(f"[{n}:a]anull[a]")
        job["status"] = "joining the slideshow"
        out = work / "slideshow.mp4"
        _run([*inputs, "-filter_complex", ";".join(chain) if len(segs) > 1 or chain else "",
              "-map", f"[{last}]" if len(segs) > 1 else "0:v", "-map", "[a]", "-t", f"{total:.3f}",
              "-c:v", "libx264", "-preset", "veryfast", "-crf", "19", "-pix_fmt", "yuv420p", "-r", str(FPS),
              "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-movflags", "+faststart", str(out)], "slideshow")
        for s in segs:
            s.unlink(missing_ok=True)
        _slides[key] = out
        cleanup = threading.Timer(3600, lambda: (_slides.pop(key, None), shutil.rmtree(work, ignore_errors=True)))
        cleanup.daemon = True
        cleanup.start()
        return out


# ---------- Facebook: a real photo post ----------
def _multipart(fields: dict, file_field: str, file_path: Path) -> tuple[bytes, str]:
    boundary = "----clipstudio" + uuid.uuid4().hex
    parts = []
    for k, v in fields.items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
    parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{file_field}"; filename="photo.jpg"\r\n'
                 f"Content-Type: image/jpeg\r\n\r\n".encode() + file_path.read_bytes() + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def _fb_upload_photo(page: str, tok: str, path: Path, **fields) -> str:
    body, ctype = _multipart({**fields, "access_token": tok}, "source", path)
    _, _, r = http("POST", f"{GRAPH}/{page}/photos", raw=body, headers={"Content-Type": ctype}, timeout=300)
    if not r.get("id"):
        raise PublishError(f"Facebook didn't accept the photo: {r}")
    return r["id"]


def publish_facebook_photos(acct_id: str, photos: list[dict], f: dict, job: dict) -> dict:
    tok = publish.access_token(acct_id)
    acct = publish.get(acct_id)
    page = acct.get("fb_page_id")
    if not page:
        raise PublishError(f"{acct['label']} has no Facebook Page connected. Connect it again.")
    at = publish._schedule_at(f)
    ids = []
    for i, p in enumerate(photos):
        job["status"] = f"uploading photo {i + 1} of {len(photos)} to Facebook"
        # Unpublished photos, attached to one post below; "temporary" is what scheduled posts need.
        ids.append(_fb_upload_photo(page, tok, photo_path(p), published="false", **({"temporary": "true"} if at else {})))
    job["status"] = "scheduling on Facebook" if at else "posting on Facebook"
    form = {"message": (f.get("caption") or "")[:63000], "access_token": tok,
            **{f"attached_media[{i}]": json.dumps({"media_fbid": x}) for i, x in enumerate(ids)}}
    if at:
        form.update(published="false", scheduled_publish_time=str(at))
    _, _, r = http("POST", f"{GRAPH}/{page}/feed", form=form)
    post_id = r.get("id") or ""
    url = f"https://www.facebook.com/{post_id}" if post_id else None
    n = f"{len(photos)} photos" if len(photos) > 1 else "a photo"
    if at:
        return {"url": url, "scheduled_at": at, "message": f"Post with {n} scheduled on {acct.get('name') or 'your Page'} for {publish._when(at)}."}
    return {"url": url, "message": f"Posted {n} on {acct.get('name') or 'your Page'}."}


# ---------- Instagram: a real photo / carousel post ----------
IG_W, IG_H = 1080, 1350  # 4:5, the tallest Instagram allows; every carousel item gets the same shape


def _ig_crop(p: dict, work: Path) -> Path:
    out = work / f"ig_{p['id']}.jpg"
    _run(["-i", str(photo_path(p)), "-frames:v", "1", "-vf",
          f"scale={IG_W}:{IG_H}:force_original_aspect_ratio=increase,crop={IG_W}:{IG_H},format=yuvj420p", "-q:v", "2", str(out)],
         "Instagram photo")
    return out


def _photo_host():
    """A connected Facebook Page that can hold photos for Instagram to fetch: (account id, page id, token) or None."""
    for a in publish.load():
        if a["platform"] == "facebook" and a.get("token") and a.get("fb_page_id"):
            try:
                return a["id"], a["fb_page_id"], publish.access_token(a["id"])
            except PublishError:
                continue
    return None


def _ig_wait(container: str, tok: str, what: str):
    for _ in range(60):
        _, _, s = http("GET", f"{GRAPH}/{container}?" + urllib.parse.urlencode({"fields": "status_code,status", "access_token": tok}))
        if s.get("status_code") == "FINISHED":
            return
        if s.get("status_code") in ("ERROR", "EXPIRED"):
            raise PublishError(f"Instagram couldn't use {what}: {s.get('status') or s['status_code']}")
        time.sleep(3)
    raise PublishError(f"Instagram took too long to process {what}. Try again later.")


def publish_instagram_photos(acct_id: str, photos: list[dict], f: dict, job: dict) -> dict:
    tok = publish.access_token(acct_id)
    ig = publish.get(acct_id)["ig_user_id"]
    host = None if all(p.get("public_url") for p in photos) else _photo_host()
    if not host and not all(p.get("public_url") for p in photos):
        # No way to give Instagram a public address for these photos: post them as a slideshow Reel instead.
        video = slideshow(photos, float(f.get("slide_seconds") or 3), f.get("music") or None, job)
        res = publish.publish_instagram(acct_id, video, f, job)
        return {**res, "message": res["message"] + " (A slideshow Reel: photo posts of your own or AI photos need a "
                                                   "connected Facebook Page to host them; add one under Accounts.)"}
    work = Path(tempfile.mkdtemp(prefix="clipstudio_ig_"))
    hosted = []  # Facebook photo IDs to delete again afterwards
    try:
        urls = []
        for i, p in enumerate(photos):
            if p.get("public_url"):
                urls.append(p["public_url"] + f"?auto=compress&cs=tinysrgb&fm=jpg&fit=crop&w={IG_W}&h={IG_H}")
                continue
            job["status"] = f"preparing photo {i + 1} of {len(photos)} for Instagram"
            _, page, page_tok = host
            fid = _fb_upload_photo(page, page_tok, _ig_crop(p, work), published="false")
            hosted.append((fid, page_tok))
            _, _, info = http("GET", f"{GRAPH}/{fid}?" + urllib.parse.urlencode({"fields": "images", "access_token": page_tok}))
            images = sorted(info.get("images") or [], key=lambda x: -x.get("width", 0))
            if not images:
                raise PublishError("Facebook didn't return an address for a photo, so Instagram can't fetch it.")
            urls.append(images[0]["source"])
        caption = (f.get("caption") or "")[:2200]
        if len(urls) == 1:
            job["status"] = "Instagram is fetching the photo"
            _, _, c = http("POST", f"{GRAPH}/{ig}/media", form={"image_url": urls[0], "caption": caption, "access_token": tok})
            container = c["id"]
            _ig_wait(container, tok, "the photo")
        else:
            children = []
            for i, u in enumerate(urls):
                job["status"] = f"Instagram is fetching photo {i + 1} of {len(urls)}"
                _, _, c = http("POST", f"{GRAPH}/{ig}/media", form={"image_url": u, "is_carousel_item": "true", "access_token": tok})
                children.append(c["id"])
            for c in children:
                _ig_wait(c, tok, "a photo")
            _, _, c = http("POST", f"{GRAPH}/{ig}/media", form={"media_type": "CAROUSEL", "children": ",".join(children),
                                                                 "caption": caption, "access_token": tok})
            container = c["id"]
            _ig_wait(container, tok, "the carousel")
        job["status"] = "publishing on Instagram"
        _, _, pub = http("POST", f"{GRAPH}/{ig}/media_publish", form={"creation_id": container, "access_token": tok})
        _, _, m = http("GET", f"{GRAPH}/{pub['id']}?" + urllib.parse.urlencode({"fields": "permalink", "access_token": tok}))
        return {"url": m.get("permalink"), "message": f"Posted as {'a carousel of ' + str(len(urls)) + ' photos' if len(urls) > 1 else 'a photo'}."}
    finally:
        for fid, page_tok in hosted:
            try:
                http("DELETE", f"{GRAPH}/{fid}?" + urllib.parse.urlencode({"access_token": page_tok}))
            except PublishError:
                pass
        shutil.rmtree(work, ignore_errors=True)


# ---------- one post, any platform ----------
def publish_post(acct_id: str, photos: list[dict], f: dict, job: dict) -> dict:
    platform = publish.get(acct_id)["platform"]
    if platform == "facebook":
        return publish_facebook_photos(acct_id, photos, f, job)
    if platform == "instagram":
        return publish_instagram_photos(acct_id, photos, f, job)
    video = slideshow(photos, float(f.get("slide_seconds") or 3), f.get("music") or None, job)
    res = publish.PUBLISHERS[platform](acct_id, video, f, job)
    return {**res, "message": "Slideshow video. " + res["message"]}


def _run_scheduled(item: dict, job: dict) -> dict:
    """Called by publish's scheduler for queued photo posts."""
    return publish_post(item["account"], [get_photo(pid) for pid in item["photos"]], item["fields"], job)


publish.PHOTO_POST = _run_scheduled


@router.post("/api/photos/publish")
def publish_photos(body: dict):
    acct = publish._find(body.get("account", ""))
    platform = acct["platform"]
    ids = [str(x) for x in body.get("photos") or []][:MAX_PHOTOS]
    if not ids:
        raise HTTPException(400, "Add at least one photo to the post.")
    try:
        photos = [get_photo(pid) for pid in ids]
    except PublishError as e:
        raise HTTPException(400, str(e))
    if platform == "youtube" and not (body.get("title") or "").strip():
        raise HTTPException(400, "YouTube needs a title.")
    try:
        body["slide_seconds"] = max(2.0, min(8.0, float(body.get("slide_seconds") or 3)))
    except (TypeError, ValueError):
        body["slide_seconds"] = 3.0
    at = publish._schedule_at(body)
    if body.get("schedule_at") and not at:
        raise HTTPException(400, "The scheduled time isn't valid.")
    if at:
        if at < time.time() + (10 * 60 if platform == "facebook" else 60):
            raise HTTPException(400, f"The scheduled time is too soon for {publish.NAMES[platform]}"
                                     + (" (Facebook needs at least 10 minutes)." if platform == "facebook" else "."))
        if platform == "facebook" and at > time.time() + 29 * 86400:
            raise HTTPException(400, "Facebook can schedule posts up to 29 days ahead.")
    job_id = uuid.uuid4().hex
    job = publish._jobs[job_id] = {"status": "starting", "platform": platform, "account": acct["id"]}

    def run():
        try:
            if at and platform not in publish.NATIVE_SCHEDULE:
                job.update(status="done", **publish._queue_photos(acct, ids, body, at))
                return
            job.update(status="done", **publish_post(acct["id"], photos, body, job))
        except PublishError as e:
            job.update(status="error", error=str(e))
        except Exception as e:
            job.update(status="error", error=f"{publish.NAMES[platform]}: {type(e).__name__}: {e}")

    publish._executor.submit(run)
    return {"job": job_id}
