"""Clip Studio backend: serves the page and transcribes uploads with faster-whisper.

Run:  .venv\\Scripts\\python server.py      then open http://127.0.0.1:8000
Env:  WHISPER_MODEL (default "large-v3-turbo" on GPU, "small" on CPU), WHISPER_DEVICE ("auto" | "cuda" | "cpu"), PORT (default 8000)
"""
import itertools
import json
import os
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

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse

import fetch
import clone
import music
import narrate
import projects
import publish
import stock
import guide
import notify
import photos

ROOT = Path(__file__).parent
# large-v3-turbo catches far more dialog (short lines, speech over music/gunfire) but is slow on CPU,
# so it's the default only on GPU. WHISPER_MODEL overrides both.
MODEL_NAME = os.environ.get("WHISPER_MODEL", "")
GPU_MODEL, CPU_MODEL = MODEL_NAME or "large-v3-turbo", MODEL_NAME or "small"
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "")  # empty = auto-pick an installed vision model
DEVICE = os.environ.get("WHISPER_DEVICE", "auto")
# Second listening pass for voices quieter than the background (GPU only). QUIET_BOOST=0 turns it off for speed.
QUIET_BOOST = os.environ.get("QUIET_BOOST", "1") != "0"
CAPTION_DEBUG = os.environ.get("CAPTION_DEBUG") == "1"  # log every line the quiet-voice pass accepts

app = FastAPI(title="Clip Studio")
app.include_router(publish.router)
app.include_router(fetch.router)
app.include_router(narrate.router)
app.include_router(clone.router)
app.include_router(projects.router)
app.include_router(stock.router)
app.include_router(music.router)
app.include_router(guide.router)
app.include_router(notify.router)
app.include_router(photos.router)
jobs: dict[str, dict] = {}


@app.middleware("http")
async def local_only(request: Request, call_next):
    # The server holds your social-media logins, so only accept requests meant for this machine
    # (blocks DNS-rebinding) and require a custom header on writes (blocks cross-site form posts).
    host = (request.headers.get("host") or "").rsplit(":", 1)[0]
    if host not in ("127.0.0.1", "localhost", "[::1]"):
        return JSONResponse({"detail": "Clip Studio only accepts local requests."}, status_code=403)
    if request.method == "POST" and request.headers.get("x-clip-studio") != "1":
        return JSONResponse({"detail": "Missing X-Clip-Studio header."}, status_code=403)
    return await call_next(request)
executor = ThreadPoolExecutor(max_workers=1)  # one transcription at a time keeps GPU/CPU memory predictable
_model = None
_model_device = None
_model_lock = threading.Lock()


def add_cuda_dll_dirs():
    """Expose DLLs from the optional nvidia-cublas-cu12 / nvidia-cudnn-cu12 pip packages on Windows."""
    if os.name != "nt":
        return
    try:
        import nvidia
    except ImportError:
        return
    for base in nvidia.__path__:
        for bin_dir in Path(base).glob("*/bin"):
            os.add_dll_directory(str(bin_dir))
            os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ["PATH"]


add_cuda_dll_dirs()


def load_model(force_cpu: bool = False):
    global _model, _model_device
    from faster_whisper import WhisperModel

    with _model_lock:
        if _model is not None and not (force_cpu and _model_device != "cpu"):
            return _model
        if not force_cpu and DEVICE in ("auto", "cuda"):
            try:
                _model = WhisperModel(GPU_MODEL, device="cuda", compute_type="float16")
                _model_device = "cuda"
                return _model
            except Exception as e:  # missing CUDA libs, unsupported GPU, ...
                if DEVICE == "cuda":
                    raise
                print(f"CUDA unavailable ({e}); falling back to CPU")
        _model = WhisperModel(CPU_MODEL, device="cpu", compute_type="int8")
        _model_device = "cpu"
        return _model


HOP = 0.25  # seconds per loudness/motion frame; must match HOP in index.html
API_VERSION = 11  # bump when the API changes; index.html refuses to talk to a mismatched server


def has_stream(src: Path, kind: str) -> bool:
    cmd = ["ffprobe", "-v", "error", "-select_streams", kind, "-show_entries", "stream=index", "-of", "csv=p=0", str(src)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    return bool(res.stdout.strip())


def video_info(src: Path) -> dict:
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name,width,height", "-of", "json", str(src)]
    try:
        return json.loads(subprocess.run(cmd, capture_output=True, text=True).stdout)["streams"][0]
    except (ValueError, KeyError, IndexError):
        return {}


def probe_duration(src: Path) -> float:
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(src)]
    try:
        return float(subprocess.run(cmd, capture_output=True, text=True).stdout.strip())
    except ValueError:
        return 0.0


def set_part(job: dict, name: str, frac: float):
    """Track progress of the parallel analysis parts and expose their average as job progress."""
    job["parts"][name] = min(1.0, frac)
    job["progress"] = 0.08 + 0.9 * sum(job["parts"].values()) / len(job["parts"])


FACE_MODEL = ROOT / "models" / "face_detection_yunet_2023mar.onnx"


def analyze_frames(src: Path, duration: float, job: dict) -> tuple[list[float], dict]:
    """One decode pass, 4 frames a second, for two things:

    - motion: mean frame difference (action, crashes and cuts spike it), used to score moments;
    - track: where the interest is horizontally (0..1) in each frame, for auto-reframing vertical crops.
      Faces win, and the camera stays locked on one person per shot (switching only when another face
      clearly dominates for 2 s); otherwise the centre of the movement; None when
      nothing stands out. Scene cuts are listed so the virtual camera can jump instead of panning.
    """
    import cv2
    import numpy as np

    W, H = 320, 180
    cpu_cmd = [
        "ffmpeg", "-v", "error", "-threads", "0",
        # Skipping the loop filter and non-reference frames roughly halves 4K decode time.
        "-skip_loop_filter", "all", "-skip_frame", "noref",
        "-i", str(src), "-an",
        "-vf", f"fps={1 / HOP:g},scale={W}:{H}:flags=fast_bilinear,format=bgr24",
        "-f", "rawvideo", "-",
    ]
    # 8K and AV1 (YouTube's 8K is AV1-only) decode slower than real time on the CPU here: this ffmpeg has
    # no dav1d. NVDEC decodes and shrinks on the GPU: 42 s of 8K60 AV1 in 14 s instead of 55 s.
    # Shrunk to 8-bit on the GPU too, so 10-bit HDR sources work the same.
    gpu_cmd = [
        "ffmpeg", "-v", "error", "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
        "-i", str(src), "-an",
        "-vf", f"scale_cuda={W}:{H}:format=nv12,hwdownload,format=nv12,fps={1 / HOP:g},format=bgr24",
        "-f", "rawvideo", "-",
    ]
    video = video_info(src)
    heavy = video.get("codec_name") == "av1" or min(video.get("width") or 0, video.get("height") or 0) > 2160
    detector = None
    if FACE_MODEL.exists():
        try:
            detector = cv2.FaceDetectorYN.create(str(FACE_MODEL), "", (W, H), 0.7)
        except cv2.error as e:
            print(f"face detector unavailable: {e}")
    total, size = max(1.0, duration / HOP), W * H * 3
    for cmd in ([gpu_cmd] if heavy else []) + [cpu_cmd]:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        first = proc.stdout.read(size)
        if len(first) == size or cmd is cpu_cmd:
            break
        proc.wait()  # no NVIDIA GPU or driver: decode on the CPU instead
        print("GPU video decoding unavailable; analyzing motion on the CPU")
    motion, xs, kinds, cuts = [], [], [], []
    prev_small = prev_gray = None
    last_face, last_face_i, rival_n = None, -99, 0
    for buf in itertools.chain([first], iter(lambda: proc.stdout.read(size), b"")):
        if len(buf) != size:
            break
        i = len(motion)
        bgr = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.int16)
        small = cv2.resize(gray.astype(np.uint8), (64, 36), interpolation=cv2.INTER_AREA).astype(np.int16)
        m = float(np.abs(small - prev_small).mean()) if prev_small is not None else 0.0
        motion.append(m)
        # A cut: the whole picture changes far more than in the last 2 s.
        recent = motion[-9:-1]
        if recent and m > 25 and m > 4 * (sorted(recent)[len(recent) // 2] + 1):
            cuts.append(i)
            last_face, last_face_i, rival_n = None, -99, 0
        diff = np.abs(gray - prev_gray) if prev_gray is not None else None

        x, kind = None, "n"
        if detector is not None and i % 2 == 0:  # faces at 2 fps is plenty
            _, faces = detector.detect(bgr)
            cands = []
            for f in faces if faces is not None else []:
                fx, fy, fw, fh = (int(v) for v in f[:4])
                if fw < 10:
                    continue
                act = float(diff[max(0, fy):fy + fh, max(0, fx):fx + fw].mean()) if diff is not None else 0.0
                cands.append((fw * fh * (1 + act / 4), (fx + fw / 2) / W, fw / W))  # bigger and moving (talking) faces win
            if cands:
                # Stay locked on one person: the face nearest the one being followed keeps the camera unless
                # another face clearly wins (3x the score) for 2 s in a row. Otherwise the crop would hop between
                # people whenever someone else moves or nods, e.g. when the narrator is off screen.
                best = max(cands)
                held = last_face is not None and i - last_face_i <= 12
                near = [c for c in cands if held and abs(c[1] - last_face) < max(0.08, c[2] * 1.2)]
                mine = min(near, key=lambda c: abs(c[1] - last_face)) if near else None
                if not held:
                    pick, rival_n = best, 0  # nobody followed yet (or lost for 3 s): take the main face
                elif mine is not None and (best is mine or best[0] < 3 * mine[0]):
                    pick, rival_n = mine, 0
                else:  # followed face hidden, or someone else clearly dominates: switch only after 2 s
                    rival_n += 1
                    pick = best if rival_n >= 4 else mine
                    if pick is best:
                        rival_n = 0
                if pick is not None:
                    last_face, last_face_i = pick[1], i
        if last_face is not None and i - last_face_i <= 12:  # hold the followed face for 3 s while it's hidden
            x, kind = last_face, "f"
        elif diff is not None:
            cols = diff.mean(axis=0) ** 2
            tot = float(cols.sum())
            if m > 1.5 and tot > 0:
                pos = np.arange(W) / W
                cx = float((cols * pos).sum() / tot)
                spread = float(np.sqrt((cols * (pos - cx) ** 2).sum() / tot))
                if spread < 0.22:  # concentrated movement; a pan or full-frame change has ~0.29
                    x, kind = cx, "m"
        xs.append(None if x is None else round(x, 3))
        kinds.append(kind)
        prev_small, prev_gray = small, gray
        if i % 40 == 0:
            set_part(job, "motion", i / total)
    proc.wait()
    if len(motion) > 1:
        motion[0] = motion[1]
    return [round(float(np.log1p(v)), 3) for v in motion], {"x": xs, "kind": "".join(kinds), "cuts": cuts}


def extract_audio(src: Path, dst: Path):
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", str(dst)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError("ffmpeg could not read the audio track: " + res.stderr.strip()[-300:])


def loudness_db(wav: Path, hop: float = 0.25) -> list[float]:
    """RMS loudness in dB per `hop` seconds, for browsers that can't decode the audio themselves."""
    import wave

    import numpy as np

    with wave.open(str(wav)) as w:
        sr = w.getframerate()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
    n = int(hop * sr)
    frames = pcm[: len(pcm) // n * n].reshape(-1, n)
    return np.round(20 * np.log10(np.sqrt((frames ** 2).mean(axis=1)) + 1e-5), 2).tolist()


def run_transcription(job_id: str, workdir: Path, src: Path, language: str | None):
    job = jobs[job_id]
    motion_thread = None
    try:
        job.update(status="extracting", progress=0.02, parts={})
        audio, video = has_stream(src, "a"), has_stream(src, "v")
        result = {"has_audio": audio, "hop": HOP, "loudness": [], "motion": [], "track": None, "segments": []}

        # Motion runs in its own thread (ffmpeg does the heavy lifting) while Whisper transcribes.
        if video:
            set_part(job, "motion", 0.0)

            def run_motion():
                try:
                    result["motion"], result["track"] = analyze_frames(src, probe_duration(src), job)
                except Exception as e:
                    print(f"motion analysis failed: {e}")
                set_part(job, "motion", 1.0)

            motion_thread = threading.Thread(target=run_motion, daemon=True)
            motion_thread.start()

        if audio:
            set_part(job, "speech", 0.0)
            wav = workdir / "audio.wav"
            extract_audio(src, wav)
            result["loudness"] = loudness_db(wav)
            job["status"] = "analyzing speech and motion" if video else "transcribing"
            try:
                result["segments"] = transcribe(load_model(), wav, language, job)
            except RuntimeError as e:
                # CUDA errors (e.g. missing cuBLAS/cuDNN DLLs) often surface only on first use.
                if _model_device != "cuda" or DEVICE == "cuda":
                    raise
                print(f"CUDA transcription failed ({e}); retrying on CPU")
                result["segments"] = transcribe(load_model(force_cpu=True), wav, language, job)
            result.update(job.pop("meta"))
            set_part(job, "speech", 1.0)

        if motion_thread:
            job["status"] = "analyzing motion"
            motion_thread.join()
        job.update(status="done", progress=1.0, result=result)
    except Exception as e:
        job.update(status="error", error=str(e))
    finally:
        if motion_thread:
            motion_thread.join()  # ffmpeg holds the upload open; Windows can't delete it until it exits
        shutil.rmtree(workdir, ignore_errors=True)


SR = 16000
# Evens out loudness so quiet voices reach the speech detector and Whisper at a usable level.
SPEECH_NORM = "highpass=f=80,lowpass=f=7800,speechnorm=e=50:r=0.0005:l=1"
# Text Whisper tends to invent over music or silence.
HALLUCINATIONS = {  # compared after hallucination_key() (lowercase, no punctuation)
    "thank you", "thanks for watching", "thank you for watching", "thank you so much for watching",
    "please subscribe", "subscribe to my channel", "like and subscribe", "we'll be right back", "you", "bye",
    "music", "gracias", "muchas gracias", "gracias por ver", "gracias por ver el video", "gracias por ver el vídeo",
    "suscríbete", "suscríbete al canal", "suscríbanse", "suscríbete y dale like", "no olvides suscribirte",
    "subtítulos realizados por la comunidad de amaraorg", "subtítulos por la comunidad de amaraorg",
    "obrigado", "inscreva-se", "merci", "sous-titrage st' 501",
}


def hallucination_key(text: str) -> str:
    return re.sub(r"[^\w\s'-]", "", text.lower()).strip()


MIN_SWITCH_S = 2.5  # lines shorter than this always use the video's main language


def primary_language(model, audio, regions) -> str | None:
    """Main spoken language: vote over up to 8 blocks of ~30 s of speech glued together."""
    import numpy as np

    blocks, cur, length = [], [], 0
    for a, b in regions:
        cur.append(audio[a:b])
        length += b - a
        if length >= 30 * SR:
            blocks.append(np.concatenate(cur))
            cur, length = [], 0
    if length >= 3 * SR or not blocks and cur:
        blocks.append(np.concatenate(cur))
    if not blocks:
        return None
    step = max(1, len(blocks) // 8)
    votes = {}
    for blk in blocks[::step][:8]:
        lang, prob, _ = model.detect_language(audio=blk)
        votes[lang] = votes.get(lang, 0) + prob * len(blk)
    return max(votes, key=votes.get)


REPEAT_LOOP = re.compile(r"\b(\w+)\b(?:\W+\1\b){3,}", re.IGNORECASE)  # "BAD BAD BAD BAD"


def transcribe(model, wav: Path, language: str | None, job: dict):
    """Transcribe speech that changes language and speech quieter than the background.

    Whisper decides one language per 30 s window and silently drops speech in any other language, so
    each stretch of speech is detected and transcribed on its own. The speech detector can't hear voices
    below the background (game sound, music), so on GPU every stretch it skipped is also listened to in
    overlapping windows, keeping only lines Whisper is confident about.
    """
    from faster_whisper import decode_audio
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    norm = wav.with_name("speech.wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(wav), "-af", SPEECH_NORM, "-ar", str(SR), "-ac", "1", str(norm)], check=True)
    audio = decode_audio(str(norm))
    duration = len(audio) / SR
    regions = [(r["start"], r["end"]) for r in get_speech_timestamps(audio, VadOptions(
        threshold=0.25, min_speech_duration_ms=200, min_silence_duration_ms=300, speech_pad_ms=250, max_speech_duration_s=15))]

    # Language guesses on a single 1-2 s line are unreliable (a Spanish "¡Hem!" can come out as Chinese,
    # German or Hungarian at 70-98% confidence), and a line given the wrong language gets *translated*.
    # So: the main language comes from the speech heard together (Whisper is reliable on ~30 s), and a
    # second/third language is only allowed with solid evidence, on lines long enough to judge.
    detected = [dict(model.detect_language(audio=audio[a:b])[2]) for a, b in regions]
    main_lang = language or primary_language(model, audio, regions)
    evidence = {}
    for (a, b), probs in zip(regions, detected):
        top = max(probs, key=probs.get)
        if top != main_lang and probs[top] >= 0.85 and (b - a) / SR >= MIN_SWITCH_S:
            evidence[top] = evidence.get(top, 0) + (b - a) / SR
    others = [] if language else [l for l, d in sorted(evidence.items(), key=lambda x: -x[1]) if d >= 3.0][:2]

    def pick_language(probs, seconds):
        """The main language, unless this line is long enough and clearly in one of the video's other languages."""
        if not main_lang:
            return max(probs, key=probs.get)
        if seconds >= MIN_SWITCH_S:
            best = max(others, key=lambda l: probs.get(l, 0), default=None)
            if best and probs.get(best, 0) >= 0.7 and probs[best] > probs.get(main_lang, 0):
                return best
        return main_lang

    allowed = [l for l in [main_lang, *others] if l]

    def run(spans, probs_list, strict, progress):
        out = []
        for i, (a, b) in enumerate(spans):
            probs = probs_list[i] if probs_list else dict(model.detect_language(audio=audio[a:b])[2])
            lang = pick_language(probs, (b - a) / SR)
            segs, _ = model.transcribe(audio[a:b], language=lang, word_timestamps=True, beam_size=5,
                                       condition_on_previous_text=False, vad_filter=False,
                                       hallucination_silence_threshold=None if strict else 2.0)
            off = a / SR
            seen_here = set()
            for s in segs:
                text = s.text.strip()
                if not s.words or hallucination_key(text) in HALLUCINATIONS or REPEAT_LOOP.search(text):
                    continue
                if strict:
                    # Fillers like "Amén." or "No sé" score like real shouts, so 1-2 word lines need more confidence.
                    min_lp = -0.5 if len(s.words) <= 2 else -0.7
                    if s.avg_logprob < min_lp or s.no_speech_prob > 0.5 or s.compression_ratio > 2.4:
                        continue
                    # Whisper sometimes loops one line through a window ("Follow me!" x3); keep it once.
                    k = re.sub(r"\W+", " ", text.lower()).strip()
                    if k in seen_here:
                        continue
                    seen_here.add(k)
                if CAPTION_DEBUG and strict:
                    print(f"gap line {a / SR + s.start:7.1f}s lp={s.avg_logprob:.2f} nsp={s.no_speech_prob:.2f} "
                          f"cr={s.compression_ratio:.2f} [{lang}] {text}")
                out.append({"start": round(s.start + off, 2), "end": round(s.end + off, 2), "text": text, "lang": lang,
                            "logprob": s.avg_logprob,
                            "words": [{"w": w.word.strip(), "s": round(w.start + off, 2), "e": round(w.end + off, 2)} for w in s.words]})
            progress(i + 1, len(spans))
        return out

    # GPU: main pass is ~40% of the work, the gap pass ~60%. CPU skips the gap pass (it would be many times slower).
    gap_pass = QUIET_BOOST and _model_device == "cuda"
    main_share = 0.4 if gap_pass else 1.0
    out = run(regions, detected, False, lambda i, n: set_part(job, "speech", main_share * i / max(1, n)))

    if gap_pass:
        win, hop = 8 * SR, 4 * SR
        gaps, t = [], 0
        for a, b in sorted((int(s["start"] * SR), int(s["end"] * SR)) for s in out) + [(len(audio), len(audio))]:
            if a - t >= SR:
                gaps.append((t, a))
            t = max(t, b)
        windows = [(x, min(x + win, b)) for a, b in gaps for x in range(a, max(a + 1, b - SR), hop)]
        # Truly silent stretches have nothing to hear; Whisper only invents text there.
        windows = [(x, y) for x, y in windows if float((audio[x:y] ** 2).mean()) ** 0.5 > 10 ** (-55 / 20)]
        cands = run(windows, None, True, lambda i, n: set_part(job, "speech", main_share + (1 - main_share) * i / max(1, n)))

        def key(text):
            return re.sub(r"\W+", " ", text.lower()).strip()

        for c in sorted(cands, key=lambda s: -s["logprob"]):  # the clearest version of each line wins
            clash = any(min(c["end"], o["end"]) - max(c["start"], o["start"]) > 0.3 * (c["end"] - c["start"])
                        or (key(c["text"]) == key(o["text"]) and abs(c["start"] - o["start"]) < 4) for o in out)
            if not clash:
                out.append(c)

    out.sort(key=lambda s: s["start"])
    for s in out:
        s.pop("logprob", None)
    counts = {}
    for s in out:
        counts[s["lang"]] = counts.get(s["lang"], 0) + len(s["words"])
    languages = sorted(counts, key=counts.get, reverse=True)
    job["meta"] = {"language": languages[0] if languages else (allowed[0] if allowed else "en"), "languages": languages,
                   "duration": duration, "device": _model_device, "model": model_name()}
    return out


def ollama(path: str, payload: dict | None = None, timeout: float = 5):
    req = urllib.request.Request(
        OLLAMA_URL + path,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def vision_model() -> str | None:
    """OLLAMA_MODEL if set, else the best installed model that can see images (None when Ollama isn't running)."""
    if OLLAMA_MODEL:
        return OLLAMA_MODEL
    try:
        names = [m["name"] for m in ollama("/api/tags")["models"]]
    except Exception:
        return None
    vision = [n for n in names if "vision" in ollama("/api/show", {"model": n}).get("capabilities", [])]
    preferred = ("qwen2.5vl", "gemma3", "llama3.2-vision", "minicpm-v", "llava")
    vision.sort(key=lambda n: next((i for i, p in enumerate(preferred) if n.startswith(p)), len(preferred)))
    return vision[0] if vision else None


# Ollama may size the context to the model's maximum (128k for qwen2.5vl), which doubles its memory and
# spills it out of a 12 GB GPU onto the CPU (minutes per call). Our prompts need far less. num_predict stops
# runaway JSON generation.
OLLAMA_OPTIONS = {"num_ctx": 8192, "num_predict": 800}

# Retention rules shared by every prompt that writes text for a short. Viewers decide in the first 3-5 seconds
# and leave as soon as the video drags or breaks its promise. No braces here: it is joined into .format() templates.
ENGAGEMENT_RULES = """Retention rules (viewers decide in the first 3-5 seconds whether to keep watching):
- Hook first: open with the most striking moment, a bold statement, an intriguing question or something unexpected. Never open with a greeting, an introduction of who is talking or a slow build-up.
- No filler: get straight to the point. Every sentence must add something new.
- Open loops: raise a question or promise a payoff early (e.g. "but that's not even the best part...") and pay it off later, so people stay for the answer.
- Deliver on the promise: the hook, title and caption must match what really happens in the clip. No empty clickbait; misled viewers leave at once."""

CAPTION_PROMPT = """You write on-screen text for a short vertical video clip (TikTok / Reels / Shorts).
The images are frames from the clip, in order.{transcript}
""" + ENGAGEMENT_RULES + """
Write:
- "hook": a scroll-stopping line for the top of the video, at most 8 words, specific to what is shown. No hashtags.
  Wrap the one or two most important words of the hook in *asterisks* so they get highlighted, e.g. "Nobody talks about *this*".
  Highlight a word that carries the meaning (a noun, verb or number), never a filler word.
- "caption": one punchy line for the bottom of the video, at most 14 words, adding context or a reaction, or teasing the payoff. May end with one emoji.
Write in {language}. Do not describe the frames literally ("a man in a kart"); react to them like a creator would.
Reply with JSON only: {{"hook": "...", "caption": "..."}}"""


@app.post("/api/caption")
def ai_caption(body: dict):
    model = vision_model()
    if not model:
        raise HTTPException(503, "No local vision model found. Install Ollama and run: ollama pull qwen2.5vl:7b")
    transcript = (body.get("transcript") or "").strip()
    prompt = CAPTION_PROMPT.format(
        transcript=f'\nWhat is said in the clip: "{transcript[:1500]}"' if transcript else "\nThe clip has no speech.",
        language=body.get("language") or "English",
    )
    frames = [f.split(",", 1)[-1] for f in body.get("frames", [])][:4]  # strip data: URL prefixes
    try:
        res = ollama("/api/chat", {
            "model": model,
            "stream": False,
            "format": {"type": "object", "properties": {"hook": {"type": "string"}, "caption": {"type": "string"}}, "required": ["hook", "caption"]},
            "options": {"temperature": 0.8, **OLLAMA_OPTIONS},
            "keep_alive": "30m",  # first load into VRAM can take minutes; keep it warm between clicks
            "messages": [{"role": "user", "content": prompt, "images": frames}],
        }, timeout=600)
        out = json.loads(res["message"]["content"])
    except Exception as e:
        raise HTTPException(502, f"{model} failed: {e}")
    clean = lambda s: str(s or "").strip().strip('"').strip()
    return {"hook": clean(out.get("hook")), "caption": clean(out.get("caption")), "model": model,
            "faces": detect_faces(frames)}


def detect_faces(frames: list[str]) -> list[list[list[float]]]:
    """Faces in each frame (base64 JPEGs) as [x, y, w, h] shares of the frame, biggest first.
    The page uses them to keep the hook and captions off the face and to pick the zoom."""
    if not FACE_MODEL.exists():
        return []
    try:
        import base64
        import cv2
        import numpy as np
    except ImportError:
        return []
    out, detector = [], None
    for b64 in frames:
        boxes = []
        try:
            img = cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                h, w = img.shape[:2]
                if detector is None:
                    detector = cv2.FaceDetectorYN.create(str(FACE_MODEL), "", (w, h), 0.7)
                detector.setInputSize((w, h))
                _, faces = detector.detect(img)
                for f in faces if faces is not None else []:
                    fx, fy, fw, fh = (float(v) for v in f[:4])
                    if fw >= 8:
                        boxes.append([round(fx / w, 4), round(fy / h, 4), round(fw / w, 4), round(fh / h, 4)])
        except Exception as e:  # a bad frame or no OpenCV face support: lay out without faces
            print(f"face detection failed: {e}")
        out.append(sorted(boxes, key=lambda b: -b[2] * b[3]))
    return out


NARRATION_STYLES = {
    "hype": "high-energy, hyped-up commentary, like a creator reacting to an epic moment",
    "story": "storytime: first person, building suspense toward a payoff",
    "explain": "clear, informative explainer that tells the viewer what is happening and why it matters",
    "funny": "funny, sarcastic commentary with a punchline",
    "documentary": "calm, dramatic documentary narrator",
    "motivational": "powerful motivational speaker talking straight to the viewer: an uplifting message about the topic, "
                    "with the footage only illustrating it, so speak about the idea, not about what each shot shows",
}

NARRATION_PROMPT = """You write the voice-over for a short vertical video clip (TikTok / Reels / Shorts).
The images are frames from the clip, in order.{context}
Style: {style}.
""" + ENGAGEMENT_RULES + """
- About {words} words in total, so it fits {seconds} seconds read aloud. Never more than {max_words} words.
- The first sentence is the hook. Short spoken sentences. No emojis, hashtags, stage directions or speaker labels.
- Strategic pause: put "..." right before the big reveal or payoff to build suspense.
- Be specific to what is shown and said. Don't guess names (games, people, places) that aren't in the source title, the speech or clearly on screen.
Write in {language}. Reply with JSON only: {{"script": "..."}}"""

# Scene by scene runs in two steps. A 7B vision model asked to look, time and write in one go reads frame
# labels as content, invents events and fills lines with generic shouts. Describing one frame at a time is
# what it's good at (positions, items, hits come out right); the writing step then only works from text.
DESCRIBE_PROMPT = """This is one frame from a video. In one short sentence in {language}, say what is happening in it.
Include anything readable that matters: a position or score on screen, the item or power-up shown, what the main character is doing, crashes, hits, finish lines, text on screen.
Call the main character "the player"; don't name characters or the game. Do not describe a small webcam of a person in a corner. Facts only, no opinions."""

SCENE_PROMPT = """You are the narrator of a short vertical video clip (TikTok / Reels / Shorts).
Style: {style}.{source}
{speech}{hint}
The clip is split into {n} parts. For each part you get what is on screen during it (from a frame-by-frame look; small details may be wrong, numbers and items shown on screen are reliable).
Write one narrator line per part, about what happens in that part, in order:
{parts}
""" + ENGAGEMENT_RULES + """
Rules:
- Line 1 is a hook that makes people keep watching. If a later part has the big moment, tease it in line 1 or 2 and pay it off in that part.
- Put "..." right before the big reveal or payoff to build suspense.
- Talk about what changes: places gained or lost, score, items picked up or used, hits, close calls, holding the lead, the ending. Don't just describe the scenery.
- Stay under each part's word limit. One or two spoken sentences per line.
- Never repeat what the person in the video says. No generic filler, emojis, hashtags, stage directions or speaker labels.
- Write numbers and positions as words people say. Don't name characters, games or people unless the source title or the speech names them.
Write every word in {language}; no words from other languages. Reply with JSON only: {{"lines": ["line 1", "line 2", ...]}}"""


def scene_schema(n: int) -> dict:
    """Exactly one line per part (the model otherwise sometimes stops after one)."""
    return {"type": "object", "required": ["lines"], "properties": {"lines": {
        "type": "array", "minItems": n, "maxItems": n, "items": {"type": "string"}}}}


def scene_parts(seconds: float, cuts: list[float]) -> list[float]:
    """Start times of the narration parts: about one every 6 s, moved onto a nearby scene cut."""
    n = max(2, round(seconds / 6))
    starts = [i * seconds / n for i in range(n)]
    for i in range(1, n):
        near = [c for c in cuts if abs(c - starts[i]) < seconds / n / 3]
        if near:
            starts[i] = min(near, key=lambda c: abs(c - starts[i]))
    return [round(t, 1) for t in starts]


def clean_speech(text: str) -> str:
    return re.sub(r"[ \t]{2,}", " ", re.sub(r"#\w+", "", str(text or ""))).strip().strip('"').strip()  # no hashtags in speech


def without_echo(text: str, said: list[str]) -> str:
    """Drop sentences that only repeat what the person in the video says ("¡Vaya, amor!")."""
    key = lambda x: re.sub(r"\W+", " ", x.lower()).strip()
    heard = {key(x) for x in said if key(x)}
    parts = re.split(r"(?<=[.!?…])\s+", text)
    return " ".join(p for p in parts if key(p) not in heard and not any(len(h) > 8 and h in key(p) for h in heard)).strip()


def ollama_json(model: str, prompt: str, schema: dict, images: list | None = None, temperature: float = 0.7) -> dict:
    msg = {"role": "user", "content": prompt, **({"images": images} if images else {})}
    request = {"model": model, "stream": False, "keep_alive": "30m", "format": schema,
               "options": {"temperature": temperature, **OLLAMA_OPTIONS}, "messages": [msg]}
    for attempt in (1, 2):  # Ollama now and then fails a structured-output request; a second try usually works
        try:
            return json.loads(ollama("/api/chat", request, timeout=600)["message"]["content"])
        except Exception as e:
            if attempt == 2:
                raise HTTPException(502, f"{model} failed: {e}")


def describe_frames(model: str, frames: list[str], times: list[float], language: str) -> list[dict]:
    # Described in the narration's language: English descriptions leak English words ("green shell") into the lines.
    schema = {"type": "object", "required": ["what"], "properties": {"what": {"type": "string"}}}
    prompt = DESCRIBE_PROMPT.format(language=language) + ' Reply with JSON: {"what": "..."}'
    return [{"t": t, "text": str(ollama_json(model, prompt, schema, [f], 0.2)
                                 .get("what", "")).strip()} for f, t in zip(frames, times)]


@app.post("/api/narration-script")
def ai_narration_script(body: dict):
    """A voice-over script sized to the clip, from its frames and transcript.

    With `frame_times` (seconds into the clip of each frame) it's scene by scene: each frame is described,
    then lines are written from that timeline, each starting at the moment it talks about.
    Without, one block of text read from the start of the clip.
    """
    model = vision_model()
    if not model:
        raise HTTPException(503, "No local vision model found. Install Ollama and run: ollama pull qwen2.5vl:7b")
    seconds = max(3.0, min(180.0, float(body.get("seconds") or 30)))
    # ~2.5 spoken words a second at normal speed, scaled by the chosen speed; a little slack for pauses.
    wps = 2.5 * (1 + max(-50, min(100, int(body.get("rate") or 0))) / 100)
    times = [round(float(t), 1) for t in body.get("frame_times") or []]
    frames = [f.split(",", 1)[-1] for f in body.get("frames", [])]  # strip data: URL prefixes
    language = body.get("language") or "English"
    style = NARRATION_STYLES.get(body.get("style"), NARRATION_STYLES["hype"])
    source = (body.get("source") or "").strip()[:150]
    hint = (body.get("hint") or "").strip()[:500]

    if times:
        timeline = describe_frames(model, frames[:12], times[:12], language)
        speech = [x for x in body.get("speech") or [] if str(x.get("text", "")).strip()]
        cuts = [round(float(t), 1) for t in body.get("cuts") or []]
        # The code decides when lines are said; the model only writes what fits each slot. A 7B model asked to
        # pick its own times bunches lines at the start or puts them all on frame times.
        starts = scene_parts(seconds, cuts)
        ends = starts[1:] + [seconds]
        caps = [max(3, int((e - t) * wps * 0.8)) for t, e in zip(starts, ends)]
        parts = []
        for i, (t, e, cap) in enumerate(zip(starts, ends, caps)):
            seen = [x["text"] for x in timeline if t - 1 <= x["t"] < e] or [min(timeline, key=lambda x: abs(x["t"] - t))["text"]]
            parts.append(f"Part {i + 1} ({t:.0f}-{e:.0f} s, at most {cap} words). On screen: " + " Then: ".join(seen))
        prompt = SCENE_PROMPT.format(
            n=len(starts), style=style, parts="\n".join(parts), language=language,
            source=f'\nThe source video is titled: "{source}"' if source else "",
            speech=("What the person in the video says, for context (second: words):\n"
                    + "\n".join(f'{float(x["t"]):.1f}s: {str(x["text"]).strip()}' for x in speech)[:1500]) if speech else "Nobody speaks in the video.",
            hint=f'\nWhat the voice-over should be about: "{hint}"' if hint else "")
        said = [str(x["text"]) for x in speech]
        out = ollama_json(model, prompt, scene_schema(len(starts)), temperature=0.5)
        lines = []
        for t, cap, text in zip(starts, caps, out.get("lines") or []):
            text = without_echo(clean_speech(text), said)
            words_ = text.split()
            if len(words_) > cap * 1.3:  # too long to finish before the next part: keep whole sentences that fit
                kept = ""
                for sent in re.split(r"(?<=[.!?…])\s+", text):
                    if len((kept + " " + sent).split()) > cap * 1.3:
                        break
                    kept = (kept + " " + sent).strip()
                text = kept or " ".join(words_[:cap])
            # The model sometimes loops the same line through every part; say it once.
            if text and re.sub(r"\W+", " ", text.lower()).strip() not in {re.sub(r"\W+", " ", l["text"].lower()).strip() for l in lines}:
                lines.append({"t": t, "text": text})
        return {"lines": lines, "words": sum(caps), "model": model, "timeline": timeline}

    # `fill`: the voice should talk for the whole clip (stock shorts), so aim for the full length, not 90%.
    fill = bool(body.get("fill"))
    words = max(5, round(seconds * wps * (1.0 if fill else 0.9)))
    context = f'\nThe source video is titled: "{source}"' if source else ""
    transcript = (body.get("transcript") or "").strip()
    context += f'\nWhat is said in the clip: "{transcript[:1500]}"' if transcript else "\nThe clip has no speech."
    if hint:
        context += f'\nWhat the voice-over should be about: "{hint}"'
    prompt = NARRATION_PROMPT.format(context=context, style=style, words=words, max_words=round(words * 1.15),
                                     seconds=round(seconds), language=language)
    schema = {"type": "object", "properties": {"script": {"type": "string"}}, "required": ["script"]}
    script = clean_speech(ollama_json(model, prompt, schema, frames[:4], 0.8).get("script"))
    # A 7B model writes well under the word budget; ask once more for a longer take and keep the longer one.
    if fill and len(script.split()) < words * 0.85:
        more = prompt + (f'\nA first draft was only {len(script.split())} words, too short: the voice would stop before the video ends. '
                         f'Write a fuller script of about {words} words (at least {round(words * 0.9)}). First draft: "{script}"')
        longer = clean_speech(ollama_json(model, more, schema, frames[:4], 0.8).get("script"))
        if len(longer.split()) > len(script.split()):
            script = longer
    # Models overshoot the word budget; drop whole trailing sentences until it fits the clip.
    sentences = re.split(r"(?<=[.!?…])\s+", script)
    while len(sentences) > 1 and len(" ".join(sentences).split()) > words * 1.1:
        sentences.pop()
    return {"script": " ".join(sentences), "words": words, "model": model}


POST_PROMPT = """You are a social media manager writing upload text for one short vertical video.
The images are frames from the clip, in order.{context}
Write text that makes people stop scrolling and is honest about what the video shows. Be specific to these frames, not generic.
""" + ENGAGEMENT_RULES + """
- youtube.title: max 70 characters, curiosity or a clear payoff, one natural keyword, ends with " #Shorts". No ALL CAPS.
- youtube.description: 2 short lines about the moment, then a blank line, then 3-5 relevant hashtags.
- youtube.tags: 6-10 short search keywords (no # sign).
- tiktok.caption: max 150 characters, conversational, ends with 3-5 hashtags mixing broad and niche.
- instagram.caption: 1-2 lines plus a question or call to action, then a blank line, then 5-8 hashtags.
- facebook.caption: 1-3 conversational lines that invite comments or shares, then 2-4 hashtags.
Never repeat a hashtag. Use #Shorts only in the YouTube title. Don't guess names (games, people, places) that aren't in the source title, the speech or clearly on screen.
Write in {language}. Reply with JSON only."""


def tidy_hashtags(text: str, drop: tuple[str, ...] = ()) -> str:
    """Remove repeated hashtags (case-insensitive) and ones that belong to another platform."""
    seen = set()

    def keep(m):
        tag = m.group(0).lower()
        if tag in seen or tag in drop:
            return ""
        seen.add(tag)
        return m.group(0)

    text = re.sub(r"#\w+", keep, text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return re.sub(r"[ \t]+\n", "\n", text).strip()

POST_SCHEMA = {
    "type": "object",
    "properties": {
        "youtube": {"type": "object", "properties": {
            "title": {"type": "string"}, "description": {"type": "string"},
            "tags": {"type": "array", "items": {"type": "string"}}}, "required": ["title", "description", "tags"]},
        "tiktok": {"type": "object", "properties": {"caption": {"type": "string"}}, "required": ["caption"]},
        "instagram": {"type": "object", "properties": {"caption": {"type": "string"}}, "required": ["caption"]},
        "facebook": {"type": "object", "properties": {"caption": {"type": "string"}}, "required": ["caption"]},
    },
    "required": ["youtube", "tiktok", "instagram", "facebook"],
}


@app.post("/api/post-text")
def ai_post_text(body: dict):
    """Titles, descriptions, captions and hashtags for each platform, from the clip's frames and transcript."""
    model = vision_model()
    if not model:
        raise HTTPException(503, "No local vision model found. Install Ollama and run: ollama pull qwen2.5vl:7b")
    context = ""
    if (body.get("source") or "").strip():
        # The original file name often names the game/show/topic, which the model would otherwise guess.
        context += f'\nThe source video is titled: "{body["source"].strip()[:150]}"'
    if (body.get("transcript") or "").strip():
        context += f'\nWhat is said in the clip: "{body["transcript"].strip()[:1500]}"'
    else:
        context += "\nThe clip has no speech."
    if body.get("hook") or body.get("caption"):
        context += f'\nOn-screen text: "{body.get("hook", "")}" / "{body.get("caption", "")}"'
    frames = [f.split(",", 1)[-1] for f in body.get("frames", [])][:4]
    try:
        res = ollama("/api/chat", {
            "model": model, "stream": False, "format": POST_SCHEMA, "keep_alive": "30m",
            "options": {"temperature": 0.7, **OLLAMA_OPTIONS},
            "messages": [{"role": "user", "content": POST_PROMPT.format(context=context, language=body.get("language") or "English"),
                          "images": frames}],
        }, timeout=600)
        out = json.loads(res["message"]["content"])
    except Exception as e:
        raise HTTPException(502, f"{model} failed: {e}")
    return clean_post_text(out, model)


def clean_post_text(out: dict, model: str) -> dict:
    """The model's per-platform text, tidied: #Shorts only in the YouTube title, plain tags, no repeated hashtags."""
    yt = out.get("youtube", {})
    title = str(yt.get("title", "")).strip().strip('"')
    if "#shorts" not in title.lower() and len(title) <= 92:
        title += " #Shorts"
    tags, seen = [], set()
    for t in yt.get("tags", []):
        t = re.sub(r"#\w+", "", str(t)).strip(" #,")  # tags are plain keywords
        if t and t.lower() not in seen:
            seen.add(t.lower())
            tags.append(t)
    return {
        "youtube": {"title": title[:100], "description": tidy_hashtags(str(yt.get("description", "")), ("#shorts",)),
                    "tags": tags[:12]},
        "tiktok": {"caption": tidy_hashtags(str(out.get("tiktok", {}).get("caption", "")), ("#shorts",))[:2200]},
        "instagram": {"caption": tidy_hashtags(str(out.get("instagram", {}).get("caption", "")), ("#shorts",))[:2200]},
        # Older or smaller models sometimes skip a platform: Facebook then reuses the Instagram caption.
        "facebook": {"caption": tidy_hashtags(str(out.get("facebook", {}).get("caption", "")
                                                  or out.get("instagram", {}).get("caption", "")), ("#shorts",))[:2200]},
        "model": model,
    }


PHOTO_POST_PROMPT = """You are a social media manager writing the text for one photo post.
The images are the {n} photo(s) of the post, in order. On Facebook and Instagram they are posted as photos
(a carousel when there are several); on TikTok and YouTube Shorts they are shown as a short slideshow video.{context}
Write text that stops the scroll and gets comments, saves and shares, and is honest about what the photos show.
Be specific to these photos, not generic.
- Hook first: the first line must grab attention (a bold statement, a surprising fact, a relatable feeling or a question).
- Invite engagement: end with a question, "save this for later", "tag someone who…" or a similar call to action that fits.
- youtube.title: max 70 characters, curiosity or a clear payoff, one natural keyword, ends with " #Shorts". No ALL CAPS.
- youtube.description: 2 short lines, then a blank line, then 3-5 relevant hashtags.
- youtube.tags: 6-10 short search keywords (no # sign).
- tiktok.caption: max 150 characters, conversational, ends with 3-5 hashtags mixing broad and niche.
- instagram.caption: a hook line, 1-3 short lines, a call to action, then a blank line, then 5-8 hashtags.
- facebook.caption: 1-3 conversational lines that invite comments or shares, then 2-4 hashtags.
Never repeat a hashtag. Use #Shorts only in the YouTube title. Don't name people, places or brands you can't be sure of.
Write in {language}. Reply with JSON only."""


@app.post("/api/photo-post-text")
def ai_photo_post_text(body: dict):
    """Captions and hashtags for each platform, from the photos of a photo post."""
    model = vision_model()
    if not model:
        raise HTTPException(503, "No local vision model found. Install Ollama and run: ollama pull qwen2.5vl:7b")
    ids = [str(x) for x in body.get("photos") or []][:4]  # the first photos are enough to know what it's about
    if not ids:
        raise HTTPException(400, "Add photos to the post first.")
    try:
        images = [photos.thumb_b64(pid) for pid in ids]
    except publish.PublishError as e:
        raise HTTPException(400, str(e))
    topic = (body.get("topic") or "").strip()[:300]
    context = f'\nThe creator says the post is about: "{topic}"' if topic else ""
    n = len(body.get("photos") or [])
    prompt = PHOTO_POST_PROMPT.format(n=n, context=context, language=body.get("language") or "English")
    return clean_post_text(ollama_json(model, prompt, POST_SCHEMA, images, 0.75), model)


VIRAL_IMAGE_PROMPT = """You design ONE scroll-stopping image for a social media photo post (Instagram, Facebook, TikTok).
The images are the {n} photo(s) it is made from, in order. The layout is "{layout}": {how}{context}
Write the text that goes ON the image, in {language}:
- "text": the hook, {words}. It must create curiosity, emotion or a clear payoff, be specific to these photos and honest
  about them. Think "NOBODY TALKS ABOUT THIS", "I WISH I KNEW THIS SOONER", "WHICH ONE WOULD YOU PICK?". No hashtags, no emojis.
- "accent": the ONE word of "text" to highlight in color (the most emotional or surprising word).
- "labels": {labels}
- "position": where the hook goes so it does not cover the main subject: "top" or "bottom".
Reply with JSON only."""

VIRAL_LAYOUTS = {
    "hook": ("one photo, full bleed, with big bold text over it.", "2 to 6 words", "an empty list."),
    "split": ("two photos side by side, a comparison: before / after, expectation / reality, this vs that.",
              "3 to 7 words, framing the comparison",
              'exactly 2 very short labels (1-2 words), one per photo, e.g. ["Before", "After"] or ["Expectation", "Reality"].'),
    "collage": ("2 to 4 photos in a grid with the hook on a band across the middle.", "3 to 7 words",
                "one 1-2 word label per photo, or an empty list if labels add nothing."),
    "ranking": ("a numbered top list, one photo per rank, the hook as the header.", 'a list title of 3 to 7 words, e.g. "TOP 5 CARS UNDER $20K"',
                "one short name (1-3 words) per photo, in rank order."),
    "meme": ("a white caption bar on top (like a tweet), the photo(s) below.",
             "one relatable or funny sentence of 6 to 16 words, in normal sentence case (not all caps)", "an empty list."),
}


@app.post("/api/photos/design-text")
def ai_design_text(body: dict):
    """Hook text, highlight word and labels for a viral image made from the post's photos."""
    model = vision_model()
    if not model:
        raise HTTPException(503, "No local vision model found. Install Ollama and run: ollama pull qwen2.5vl:7b")
    ids = [str(x) for x in body.get("photos") or []][:5]
    if not ids:
        raise HTTPException(400, "Add photos to the post first.")
    layout = body.get("layout") if body.get("layout") in VIRAL_LAYOUTS else "hook"
    if layout == "hook":
        ids = ids[:1]
    elif layout == "split":
        ids = ids[:2]
    try:
        images = [photos.thumb_b64(pid, 512) for pid in ids]
    except publish.PublishError as e:
        raise HTTPException(400, str(e))
    how, words, labels = VIRAL_LAYOUTS[layout]
    topic = (body.get("topic") or "").strip()[:300]
    context = f'\nThe creator says the post is about: "{topic}"' if topic else ""
    schema = {"type": "object", "required": ["text", "accent", "labels", "position"], "properties": {
        "text": {"type": "string"}, "accent": {"type": "string"}, "labels": {"type": "array", "items": {"type": "string"}},
        "position": {"type": "string", "enum": ["top", "bottom"]}}}
    prompt = VIRAL_IMAGE_PROMPT.format(n=len(ids), layout=layout, how=how, words=words, labels=labels, context=context,
                                       language=body.get("language") or "English")
    try:
        out = ollama_json(model, prompt, schema, images, 0.8)
    except Exception as e:
        raise HTTPException(502, f"{model} failed: {e}")
    text = re.sub(r"#\w+", "", str(out.get("text") or "")).strip().strip('"').strip()
    accent = str(out.get("accent") or "").strip().strip('"')
    if accent.lower() not in text.lower():
        accent = max(text.split(), key=len, default="")
    labels_out = [re.sub(r"#\w+", "", str(x)).strip()[:30] for x in (out.get("labels") or []) if str(x).strip()]
    return {"text": text, "accent": accent, "labels": labels_out[:len(ids)] if layout != "hook" else [],
            "position": out.get("position") if out.get("position") in ("top", "bottom") else "top", "model": model}


THUMB_PROMPT = """You design the thumbnail (cover image) for a short vertical video (TikTok / Reels / Shorts).
The images are {n} candidate frames from the clip, numbered 1 to {n} in order.{context}
Pick the ONE frame that makes the strongest thumbnail. A great thumbnail frame has:
- a clear, sharp subject that is big in the frame, ideally a face with a strong emotion (shock, joy, intensity, tears) looking at or near the camera;
- the peak moment or a strong hint of it (the action, the reveal, the reaction), not a calm or in-between moment;
- good light and contrast. Avoid blurry frames, motion smear, closed eyes, half-faces, black/transition frames, menus and frames covered by text.
Then write the thumbnail text, in {language}:
- "text": 2 to 4 words, ALL about the curiosity or emotion of the clip, big and bold. It must not repeat the title word for word and must not describe the picture ("man on stage"). Think "I QUIT.", "SHE SAID YES", "DON'T WAIT", "100 VS 1". No hashtags, no emojis.
- "accent": the ONE word of "text" to highlight in color (the most emotional or surprising word).
- "position": where the text goes so it does not cover the subject's face: "top" or "bottom".
""" + ENGAGEMENT_RULES + """
Reply with JSON only: {{"frame": <number 1-{n}>, "text": "...", "accent": "...", "position": "top|bottom", "why": "one short sentence"}}"""


@app.post("/api/thumbnail")
def ai_thumbnail(body: dict):
    """Picks the best cover frame among candidates and writes short, bold thumbnail text for it."""
    model = vision_model()
    if not model:
        raise HTTPException(503, "No local vision model found. Install Ollama and run: ollama pull qwen2.5vl:7b")
    frames = [f.split(",", 1)[-1] for f in body.get("frames", [])][:6]
    if not frames:
        raise HTTPException(400, "No frames to choose from.")
    context = ""
    if (body.get("source") or "").strip():
        context += f'\nThe source video is titled: "{body["source"].strip()[:150]}"'
    if (body.get("title") or "").strip():
        context += f'\nThe post title is: "{body["title"].strip()[:150]}"'
    if (body.get("transcript") or "").strip():
        context += f'\nWhat is said in the clip: "{body["transcript"].strip()[:1500]}"'
    if (body.get("hook") or "").strip():
        context += f'\nOn-screen hook: "{body["hook"].strip()[:150]}"'
    n = len(frames)
    schema = {"type": "object", "required": ["frame", "text", "accent", "position"], "properties": {
        "frame": {"type": "integer", "minimum": 1, "maximum": n}, "text": {"type": "string"},
        "accent": {"type": "string"}, "position": {"type": "string", "enum": ["top", "bottom"]}, "why": {"type": "string"}}}
    prompt = THUMB_PROMPT.format(n=n, context=context, language=body.get("language") or "English")
    try:
        out = ollama_json(model, prompt, schema, frames, 0.6)
    except Exception as e:
        raise HTTPException(502, f"{model} failed: {e}")
    text = re.sub(r"#\w+", "", str(out.get("text") or "")).strip().strip('"').strip()
    words = text.split()
    if len(words) > 5:  # the model sometimes writes a sentence; a thumbnail reads at a glance
        text = " ".join(words[:4])
    accent = str(out.get("accent") or "").strip().strip('"')
    if accent.lower() not in text.lower():
        accent = max(text.split(), key=len, default="")
    frame = int(out.get("frame") or 1)
    return {"frame": min(max(frame, 1), n) - 1, "text": text, "accent": accent,
            "position": out.get("position") if out.get("position") in ("top", "bottom") else "top",
            "why": str(out.get("why") or "").strip(), "model": model}


@app.get("/api/health")
def health():
    return {"ok": True, "version": API_VERSION, "model": model_name(), "device": _model_device or DEVICE, "ai": vision_model()}


def model_name() -> str:
    """The Whisper model in use, or the one that will load (GPU model unless forced onto CPU)."""
    if _model_device:
        return GPU_MODEL if _model_device == "cuda" else CPU_MODEL
    return CPU_MODEL if DEVICE == "cpu" else GPU_MODEL


@app.post("/api/transcribe")
async def start_transcription(request: Request, name: str = "", language: str | None = None, media: str = ""):
    # The body is the raw video file, streamed straight to disk (multi-GB uploads stay cheap).
    # A video imported from a link is already on disk: `media` names it and the body is empty.
    if media:
        src = fetch.media_path(media)
        workdir = Path(tempfile.mkdtemp(prefix="clipstudio_"))  # holds only the extracted audio
        job_id = uuid.uuid4().hex
        jobs[job_id] = {"status": "queued", "progress": 0.0}
        executor.submit(run_transcription, job_id, workdir, src, language)
        return {"job": job_id}
    # Uploads are kept in downloads/ (like imported links) so the clips found can be saved and reopened later.
    workdir = Path(tempfile.mkdtemp(prefix="clipstudio_"))
    media = uuid.uuid4().hex
    suffix = Path(name).suffix.lower()
    suffix = suffix if re.fullmatch(r"\.[a-z0-9]{1,5}", suffix) else ".mp4"
    fetch.DOWNLOADS.mkdir(exist_ok=True)
    src = fetch.DOWNLOADS / (media + suffix)
    try:
        with src.open("wb") as f:
            async for chunk in request.stream():
                f.write(chunk)
    except BaseException:
        shutil.rmtree(workdir, ignore_errors=True)
        src.unlink(missing_ok=True)
        raise
    if src.stat().st_size == 0:
        shutil.rmtree(workdir, ignore_errors=True)
        src.unlink(missing_ok=True)
        raise HTTPException(400, "empty upload")
    job_id = uuid.uuid4().hex
    jobs[job_id] = {"status": "queued", "progress": 0.0}
    executor.submit(run_transcription, job_id, workdir, src, language)
    return {"job": job_id, "media": media}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "unknown job")
    if job["status"] in ("done", "error"):
        jobs.pop(job_id)  # results are handed out once
    return {k: v for k, v in job.items() if k != "meta"}


# ---------- import a link and find its moments, in the background ----------
# One job downloads the video, then transcribes it and measures loudness and motion. The bell (notify.py)
# follows it, so the page can be closed meanwhile. The page scores the moments from the saved analysis
# when the result is opened (that part takes a second).
ANALYSIS = ROOT / "data" / "analysis"
auto_jobs: dict[str, dict] = {}


def analysis_file(media: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{32}", media or ""):
        raise HTTPException(404, "unknown video")
    return ANALYSIS / f"{media}.json"


def run_auto(job_id: str, fetch_id: str, opts: dict):
    job, nid = auto_jobs[job_id], auto_jobs[job_id]["nid"]

    def report(progress: float, phase: str, step: str = ""):
        notify.live[nid] = {"progress": round(progress, 3), "phase": phase, "step": step}

    try:
        # 1. Download (0-50 %). start_fetch already started the yt-dlp thread.
        while True:
            f = fetch.fetch_jobs.get(fetch_id) or {"status": "error", "error": "The download was lost."}
            if f["status"] == "done":
                break
            if f["status"] == "error":
                raise RuntimeError("cancelled" if f.get("cancelled") else f.get("error") or "Download failed.")
            if f.get("title") and job.get("title") != f["title"]:
                job["title"] = f["title"]
                notify.update(nid, title=f["title"])
            clock = f.get("clock")
            p = min(0.99, (time.time() - clock[0]) / clock[1]) if clock else f.get("progress", 0.0)
            report(0.5 * p, "recording" if f["status"] == "recording" else "download", f"{round(p * 100)}%")
            time.sleep(1)
        fetch.fetch_jobs.pop(fetch_id, None)
        r = job["result"] = f["result"]
        if job.get("cancel"):
            raise RuntimeError("cancelled")

        # 2. Speech, loudness and motion (50-100 %), on the same single worker as uploads.
        job["analyzing"] = True
        tid = uuid.uuid4().hex
        jobs[tid] = {"status": "queued", "progress": 0.0}
        workdir = Path(tempfile.mkdtemp(prefix="clipstudio_"))
        future = executor.submit(run_transcription, tid, workdir, fetch.media_path(r["media"]), None)
        while not future.done():
            t = jobs.get(tid, {})
            report(0.5 + 0.49 * t.get("progress", 0.0), "analyze", str(t.get("status", "")))
            if job.get("cancel"):
                raise RuntimeError("cancelled")
            time.sleep(1)
        t = jobs.pop(tid, {})
        if t.get("status") != "done":
            raise RuntimeError("Analysis failed: " + str(t.get("error") or "unknown error"))
        ANALYSIS.mkdir(parents=True, exist_ok=True)
        analysis_file(r["media"]).write_text(json.dumps(t["result"]), "utf-8")
        result = {**r, **opts}
        words = sum(len(s.get("words") or []) for s in t["result"].get("segments") or [])
        res = f"{min(r['width'], r['height'])}p · " if r.get("height") else ""
        notify.update(nid, status="done", read=False, progress=1.0, result=result,
                      body=f"{res}{words} words transcribed. Open it to see the {opts['count']} best moments.")
    except Exception as e:
        cancelled = str(e) == "cancelled"
        if cancelled and job.get("result") and not job.get("analyzing"):
            fetch.delete_media(job["result"]["media"])  # while analyzing, ffmpeg still holds it open
        notify.update(nid, status="cancelled" if cancelled else "error", read=cancelled,
                      body="Cancelled." if cancelled else str(e))
    finally:
        auto_jobs.pop(job_id, None)


@app.post("/api/import-moments")
def start_import_moments(body: dict):
    opts = {"len": int(body.get("len") or 60), "count": int(body.get("count") or 5),
            "aspect": "9:16"}  # clips are always Shorts (1080x1920)
    fetch_id = fetch.start_fetch(body)["job"]  # validates the link and the times, then starts downloading
    job_id = uuid.uuid4().hex
    nid = notify.add("moments", str(body.get("url", "")).strip()[:120], "Starting…", status="running",
                     job=job_id, progress=0.0)
    auto_jobs[job_id] = {"nid": nid, "fetch": fetch_id}
    threading.Thread(target=run_auto, args=(job_id, fetch_id, opts), daemon=True).start()
    return {"job": job_id, "notification": nid}


@app.post("/api/import-moments/{job_id}/stop")
def stop_import_moments(job_id: str):
    job = auto_jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Nothing to stop.")
    job["cancel"] = True
    try:
        fetch.stop_fetch(job["fetch"])
    except HTTPException:
        pass
    return {"ok": True}


@app.get("/api/analysis/{media}")
def get_analysis(media: str):
    f = analysis_file(media)
    if not f.exists():
        raise HTTPException(404, "The analysis of this video is gone. Import it again.")
    return FileResponse(f, media_type="application/json")


@app.get("/")
def index():
    return FileResponse(ROOT / "index.html")


@app.get("/reader")
def reader():
    return FileResponse(ROOT / "reader.html")


READ_PROMPT = """This is a screenshot from a video game. Transcribe the text a player would read: dialogue, subtitles,
menus, item descriptions, quest text. Copy it exactly, in its original language, in reading order. Join lines that
belong to the same sentence. Skip HUD numbers, button prompts and decorative text. If there is no readable text, return "".
Reply with JSON only: {"text": "..."}"""


@app.post("/api/read-image")
def read_image(body: dict):
    """Game-screen OCR with the local vision model (better than Tesseract on stylized game fonts)."""
    model = vision_model()
    if not model:
        raise HTTPException(503, "No local vision model found. Install Ollama and run: ollama pull qwen2.5vl:7b")
    image = str(body.get("image") or "").split(",", 1)[-1]
    if not image:
        raise HTTPException(400, "No image.")
    try:
        res = ollama("/api/chat", {
            "model": model,
            "stream": False,
            "format": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
            "options": {"temperature": 0, **OLLAMA_OPTIONS},
            "keep_alive": "30m",
            "messages": [{"role": "user", "content": READ_PROMPT, "images": [image]}],
        }, timeout=300)
        text = str(json.loads(res["message"]["content"]).get("text") or "").strip()
    except Exception as e:
        raise HTTPException(502, f"{model} failed: {e}")
    return {"text": text, "model": model}


# Terms and privacy pages that developer consoles (TikTok, Meta) ask for when registering the app.
@app.get("/terms")
def terms():
    return FileResponse(ROOT / "legal" / "terms.html")


@app.get("/privacy")
def privacy():
    return FileResponse(ROOT / "legal" / "privacy.html")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", 8000)))
