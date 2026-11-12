"""Voice-cloning worker: speaks text in a voice copied from a short recording (Chatterbox Multilingual).

Runs in its own environment (.venv-voice, Python 3.14 + PyTorch for the RTX 50 series) so its pinned
packages can't clash with Whisper in .venv. server.py starts it on first use and proxies to it; you
don't run it yourself. Everything stays on this machine. Chatterbox adds an inaudible watermark to
what it generates, so cloned speech can be identified as synthetic.

    .venv-voice/Scripts/python voice_server.py   # listens on 127.0.0.1:8765
"""
import os
import re
import subprocess
import threading
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

PORT = int(os.environ.get("VOICE_PORT", 8765))
LANGS = {"ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi", "it", "ja", "ko", "ms", "nl", "no",
         "pl", "pt", "ru", "sv", "sw", "tr", "zh"}

app = FastAPI(title="Clip Studio voice cloning")
_model = None
_status = {"state": "idle", "error": ""}
_lock = threading.Lock()          # one generation at a time: the model holds a single set of voice conditionals
_conds_key = None                 # (sample path, mtime, emotion) the model is currently conditioned on


@app.middleware("http")
async def local_only(request: Request, call_next):
    # Same rule as server.py: only this machine, and writes must carry the custom header (no cross-site posts).
    host = (request.headers.get("host") or "").rsplit(":", 1)[0]
    if host not in ("127.0.0.1", "localhost") or (request.method == "POST" and request.headers.get("x-clip-studio") != "1"):
        return JSONResponse({"detail": "forbidden"}, status_code=403)
    return await call_next(request)


def model():
    global _model
    if _model is None:
        _status["state"] = "loading"
        try:
            import torch
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS

            device = "cuda" if torch.cuda.is_available() else "cpu"
            m = ChatterboxMultilingualTTS.from_pretrained(device=torch.device(device))
            # The first generation sets up the GPU kernels (~20 s); do it now, not on the player's first line.
            _status["state"] = "warming up"
            with _lock:
                m.generate("Ready.", language_id="en")
            _model = m
            _status.update(state="ready", device=device)
        except Exception as e:
            _status.update(state="error", error=str(e))
            raise
    return _model


@app.get("/health")
def health():
    return {"ok": True, **_status}


@app.post("/load")
def load():
    """Load the model in the background (first time also downloads it, ~3 GB)."""
    if _status["state"] in ("idle", "error"):
        threading.Thread(target=lambda: _safe(model), daemon=True).start()
    return {"ok": True, **_status}


def _safe(fn):
    try:
        fn()
    except Exception:
        pass


def sentences(text: str) -> list[str]:
    parts = [p.strip() for p in re.split(r"(?<=[.!?…。！？])\s+", text) if p.strip()]
    # Very long sentences make the model rush and drift; split them at commas.
    out = []
    for p in parts:
        while len(p) > 220 and "," in p[:220]:
            cut = p.rfind(",", 0, 220) + 1
            out.append(p[:cut].strip()); p = p[cut:].strip()
        out.append(p)
    return out


def to_mp3(wav: np.ndarray, sr: int) -> bytes:
    pcm = (np.clip(wav, -1, 1) * 32767).astype("<i2").tobytes()
    return subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-f", "s16le", "-ar", str(sr), "-ac", "1", "-i", "pipe:0",
         # No Xing/ID3 headers: they state the length of this one piece, and players would stop
         # after the first sentence of the concatenated stream.
         "-c:a", "libmp3lame", "-b:a", "96k", "-write_xing", "0", "-id3v2_version", "0", "-f", "mp3", "pipe:1"],
        input=pcm, capture_output=True, check=True).stdout


@app.post("/speak")
def speak(body: dict):
    sample = Path(str(body.get("sample") or ""))
    if not sample.is_file():
        raise HTTPException(400, "Voice sample not found.")
    text = re.sub(r"\s+", " ", str(body.get("text") or "")).strip()[:3000]
    if not text:
        raise HTTPException(400, "Nothing to say.")
    lang = str(body.get("language") or "en")[:2].lower()
    lang = lang if lang in LANGS else "en"
    emotion = max(0.25, min(1.5, float(body.get("emotion") or 0.5)))
    # Lower guidance gives slower, more relaxed delivery; it pairs well with more emotion. When the sample
    # is in another language than the text, guidance 0 stops the model inventing words in the sample's
    # language (Chatterbox's own advice for cross-language cloning).
    if str(body.get("sample_language") or lang)[:2].lower() != lang:
        cfg = 0.0
    else:
        cfg = 0.5 if emotion <= 0.6 else 0.35
    m = model()

    def audio():
        global _conds_key
        silence = np.zeros(int(m.sr * 0.12), dtype=np.float32)   # a breath between sentences
        for i, part in enumerate(sentences(text)):
            with _lock:
                key = (str(sample), sample.stat().st_mtime, emotion)
                if key != _conds_key:
                    m.prepare_conditionals(str(sample), exaggeration=emotion)
                    _conds_key = key
                wav = m.generate(part, language_id=lang, exaggeration=emotion, cfg_weight=cfg, temperature=0.8)
            wav = wav.squeeze().detach().cpu().numpy().astype(np.float32)
            yield to_mp3(np.concatenate([silence, wav]) if i else wav, m.sr)

    return StreamingResponse(audio(), media_type="audio/mpeg")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
