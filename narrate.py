"""Voice-over narration: turns a script into speech with word timings, in any of ~300 voices.

Uses Microsoft Edge's online neural voices through edge-tts (free, no key). The script text is sent
to Microsoft's speech service; the video never leaves this machine.
"""
import asyncio
import base64
import re
import time

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

router = APIRouter()
_voices: tuple[float, list] = (0.0, [])
TICKS = 1e7  # edge-tts times are in 100 ns units


@router.get("/api/voices")
async def voices():
    global _voices
    import edge_tts

    if time.time() - _voices[0] > 24 * 3600 or not _voices[1]:
        try:
            raw = await edge_tts.list_voices()
        except Exception as e:
            raise HTTPException(502, f"Couldn't load the voice list (needs internet): {e}")
        out = []
        for v in raw:
            # "Microsoft Jorge Online (Natural) - Spanish (Mexico)" -> "Jorge"
            name = re.sub(r"^Microsoft\s+|\s+Online.*$", "", v.get("FriendlyName", "")) or v["ShortName"]
            out.append({"id": v["ShortName"], "name": name, "gender": v.get("Gender", ""), "locale": v["Locale"],
                        "multilingual": "Multilingual" in v["ShortName"],
                        "tags": v.get("VoiceTag", {}).get("VoicePersonalities", [])})
        _voices = (time.time(), out)
    return {"voices": _voices[1]}


def attach_punctuation(text: str, words: list[dict]) -> list[dict]:
    """Edge reports bare words ("narración"); put back the script's punctuation ("narración.") so captions
    break on sentences like the transcript captions do."""
    tokens, k = text.split(), 0
    norm = lambda s: re.sub(r"\W+", "", s.lower())
    for w in words:
        target = norm(w["w"])
        for j in range(k, min(k + 4, len(tokens))):
            if target and target in norm(tokens[j]):
                if norm(tokens[j]) == target:
                    w["w"] = tokens[j]
                k = j + 1
                break
    return words


@router.get("/api/speak")
async def speak(text: str, voice: str = "en-US-AndrewMultilingualNeural", rate: int = 0):
    """Streams MP3 as the speech service produces it, so a player can start on the first chunk
    (the Game Reader uses it; /api/narration waits for the whole clip because it needs word timings)."""
    import edge_tts

    text = re.sub(r"\s+", " ", text).strip()[:3000]
    if not text:
        raise HTTPException(400, "Nothing to say.")
    rate = max(-50, min(100, rate))

    # Edge takes longer to start on long text (~2 s for a paragraph vs ~0.7 s for a sentence), so each
    # sentence is synthesized separately, a few at a time, and streamed in order: the first sentence
    # plays almost at once while the next ones are already being made. MP3 frames concatenate cleanly.
    parts = [p for p in re.split(r"(?<=[.!?…])\s+", text) if p.strip()]
    sem = asyncio.Semaphore(3)

    async def synth(part: str, out: asyncio.Queue):
        async with sem:
            for attempt in range(3):
                sent = False
                try:
                    async for chunk in edge_tts.Communicate(part, voice, rate=f"{rate:+d}%").stream():
                        if chunk["type"] == "audio":
                            sent = True
                            await out.put(chunk["data"])
                    break
                except Exception:
                    # Retry dropped requests only while nothing was sent for this sentence.
                    if sent or attempt == 2:
                        break
                    await asyncio.sleep(0.5 + attempt)
        await out.put(None)

    async def audio():
        queues = [asyncio.Queue() for _ in parts]
        tasks = [asyncio.create_task(synth(p, q)) for p, q in zip(parts, queues)]
        try:
            for q in queues:
                while (data := await q.get()) is not None:
                    yield data
        finally:  # the player stopped or moved on: don't keep synthesizing
            for t in tasks:
                t.cancel()

    return StreamingResponse(audio(), media_type="audio/mpeg", headers={"Cache-Control": "no-store"})


@router.post("/api/narration")
async def narration(body: dict):
    import edge_tts

    text = re.sub(r"\s+", " ", str(body.get("text") or "")).strip()
    if not text:
        raise HTTPException(400, "Write something for the voice to say first.")
    if len(text) > 3000:
        raise HTTPException(400, "The script is too long (max 3000 characters).")
    voice = str(body.get("voice") or "en-US-AndrewMultilingualNeural")
    rate, pitch = int(body.get("rate") or 0), int(body.get("pitch") or 0)
    rate, pitch = max(-50, min(100, rate)), max(-50, min(50, pitch))
    # The service drops requests now and then, in bursts ("No audio was received"); the same text works
    # a moment later. Three tries over ~4 s get through.
    for attempt in range(3):
        com = edge_tts.Communicate(text, voice, rate=f"{rate:+d}%", pitch=f"{pitch:+d}Hz", boundary="WordBoundary")
        audio, words = bytearray(), []
        try:
            async for chunk in com.stream():
                if chunk["type"] == "audio":
                    audio += chunk["data"]
                elif chunk["type"] == "WordBoundary":
                    s = chunk["offset"] / TICKS
                    words.append({"w": chunk["text"], "s": round(s, 3), "e": round(s + chunk["duration"] / TICKS, 3)})
            break
        except Exception as e:
            if attempt == 2:
                raise HTTPException(502, f"Text-to-speech failed after 3 tries (needs internet; if it keeps failing, try another voice): {e}")
            await asyncio.sleep(1 + 2 * attempt)
    if not audio:
        raise HTTPException(502, "The speech service returned no audio. Try another voice.")
    return {
        "audio": "data:audio/mpeg;base64," + base64.b64encode(bytes(audio)).decode(),
        "words": attach_punctuation(text, words),
        "voice": voice,
    }
