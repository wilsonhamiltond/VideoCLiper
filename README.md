# Clip Studio

Turn a long video into short vertical clips: find the best moments, add word-by-word captions, reframe, export.

- `index.html` – the app (moment scoring, editor, export all run in the browser)
- `server.py` – FastAPI backend: serves the page and transcribes uploads locally with [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- `narrate.py` – voice-over narration: text-to-speech with word timings ([edge-tts](https://github.com/rany2/edge-tts))
- `reader.html` – Game Reader at `/reader`: shares the game screen, reads on-screen text (Tesseract.js, or the local Ollama vision model) and speaks it with the narration voices
- `clone.py` / `voice_server.py` – voice cloning for the reader (Chatterbox, runs in `.venv-voice`)
- `fetch.py` – imports videos and live streams from a link with [yt-dlp](https://github.com/yt-dlp/yt-dlp)
- `stock.py` – makes a short from stock footage: searches [Pexels](https://www.pexels.com/api/) by topic and merges the clips
- `music.py` – background music library (`data/music/`)
- `ANALYSIS.md` – notes on the reference product (viralclip.tech)

## Run

Needs Python 3.10+ and `ffmpeg` on PATH.

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python server.py
```

Open http://127.0.0.1:8000. The first transcription downloads the Whisper model (~460 MB for `small`).

Opening `index.html` directly still works, but without captions, transcript or keyword search.

## Import from a link (YouTube, Twitch, Kick, X…)
Paste a link under the drop zone and press **Import link**. Needs `server.py`. The video downloads to `downloads/` and then works like a dropped file.
- **Quality:** the dropdown next to the link. **Best quality** (the default) takes the highest resolution the video has, 8K included. "Up to 4K / 1440p / 1080p" download smaller and faster. The import message shows the resolution you got.
  - **Why 8K helps vertical clips:** a 9:16 crop of 4K is only 1215×2160, so a 4K vertical export is upscaled. A crop of 8K is 2430×4320, enough for a sharp 4K vertical export.
  - **Codec, same resolution:** SDR is preferred over HDR, and H.264 over VP9 over AV1. In practice, 1080p and below is H.264, 4K is VP9, and YouTube's 8K is AV1 (its only 8K format).
  - **HDR-only videos** are still taken at full resolution. Colors can look flat in clips, and the import message warns you.
  - **8K and AV1 analysis** decodes on the NVIDIA GPU: 42 s of 8K60 analyzed in 15 s, against 55 s on the CPU. Without an NVIDIA GPU it falls back to the CPU, which runs slower than real time for 8K.
  - **Size and time:** 5½ minutes of 8K60 is about 1.3 GB and took ~2 min to download.
  - **Default for API calls without a quality:** `IMPORT_MAX_HEIGHT`, e.g. `1080`. The default is no cap.
- **Long ones:** open **Long video or live stream?** and set **From / To** (e.g. `1:05:00` to `1:35:00`) to keep only that stretch. Clip times then count from "From".
  - On YouTube the whole video is downloaded, then the range is cut out locally. YouTube's own fast download is ~20× real time; asking it for just a range runs at ~1.5×.
  - On other sites only the range is downloaded.
- **Live streams:** records the next 5 min – 2 h from now, then stops by itself. Channel live links work too (`https://www.youtube.com/@channel/live`).
  - **Stop recording** ends it early and keeps what's recorded so far, ready to analyze in about 2 s.
  - The server records with ffmpeg directly, not through yt-dlp, so stopping still leaves a playable MP4. YouTube live sends picture and sound separately, and both are recorded.
- **"Confirm you're not a bot":** give yt-dlp your YouTube login cookies, then restart `server.py`.
  - **cookies.txt (works with any browser):** while signed in to youtube.com, export cookies with a "Get cookies.txt LOCALLY" extension and save the file as `cookies.txt` next to `server.py`. Another path: set `YTDLP_COOKIES`.
  - **Straight from Firefox:** set `YTDLP_COOKIES_FROM_BROWSER=firefox`. Chrome and Edge on Windows lock and encrypt their cookies, so use cookies.txt for those.
  - The cookies are a login: keep `cookies.txt` private. If the message comes back, the cookies expired; export them again.
- **Cancel** stops a normal download and deletes the partial file.
- **Not supported:** playlists and channel pages (paste one video's link), members-only or sign-in-only videos.
- **YouTube** needs a JavaScript runtime for most formats. `fetch.py` uses Deno or Node, whichever is installed.
- **Disk:** downloads are kept so you can re-analyze them. Delete `downloads/` to free the space. A 1-hour video is about 1–2 GB at 1080p, 4–8 GB at 4K and 10–15 GB at 8K.
- **Updates:** if YouTube downloads start failing, update yt-dlp first: `.venv\Scripts\python -m pip install -U "yt-dlp[default]"`.

Only import videos you have the rights to clip and repost.

## Short from stock videos
Press **Stock video** next to **Import link**, type a topic (e.g. `motivation`) and press **Create the short**.
- **What happens:** the server searches Pexels, takes a random set of clips (one per creator where possible), crops each to 9:16, and merges them with 0.5 s crossfades into one 1080×1920 video of the length you picked. A 30 s video from 6 clips takes about 10–30 s, mostly downloading.
- **Then:** it opens as one clip. With **Add an AI voice-over** ticked, the AI writes a script from the frames and your topic (plus anything under "What the voice should say"), and the voice is generated in the chosen language. The **Motivational** style talks to the viewer about the idea instead of describing each shot. Edit it all afterwards under **Narration**.
- **Progress:** a window shows each step (search, download, merge, script, voice) with the overall %, elapsed time and **Cancel**.
- **Voice fills the video:** each take is measured; if it's short, the AI writes a longer one (up to 3 takes), then the voice reads up to ~15% slower. If it's still short, the video ends a second after the voice so there's no silent tail.
- **Each run is different:** press it again for a new mix of clips.
- **API key:** needs a free Pexels key from https://www.pexels.com/api/. Paste it in the dialog once; it's kept in `data/stock.json`, or set `PEXELS_API_KEY`. Pexels allows 200 searches an hour.
- **Rights:** Pexels videos are free to use and edit, with no attribution required. The progress message names the creators if you want to credit them.
- English keywords find the most clips, even for a voice-over in another language.

## Background music
Press **Music** in the editor toolbar, tick **Add background music** and pick a track, or **Add music file…** (MP3, M4A, WAV, OGG, OPUS, FLAC, up to 60 MB).
- Tracks stay in a library in `data/music/`, ready for any clip. The stock video dialog can pick one too.
- Per clip: **Music volume** (25% by default), **Original audio**, and **Start the song at** to skip an intro.
- The song loops if the clip is longer and fades out over the last second, in the preview and in the export.
- Use music you have the rights to, such as the YouTube Audio Library or Pixabay Music.

## Auto-reframe ("Follow the action")
When a 16:9 video is cropped to 9:16 or 1:1, the crop follows what matters instead of staying in one place:
- **Faces first:** the server finds faces 2× a second (OpenCV's YuNet model, in `models/`). With several faces, it follows the one moving most, usually whoever is talking.
- **Then movement:** with no face, it follows the centre of concentrated movement. Whole-frame motion like camera pans is ignored.
- **Like a camera operator:** it holds still inside a dead zone, pans smoothly (at most ~12% of the width per second), and jumps straight to the subject on scene cuts. A new shot with nothing to track is framed centrally.
- **Same in preview and export:** the preview and the exported video use the same path.
- **Manual override:** drag the focus slider to set the crop by hand for a clip; tick **Follow the action** to go back.
- Needs `server.py`. It adds about 2 s per 5 minutes of 1080p to the analysis.

## How captions are made
- **Mixed languages:** Whisper normally picks one language per 30 s and silently drops speech in any other. Here each stretch of speech gets its own language, chosen from the video's main languages (up to 3), so English and Spanish in the same clip are both captioned.
- **Quiet voices:**
  - The audio is loudness-evened (ffmpeg `speechnorm`) before anything listens to it.
  - On GPU, every stretch the speech detector skipped is listened to again in overlapping 8 s windows. Only lines Whisper is confident about are kept, and repeats, loops and known filler phrases ("Thank you.", "We'll be right back.") are dropped.
- **Measured** on a test track of alternating English and Spanish lines, 6 of 10 spoken 22–28 dB under a music-like background:

  | | Before | Now |
  |---|---|---|
  | All words | 22% | **97%** |
  | Quiet English | 0% | **96%** |
  | Quiet Spanish | 0% | **92%** |
  | Pure background | — | 0 invented words |

- **Cost:** the quiet-voice pass makes transcription several times slower. A 2.5-minute trailer takes ~23 s instead of ~2 s; a 27-minute video about 3–4 minutes on an RTX 5070.

## What the backend adds
- **Transcript** with word timestamps (language auto-detected)
- **Auto captions**: word-by-word, the spoken word highlighted; burned into exports
- **Sentence-aligned clips**: starts/ends snap to sentence boundaries
- **Keyword search**: "Find moments about…" builds clips around sentences that mention the words
- **Clickable transcript** per clip to jump around
- **Motion scoring**: ffmpeg samples tiny frames 4×/second and measures how much the picture changes (action, crashes, cuts). It's blended with loudness: 30% motion for talk videos, 60% for speechless ones like gameplay or sports. A 15-minute 4K60 video takes ~1.5 minutes on CPU.

## AI hooks & captions (local, free)
"Write hook & caption with AI" in the editor, or "AI for all" in the clip list, sends 3 frames from the clip and its transcript to a local vision model through [Ollama](https://ollama.com). The model writes a top hook line and a bottom caption, which works for speechless clips like gameplay too. Clips with speech keep their word-by-word captions at the bottom.

Install Ollama, pull a vision model, then restart `server.py`:

```bash
ollama pull qwen2.5vl:7b
```

The server picks the best installed vision model, in this order: qwen2.5vl, gemma3, llama3.2-vision, minicpm-v, llava. Set `OLLAMA_MODEL` to force one. The header shows which model is in use.

On an RTX 5070, qwen2.5vl:7b takes about 45 s to load on the first call, then about 2 s per clip. It stays loaded for 30 minutes after the last call. In side-by-side tests it read the race position and items on screen and kept hooks short. llava:13b mislabeled the game and wrote 15-word hooks.

## Hook intro (best 3–10 s first)
Press **Hook intro** in the editor toolbar and tick **Open with the best 3–10 s of the clip**. The short opens with the clip's strongest 3 to 10 seconds (4 by default), then plays the clip from its start. A dot on the button shows the clip has one.
- **Which moment:** the window with the highest loudness + motion score (its average plus its biggest peak), moved to the nearest word start so it doesn't open mid-word. When the clip is long enough, its own opening seconds are skipped, so the intro shows something new. Drag **Intro taken from** to choose another moment, or press **Pick the best moment** again.
- **Preview:** the modal shows the short as it will be exported, with its own play button and timeline. The purple part of the timeline is the intro, and an **Intro** badge shows while it plays. Changing the intro jumps the preview back to the start of the short.
- **Export:** the intro is recorded first, so the file is the clip length plus the intro (e.g. 30 s + 4 s = 34 s). The intro keeps the original sound and the hook text. The voice-over, its captions and the music start with the clip.

## Reframe and account banner
- **Reframe:** left ↔ right, up ↕ down and zoom (100–300%) are set per clip. Up ↕ down needs zoom first, because a 9:16 crop of a 16:9 video already uses the full height. Zooming in uses fewer source pixels: a 4K source has room for about 2× at 1080p before the export warns that it's enlarged.
- **Account banner:** a banner image, your handle, or both, on every clip. Tick **Show on all clips** under the editor.
  - **Banner image:** click **+ Banner image** and pick a wide graphic, like your channel banner. It covers that part of the video.
    - **Place:** flush with the top or bottom edge, or dragged anywhere.
    - **Width:** 30–100% of the frame. The height follows the image's shape.
    - **Opacity:** adjustable. Transparent PNGs keep their transparency.
    - Shift+click the image button to remove it.
  - **Handle:** your @name in a pill, with an optional round logo (**+ Logo**, cropped to a circle; Shift+click to remove).
    - The handle box suggests your connected accounts' handles and channel names.
    - **Placing it:** drag it, or pick one of 9 positions. It keeps a small gap from the frame edge and is drawn on top of the image.
    - **Look:** dark pill, light pill or text only, plus size and opacity.
  - Dropped near an edge or the centre, either one snaps there.
  - Positions are relative to the frame, so the banner lands in the same spot at 720p, 1080p and 4K. The preview and the export use the same drawing code.
  - Banner settings and images are remembered in this browser. Images are shrunk first (up to 1440 px for the banner, 256 px for the logo) so they fit in the browser's storage.

## Narration (voice-over)
Tick **Add a voice-over** under the captions in the editor.
- **Script:** type what the voice should say, or press **Write script with AI**. The AI (the same local Ollama model as AI hooks) writes a script sized to the clip length, in the style you pick: Hype, Storytime, Explainer, Funny or Documentary. A few words you type first steer what it writes.
- **Timed lines:** start a line with a time to say it at that moment of the clip, counted from the clip start:
  ```
  [0:00] Watch the leader on this corner…
  [0:06] What a save!
  [0:12.5] Now the gap is closing.
  ```
  Each line is voiced on its own and placed at its time. If the line before is still talking, it starts right after, and the note says which lines start late. A script with no times is read in one go from the start.
- **Scene by scene** (the default AI mode) runs in two steps, about 5–15 s per clip:
  1. The AI describes up to 12 frames one at a time, in the narration's language: one frame just after each scene cut, the rest every ~2.5 s.
  2. The clip is split into parts of about 6 s, moved onto nearby scene cuts. The AI writes one line per part from what was seen in it, with a word limit so each line ends before the next.
  - **Why two steps:** asked to look, time and write in one go, the 7B model invented events, put lines on every frame and filled them with generic shouts. One frame at a time, it reads positions and items correctly.
  - **Cleanup:** lines that repeat the speaker's words, or repeat an earlier line, are dropped.
  - **Expect to edit:** quality varies from run to run. Press the button again for another take, and fix odd words by hand. Tested on 8.6 min of kart-racing gameplay in Spanish: about 1 run in 3 was good as is, telling the 3rd→2nd overtake, the items and holding the lead. The others had stray English words or repeated lines.
  - **One block** writes plain text from 4 frames, as before.
- **Script from captions** fills the script with the clip's captions as timed lines, each starting when it's said in the clip, so the voice re-says the speech at the same moments. Turn **Original audio** down to replace the speaker's voice. A clip without a transcript gets its typed hook and caption text, read from the start. Press **Write script with AI** afterwards to turn the captions into commentary: what's in the script box tells the AI what to talk about.
- **Voice requests** go one line at a time, and each is tried up to 3 times over ~4 s: the speech service drops requests in bursts ("No audio was received"), more often when several are sent at once.
- **Voice:** pick a language, then one of its voices (322 voices in 75+ languages). **Multilingual** voices read any language. **Hear this voice** plays your first line, or a sample. **Speed** and **Pitch** change how it sounds.
- **Mix:** **Original audio** sets how loud the video's own sound stays under the voice (20% by default). **Voice volume** sets the narration level.
- **Generate voice & play** makes the audio and plays the clip with it. The voice starts with the clip. The note warns if it runs past the clip end, where it gets cut.
- **Captions** switch to **Narration · word-by-word**, timed to the voice, unless you'd typed a custom caption.
- **Export** mixes the voice in. If you changed the script or voice since generating, export makes the voice again first.
- The voices are Microsoft Edge's online neural voices, through [edge-tts](https://github.com/rany2/edge-tts). They're free with no key, but need internet, and the script text is sent to Microsoft. The video stays on your machine.

## Game Reader (/reader) and voice cloning
Open http://127.0.0.1:8000/reader, share the game window and drag a box around the dialogue. **Auto-read** queues every new screen of text and reads them in order, like a conversation.
- **Characters:** lines like `Name: text` get a voice per character (change them in the Characters panel). ALL-CAPS text, `...` and "next page" arrows are cleaned up before speaking.
- **Clone your voice:** in the reader, tick the consent box, press **Record** and read the script for 10–20 s (or upload a clip). Then pick "★ My voice" as the narrator or for any character. **Emotion** goes from calm to expressive.
  - Runs locally with [Chatterbox Multilingual](https://github.com/resemble-ai/chatterbox) (MIT, 23 languages) in its own environment, so it can't break Whisper. Output carries Chatterbox's inaudible watermark.
  - First use downloads the model (~3 GB, a few minutes); each server start loads it in ~30 s. After that a line starts in ~2–3 s, and queued lines start instantly because they're made while the previous one plays.
  - Record in the language you play in: an English sample reading Spanish can add stray words.
  - Samples are in `voices/`; the worker logs to `voice_server.log`.
- **Install** (once; needs an NVIDIA GPU, ~6 GB disk; Python 3.14 because the RTX 50 series needs PyTorch 2.9+):

```bash
py -3.14 -m venv .venv-voice
.venv-voice\Scripts\python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
.venv-voice\Scripts\python -m pip install chatterbox-tts fastapi "uvicorn[standard]" soundfile
```

## Publish to YouTube, TikTok, Instagram, Facebook
The **Publish** panel under Export writes per-platform titles, descriptions and hashtags with the local AI. It then uploads the clip through each platform's official API, using developer apps you register yourself. Setup steps and each platform's limits are in [PUBLISHING.md](PUBLISHING.md).

**Thumbnail:** **Make thumbnail with AI** in the Publish panel samples 14 frames of the clip with its crop and scores them for sharpness, contrast and exposure. The local vision model then picks the most striking of the best 6 and writes 2–4 bold words for it, with one word highlighted. You can change the words, the color and the text position, pick another frame, or use the frame at the playhead. It takes about 12 s on an RTX 5070.
- **YouTube** gets the designed image as a JPEG through `thumbnails.set`. Custom thumbnails need a phone-verified channel; if the channel isn't verified, the video still uploads and the result says why the thumbnail wasn't set.
- **TikTok** (direct posts) and **Instagram** can't take an image here, so they get that moment of the video as the cover (`video_cover_timestamp_ms` / `thumb_offset`). TikTok drafts pick their cover in the app.
- **Facebook** Reels take no cover through the API; Facebook picks one.

## Settings (env vars)
| Var | Default | |
|---|---|---|
| `WHISPER_MODEL` | `large-v3-turbo` on GPU, `small` on CPU | turbo catches ~35% more dialog than small, but is ~6× slower on CPU |
| `QUIET_BOOST` | `1` | second listening pass for voices quieter than the background (GPU only); `0` is ~4× faster |
| `CAPTION_DEBUG` | off | `1` logs each line the quiet-voice pass adds, with Whisper's confidence |
| `WHISPER_DEVICE` | `auto` | `auto` tries CUDA then falls back to CPU |
| `PORT` | `8000` | |
| `OLLAMA_MODEL` | auto | vision model for AI hooks, e.g. `qwen2.5vl:7b` |
| `OLLAMA_URL` | `http://127.0.0.1:11434` | |

### GPU (optional)
CUDA needs the cuBLAS and cuDNN libraries. Easiest on Windows:

```bash
.venv\Scripts\python -m pip install nvidia-cublas-cu12 nvidia-cudnn-cu12==9.*
```

`server.py` picks their DLLs up automatically and switches to `large-v3-turbo`. On an RTX 5070 it transcribes a 2.5-minute trailer in about 2 s, after a ~25 s warm-up on the first job. Without the libraries it runs `small` on CPU, which drops more short lines and speech over music.

## Large files
Files over 1 GB (e.g. 4K recordings) aren't loaded into browser memory. The server measures their audio with ffmpeg instead, so keep `server.py` running for them. Uploads stream to the system temp folder, which needs free space equal to the file size; they're deleted once the audio has been extracted.

## Limits
- Moment scoring is a loudness + motion heuristic, not an LLM; it can't tell a great overtake from a busy menu screen with animation.
- Export runs at playback speed and saves MP4 (Chrome/Edge) or WebM.
- Jobs are held in memory; one transcription runs at a time.
