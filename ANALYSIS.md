# Reference analysis: viralclip.tech

Reviewed 2026-11-03 (public landing page only; the logged-in app was not inspected).

## What the product does
Turns long videos (podcasts, streams, courses) into short vertical clips:
1. **Find moments** – AI picks the most engaging segments ("virality score" 0–100, based on hook strength, emotional intensity, trends).
2. **Keyword-directed clipping** – user types a topic/phrase; AI extracts matching moments (needs a transcript).
3. **Auto captions** – word-by-word animated captions, 50+ languages, custom styling.
4. **Reformat** – auto-resize to 9:16 / 1:1 / 16:9 per platform.
5. **Publish & schedule** – OAuth to TikTok, Instagram, YouTube Shorts, X, Facebook, Threads; AI picks posting times.
6. **Analytics** – views, engagement rate, watch time, clips created.

Monetisation: free plan + Pro/Team subscriptions, prepaid AI tokens ($4.99 / 500), or BYOK (own OpenAI / Anthropic / Gemini key).

## Landing page structure
Hero ("Create less. Grow more.") → social proof stats → testimonials → 7 numbered features → platform integrations → AI access (tokens vs BYOK + comparison table) → analytics preview → CTA → lead-magnet email form → FAQ.
Visual style: light background, violet→pink→orange gradient accents, rounded cards, floating "Hook detected / Auto captions / 3X faster" badges.

## What the core "create clip" flow needs
| Step | Needs a backend? | In `index.html` |
|---|---|---|
| Upload video | No (local file) | ✅ drag & drop |
| Import from YouTube URL | Yes (download server) | ❌ not possible client-side |
| Find best moments | Basic: no / AI: yes | ✅ audio-energy + hook scoring in browser |
| Virality score | Same | ✅ heuristic 0–100 |
| Transcript, captions, keyword search | Yes (speech-to-text, e.g. Whisper) | ⚠️ manual hook/caption text overlay only |
| Reframe to 9:16 / 1:1 | No | ✅ crop with focus slider |
| Export clip | No | ✅ real-time WebM render (canvas + MediaRecorder) |
| Publish / schedule / analytics | Yes (OAuth + DB) | ❌ |

## Suggested next steps for a real product
- Backend: upload to object storage, FFmpeg workers for cutting/encoding to MP4.
- Transcription (Whisper or similar) → word timestamps → captions + keyword search.
- LLM pass over the transcript to rank moments by hook/story/payoff and write titles.
- Face tracking for automatic reframing.
- OAuth publishing + scheduler + analytics store.
