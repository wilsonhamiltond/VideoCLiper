"""yt-dlp with a rewind: download the last N seconds of a YouTube stream that is still live.

    python live_rewind.py <seconds> <yt-dlp args...>

YouTube keeps a live stream's earlier footage (up to 120 hours) as numbered segments, which yt-dlp
reaches with --live-from-start. But yt-dlp always starts at segment 0 and keeps going until the stream
ends. This wraps its segment list so the download starts `seconds` before the live edge and stops at the
live edge as it was when the download began. Run with --live-from-start and a format filter that keeps
only the from-start formats (protocol http_dash_segments_generator), so nothing falls back to "record
from now".
"""

import sys

import yt_dlp
from yt_dlp.extractor.youtube import YoutubeIE


def main():
    lookback = float(sys.argv[1])
    original = YoutubeIE._live_adaptive_fragments

    def rewound(self, video_id, itag, client_name, live_start_time, url_feed, base_url, fragment_duration,
                last_seq_cache, ctx):
        per = float(fragment_duration or 5)
        first = last = None
        for frag in original(self, video_id, itag, client_name, live_start_time, url_feed, base_url,
                             fragment_duration, last_seq_cache, ctx):
            if last is None:
                # fragment_count is the live edge's segment number; segments are `per` seconds long.
                last = int(frag["fragment_count"])
                first = max(0, last - int(lookback / per))
                # yt-dlp reports the segment total only now and then; tell the app ourselves (see fetch.py).
                print(f"REWIND {last - first}", flush=True)
            sq = int(frag["url"].rsplit("sq=", 1)[1].split("&", 1)[0])
            if sq < first:
                continue
            if sq >= last:
                return
            # Count only our window, so the progress bar fills over the part we actually fetch.
            yield {**frag, "fragment_count": last - first}

    YoutubeIE._live_adaptive_fragments = rewound
    sys.exit(yt_dlp.main(sys.argv[2:]))


if __name__ == "__main__":
    main()
