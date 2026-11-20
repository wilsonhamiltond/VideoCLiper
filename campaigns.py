"""Clipping campaigns: a brief you paste in (e.g. from Clipped & Shipped / Whop), used when the AI writes the post text.

Saved in data/campaigns.json as [{"id", "name", "brief", "hashtags": ["#KaiCenat"], "updated"}].
The brief goes into the prompts; the required hashtags are also added to every caption afterwards, so a
campaign rule is never missed because the model forgot it.
"""
import json
import re
import threading
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException

FILE = Path(__file__).parent / "data" / "campaigns.json"
_lock = threading.Lock()
router = APIRouter()


def load() -> list[dict]:
    try:
        items = json.loads(FILE.read_text("utf-8"))
        return items if isinstance(items, list) else []
    except Exception:
        return []


def _write(items: list[dict]):
    FILE.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        FILE.write_text(json.dumps(items, indent=2, ensure_ascii=False), "utf-8")


def clean_tags(v) -> list[str]:
    """'#KaiCenat, gta6 #GTA6' or a list -> ['#KaiCenat', '#gta6', '#GTA6'] without case-insensitive repeats."""
    if isinstance(v, list):
        v = " ".join(str(x) for x in v)
    out, seen = [], set()
    for t in re.findall(r"#?([\w]+)", str(v or "")):
        if t.lower() not in seen:
            seen.add(t.lower())
            out.append("#" + t)
    return out[:10]


def brief_tags(brief: str) -> list[str]:
    """Hashtags written in the brief itself (e.g. 'must use the hashtag: #KaiCenat')."""
    return clean_tags(" ".join(re.findall(r"#\w+", brief or "")))


def from_body(body: dict) -> dict | None:
    """The campaign sent with an AI request: {"id"} for a saved one, or {"name", "brief", "hashtags"} inline."""
    c = body.get("campaign")
    if not isinstance(c, dict):
        return None
    if c.get("id") and not c.get("brief"):
        c = next((x for x in load() if x["id"] == c["id"]), None)
        if not c:
            return None
    brief = str(c.get("brief") or "").strip()[:3000]
    name = str(c.get("name") or "").strip()[:120]
    if not brief and not name:
        return None
    return {"name": name, "brief": brief, "hashtags": clean_tags(c.get("hashtags")) or brief_tags(brief)}


def prompt_block(c: dict | None) -> str:
    """Text added to a prompt's context (no braces: it goes through .format())."""
    if not c:
        return ""
    s = "\n\nTHIS CLIP IS FOR A PAID CLIPPING CAMPAIGN. Follow its brief exactly; it outranks the style tips below."
    if c["name"]:
        s += f'\nCampaign: "{c["name"]}"'
    if c["brief"]:
        s += f'\nCampaign brief:\n"""\n{c["brief"]}\n"""'
    if c["hashtags"]:
        s += ("\nRequired hashtags (put ALL of them in every caption and the YouTube description): "
              + " ".join(c["hashtags"]))
    s += ("\nUse the names, game/show and context the brief gives (e.g. the streamer's name and what they are playing),"
          " so the post is clearly about the campaign's subject.")
    return s.replace("{", "(").replace("}", ")")


def ensure_tags(text: str, tags: list[str]) -> str:
    """Append any required hashtag the text is missing."""
    have = {t.lower() for t in re.findall(r"#\w+", text or "")}
    missing = [t for t in tags if t.lower() not in have]
    if not missing:
        return text
    text = (text or "").rstrip()
    sep = " " if re.search(r"#\w+\s*$", text) else ("\n\n" if text else "")
    return text + sep + " ".join(missing)


def apply(post: dict, c: dict | None) -> dict:
    """Make sure every caption, description and tag list carries the campaign's required hashtags."""
    if not c or not c["hashtags"]:
        return post
    tags = c["hashtags"]
    yt = post.get("youtube") or {}
    if "description" in yt:
        yt["description"] = ensure_tags(yt["description"], tags)[:4900]
    if isinstance(yt.get("tags"), list):
        have = {t.lower() for t in yt["tags"]}
        yt["tags"] = [t[1:] for t in tags if t[1:].lower() not in have] + yt["tags"]
        yt["tags"] = yt["tags"][:12]
    for p in ("tiktok", "instagram", "facebook"):
        if isinstance(post.get(p), dict) and "caption" in post[p]:
            post[p]["caption"] = ensure_tags(post[p]["caption"], tags)[:2200]
    return post


# ---------- API ----------

@router.get("/api/campaigns")
def list_campaigns():
    return sorted(load(), key=lambda c: -c.get("updated", 0))


@router.post("/api/campaigns")
def save_campaign(body: dict):
    """Create ({"name", "brief", "hashtags"}) or update (with "id") a campaign."""
    name = str(body.get("name") or "").strip()[:120]
    brief = str(body.get("brief") or "").strip()[:3000]
    if not name:
        raise HTTPException(400, "Give the campaign a name.")
    if not brief:
        raise HTTPException(400, "Paste the campaign brief.")
    tags = clean_tags(body.get("hashtags")) or brief_tags(brief)
    items = load()
    item = next((c for c in items if c["id"] == body.get("id")), None) if body.get("id") else None
    if not item:
        item = {"id": uuid.uuid4().hex}
        items.append(item)
    item.update(name=name, brief=brief, hashtags=tags, updated=time.time())
    _write(items)
    return item


@router.post("/api/campaigns/{cid}/delete")  # POST, so the X-Clip-Studio header check applies
def delete_campaign(cid: str):
    items = load()
    left = [c for c in items if c["id"] != cid]
    if len(left) == len(items):
        raise HTTPException(404, "That campaign is gone.")
    _write(left)
    return {"ok": True}
