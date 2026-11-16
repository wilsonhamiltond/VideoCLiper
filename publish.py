"""Publishing clips to YouTube Shorts, TikTok, Instagram Reels and Facebook Page Reels through each platform's official API.

You register your own developer app on each platform (see PUBLISHING.md). App keys and the
OAuth tokens are kept in data/accounts.json on this machine only.
"""
import base64
import hashlib
import json
import os
import secrets
import shutil
import subprocess
import tempfile
import threading
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse

DATA = Path(os.environ.get("CLIPSTUDIO_DATA") or Path(__file__).parent / "data")
ACCOUNTS = DATA / "accounts.json"
GRAPH = "https://graph.facebook.com/v26.0"
PLATFORMS = ("youtube", "tiktok", "instagram", "facebook")
NAMES = {"youtube": "YouTube", "tiktok": "TikTok", "instagram": "Instagram", "facebook": "Facebook"}
FB_REEL_MAX_S = 90  # Facebook Reels made through the API: 3 to 90 seconds

router = APIRouter()
_lock = threading.Lock()
_oauth_pending: dict[str, dict] = {}  # state -> {platform, verifier, redirect_uri}
_media: dict[str, Path] = {}  # rendered clips, transcoded to MP4, waiting to be published
_jobs: dict[str, dict] = {}
_executor = ThreadPoolExecutor(max_workers=3)


class PublishError(Exception):
    pass


# ---------- storage ----------
# data/accounts.json:
#   "apps":     API credentials, one per developer app: {id, platform, label, client_id, client_secret, config_id?}
#   "accounts": user accounts that sign in through an app: {id, platform, label, app_id, login?, ig_username?, token?, name?, ig_user_id?,
#               fb_page? (wanted Facebook Page: name or ID), fb_page_id? (the Page actually connected),
#               yt_channel? (wanted channel: "@handle" or "UC…"), yt_channel_id?, yt_handle? (the channel actually connected),
#               brand_banner?, brand_logo? (branding images as data: URLs, the defaults for the clip banner),
#               brand_handle? (the @name drawn on clips),
#               follow_on?, follow_text?, follow_btn?, follow_at?, follow_dur? (the "Follow me" scene in clips)}
# Any number of each per platform; several accounts can share one app.
KEYS = ("client_id", "client_secret", "config_id")


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


def _read() -> dict:
    data = json.loads(ACCOUNTS.read_text()) if ACCOUNTS.exists() else {}
    changed = False
    if "accounts" not in data:  # oldest files: one account per platform, keyed by platform name
        data = {"accounts": [{"id": _new_id(), "platform": p, "label": NAMES[p], **v}
                             for p, v in data.items() if p in PLATFORMS and isinstance(v, dict)]}
        changed = True
    if "apps" not in data:  # app keys used to live on each account: move them out, one app per distinct key set
        apps = []
        for a in data["accounts"]:
            keys = {k: a.pop(k) for k in KEYS if k in a}
            if not keys.get("client_id"):
                continue
            app = next((x for x in apps if x["platform"] == a["platform"] and all(x.get(k) == keys.get(k) for k in KEYS)), None)
            if not app:
                n = sum(x["platform"] == a["platform"] for x in apps) + 1
                app = {"id": _new_id(), "platform": a["platform"], "label": f"{NAMES[a['platform']]} app" + (f" {n}" if n > 1 else ""), **keys}
                apps.append(app)
            a["app_id"] = app["id"]
        data = {"apps": apps, "accounts": data["accounts"]}
        changed = True
    if changed and (data["apps"] or data["accounts"]):
        _write(data)
    return data


def _write(data: dict):
    DATA.mkdir(exist_ok=True)
    ACCOUNTS.write_text(json.dumps(data, indent=2))


def load() -> list[dict]:
    with _lock:
        return _read()["accounts"]


def load_apps() -> list[dict]:
    with _lock:
        return _read()["apps"]


def get(acct_id: str) -> dict:
    """An account merged with its app's keys (what OAuth and publishing need)."""
    with _lock:
        data = _read()
    acct = next((a for a in data["accounts"] if a["id"] == acct_id), None)
    if acct is None:
        raise PublishError("That account was removed. Pick another one under Accounts.")
    app = next((x for x in data["apps"] if x["id"] == acct.get("app_id")), {})
    return {**{k: app.get(k) for k in KEYS}, "app_label": app.get("label"), **acct}


def update(acct_id: str, **fields) -> dict:
    with _lock:
        data = _read()
        acct = next((a for a in data["accounts"] if a["id"] == acct_id), None)
        if acct is None:
            raise PublishError("That account was removed. Pick another one under Accounts.")
        for k, v in fields.items():
            if v is None:
                acct.pop(k, None)
            else:
                acct[k] = v
        _write(data)
        return acct


# ---------- http ----------
def http(method: str, url: str, *, form=None, body=None, raw: bytes | None = None, headers=None, timeout=60):
    """Small urllib wrapper. Returns (status, headers, parsed JSON or {})."""
    h = dict(headers or {})
    data = None
    if form is not None:
        data = urllib.parse.urlencode(form).encode()
        h.setdefault("Content-Type", "application/x-www-form-urlencoded")
    elif body is not None:
        data = json.dumps(body).encode()
        h.setdefault("Content-Type", "application/json; charset=UTF-8")
    elif raw is not None:
        data = raw
        h.setdefault("Content-Type", "application/octet-stream")
    req = urllib.request.Request(url, data=data, headers=h, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            content = r.read()
            try:
                parsed = json.loads(content) if content else {}
            except ValueError:
                parsed = {}
            return r.status, r.headers, parsed
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:600]
        raise PublishError(f"{_service_of(url)} returned {e.code}: {detail}")
    except urllib.error.URLError as e:
        raise PublishError(f"Couldn't reach {urllib.parse.urlparse(url).netloc}: {e.reason}")


def _service_of(url: str) -> str:
    """Who answered, for error messages. Instagram and Facebook both go through Meta's Graph API."""
    host = urllib.parse.urlparse(url).netloc
    return ("YouTube" if "google" in host else "TikTok" if "tiktok" in host
            else "Meta (Facebook / Instagram)" if "facebook" in host else host)


# ---------- OAuth ----------
def redirect_uri(request: Request, platform: str) -> str:
    # "localhost" is what the developer consoles accept for local testing; register exactly this.
    # It is per platform, not per account, so one developer app can serve several accounts.
    return f"http://localhost:{request.url.port or 80}/api/auth/{platform}/callback"


def auth_url(acct: dict, redirect: str, state: str, verifier: str) -> str:
    platform = acct["platform"]
    if platform == "youtube":
        login = acct.get("login") or ""
        return "https://accounts.google.com/o/oauth2/v2/auth?" + urllib.parse.urlencode({
            "client_id": acct["client_id"], "redirect_uri": redirect, "response_type": "code", "state": state,
            "scope": "https://www.googleapis.com/auth/youtube.upload https://www.googleapis.com/auth/youtube.readonly",
            "access_type": "offline",
            # Always show the chooser: a Google login can own several channels (brand accounts), and the
            # channel picked there is the one the token uploads to. A login email just preselects the login.
            "prompt": "select_account consent", **({"login_hint": login} if "@" in login else {}),
        })
    if platform == "tiktok":
        # Login Kit for Desktop: PKCE with a hex-encoded SHA-256 challenge (not base64url).
        return "https://www.tiktok.com/v2/auth/authorize/?" + urllib.parse.urlencode({
            "client_key": acct["client_id"], "redirect_uri": redirect, "response_type": "code", "state": state,
            # video.publish posts directly (public only once TikTok audits the app); video.upload sends a draft
            # to the creator's TikTok inbox, which works for unaudited apps and can then be posted publicly.
            "scope": "user.info.basic,video.publish,video.upload",
            "code_challenge": hashlib.sha256(verifier.encode()).hexdigest(), "code_challenge_method": "S256",
        })
    # Facebook Login for Business apps use a login configuration ID; classic Facebook Login uses scopes.
    scope = ("pages_show_list,pages_read_engagement,pages_manage_posts,business_management" if platform == "facebook" else
             "instagram_basic,instagram_content_publish,pages_show_list,pages_read_engagement,business_management")
    perms = {"config_id": acct["config_id"]} if acct.get("config_id") else {"scope": scope}
    return "https://www.facebook.com/v26.0/dialog/oauth?" + urllib.parse.urlencode({
        "client_id": acct["client_id"], "redirect_uri": redirect, "response_type": "code", "state": state, **perms,
        # Facebook remembers which Pages / Instagram accounts were shared last time and skips that screen;
        # rerequest shows it again on every Connect, so a Page left unticked can be added.
        "auth_type": "rerequest",
    })


def exchange_code(acct: dict, code: str, redirect: str, verifier: str):
    now, platform, aid = time.time(), acct["platform"], acct["id"]
    if platform == "youtube":
        _, _, t = http("POST", "https://oauth2.googleapis.com/token", form={
            "code": code, "client_id": acct["client_id"], "client_secret": acct["client_secret"],
            "redirect_uri": redirect, "grant_type": "authorization_code",
        })
        token = {"access_token": t["access_token"], "refresh_token": t.get("refresh_token"), "expires_at": now + t.get("expires_in", 3600)}
        ch = youtube_channel(token["access_token"])
        if not ch:
            raise PublishError("This Google account has no YouTube channel yet. Create one on youtube.com first.")
        # Same channel ID as before = same channel, even if its @handle was renamed since.
        if ch["id"] != acct.get("yt_channel_id") and not _channel_matches(acct.get("yt_channel"), ch):
            raise PublishError(f"You picked the channel {_channel_text(ch)}, but this account is set to {acct['yt_channel']}. "
                               "Click Connect again and choose that channel in Google's account chooser "
                               "(channels you manage appear there as separate entries).")
        update(aid, token=token)
        _sync_channel(get(aid), ch)
    elif platform == "tiktok":
        _, _, t = http("POST", "https://open.tiktokapis.com/v2/oauth/token/", form={
            "client_key": acct["client_id"], "client_secret": acct["client_secret"], "code": code,
            "grant_type": "authorization_code", "redirect_uri": redirect, "code_verifier": verifier,
        })
        if "access_token" not in t:
            raise PublishError(f"TikTok token exchange failed: {t}")
        token = {"access_token": t["access_token"], "refresh_token": t["refresh_token"], "expires_at": now + t["expires_in"],
                 "refresh_expires_at": now + t.get("refresh_expires_in", 0), "open_id": t.get("open_id")}
        update(aid, token=token)
        info = tiktok_creator(token["access_token"])
        update(aid, name="@" + info["creator_username"] if info.get("creator_username") else info.get("creator_nickname"))
    else:
        q = {"client_id": acct["client_id"], "client_secret": acct["client_secret"]}
        _, _, short = http("GET", f"{GRAPH}/oauth/access_token?" + urllib.parse.urlencode({**q, "redirect_uri": redirect, "code": code}))
        _, _, t = http("GET", f"{GRAPH}/oauth/access_token?" + urllib.parse.urlencode(
            {**q, "grant_type": "fb_exchange_token", "fb_exchange_token": short["access_token"]}))
        tok = t["access_token"]
        if platform == "facebook":
            _connect_facebook_page(acct, tok, now)
            return
        _, _, pages = http("GET", f"{GRAPH}/me/accounts?" + urllib.parse.urlencode(
            {"fields": "name,instagram_business_account{id,username}", "access_token": tok}))
        igs = [p["instagram_business_account"] for p in pages.get("data", []) if p.get("instagram_business_account")]
        if not igs:
            raise PublishError("No Instagram professional account found. Link a Business or Creator account to a Facebook Page you manage, then connect again.")
        # One Facebook login can manage several Instagram accounts: use the one named on this account,
        # else the first one no other Clip Studio account uses yet.
        want = (acct.get("ig_username") or "").strip().lstrip("@").lower()
        if want:
            ig = next((i for i in igs if (i.get("username") or "").lower() == want), None)
            if not ig:
                found = ", ".join("@" + (i.get("username") or "?") for i in igs)
                raise PublishError(f"@{want} isn't among the Instagram accounts this Facebook login manages ({found}).")
        else:
            taken = {a.get("ig_user_id") for a in load() if a["id"] != aid}
            ig = next((i for i in igs if i["id"] not in taken), igs[0])
        update(aid, token={"access_token": tok, "expires_at": now + t.get("expires_in", 60 * 86400)},
               ig_user_id=ig["id"], name="@" + ig.get("username", ""))


def _business_pages(user_token: str) -> list[dict]:
    """Pages reached through a business portfolio rather than a personal Page role: /me/accounts leaves these out."""
    fields = "id,name,access_token,tasks"
    try:
        _, _, r = http("GET", f"{GRAPH}/me/businesses?" + urllib.parse.urlencode({
            "fields": f"owned_pages.limit(100){{{fields}}},client_pages.limit(100){{{fields}}}",
            "limit": "50", "access_token": user_token}))
    except PublishError:
        return []  # no business_management permission or no portfolio: the personal Page list is all there is
    out = []
    for b in r.get("data", []):
        for kind in ("owned_pages", "client_pages"):
            out += [p for p in (b.get(kind) or {}).get("data", []) if p.get("access_token")]
    return out


def _connect_facebook_page(acct: dict, user_token: str, now: float):
    """Pick the Page this account posts to and keep its Page token (posting as a Page needs the Page's own token)."""
    _, _, r = http("GET", f"{GRAPH}/me/accounts?" + urllib.parse.urlencode(
        {"fields": "id,name,access_token,tasks", "limit": "100", "access_token": user_token}))
    pages = [p for p in r.get("data", []) if p.get("access_token")]
    pages += [p for p in _business_pages(user_token) if p["id"] not in {x["id"] for x in pages}]
    if not pages:
        raise PublishError("No Facebook Page found. Reels can only be posted to a Page you manage (not a personal profile). "
                           "Create a Page, or tick it in Facebook's login popup, then connect again.")
    want = " ".join((acct.get("fb_page") or "").split()).lower()
    if want:
        page = next((p for p in pages if p["id"] == want or " ".join((p.get("name") or "").split()).lower() == want), None)
        if not page:
            found = ", ".join(f"\"{p.get('name')}\"" for p in pages)
            raise PublishError(f"The Page \"{acct['fb_page']}\" isn't among the Pages this Facebook login gave access to ({found}). "
                               "Click Connect again and tick it under Edit access in Facebook's popup. If it isn't offered there, "
                               "your Facebook profile needs content access to that Page (Page settings → Page access).")
    else:
        taken = {a.get("fb_page_id") for a in load() if a["id"] != acct["id"]}
        page = next((p for p in pages if p["id"] not in taken), pages[0])
    if page.get("tasks") and "CREATE_CONTENT" not in page["tasks"]:
        raise PublishError(f"Your role on the Page \"{page.get('name')}\" can't create posts. Ask a Page admin for content access.")
    # A Page token obtained from a long-lived user token doesn't expire; it stops working only if the
    # login is revoked or the password changes, and then the Graph API's error says so.
    update(acct["id"], token={"access_token": page["access_token"], "expires_at": now + 10 * 365 * 86400},
           fb_page_id=page["id"], name=page.get("name"))


def youtube_channel(token: str) -> dict | None:
    """The one channel an OAuth token acts for (the one picked in Google's chooser)."""
    _, _, r = http("GET", "https://www.googleapis.com/youtube/v3/channels?part=snippet&mine=true",
                   headers={"Authorization": f"Bearer {token}"})
    items = r.get("items") or []
    if not items:
        return None
    s = items[0]["snippet"]
    return {"id": items[0]["id"], "title": s.get("title") or "", "handle": s.get("customUrl")}


def _norm_channel(v: str | None) -> str:
    """'@Handle', 'handle', 'UC…' or a youtube.com/@handle or /channel/UC… link -> comparable form."""
    # Handles can hold accents (@raícesdeacero1): decode %C3%AD from pasted links and use one Unicode form.
    v = unicodedata.normalize("NFC", urllib.parse.unquote((v or "").strip()))
    for marker in ("/channel/", "youtube.com/"):
        if marker in v:
            v = v.split(marker, 1)[1]
    v = v.split("/")[0].split("?")[0]
    if v.startswith("UC") and len(v) == 24:
        return v  # channel IDs are case-sensitive
    return "@" + v.lstrip("@").lower() if v else ""


def _channel_matches(want: str | None, ch: dict) -> bool:
    want = _norm_channel(want)
    if not want:
        return True
    return want == ch["id"] or (bool(ch.get("handle")) and want == _norm_channel(ch["handle"]))


def _sync_channel(acct: dict, ch: dict) -> None:
    """Store the channel's current title/@handle; a wanted @handle that was renamed follows the new one."""
    fields = {"yt_channel_id": ch["id"], "name": ch["title"], "yt_handle": ch.get("handle")}
    want = _norm_channel(acct.get("yt_channel"))
    if want.startswith("@") and ch.get("handle") and not _channel_matches(want, ch):
        fields["yt_channel"] = _norm_channel(ch["handle"])
    if any(acct.get(k) != v for k, v in fields.items()):
        update(acct["id"], **fields)


def _channel_text(ch: dict) -> str:
    return f"\"{ch['title']}\"" + (f" ({ch['handle']})" if ch.get("handle") else "")


def access_token(acct_id: str) -> str:
    acct = get(acct_id)
    platform, token = acct["platform"], acct.get("token")
    if not token:
        raise PublishError(f"{acct['label']} isn't connected.")
    if token["expires_at"] > time.time() + 60:
        return token["access_token"]
    if platform == "youtube" and token.get("refresh_token"):
        _, _, t = http("POST", "https://oauth2.googleapis.com/token", form={
            "client_id": acct["client_id"], "client_secret": acct["client_secret"],
            "refresh_token": token["refresh_token"], "grant_type": "refresh_token",
        })
        token.update(access_token=t["access_token"], expires_at=time.time() + t.get("expires_in", 3600))
    elif platform == "tiktok" and token.get("refresh_expires_at", 0) > time.time():
        _, _, t = http("POST", "https://open.tiktokapis.com/v2/oauth/token/", form={
            "client_key": acct["client_id"], "client_secret": acct["client_secret"],
            "grant_type": "refresh_token", "refresh_token": token["refresh_token"],
        })
        # TikTok rotates refresh tokens: always store the new one.
        token.update(access_token=t["access_token"], refresh_token=t["refresh_token"],
                     expires_at=time.time() + t["expires_in"], refresh_expires_at=time.time() + t.get("refresh_expires_in", 0))
    else:
        raise PublishError(f"The login for {acct['label']} expired. Connect it again.")
    update(acct_id, token=token)
    return token["access_token"]


# ---------- API credentials and accounts ----------
def _public_app(x: dict, accounts: list[dict]) -> dict:
    """What the page may see of an app: never the secret."""
    return {"id": x["id"], "platform": x["platform"], "label": x["label"], "client_id": x.get("client_id"),
            "config_id": x.get("config_id"), "accounts": sum(a.get("app_id") == x["id"] for a in accounts)}


def _public(a: dict, apps: list[dict]) -> dict:
    """What the page may see of an account: never tokens."""
    app = next((x for x in apps if x["id"] == a.get("app_id")), None)
    return {"id": a["id"], "platform": a["platform"], "label": a.get("label") or NAMES[a["platform"]],
            "app_id": app and app["id"], "app_label": app and app["label"], "ig_username": a.get("ig_username"), "login": a.get("login"),
            "yt_channel": a.get("yt_channel"), "yt_channel_id": a.get("yt_channel_id"), "yt_handle": a.get("yt_handle"),
            "fb_page": a.get("fb_page"), "fb_page_id": a.get("fb_page_id"),
            "configured": bool(app and app.get("client_id") and app.get("client_secret")),
            "connected": bool(a.get("token")), "name": a.get("name"),
            "brand_banner": a.get("brand_banner"), "brand_logo": a.get("brand_logo"), "brand_handle": a.get("brand_handle"),
            **{k: a.get(k) for k in ("follow_on", "follow_text", "follow_btn", "follow_at", "follow_dur")}}


def _find(acct_id: str) -> dict:
    try:
        return get(acct_id)
    except PublishError as e:
        raise HTTPException(404, str(e))


def _find_app(app_id: str, platform: str | None = None) -> dict:
    app = next((x for x in load_apps() if x["id"] == app_id), None)
    if app is None:
        raise HTTPException(404 if platform is None else 400, "Pick API credentials for this account.")
    if platform and app["platform"] != platform:
        raise HTTPException(400, f"Those API credentials are for {NAMES[app['platform']]}, not {NAMES[platform]}.")
    return app


def _clean(v) -> str | None:
    return (str(v).strip() or None) if v is not None else None


def _label(body: dict, what: str) -> str:
    label = _clean(body.get("label"))
    if not label:
        raise HTTPException(400, f"Give the {what} a name.")
    return label[:60]


@router.get("/api/accounts")
def accounts(request: Request):
    with _lock:
        data = _read()
    return {"apps": [_public_app(x, data["accounts"]) for x in data["apps"]],
            "accounts": [_public(a, data["apps"]) for a in data["accounts"]],
            "platforms": {p: {"name": NAMES[p], "redirect_uri": redirect_uri(request, p)} for p in PLATFORMS}}


# --- API credentials (developer apps) ---
def _app_fields(body: dict, current: dict | None = None) -> dict:
    cid, secret = _clean(body.get("client_id")), _clean(body.get("client_secret"))
    if not secret and current and current.get("client_id") == cid:
        secret = current.get("client_secret")  # a blank secret keeps the saved one
    if not cid or not secret:
        raise HTTPException(400, "Both the app ID / client key and the secret are required.")
    return {"label": _label(body, "credentials"), "client_id": cid, "client_secret": secret, "config_id": _clean(body.get("config_id"))}


@router.post("/api/apps")
def create_app(body: dict):
    platform = body.get("platform", "")
    _check(platform)
    app = {"id": _new_id(), "platform": platform, **{k: v for k, v in _app_fields(body).items() if v is not None}}
    with _lock:
        data = _read()
        data["apps"].append(app)
        _write(data)
        return _public_app(app, data["accounts"])


@router.post("/api/apps/{app_id}/config")
def configure_app(app_id: str, body: dict):
    with _lock:
        data = _read()
        app = next((x for x in data["apps"] if x["id"] == app_id), None)
        if app is None:
            raise HTTPException(404, "Those API credentials were removed.")
        fields = _app_fields(body, app)
        if any(fields.get(k) != app.get(k) for k in KEYS):
            # Logins were issued to the old app; they won't refresh with new keys.
            for a in data["accounts"]:
                if a.get("app_id") == app_id:
                    for k in ("token", "name", "ig_user_id", "yt_channel_id", "yt_handle", "fb_page_id"):
                        a.pop(k, None)
        for k, v in fields.items():
            if v is None:
                app.pop(k, None)
            else:
                app[k] = v
        _write(data)
        return _public_app(app, data["accounts"])


@router.post("/api/apps/{app_id}/delete")  # POST, so the X-Clip-Studio header check applies
def delete_app(app_id: str):
    with _lock:
        data = _read()
        users = [a["label"] for a in data["accounts"] if a.get("app_id") == app_id]
        if users:
            raise HTTPException(400, f"Still used by {', '.join(users)}. Remove those accounts or switch them to other credentials first.")
        data["apps"] = [x for x in data["apps"] if x["id"] != app_id]
        _write(data)
    return {"ok": True}


# --- user accounts ---
BRAND_KEYS = ("brand_banner", "brand_logo")
BRAND_MAX = 3_000_000  # characters of data: URL; the page shrinks images well below this
BRAND_TYPES = ("data:image/png;base64,", "data:image/jpeg;base64,", "data:image/webp;base64,")


def _brand_image(v, what: str) -> str | None:
    """A branding image from the page: a small data: URL, or empty to remove it."""
    v = (v or "").strip() if isinstance(v, str) else ""
    if not v:
        return None
    if not v.startswith(BRAND_TYPES):
        raise HTTPException(400, f"The {what} must be a PNG, JPG or WebP image.")
    if len(v) > BRAND_MAX:
        raise HTTPException(400, f"The {what} is too large. Use a smaller image.")
    return v


def _account_fields(body: dict, platform: str) -> dict:
    out = {"label": _label(body, "account"), "app_id": _find_app(body.get("app_id") or "", platform)["id"],
           # The email / username you sign in with. Never a password: the platforms only allow OAuth sign-in.
           "login": (_clean(body.get("login")) or "")[:120] or None}
    if platform == "instagram":
        out["ig_username"] = (_clean(body.get("ig_username")) or "").lstrip("@") or None
    if platform == "facebook":
        out["fb_page"] = (_clean(body.get("fb_page")) or "")[:120] or None
    if platform == "youtube":
        raw = (_clean(body.get("yt_channel")) or "")[:200]
        norm = _norm_channel(raw)
        if raw and norm in ("", "@"):
            raise HTTPException(400, "Enter the channel as @handle, a channel ID (UC…) or its youtube.com link.")
        out["yt_channel"] = norm or None
    # Branding is only changed when sent: a missing key keeps the saved image, an empty one removes it.
    for k, what in zip(BRAND_KEYS, ("banner", "logo")):
        if k in body:
            out[k] = _brand_image(body[k], what)
    if "brand_handle" in body:  # the @name drawn on clips; empty means the channel handle / username
        out["brand_handle"] = (_clean(body.get("brand_handle")) or "")[:40] or None
    # "Follow me" scene: a card with the logo, handle and a Follow button shown for a few seconds in each clip.
    if "follow_on" in body:
        out["follow_on"] = bool(body.get("follow_on")) or None
    if "follow_text" in body:
        out["follow_text"] = (_clean(body.get("follow_text")) or "")[:60] or None
    if "follow_btn" in body:
        out["follow_btn"] = (_clean(body.get("follow_btn")) or "")[:20] or None
    if "follow_at" in body:  # where in the clip: start, middle or end
        out["follow_at"] = body.get("follow_at") if body.get("follow_at") in ("start", "middle", "end") else None
    if "follow_dur" in body:
        try:
            out["follow_dur"] = min(8.0, max(1.5, float(body.get("follow_dur"))))
        except (TypeError, ValueError):
            out["follow_dur"] = None
    return out


@router.post("/api/accounts")
def create_account(body: dict):
    platform = body.get("platform", "")
    _check(platform)
    acct = {"id": _new_id(), "platform": platform, **{k: v for k, v in _account_fields(body, platform).items() if v is not None}}
    with _lock:
        data = _read()
        data["accounts"].append(acct)
        _write(data)
        return _public(acct, data["apps"])


@router.post("/api/accounts/{acct_id}/config")
def configure(acct_id: str, body: dict):
    acct = _find(acct_id)
    fields = _account_fields(body, acct["platform"])
    if (fields["app_id"] != acct.get("app_id") or fields.get("ig_username") != acct.get("ig_username")
            or (fields.get("fb_page") or "").lower() != (acct.get("fb_page") or "").lower()
            or (acct.get("yt_channel_id") and not _channel_matches(fields.get("yt_channel"),
                {"id": acct["yt_channel_id"], "title": acct.get("name") or "", "handle": acct.get("yt_handle")}))):
        # the old login belongs to the old app / account / channel
        fields.update(token=None, name=None, ig_user_id=None, yt_channel_id=None, yt_handle=None, fb_page_id=None)
    update(acct_id, **fields)
    return _public(get(acct_id), load_apps())


@router.post("/api/accounts/{acct_id}/disconnect")
def disconnect(acct_id: str):
    _find(acct_id)
    update(acct_id, token=None, name=None, ig_user_id=None, yt_channel_id=None, yt_handle=None, fb_page_id=None)
    return {"ok": True}


@router.post("/api/accounts/{acct_id}/delete")  # POST, so the X-Clip-Studio header check applies
def delete_account(acct_id: str):
    _find(acct_id)
    with _lock:
        data = _read()
        data["accounts"] = [a for a in data["accounts"] if a["id"] != acct_id]
        _write(data)
    return {"ok": True}


@router.get("/api/accounts/{acct_id}/connect")
def auth_start(acct_id: str, request: Request):
    acct = _find(acct_id)
    if not (acct.get("client_id") and acct.get("client_secret")):
        raise HTTPException(400, f"Pick API credentials for {acct['label']} first.")
    state, verifier = secrets.token_urlsafe(24), secrets.token_urlsafe(48)
    redirect = redirect_uri(request, acct["platform"])
    _oauth_pending[state] = {"account": acct_id, "platform": acct["platform"], "verifier": verifier,
                             "redirect_uri": redirect, "at": time.time()}
    return RedirectResponse(auth_url(acct, redirect, state, verifier))


@router.get("/api/auth/{platform}/callback")
def auth_callback(platform: str, request: Request):
    q = request.query_params
    pending = _oauth_pending.pop(q.get("state", ""), None)
    if not pending or pending["platform"] != platform or time.time() - pending["at"] > 900:
        return _done_page(False, "This login link expired or wasn't started from Clip Studio. Try Connect again.")
    if q.get("error"):
        return _done_page(False, f"{NAMES[platform]} said: {q.get('error_description') or q.get('error')}")
    try:
        exchange_code(get(pending["account"]), q["code"], pending["redirect_uri"], pending["verifier"])
        acct = get(pending["account"])
    except (PublishError, KeyError) as e:
        return _done_page(False, str(e))
    return _done_page(True, f"{acct['label']} is connected to {NAMES[platform]} as {acct.get('name')}.")


def _done_page(ok: bool, message: str):
    safe = message.replace("&", "&amp;").replace("<", "&lt;")
    return HTMLResponse(f"""<!doctype html><meta charset="utf-8"><title>Clip Studio</title>
<body style="font:16px system-ui;display:grid;place-items:center;height:90vh;margin:0">
<div style="max-width:460px;text-align:center"><h2>{'✅ Connected' if ok else '⚠️ Not connected'}</h2><p>{safe}</p>
<p style="color:#888">You can close this window.</p></div>
<script>try {{ window.opener && window.opener.postMessage('clipstudio:accounts', '*'); }} catch (e) {{}}
{'setTimeout(() => window.close(), 1500);' if ok else ''}</script>""")


def _check(platform: str):
    if platform not in PLATFORMS:
        raise HTTPException(404, "unknown platform")


# ---------- media ----------
@router.post("/api/media")
async def upload_media(request: Request):
    """Receive a rendered clip from the page and transcode it to a platform-friendly MP4 (H.264/AAC, faststart)."""
    work = Path(tempfile.mkdtemp(prefix="clipstudio_pub_"))
    src = work / "render.bin"
    with src.open("wb") as f:
        async for chunk in request.stream():
            f.write(chunk)
    out = work / "clip.mp4"
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(src), "-c:v", "libx264", "-preset", "veryfast", "-crf", "19",
           "-pix_fmt", "yuv420p", "-r", "30", "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-movflags", "+faststart", str(out)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    src.unlink(missing_ok=True)
    if res.returncode != 0:
        shutil.rmtree(work, ignore_errors=True)
        raise HTTPException(400, "Couldn't convert the clip to MP4: " + res.stderr.strip()[-300:])
    media_id = uuid.uuid4().hex
    _media[media_id] = out
    # Drop rendered clips after an hour.
    cleanup = threading.Timer(3600, lambda: (_media.pop(media_id, None), shutil.rmtree(work, ignore_errors=True)))
    cleanup.daemon = True  # don't keep the server alive on shutdown
    cleanup.start()
    return {"media_id": media_id, "size": out.stat().st_size}


# ---------- publishing ----------
def tiktok_creator(token: str) -> dict:
    _, _, r = http("POST", "https://open.tiktokapis.com/v2/post/publish/creator_info/query/", body={},
                   headers={"Authorization": f"Bearer {token}"})
    if r.get("error", {}).get("code") not in (None, "ok"):
        raise PublishError(f"TikTok: {r['error'].get('message') or r['error']['code']}")
    return r.get("data", {})


@router.get("/api/accounts/{acct_id}/creator")
def tiktok_creator_info(acct_id: str):
    if _find(acct_id)["platform"] != "tiktok":
        raise HTTPException(400, "Not a TikTok account.")
    try:
        return tiktok_creator(access_token(acct_id))
    except PublishError as e:
        raise HTTPException(400, str(e))


def publish_youtube(acct_id: str, path: Path, f: dict, job: dict) -> dict:
    tok = access_token(acct_id)
    # The token decides the channel. Check it is still the one this account was connected to
    # (and the one set on the account), so a clip never lands on a different channel.
    acct = get(acct_id)
    job["status"] = "checking the YouTube channel"
    ch = youtube_channel(tok)
    if not ch:
        raise PublishError(f"The login for {acct['label']} no longer has a YouTube channel. Connect it again.")
    same_id = bool(acct.get("yt_channel_id")) and ch["id"] == acct["yt_channel_id"]
    if (acct.get("yt_channel_id") and not same_id) or not (same_id or _channel_matches(acct.get("yt_channel"), ch)):
        raise PublishError(f"{acct['label']} would upload to {_channel_text(ch)}, not the channel it's set to. Connect it again and pick the right channel.")
    _sync_channel(acct, ch)  # picks up renames (title / @handle) made on YouTube
    tags, total = [], 0
    for t in f.get("tags") or []:  # YouTube caps tags at ~500 characters in total
        t = str(t).strip().lstrip("#")
        if t and total + len(t) + 2 <= 480:
            tags.append(t)
            total += len(t) + 2
    meta = {
        "snippet": {"title": f["title"][:100], "description": (f.get("description") or "")[:4900],
                    "tags": tags, "categoryId": str(f.get("category") or "22")},
        "status": {"privacyStatus": f.get("privacy") or "private", "selfDeclaredMadeForKids": False},
    }
    at = _schedule_at(f)
    if at:  # YouTube keeps it private and makes it public at publishAt
        meta["status"].update(privacyStatus="private", publishAt=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(at)))
    _, h, _ = http("POST", "https://www.googleapis.com/upload/youtube/v3/videos?uploadType=resumable&part=snippet,status",
                   body=meta, headers={"Authorization": f"Bearer {tok}", "X-Upload-Content-Type": "video/mp4",
                                       "X-Upload-Content-Length": str(path.stat().st_size)})
    job["status"] = "uploading to YouTube"
    _, _, video = http("PUT", h["Location"], raw=path.read_bytes(), timeout=900,
                       headers={"Authorization": f"Bearer {tok}", "Content-Type": "video/mp4"})
    message = (f"Uploaded to {_channel_text(ch)}; private until it goes public {_when(at)}." if at else
               f"Uploaded to {_channel_text(ch)} as {video.get('status', {}).get('privacyStatus', meta['status']['privacyStatus'])}.")
    out = {"url": f"https://youtube.com/shorts/{video['id']}", "video_id": video["id"], **({"scheduled_at": at} if at else {})}
    thumb = _thumb_bytes(f.get("thumb"))
    if f.get("thumb") and not thumb:
        return {**out, "thumb_set": False, "message": message + " The thumbnail wasn't set: the image was empty or over 2 MB."}
    if thumb:
        ok, note = set_youtube_thumbnail(acct_id, video["id"], thumb, job)
        return {**out, "thumb_set": ok, "message": message + " " + note}
    return {**out, "message": message}


# Waits between thumbnail attempts. YouTube refuses thumbnails for a while after a channel has set many in a
# short time ("uploadRateLimitExceeded"), and now and then right after an upload; both pass if we wait.
THUMB_RETRY_WAITS = (20, 40, 60, 120, 180, 300)


def set_youtube_thumbnail(acct_id: str, video_id: str, thumb: bytes, job: dict) -> tuple[bool, str]:
    """Set a video's custom thumbnail, retrying while YouTube is rate-limiting. Returns (set?, message)."""
    for attempt in range(len(THUMB_RETRY_WAITS) + 1):
        job["status"] = "setting the YouTube thumbnail" + (f" (try {attempt + 1})" if attempt else "")
        try:
            http("POST", f"https://www.googleapis.com/upload/youtube/v3/thumbnails/set?videoId={video_id}", raw=thumb,
                 timeout=120, headers={"Authorization": f"Bearer {access_token(acct_id)}", "Content-Type": "image/jpeg"})
            return True, "Thumbnail set." + (f" (YouTube accepted it on try {attempt + 1}.)" if attempt else "")
        except PublishError as e:
            err = str(e)
            retry = "uploadRateLimitExceeded" in err or "rateLimitExceeded" in err or any(
                f"returned {code}" in err for code in (429, 500, 502, 503, 504))
            if retry:
                if attempt < len(THUMB_RETRY_WAITS):
                    wait = THUMB_RETRY_WAITS[attempt]
                    for left in range(wait, 0, -5):
                        job["status"] = f"YouTube is limiting thumbnail uploads; trying again in {left}s"
                        time.sleep(5)
                    continue
                return False, ("The thumbnail wasn't set: YouTube is still limiting thumbnail uploads on this channel. "
                               "Use Retry thumbnail later.")
            if "403" in err:
                # Custom thumbnails need a verified channel (phone check in YouTube Studio).
                return False, "The thumbnail wasn't set: verify the channel by phone at youtube.com/verify to allow custom thumbnails."
            return False, f"The thumbnail wasn't set ({err})."
    return False, "The thumbnail wasn't set."


@router.post("/api/accounts/{acct_id}/youtube-thumbnail")
def retry_youtube_thumbnail(acct_id: str, body: dict):
    """Set (again) the custom thumbnail of a video already on the channel. Runs as a job, like a publish."""
    if _find(acct_id)["platform"] != "youtube":
        raise HTTPException(400, "Not a YouTube account.")
    video_id, thumb = str(body.get("video_id") or ""), _thumb_bytes(body.get("thumb"))
    if not video_id or not thumb:
        raise HTTPException(400, "Need the video and a thumbnail image under 2 MB.")
    job_id = uuid.uuid4().hex
    job = _jobs[job_id] = {"status": "starting", "platform": "youtube", "account": acct_id}

    def run():
        try:
            ok, note = set_youtube_thumbnail(acct_id, video_id, thumb, job)
            job.update(status="done", thumb_set=ok, video_id=video_id, message=note,
                       url=f"https://youtube.com/shorts/{video_id}")
        except Exception as e:
            job.update(status="error", error=f"YouTube: {e}")

    _executor.submit(run)
    return {"job": job_id}


def _thumb_bytes(data_url) -> bytes | None:
    """The page's JPEG thumbnail (a data: URL), or None. YouTube takes up to 2 MB."""
    if not data_url:
        return None
    try:
        raw = base64.b64decode(str(data_url).split(",", 1)[-1])
    except ValueError:
        return None
    return raw if 0 < len(raw) <= 2 * 1024 * 1024 else None


def _cover_ms(f: dict) -> int | None:
    """Where in the clip the chosen cover frame is, in milliseconds."""
    try:
        return max(0, int(float(f["cover_ms"])))
    except (KeyError, TypeError, ValueError):
        return None


def publish_tiktok(acct_id: str, path: Path, f: dict, job: dict) -> dict:
    tok = access_token(acct_id)
    draft = f.get("mode") == "draft"
    size = path.stat().st_size
    # Files up to 64 MB go in one chunk; bigger ones in 10 MB chunks with the remainder in the last.
    chunk = size if size <= 64 * 1024 ** 2 else 10 * 1024 ** 2
    count = max(1, size // chunk)
    source = {"source": "FILE_UPLOAD", "video_size": size, "chunk_size": chunk, "total_chunk_count": count}
    try:
        privacy, r = _tiktok_init(tok, f, source, draft)
    except PublishError as e:
        if "scope_not_authorized" in str(e):
            raise PublishError("TikTok: this account hasn't allowed " + ("draft uploads" if draft else "direct posting")
                               + ". In Accounts, disconnect it and Connect again to grant the new permission.")
        if "unaudited_client_can_only_post_to_private_accounts" in str(e):
            raise PublishError("TikTok: until TikTok audits your app, direct posts only work on a private account as "
                               "\"Only me\". Choose \"Send as a draft\" to post publicly from the TikTok app.")
        raise
    err = r.get("error", {})
    if err.get("code") != "ok":
        raise PublishError(f"TikTok: {err.get('message') or err}")
    upload_url, publish_id = r["data"]["upload_url"], r["data"]["publish_id"]
    return _tiktok_upload(tok, path, size, chunk, count, upload_url, publish_id, privacy, draft, acct_id, job)


def _tiktok_init(tok: str, f: dict, source: dict, draft: bool):
    """Start a TikTok upload: a draft in the creator's inbox, or a direct post. Returns (privacy, response)."""
    if draft:
        # Inbox upload: no caption or visibility here; the creator adds them in the TikTok app and posts.
        privacy = None
        _, _, r = http("POST", "https://open.tiktokapis.com/v2/post/publish/inbox/video/init/",
                       headers={"Authorization": f"Bearer {tok}"}, body={"source_info": source})
    else:
        info = tiktok_creator(tok)
        privacy = f.get("privacy")
        if privacy not in info.get("privacy_level_options", []):
            raise PublishError(f"Choose who can view the TikTok post (allowed: {', '.join(info.get('privacy_level_options', []))}).")
        _, _, r = http("POST", "https://open.tiktokapis.com/v2/post/publish/video/init/", headers={"Authorization": f"Bearer {tok}"}, body={
            "post_info": {
                "title": (f.get("caption") or "")[:2200], "privacy_level": privacy,
                "disable_comment": not f.get("allow_comment", True) or info.get("comment_disabled", False),
                "disable_duet": not f.get("allow_duet", True) or info.get("duet_disabled", False),
                "disable_stitch": not f.get("allow_stitch", True) or info.get("stitch_disabled", False),
                "brand_content_toggle": False, "brand_organic_toggle": False,
                **({"video_cover_timestamp_ms": _cover_ms(f)} if _cover_ms(f) is not None else {}),
            },
            "source_info": source,
        })
    return privacy, r


def _tiktok_upload(tok, path: Path, size, chunk, count, upload_url, publish_id, privacy, draft, acct_id, job) -> dict:
    job["status"] = "uploading to TikTok"
    with path.open("rb") as fh:
        for i in range(count):
            start = i * chunk
            end = size - 1 if i == count - 1 else start + chunk - 1
            fh.seek(start)
            http("PUT", upload_url, raw=fh.read(end - start + 1), timeout=900, headers={
                "Content-Type": "video/mp4", "Content-Range": f"bytes {start}-{end}/{size}"})
    job["status"] = "TikTok is processing"
    for _ in range(60):
        time.sleep(5)
        _, _, s = http("POST", "https://open.tiktokapis.com/v2/post/publish/status/fetch/",
                       body={"publish_id": publish_id}, headers={"Authorization": f"Bearer {tok}"})
        status = s.get("data", {}).get("status")
        if draft and status in ("SEND_TO_USER_INBOX", "PUBLISH_COMPLETE"):
            return {"url": None, "draft": True,
                    "message": "Sent to your TikTok inbox. Open TikTok, tap the notification, paste the caption, choose Everyone and post."}
        if status == "PUBLISH_COMPLETE":
            name = get(acct_id).get("name") or ""
            return {"url": f"https://www.tiktok.com/{name}" if name.startswith("@") else None,
                    "message": "Posted" + (" (visible only to you)" if privacy == "SELF_ONLY" else "") + "."}
        if status == "FAILED":
            raise PublishError(f"TikTok couldn't post it: {s.get('data', {}).get('fail_reason', 'unknown reason')}")
    if draft:
        return {"url": None, "draft": True, "message": "Uploaded. TikTok is still processing it; the draft will appear in your TikTok inbox shortly."}
    return {"url": None, "message": "Uploaded. TikTok is still processing it; check your profile in a few minutes."}


def publish_instagram(acct_id: str, path: Path, f: dict, job: dict) -> dict:
    tok = access_token(acct_id)
    ig = get(acct_id)["ig_user_id"]
    size = path.stat().st_size
    _, _, c = http("POST", f"{GRAPH}/{ig}/media", form={
        "media_type": "REELS", "upload_type": "resumable", "caption": (f.get("caption") or "")[:2200],
        "share_to_feed": "true" if f.get("share_to_feed", True) else "false", "access_token": tok,
        **({"thumb_offset": str(_cover_ms(f))} if _cover_ms(f) is not None else {}),
    })
    container = c["id"]
    job["status"] = "uploading to Instagram"
    http("POST", c.get("uri") or f"https://rupload.facebook.com/ig-api-upload/v26.0/{container}", raw=path.read_bytes(),
         timeout=900, headers={"Authorization": f"OAuth {tok}", "offset": "0", "file_size": str(size)})
    job["status"] = "Instagram is processing"
    for _ in range(60):
        time.sleep(5)
        _, _, s = http("GET", f"{GRAPH}/{container}?" + urllib.parse.urlencode({"fields": "status_code,status", "access_token": tok}))
        if s.get("status_code") == "FINISHED":
            break
        if s.get("status_code") in ("ERROR", "EXPIRED"):
            raise PublishError(f"Instagram couldn't process the video: {s.get('status') or s['status_code']}")
    else:
        raise PublishError("Instagram took more than 5 minutes to process the video. Try again later.")
    _, _, pub = http("POST", f"{GRAPH}/{ig}/media_publish", form={"creation_id": container, "access_token": tok})
    _, _, m = http("GET", f"{GRAPH}/{pub['id']}?" + urllib.parse.urlencode({"fields": "permalink", "access_token": tok}))
    return {"url": m.get("permalink"), "message": "Posted as a Reel."}


def _video_seconds(path: Path) -> float | None:
    res = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True)
    try:
        return float(res.stdout.strip())
    except ValueError:
        return None


def publish_facebook(acct_id: str, path: Path, f: dict, job: dict) -> dict:
    """A Reel on the account's Facebook Page, through the Page's video_reels endpoint (start, upload, finish)."""
    tok = access_token(acct_id)
    acct = get(acct_id)
    page = acct.get("fb_page_id")
    if not page:
        raise PublishError(f"{acct['label']} has no Facebook Page connected. Connect it again.")
    secs = _video_seconds(path)
    if secs is not None and not 3 <= secs <= FB_REEL_MAX_S + 0.5:
        raise PublishError(f"Facebook Reels must be 3 to {FB_REEL_MAX_S} seconds long; this clip is {secs:.0f}s.")
    _, _, start = http("POST", f"{GRAPH}/{page}/video_reels", form={"upload_phase": "start", "access_token": tok})
    video_id = start["video_id"]
    job["status"] = "uploading to Facebook"
    http("POST", start.get("upload_url") or f"https://rupload.facebook.com/video-upload/v26.0/{video_id}",
         raw=path.read_bytes(), timeout=900,
         headers={"Authorization": f"OAuth {tok}", "offset": "0", "file_size": str(path.stat().st_size)})
    at = _schedule_at(f)
    job["status"] = "scheduling on Facebook" if at else "publishing on Facebook"
    _, _, fin = http("POST", f"{GRAPH}/{page}/video_reels", form={
        "upload_phase": "finish", "video_id": video_id, "video_state": "SCHEDULED" if at else "PUBLISHED",
        **({"scheduled_publish_time": str(at)} if at else {}),
        "description": (f.get("caption") or "")[:2200], "access_token": tok})
    if fin.get("success") is False:
        raise PublishError(f"Facebook didn't accept the Reel: {fin}")
    if at:
        return {"url": f"https://www.facebook.com/reel/{video_id}", "video_id": video_id, "scheduled_at": at,
                "message": f"Scheduled on {acct.get('name') or 'your Page'} for {_when(at)}."}
    job["status"] = "Facebook is processing"
    for _ in range(60):
        time.sleep(5)
        _, _, s = http("GET", f"{GRAPH}/{video_id}?" + urllib.parse.urlencode(
            {"fields": "status,permalink_url", "access_token": tok}))
        st = s.get("status") or {}
        phases = [st.get(k) or {} for k in ("uploading_phase", "processing_phase", "publishing_phase")]
        if st.get("video_status") == "error" or any(ph.get("status") == "error" for ph in phases):
            errs = [e.get("message") for ph in phases for e in ph.get("errors") or []]
            raise PublishError("Facebook couldn't publish the Reel: " + ("; ".join(filter(None, errs)) or "processing failed"))
        if (st.get("publishing_phase") or {}).get("status") == "complete" or st.get("video_status") == "ready":
            link = s.get("permalink_url")
            url = ("https://www.facebook.com" + link) if link and link.startswith("/") else link
            return {"url": url or f"https://www.facebook.com/reel/{video_id}", "video_id": video_id,
                    "message": f"Posted as a Reel on {acct.get('name') or 'your Page'}."}
    return {"url": f"https://www.facebook.com/reel/{video_id}", "video_id": video_id,
            "message": "Uploaded. Facebook is still processing it; the Reel will appear on your Page shortly."}


PUBLISHERS = {"youtube": publish_youtube, "tiktok": publish_tiktok, "instagram": publish_instagram, "facebook": publish_facebook}


@router.post("/api/publish")
def publish(body: dict):
    acct = _find(body.get("account", ""))
    platform = acct["platform"]
    path = _media.get(body.get("media_id", ""))
    if not path or not path.exists():
        raise HTTPException(400, "The rendered clip expired. Publish again to re-render it.")
    if platform == "youtube" and not (body.get("title") or "").strip():
        raise HTTPException(400, "YouTube needs a title.")
    at = _schedule_at(body)
    if body.get("schedule_at") and not at:
        raise HTTPException(400, "The scheduled time isn't valid.")
    if at:
        soonest = time.time() + (10 * 60 if platform == "facebook" else 60)
        if at < soonest:
            raise HTTPException(400, f"The scheduled time is too soon for {NAMES[platform]}"
                                     + (" (Facebook needs at least 10 minutes)." if platform == "facebook" else "."))
        if platform == "facebook" and at > time.time() + 29 * 86400:
            raise HTTPException(400, "Facebook can schedule Reels up to 29 days ahead.")
    job_id = uuid.uuid4().hex
    job = _jobs[job_id] = {"status": "starting", "platform": platform, "account": acct["id"]}

    def run():
        try:
            if at and platform not in NATIVE_SCHEDULE:
                job.update(status="done", **_queue(acct, path, body, at))
                return
            job.update(status="done", **PUBLISHERS[platform](acct["id"], path, body, job))
        except PublishError as e:
            job.update(status="error", error=str(e))
        except Exception as e:  # unexpected response shape etc.
            job.update(status="error", error=f"{NAMES[platform]}: {type(e).__name__}: {e}")

    _executor.submit(run)
    return {"job": job_id}


@router.get("/api/publish/{job_id}")
def publish_status(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "unknown job")
    if job["status"] in ("done", "error"):
        _jobs.pop(job_id)
    return job


# ---------- scheduled posts ----------
# YouTube and Facebook schedule natively: the video is uploaded now and goes live at the chosen time.
# Instagram and TikTok have no scheduling in their APIs, so the clip is kept in data/scheduled/ and this
# server posts it when the time comes (it has to be running then). Queue: data/scheduled.json.
NATIVE_SCHEDULE = ("youtube", "facebook")
SCHEDULED = DATA / "scheduled.json"
SCHEDULED_DIR = DATA / "scheduled"
_sched_lock = threading.Lock()


def _schedule_at(f: dict) -> int | None:
    """The time to publish at (Unix seconds), or None to publish now."""
    try:
        at = int(float(f.get("schedule_at") or 0))
    except (TypeError, ValueError):
        return None
    return at if at > 0 else None


def _when(at: int) -> str:
    return time.strftime("%a %d %b %Y, %H:%M", time.localtime(at))


def _sched_read() -> list[dict]:
    try:
        return json.loads(SCHEDULED.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return []


def _sched_write(items: list[dict]):
    DATA.mkdir(parents=True, exist_ok=True)
    tmp = SCHEDULED.with_suffix(".tmp")
    tmp.write_text(json.dumps(items, indent=1), encoding="utf-8")
    tmp.replace(SCHEDULED)


def _queue(acct: dict, path: Path, body: dict, at: int) -> dict:
    SCHEDULED_DIR.mkdir(parents=True, exist_ok=True)
    item_id = _new_id()
    dst = SCHEDULED_DIR / f"{item_id}.mp4"
    shutil.copyfile(path, dst)
    fields = {k: v for k, v in body.items() if k not in ("account", "media_id", "thumb", "schedule_at")}
    with _sched_lock:
        items = _sched_read()
        items.append({"id": item_id, "account": acct["id"], "platform": acct["platform"], "at": at,
                      "file": dst.name, "fields": fields, "status": "waiting"})
        _sched_write(items)
    return {"url": None, "scheduled_at": at, "scheduled_id": item_id,
            "message": f"Scheduled for {_when(at)}. ViralClip posts it then, so keep server.py running at that time."}


def _run_due():
    with _sched_lock:
        items = _sched_read()
        due = [i for i in items if i["status"] == "waiting" and i["at"] <= time.time()]
        for i in due:
            i["status"] = "posting"
        if due:
            _sched_write(items)
    for item in due:
        path = SCHEDULED_DIR / item["file"]
        job = {"status": "starting"}
        try:
            if not path.exists():
                raise PublishError("the saved clip is missing")
            res = PUBLISHERS[item["platform"]](item["account"], path, item["fields"], job)
            update = {"status": "done", "url": res.get("url"), "message": res.get("message"), "posted": time.time()}
        except Exception as e:
            update = {"status": "error", "error": str(e), "posted": time.time()}
        path.unlink(missing_ok=True)
        with _sched_lock:
            items = _sched_read()
            for i in items:
                if i["id"] == item["id"]:
                    i.update(update)
            _sched_write(items)


def _scheduler():
    # A post that was mid-way when the server stopped isn't retried blindly (it may have gone out).
    with _sched_lock:
        items = _sched_read()
        if any(i["status"] == "posting" for i in items):
            for i in items:
                if i["status"] == "posting":
                    i.update(status="error", error="The server stopped while posting; check the account and post again if needed.")
            _sched_write(items)
    while True:
        try:
            _run_due()
        except Exception as e:  # keep the scheduler alive
            print("scheduled posts:", e)
        time.sleep(30)


@router.get("/api/scheduled")
def scheduled():
    """Posts this server will publish later (Instagram, TikTok), and how the past ones went."""
    with _sched_lock:
        return [{k: v for k, v in i.items() if k != "fields"} for i in _sched_read()]


@router.delete("/api/scheduled/{item_id}")
def cancel_scheduled(item_id: str):
    with _sched_lock:
        items = _sched_read()
        item = next((i for i in items if i["id"] == item_id), None)
        if not item:
            raise HTTPException(404, "unknown scheduled post")
        if item["status"] != "waiting":
            raise HTTPException(400, "That post isn't waiting any more.")
        (SCHEDULED_DIR / item["file"]).unlink(missing_ok=True)
        items.remove(item)
        _sched_write(items)
    return {"ok": True}


threading.Thread(target=_scheduler, daemon=True, name="scheduled-posts").start()
