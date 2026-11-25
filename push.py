"""Web Push: tells your phone (or any browser that turned it on) when clips are ready or an import fails,
even with the Clip Studio tab closed.

Keys live in data/vapid.pem (made on first use); subscribed browsers in data/push_subs.json.
The browser must see the page as secure: localhost works as-is; on a phone over the LAN
(http://<pc-ip>:8000) turn on the Chrome flag "Insecure origins treated as secure" for that address.
"""

import base64
import json
import threading
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

ROOT = Path(__file__).parent
KEY = ROOT / "data" / "vapid.pem"
SUBS = ROOT / "data" / "push_subs.json"
CLAIMS = {"sub": "mailto:clip-studio@example.com"}  # push services want a contact; nothing is sent there

router = APIRouter()
_lock = threading.Lock()


def _vapid():
    from py_vapid import Vapid01
    if not KEY.exists():
        KEY.parent.mkdir(parents=True, exist_ok=True)
        v = Vapid01()
        v.generate_keys()
        v.save_key(str(KEY))
    return Vapid01.from_file(str(KEY))


def public_key() -> str:
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
    raw = _vapid().public_key.public_bytes(Encoding.X962, PublicFormat.UncompressedPoint)
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _load() -> list[dict]:
    try:
        return json.loads(SUBS.read_text("utf-8"))
    except (OSError, ValueError):
        return []


def _save(subs: list[dict]):
    SUBS.parent.mkdir(parents=True, exist_ok=True)
    SUBS.write_text(json.dumps(subs), "utf-8")


def send(title: str, body: str, nid: str = "", test: bool = False):
    """Push to every subscribed browser, in the background. Gone subscriptions are dropped."""
    subs = _load()
    if not subs:
        return

    def run():
        from pywebpush import WebPushException, webpush
        data = json.dumps({"title": title, "body": body, "nid": nid, "test": test})
        dead = []
        for s in subs:
            try:
                webpush(s, data, vapid_private_key=str(KEY), vapid_claims=dict(CLAIMS), ttl=6 * 3600)
            except WebPushException as e:
                if e.response is not None and e.response.status_code in (404, 410):
                    dead.append(s["endpoint"])
                else:
                    print("push:", e, flush=True)
            except Exception as e:  # no internet, etc.: never break the job that finished
                print("push:", e, flush=True)
        if dead:
            with _lock:
                _save([s for s in _load() if s["endpoint"] not in dead])

    _vapid()  # make sure the key file exists before the thread reads it
    threading.Thread(target=run, daemon=True).start()


@router.get("/api/push/key")
def get_key():
    return {"key": public_key()}


@router.post("/api/push/subscribe")
def subscribe(body: dict):
    if not str(body.get("endpoint", "")).startswith("https://") or not (body.get("keys") or {}).get("p256dh"):
        raise HTTPException(400, "Not a push subscription.")
    sub = {"endpoint": body["endpoint"], "keys": {"p256dh": body["keys"]["p256dh"], "auth": body["keys"]["auth"]}}
    with _lock:
        _save([sub] + [s for s in _load() if s["endpoint"] != sub["endpoint"]])
    return {"ok": True}


@router.post("/api/push/unsubscribe")
def unsubscribe(body: dict):
    with _lock:
        _save([s for s in _load() if s["endpoint"] != body.get("endpoint")])
    return {"ok": True}


@router.post("/api/push/test")
def test():
    if not _load():
        raise HTTPException(400, "No browser has turned on notifications yet.")
    send("Clip Studio", "Notifications work on this device.", test=True)
    return {"ok": True}


SW = """
self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', e => e.waitUntil(self.clients.claim()));
self.addEventListener('push', e => {
  const d = e.data ? e.data.json() : {};
  e.waitUntil(self.clients.matchAll({ type: 'window', includeUncontrolled: true }).then(list => {
    // The open, visible page already shows its own toast: don't pop a second notification there.
    if (!d.test && list.some(c => c.visibilityState === 'visible' && c.focused)) return;
    return self.registration.showNotification(d.title || 'Clip Studio', {
      body: d.body || '', tag: d.nid || undefined, data: { nid: d.nid || '' },
    });
  }));
});
self.addEventListener('notificationclick', e => {
  e.notification.close();
  const url = '/' + (e.notification.data.nid ? '?notif=' + e.notification.data.nid : '');
  e.waitUntil(self.clients.matchAll({ type: 'window', includeUncontrolled: true }).then(list => {
    const c = list[0];
    if (c) return c.focus().then(c => c.navigate ? c.navigate(url) : c);
    return self.clients.openWindow(url);
  }));
});
"""


@router.get("/sw.js")
def service_worker():
    return Response(SW, media_type="application/javascript", headers={"Cache-Control": "no-cache"})
