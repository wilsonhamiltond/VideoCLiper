"""Saved videos: the clips found for a video (plus transcript, scores and your edits) kept on disk,
so you can come back later and keep editing, export or publish without analyzing the video again.

Each project is data/projects/<id>.json and points at its video in downloads/ (imported links already
live there; uploads are kept there too once they've been analyzed).
"""

import json
import re
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException

import fetch

ROOT = Path(__file__).parent
PROJECTS = ROOT / "data" / "projects"
PROJECT_ID = re.compile(r"^[0-9a-f]{32}$")

router = APIRouter()


def project_path(pid: str) -> Path:
    if not PROJECT_ID.match(pid):
        raise HTTPException(404, "unknown project")
    return PROJECTS / f"{pid}.json"


def load(pid: str) -> dict:
    path = project_path(pid)
    if not path.exists():
        raise HTTPException(404, "That saved video is gone.")
    return json.loads(path.read_text("utf-8"))


def store(project: dict):
    PROJECTS.mkdir(parents=True, exist_ok=True)
    path = project_path(project["id"])
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(project, separators=(",", ":")), "utf-8")
    tmp.replace(path)  # atomic, so a crash mid-save never leaves a half-written project


@router.get("/api/projects")
def list_projects():
    out = []
    for path in PROJECTS.glob("*.json") if PROJECTS.exists() else []:
        try:
            p = json.loads(path.read_text("utf-8"))
        except (OSError, ValueError):
            continue
        clips = p.get("clips") or []
        out.append({
            "id": p["id"], "name": p.get("name", "Video"), "created": p.get("created"), "updated": p.get("updated"),
            "duration": p.get("duration", 0), "clips": len(clips), "aspect": p.get("aspect"),
            "thumb": next((c.get("thumb") for c in clips if c.get("thumb")), None),
            "available": fetch.media_exists(p.get("media", "")),
        })
    out.sort(key=lambda p: p.get("updated") or 0, reverse=True)
    return {"projects": out}


@router.get("/api/projects/{pid}")
def get_project(pid: str):
    return load(pid)


@router.post("/api/projects")
def create_project(body: dict):
    media = str(body.get("media") or "")
    if not fetch.media_exists(media):
        raise HTTPException(400, "The video isn't stored on the server, so it can't be saved.")
    now = time.time()
    project = {**body, "id": uuid.uuid4().hex, "media": media, "created": now, "updated": now}
    store(project)
    return {"id": project["id"], "updated": now}


@router.post("/api/projects/{pid}")
def update_project(pid: str, body: dict):
    """Merges the given top-level fields (the editor autosaves just the clips)."""
    project = load(pid)
    body = {k: v for k, v in body.items() if k not in ("id", "media", "created")}
    project.update(body, updated=time.time())
    store(project)
    return {"id": pid, "updated": project["updated"]}


@router.post("/api/projects/{pid}/delete")
def delete_project(pid: str):
    project = load(pid)
    project_path(pid).unlink(missing_ok=True)
    # Free the disk space too, unless another saved project still uses the same video.
    media = project.get("media", "")
    others = [json.loads(p.read_text("utf-8")).get("media") for p in PROJECTS.glob("*.json")]
    if media not in others:
        fetch.delete_media(media)
    return {"ok": True}
