"""Local chord annotation server.

Downloads YouTube audio to a temp folder, detects beats, and serves the
Chordify-style editor in ./static. Draft .lab files are read from the repo's
label folder and overwritten in place by the editor's autosave, in Harte
syntax (validated with labeling/chord_utils.parse_harte before writing).

Audio files are temporary: they are deleted when the tab closes or another
song is loaded (DELETE /api/session/<id>) and on server start.
"""
import datetime
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import uuid
from typing import Optional

import librosa
import numpy as np
import yt_dlp
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(ROOT)
TMP = os.path.join(ROOT, "tmp")
# Overridable so the editor can be pointed at a scratch copy of the labels.
LAB_DIR = os.path.realpath(os.environ.get("LAB_DIR", os.path.join(REPO, "btc_predictions 2")))
SONGS_JSON = os.environ.get("SONGS_JSON", os.path.join(ROOT, "songs.json"))
# Personal work history (gitignored, this machine only); songs.json is the shared one.
HISTORY_JSON = os.environ.get("HISTORY_JSON", os.path.join(ROOT, "history.json"))
FFMPEG = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
SR = 44100

sys.path.insert(0, os.path.join(REPO, "labeling"))
from chord_utils import parse_harte  # noqa: E402

app = FastAPI()
json_lock = threading.Lock()


def clean_tmp():
    shutil.rmtree(TMP, ignore_errors=True)
    os.makedirs(TMP, exist_ok=True)


def session_dir(sid):
    if not re.fullmatch(r"[0-9a-f]{32}", sid):
        raise HTTPException(400, "invalid session id")
    return os.path.join(TMP, sid)


def lab_file(rel):
    """Resolve a .lab path relative to LAB_DIR; only existing files inside it are allowed."""
    path = os.path.realpath(os.path.join(LAB_DIR, rel))
    if not path.startswith(LAB_DIR + os.sep) or not path.endswith(".lab") or not os.path.isfile(path):
        raise HTTPException(404, "file .lab tidak ditemukan: %s" % rel)
    return path


def now_iso():
    return datetime.datetime.now().isoformat(timespec="seconds")


def read_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, ValueError):
        return {}


def update_json(path, rel, fn):
    with json_lock:
        data = read_json(path)
        fn(data.setdefault(rel, {}))
        write_atomic(path, json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


# songs.json: {"<lab path>": {"url": "<YouTube link>", "saved": "<ISO time of last autosave>"}}
def read_songs():
    return read_json(SONGS_JSON)


def update_song(rel, **fields):
    update_json(SONGS_JSON, rel, lambda e: e.update(fields))


# history.json: {"<lab path>": {"url", "title", "opened", "saved", "saves"}}
def record_history(rel, saved=False, **fields):
    def apply(e):
        e.update(fields)
        if saved:
            e["saved"] = now_iso()
            e["saves"] = e.get("saves", 0) + 1
        else:
            e["opened"] = now_iso()
    update_json(HISTORY_JSON, rel, apply)


def write_atomic(path, text):
    tmp = os.path.join(os.path.dirname(path), ".%s.tmp" % os.path.basename(path))
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def check_lab(text):
    """Contiguous `start end label` lines with valid Harte labels, or raise 400."""
    prev_end = 0.0
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise HTTPException(400, "isi .lab kosong")
    for n, line in enumerate(lines, 1):
        parts = line.split()
        try:
            start, end, label = float(parts[0]), float(parts[1]), parts[2]
            if len(parts) != 3 or end <= start or abs(start - prev_end) > 0.0015:
                raise ValueError("waktu tidak berurutan")
            parse_harte(label)
        except (ValueError, IndexError, KeyError) as e:
            raise HTTPException(400, "baris %d tidak valid (%s): %s" % (n, e, line))
        prev_end = end


class LoadReq(BaseModel):
    url: str
    lab: Optional[str] = None


class SaveReq(BaseModel):
    text: str


def detect_beats(path):
    y, sr = librosa.load(path, sr=22050, mono=True)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units="time", trim=False)
    tempo = float(np.atleast_1d(tempo)[0])
    return tempo, [round(float(b), 3) for b in beats]


@app.get("/api/labs")
def list_labs():
    songs = read_songs()
    out = []
    for dirpath, dirnames, files in os.walk(LAB_DIR):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for f in files:
            if f.endswith(".lab") and not f.startswith("."):
                rel = os.path.relpath(os.path.join(dirpath, f), LAB_DIR)
                out.append({"path": rel, **songs.get(rel, {})})
    return sorted(out, key=lambda s: s["path"].lower())


@app.get("/api/history")
def history():
    """Songs worked on from this machine, most recent first."""
    out = []
    for rel, e in read_json(HISTORY_JSON).items():
        try:
            lab_file(rel)
        except HTTPException:
            continue
        out.append({"path": rel, **e, "last": max(e.get("saved", ""), e.get("opened", ""))})
    return sorted(out, key=lambda e: e["last"], reverse=True)


@app.get("/api/lab")
def get_lab(path: str):
    with open(lab_file(path), encoding="utf-8") as f:
        return {"text": f.read(), **read_songs().get(path, {})}


@app.put("/api/lab")
def save_lab(path: str, req: SaveReq):
    full = lab_file(path)
    check_lab(req.text)
    write_atomic(full, req.text)
    saved = now_iso()
    update_song(path, saved=saved)
    record_history(path, saved=True)
    return {"saved": saved}


@app.post("/api/load")
def load(req: LoadReq):
    if req.lab is not None:
        lab_file(req.lab)
    sid = uuid.uuid4().hex
    d = os.path.join(TMP, sid)
    os.makedirs(d)
    opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(d, "src.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(req.url, download=True)
        src = next(os.path.join(d, f) for f in os.listdir(d) if f.startswith("src."))
        wav = os.path.join(d, "audio.wav")
        # Decode once to PCM WAV so the browser and beat tracker see identical
        # sample timing (no MP3/AAC encoder-delay ambiguity).
        subprocess.run([FFMPEG, "-v", "error", "-y", "-i", src, "-ac", "1", "-ar", str(SR), wav], check=True)
        os.remove(src)
        tempo, beats = detect_beats(wav)
    except Exception as e:
        shutil.rmtree(d, ignore_errors=True)
        raise HTTPException(500, "Gagal memuat audio: %s" % e)
    if req.lab is not None:
        update_song(req.lab, url=req.url)
        record_history(req.lab, url=req.url, title=info.get("title") or "")
    return {
        "id": sid,
        "videoId": info.get("id"),
        "title": info.get("title") or "untitled",
        "duration": round(librosa.get_duration(path=wav), 3),
        "tempo": round(tempo, 2),
        "beats": beats,
    }


@app.get("/api/audio/{sid}")
def audio(sid: str):
    path = os.path.join(session_dir(sid), "audio.wav")
    if not os.path.exists(path):
        raise HTTPException(404, "audio sudah dihapus")
    return FileResponse(path, media_type="audio/wav")


@app.delete("/api/session/{sid}")
def delete_session(sid: str):
    shutil.rmtree(session_dir(sid), ignore_errors=True)
    return {"deleted": sid}


app.mount("/", StaticFiles(directory=os.path.join(ROOT, "static"), html=True), name="static")

clean_tmp()
