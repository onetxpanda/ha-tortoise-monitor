"""Collection: full-resolution frames from Frigate's recordings, through
Frigate's API (never the cameras directly).

  continuous  one frame every collect_every_s over the window Frigate keeps
              (about a day), so every ordinary scene ends up in the set;
              night-time yard frames only one per 5 minutes.
  event       frames through every Frigate event on these cameras (people,
              dogs, rabbits, birds ... and him): the negatives.
  snapshot    each event's clean 640x360 snapshot, kept by Frigate for weeks.

Frames go to frames/<camera>/<YYYY-MM-DD>/ under the media folder, one row
each in the database. Already-saved frames are skipped."""
import json, os, threading, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor

from . import config, db, rules

WINDOW = int(23.5 * 3600)
SETTLE = 120               # the newest 2 minutes may still be being written
EVENT_DAYS = 120          # Frigate keeps event snapshots for months; frames only for about a day
EVENT_STEP = 5
EVENT_MAX_FRAMES = 24
PARALLEL = 2               # Frigate decodes each frame: keep its load light

state = {"running": False, "last_start": None, "last_end": None, "last_ok": None, "last_counts": {}, "error": None}
_lock = threading.Lock()


def _get(path):
    with urllib.request.urlopen(config.options()["frigate_url"] + "/api" + path, timeout=60) as r:
        return json.load(r)


def _jpeg_size(data):
    i = 2
    while i < len(data) - 9:
        if data[i] != 0xFF:
            return None
        m = data[i + 1]; ln = int.from_bytes(data[i + 2:i + 4], "big")
        if m in (0xC0, 0xC1, 0xC2):
            return int.from_bytes(data[i + 7:i + 9], "big"), int.from_bytes(data[i + 5:i + 7], "big")
        i += 2 + ln
    return None


def store(rel, data, meta):
    """Write a frame file and its row (used by collection and by the import)."""
    path = os.path.join(config.FRAMES, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".part", "wb") as f:
        f.write(data)
    os.replace(path + ".part", path)
    w, h = _jpeg_size(data) or (None, None)
    c = db.connect()
    c.execute("INSERT OR REPLACE INTO frames(file, camera, ts, source, event_id, frigate_label, frigate_score, w, h, bytes, added)"
              " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
              (rel, meta["camera"], meta["ts"], meta["source"], meta.get("event_id"), meta.get("frigate_label"),
               meta.get("frigate_score"), w, h, len(data), time.time()))
    c.commit()


def _have(c):
    return {r["file"] for r in c.execute("SELECT file FROM frames")}


def _fetch(job, have, missing):
    cam, ts, meta = job
    day = rules.local(ts).strftime("%Y-%m-%d")
    rel = f"{cam}/{day}/{cam}-{ts:.1f}.jpg".replace(".0.jpg", ".jpg")
    if rel in have:
        return "have"
    if (cam, float(ts)) in missing:
        return "missing"
    url = f"{config.options()['frigate_url']}/api/{cam}/recordings/{ts}/snapshot.jpg"
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            data = r.read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            c = db.connect(); c.execute("INSERT OR IGNORE INTO missing VALUES (?, ?)", (cam, float(ts))); c.commit()
            return "missing"
        return "error"
    except Exception:
        return "error"
    if len(data) < 10_000 or not data.startswith(b"\xff\xd8"):
        return "error"
    store(rel, data, {"camera": cam, "ts": ts, **meta})
    return "saved"


def run_once(log=print):
    if not _lock.acquire(blocking=False):
        return
    state.update(running=True, last_start=time.time(), error=None)
    counts = {}
    try:
        now = time.time()
        c = db.connect(); have = _have(c)
        missing = {(r["camera"], r["ts"]) for r in c.execute("SELECT camera, ts FROM missing")}
        after = int(now) - EVENT_DAYS * 86400
        jobs, snaps = [], []
        for cam, cc in config.cameras().items():
            events = _get(f"/events?cameras={cam}&after={after}&limit=10000&include_thumbnails=0")
            for e in events:
                if e.get("end_time") is None:
                    continue
                data = e.get("data") or {}
                meta = {"source": "event", "event_id": e["id"], "frigate_label": e["label"], "frigate_score": data.get("top_score")}
                t0, t1 = float(e["start_time"]), float(e["end_time"])
                if t1 > now - WINDOW:
                    n = min(EVENT_MAX_FRAMES, max(1, int((t1 - t0) // EVENT_STEP) + 1))
                    for ts in sorted({round(t0 + (t1 - t0) * i / max(1, n - 1), 1) for i in range(n)}):
                        jobs.append((cam, ts, meta))
                if e.get("has_snapshot"):
                    snap = data.get("snapshot") or {}
                    snaps.append((cam, e["id"], {**meta, "source": "event_snapshot", "ts": snap.get("frame_time") or e["start_time"]}))
            step = int(cc.get("collect_every_s", 60))
            for ts in range((int(now) - WINDOW) // step * step + step, int(now) - SETTLE, step):
                if rules.wanted(cam, "continuous", ts):
                    jobs.append((cam, ts, {"source": "continuous"}))
        with ThreadPoolExecutor(PARALLEL) as pool:
            for (cam, _, meta), res in zip(jobs, pool.map(lambda j: _fetch(j, have, missing), jobs)):
                k = f"{cam} {meta['source']} {res}"; counts[k] = counts.get(k, 0) + 1
        for cam, eid, meta in snaps:
            rel = f"{cam}/snapshots/{cam}-snap-{eid}.jpg"
            if rel in have:
                res = "have"
            else:
                try:
                    with urllib.request.urlopen(f"{config.options()['frigate_url']}/api/events/{eid}/snapshot.jpg?bbox=0&timestamp=0&crop=0", timeout=60) as r:
                        data = r.read()
                    if data.startswith(b"\xff\xd8"):
                        store(rel, data, {"camera": cam, **meta}); res = "saved"
                    else:
                        res = "error"
                except Exception:
                    res = "error"
            k = f"{cam} snapshot {res}"; counts[k] = counts.get(k, 0) + 1
        state.update(last_ok=True)
        log("collection: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items()) if "have" not in k))
    except Exception as e:
        state.update(last_ok=False, error=str(e)); log(f"collection failed: {e}")
    finally:
        state.update(running=False, last_end=time.time(), last_counts=counts)
        _lock.release()
