"""The add-on's database (SQLite, one connection per call site)."""
import json, sqlite3, threading, time

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS frames(
  file TEXT PRIMARY KEY, camera TEXT NOT NULL, ts REAL NOT NULL, source TEXT NOT NULL,
  event_id TEXT, frigate_label TEXT, frigate_score REAL, w INTEGER, h INTEGER, bytes INTEGER,
  added REAL NOT NULL, deleted REAL, deleted_why TEXT);
CREATE INDEX IF NOT EXISTS frames_cam_ts ON frames(camera, ts);
CREATE TABLE IF NOT EXISTS missing(camera TEXT NOT NULL, ts REAL NOT NULL, PRIMARY KEY(camera, ts));
CREATE TABLE IF NOT EXISTS owl(file TEXT NOT NULL, mode TEXT NOT NULL, secs REAL, dets TEXT NOT NULL, at REAL NOT NULL, PRIMARY KEY(file, mode));
CREATE TABLE IF NOT EXISTS preds(file TEXT NOT NULL, run INTEGER NOT NULL, dets TEXT NOT NULL, at REAL NOT NULL, PRIMARY KEY(file, run));
CREATE TABLE IF NOT EXISTS reviews(file TEXT PRIMARY KEY, verdict TEXT NOT NULL, box TEXT, rejected TEXT, at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS runs(
  id INTEGER PRIMARY KEY, camera TEXT NOT NULL, status TEXT NOT NULL, created REAL NOT NULL,
  decided REAL, started REAL, finished REAL, config TEXT NOT NULL, counts TEXT, result TEXT,
  server TEXT, progress TEXT, items TEXT);
CREATE TABLE IF NOT EXISTS servers(
  id TEXT PRIMARY KEY, name TEXT NOT NULL, key_hash TEXT NOT NULL, created REAL NOT NULL,
  last_seen REAL, info TEXT, settings TEXT);
CREATE TABLE IF NOT EXISTS jobs(
  id INTEGER PRIMARY KEY, kind TEXT NOT NULL, status TEXT NOT NULL, server TEXT, run INTEGER,
  payload TEXT, created REAL NOT NULL, started REAL, finished REAL, progress TEXT, result TEXT);
CREATE TABLE IF NOT EXISTS detections(
  id INTEGER PRIMARY KEY, camera TEXT NOT NULL, ts REAL NOT NULL, run INTEGER, source TEXT,
  score REAL, box TEXT, w INTEGER, h INTEGER, fx REAL, fy REAL, place TEXT);
CREATE INDEX IF NOT EXISTS det_cam_ts ON detections(camera, ts);
CREATE TABLE IF NOT EXISTS checks(ts REAL PRIMARY KEY, house_in INTEGER NOT NULL, outside INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS cycles(camera TEXT NOT NULL, hour REAL NOT NULL, n INTEGER NOT NULL, seen INTEGER NOT NULL, outside INTEGER NOT NULL, PRIMARY KEY(camera, hour));
CREATE TABLE IF NOT EXISTS kv(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS deletions(at REAL NOT NULL, tier TEXT NOT NULL, files INTEGER NOT NULL, bytes INTEGER NOT NULL);
"""

_init = threading.Lock()
_done = False


def connect():
    global _done
    c = sqlite3.connect(config.DB, timeout=60)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA synchronous=NORMAL")
    if not _done:
        with _init:
            if not _done:
                c.executescript(SCHEMA); c.commit(); _done = True
    return c


def kv_get(key, default=None, c=None):
    c = c or connect()
    r = c.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
    return json.loads(r["value"]) if r else default


def kv_set(key, value, c=None):
    own = c is None
    c = c or connect()
    if value is None:
        c.execute("DELETE FROM kv WHERE key=?", (key,))
    else:
        c.execute("INSERT INTO kv(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, json.dumps(value)))
    if own:
        c.commit()


def now():
    return time.time()
