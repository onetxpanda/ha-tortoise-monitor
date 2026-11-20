"""Paths and options. The Supervisor writes the add-on's options to
/data/options.json; outside Home Assistant (tests) TM_DATA and TM_MEDIA point
elsewhere.

/data (backed up with the add-on): the database and compiled-model cache.
/media/tortoise_monitor: frames, models and live snapshots (large; the
backup can leave the media folder out)."""
import json, os, re

DATA = os.environ.get("TM_DATA", "/data")
MEDIA = os.environ.get("TM_MEDIA", "/media/tortoise_monitor")
FRAMES = os.path.join(MEDIA, "frames")
MODELS = os.path.join(MEDIA, "models")
LIVE = os.path.join(MEDIA, "live")
DB = os.path.join(DATA, "tortoise_monitor.db")
OV_CACHE = os.path.join(DATA, "openvino_cache")
HERE = os.path.dirname(os.path.abspath(__file__))
ADDON_DIR = os.path.dirname(HERE)
WORKER_DIR = os.environ.get("TM_WORKER_DIR", os.path.join(ADDON_DIR, "worker"))
WEB = os.path.join(HERE, "web")

DEFAULTS = {
    "frigate_url": "http://ccab4aaf-frigate-fa:5000",      # Frigate's API (the Frigate add-on's internal name)
    "restream_url": "rtsp://ccab4aaf-frigate-fa:8554",     # Frigate's restream, for detection in this add-on
    "server_restream_url": "",                             # the same for training servers ("" = rtsp://<this host>:8554)
    "cameras": [
        {"name": "yard_camera", "role": "yard", "collect_every_s": 30, "imgsz": 960},
        {"name": "tortoise_camera", "role": "house", "collect_every_s": 60, "imgsz": 640},
    ],
    # the safety checks (the location automation's LLM answers), read from Home Assistant's history:
    "check_entity": "input_datetime.tortoise_vision_last_check",      # set at every check that got an answer
    "sighting_entity": "input_datetime.tortoise_last_seen",           # set when a check saw him
    "sighting_camera_entity": "input_text.tortoise_last_seen_camera", # "House camera" / "Yard camera"
    "detect_every_s": 5,
    "server_port": 8765,
    "advertise_url": "",                                   # how training servers reach this add-on ("" = http://<this host>:server_port)
    "prune_at_percent": 95,
}
ROLES = {"yard": "the yard", "house": "his house"}


def options():
    o = dict(DEFAULTS)
    try:
        with open(os.path.join(DATA, "options.json")) as f:
            o.update({k: v for k, v in json.load(f).items() if v not in (None, "")})
    except (OSError, ValueError):
        pass
    return o


def cameras():
    return {c["name"]: c for c in options()["cameras"]}


def version():
    for p in (os.path.join(ADDON_DIR, "config.yaml"), "/app/config.yaml"):
        try:
            with open(p) as f:
                m = re.search(r'^version:\s*"?([\w.\-]+)"?', f.read(), re.M)
            if m:
                return m.group(1)
        except OSError:
            pass
    return "0.0.0"


for d in (DATA, FRAMES, MODELS, LIVE, OV_CACHE):
    os.makedirs(d, exist_ok=True)
