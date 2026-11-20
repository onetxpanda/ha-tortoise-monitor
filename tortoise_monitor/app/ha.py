"""Home Assistant and Supervisor, through the Supervisor's proxy with the
add-on's own token (no user token anywhere)."""
import json, os, time, urllib.error, urllib.parse, urllib.request
from datetime import datetime, timezone

from . import config, db

TOKEN = os.environ.get("SUPERVISOR_TOKEN", "")


def _get(url):
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {TOKEN}"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.load(r)


def available():
    return bool(TOKEN)


def mqtt_service():
    """Broker host, port, username and password, as the Mosquitto add-on publishes them."""
    if not TOKEN:
        return None
    try:
        return _get("http://supervisor/services/mqtt").get("data")
    except Exception:
        return None


def host_address():
    """This host's primary IPv4 address (for training servers to reach the add-on)."""
    if not TOKEN:
        return None
    try:
        for i in _get("http://supervisor/network/info")["data"]["interfaces"]:
            if i.get("primary") and (i.get("ipv4") or {}).get("address"):
                return i["ipv4"]["address"][0].split("/")[0]
    except Exception:
        pass
    return None


def refresh_record(log=print):
    """The in/out record (the record entity's history, on = out of his
    house) as intervals, kept in the database for the review rules."""
    ent = config.options().get("record_entity")
    if not TOKEN or not ent:
        return
    c = db.connect()
    rec = db.kv_get("record", {"covered": [0, 0], "out": []}, c)
    start = max(rec["covered"][1] - 86400, time.time() - 30 * 86400)
    since = datetime.fromtimestamp(start, timezone.utc).isoformat()
    q = urllib.parse.urlencode({"filter_entity_id": ent, "minimal_response": "1", "no_attributes": "1"})
    try:
        hist = _get(f"http://supervisor/core/api/history/period/{urllib.parse.quote(since)}?{q}")
    except Exception as e:
        log(f"in/out record: {e}"); return
    rows = hist[0] if hist else []
    out = [iv for iv in rec["out"] if iv[1] < start]
    on_since = None; first = None
    for r in rows:
        t = datetime.fromisoformat(r["last_changed"]).timestamp()
        first = first if first is not None else t
        if r["state"] == "on" and on_since is None:
            on_since = t
        elif r["state"] != "on" and on_since is not None:
            out.append([on_since, t]); on_since = None
    now = time.time()
    if on_since is not None:
        out.append([on_since, now])
    covered = [rec["covered"][0] or (first or start), now]
    db.kv_set("record", {"covered": covered, "out": out}, c)
    c.commit()
