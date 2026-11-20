"""Home Assistant and Supervisor, through the Supervisor's proxy with the
add-on's own token (no user token anywhere)."""
import bisect, json, os, time, urllib.error, urllib.parse, urllib.request
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


def refresh_checks(log=print):
    """The safety checks' answers, from Home Assistant's history of the
    location automation's helpers: each change of check_entity is a check; a
    change of sighting_entity within the next 110 s means that check saw him,
    on the camera sighting_camera_entity named then. Stored in `checks`."""
    o = config.options()
    ents = [o.get("check_entity"), o.get("sighting_entity"), o.get("sighting_camera_entity")]
    if not TOKEN or not all(ents):
        return
    c = db.connect()
    last = c.execute("SELECT MAX(ts) FROM checks").fetchone()[0]
    start = (last - 600) if last else time.time() - 10 * 86400
    since = datetime.fromtimestamp(start, timezone.utc).isoformat()
    q = urllib.parse.urlencode({"filter_entity_id": ",".join(ents), "minimal_response": "1", "no_attributes": "1"})
    try:
        hist = _get(f"http://supervisor/core/api/history/period/{urllib.parse.quote(since)}?{q}")
    except Exception as e:
        log(f"safety checks: {e}"); return
    series = {}
    for rows in hist:
        if rows:
            series[rows[0]["entity_id"]] = [(datetime.fromisoformat(r["last_changed"]).timestamp(), r["state"]) for r in rows]
    checks = sorted(t for t, _ in series.get(ents[0], []) if t >= start)
    seen = sorted(t for t, _ in series.get(ents[1], []) if t >= start)
    cams = sorted(series.get(ents[2], []))
    cam_at = lambda t: next((s for tt, s in reversed(cams) if tt <= t + 1), cams[0][1] if cams else "")
    hit = {}
    for s in seen:
        i = bisect.bisect_right(checks, s + 5) - 1
        if i >= 0 and s - checks[i] < 110:
            hit[i] = cam_at(s)
    for i, t in enumerate(checks):
        cam = (hit.get(i) or "").lower()
        c.execute("INSERT OR REPLACE INTO checks VALUES (?,?,?)", (t, int("house" in cam), int("yard" in cam)))
    c.commit()
    if checks:
        log(f"safety checks: {len(checks)} read, up to {datetime.fromtimestamp(checks[-1]).astimezone().strftime('%a %H:%M')}")
