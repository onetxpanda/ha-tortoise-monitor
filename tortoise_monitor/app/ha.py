"""Home Assistant and Supervisor, through the Supervisor's proxy with the
add-on's own token (no user token anywhere)."""
import bisect, json, os, time, urllib.error, urllib.parse, urllib.request
from datetime import datetime, timezone

from . import config, db

TOKEN = os.environ.get("SUPERVISOR_TOKEN", "")


def _get(url, timeout=20):
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {TOKEN}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
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


def location():
    """Home Assistant's latitude and longitude, or None."""
    if not TOKEN:
        return None
    d = _get("http://supervisor/core/api/config")
    return float(d["latitude"]), float(d["longitude"])


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
    try:
        _refresh_checks(log)
    except Exception as e:
        import traceback
        log(f"safety checks: reading them failed: {type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}")


def _refresh_checks(log):
    o = config.options()
    ents = [o.get("check_entity"), o.get("sighting_entity"), o.get("sighting_camera_entity")]
    if not TOKEN or not all(ents):
        log("safety checks: not read (no Supervisor token, or the check helpers aren't set in the options)")
        return
    c = db.connect()
    last = c.execute("SELECT MAX(ts) FROM checks").fetchone()[0]
    start = (last - 600) if last else time.time() - 10 * 86400
    since = datetime.fromtimestamp(start, timezone.utc).isoformat()
    # end_time must be given: without it Home Assistant returns only one day from the start
    q = urllib.parse.urlencode({"filter_entity_id": ",".join(ents), "end_time": datetime.now(timezone.utc).isoformat(),
                                "minimal_response": "1", "no_attributes": "1"})
    hist = _get(f"http://supervisor/core/api/history/period/{urllib.parse.quote(since)}?{q}", timeout=120)
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
    log(f"safety checks: {len(checks)} read since {datetime.fromtimestamp(start).astimezone().strftime('%a %d %b %H:%M')}"
        f" ({sum(1 for v in hit.values() if 'house' in (v or '').lower())} saw him in his house, {sum(1 for v in hit.values() if 'yard' in (v or '').lower())} outside the walls);"
        f" {c.execute('SELECT COUNT(*) FROM checks').fetchone()[0]} in all; history had {', '.join(f'{k.split(chr(46))[1]} {len(v)}' for k, v in series.items()) or 'nothing'}")
