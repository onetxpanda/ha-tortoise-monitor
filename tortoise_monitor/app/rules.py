"""What goes into training, what goes to Review, and what a detection means
against the pen outline.

Training frames come from the safety checks: every 2 minutes the location
automation asks the local LLM whether he is in his house (house camera) and,
when he isn't, whether he is outside the pen walls (yard camera). The frame
collected at each check's moment, with the first pass's (and the detector's)
boxes on it, is labelled by that answer:

  house camera   in his house + a box on him         -> him, with that box
                 out of his house + no clear box     -> not him
  yard camera    in his house + no clear tortoise    -> not him (look-alikes
                                                        such as the globe become
                                                        hard negatives)
                 out of his house + a clear box      -> him, with that box
                 out of his house + no box           -> not used (he may be
                                                        hidden)
  either         the answer and the boxes disagree,  -> Review
                 or "outside the pen walls"

Frigate's events (people, cats, dogs, deer ... on either camera; not the house
camera's "cat", which is him) are labelled by where the detector in use saw
him at that moment: on the yard camera while he was in his house, or on the
house camera while it didn't see him, the frame is "not him" unless something
in it strongly looks like him (then Review); with no record of where he was,
anything that looks like him goes to Review.

A frame only joins the training set if it teaches the detector in use
something: a frame with him that it misses or is less than SURE of, or a
frame without him where it puts a box on something. Frames of the run in use
stay in the set, so the next model can't forget what this one learned; frames
it already gets right are left out. Everything new that goes into a run is
made diverse first: frames with him, at most one per hour for each part of the
picture he is in, and none where he hasn't moved within 2 hours; frames
without him, an empty scene once per 20 minutes and each look-alike once per
hour.

Each camera has test days that no run trains on, so every run and the
detector in use are scored on the same frames (see choose_test_days()). Only
full frames are test frames -- what live detection sees -- never Frigate's
low-resolution event snapshots. The days are fixed when the next run is
approved."""
import bisect, collections, hashlib, json, math, threading, time
from datetime import datetime

from . import config, db, ha

POS, NEG, CAND = 0.40, 0.15, 0.10           # first-pass score: clearly him / clearly not / worth drawing
MODEL_YES, MODEL_SHOW = 0.50, 0.25          # detector in use: found / worth drawing
SURE = 0.75                                 # detector in use: a frame with him it scores this or more teaches it nothing
EVENT_NEAR = 60                             # a detector sighting within this many seconds says where he was at an event
CHECK_NEAR = 40                             # a collected frame within this many seconds of a check is that check's frame
SNAP_NEAR = 120                             # for Frigate's event snapshots
SAME_IOU, SAME_SECS = 0.8, 7200             # him in the same spot within 2 hours: not a new frame
GRID = (4, 3)                               # parts of the picture, for "one per hour for each part"
TEST_MIN, TEST_MAX = 0.20, 0.40             # test days: at least / never more than this share of the frames with him
NIGHT_SUN = -6.0                            # sun below this many degrees: night


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy; u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def local(ts):
    return datetime.fromtimestamp(ts).astimezone()


def role(cam):
    return config.cameras().get(cam, {}).get("role", "yard")


def outlined(cam):
    return role(cam) == "yard"     # only the yard has an outline: the pen walls


def wanted(camera, source, ts):
    """Frames worth collecting: all except yard frames at night (19:00-06:00),
    of which one per 5 minutes."""
    if source != "continuous" or role(camera) != "yard":
        return True
    h = local(ts).hour
    return 6 <= h < 19 or int(ts) % 300 < 30


def inside(pt, poly):
    x, y = pt; n = len(poly); hit = False
    for i in range(n):
        x1, y1 = poly[i]; x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            hit = not hit
    return hit


def foot(box, size):
    """Where his feet are, as fractions of the frame: the bottom centre of his box."""
    return ((box[0] + box[2]) / 2 / size[0], box[3] / size[1])


def outline(cam, c=None):
    o = db.kv_get(f"outline:{cam}", None, c)
    return o["poly"] if o and len(o.get("poly") or []) >= 3 else None


def place(cam, box, size, poly):
    if role(cam) == "house":
        return "house"
    if not poly:
        return "yard"
    return "pen" if inside(foot(box, size), poly) else "outside"


PLACE_TEXT = {"house": "in his house", "pen": "inside the pen walls", "outside": "outside the pen walls", "yard": "in the yard"}


# ---------------- frames, with their boxes and their safety check ----------------
class Frames:
    """Everything known about each frame with first-pass boxes. Reloaded when
    the tables change, checked at most every 30 s."""

    def __init__(self):
        self.lock = threading.Lock(); self.sig = None; self.checked = 0
        self.frames = {}; self.index = {}

    def refresh(self, force=False):
        with self.lock:
            if not force and time.time() - self.checked < 30:
                return
            self.checked = time.time()
            c = db.connect()
            sig = tuple(tuple(r) for r in c.execute(
                "SELECT (SELECT COUNT(*) FROM frames WHERE deleted IS NULL), (SELECT MAX(added) FROM frames),"
                " (SELECT COUNT(*) FROM owl), (SELECT MAX(at) FROM owl), (SELECT COUNT(*) FROM preds), (SELECT MAX(at) FROM preds),"
                " (SELECT COUNT(*) FROM checks), (SELECT group_concat(value) FROM kv WHERE key LIKE 'model:%')"))
            if sig != self.sig:
                self.frames, self.index = build(c)
                self.sig = sig


def load_checks(c):
    rows = c.execute("SELECT ts, house_in, outside FROM checks ORDER BY ts").fetchall()
    return [r["ts"] for r in rows], [{"ts": r["ts"], "house_in": bool(r["house_in"]), "outside": bool(r["outside"])} for r in rows]


def nearest_check(tss, chk, ts, near):
    i = bisect.bisect_left(tss, ts)
    best = None
    for j in (i - 1, i):
        if 0 <= j < len(tss) and abs(tss[j] - ts) <= near and (best is None or abs(tss[j] - ts) < abs(best["ts"] - ts)):
            best = chk[j]
    return best


def check_frames(c, idx=None):
    """The collected frame nearest each check, per camera (within CHECK_NEAR)."""
    tss, chk = load_checks(c)
    idx = idx if idx is not None else {r["file"]: dict(r) for r in c.execute("SELECT file, camera, ts, source FROM frames WHERE deleted IS NULL")}
    by_cam = {}
    for f, r in idx.items():
        if r["source"] == "continuous":
            by_cam.setdefault(r["camera"], []).append((r["ts"], f))
    out = set()
    for cam, seq in by_cam.items():
        seq.sort(); ts = [t for t, _ in seq]
        for t in tss:
            i = bisect.bisect_left(ts, t); best = None
            for j in (i - 1, i):
                if 0 <= j < len(ts) and abs(ts[j] - t) <= CHECK_NEAR and (best is None or abs(ts[j] - t) < abs(ts[best] - t)):
                    best = j
            if best is not None:
                out.add(seq[best][1])
    return out


def build(c):
    idx = {r["file"]: dict(r) for r in c.execute("SELECT * FROM frames WHERE deleted IS NULL")}
    owl = {}
    for r in c.execute("SELECT file, dets FROM owl"):
        o = owl.setdefault(r["file"], [])
        o += [[d[1], d[2]] for d in json.loads(r["dets"]) if d[0] in ("tortoise", "turtle") and d[1] >= CAND]
    use = {k[6:]: v for k, v in ((r["key"], json.loads(r["value"])) for r in c.execute("SELECT key, value FROM kv WHERE key LIKE 'model:%'"))}
    pred = {}
    for cam, m in use.items():
        for r in c.execute("SELECT p.file, p.dets FROM preds p JOIN frames f ON f.file = p.file WHERE p.run=? AND f.camera=?", (m["run"], cam)):
            pred[r["file"]] = json.loads(r["dets"])
    tss, chk = load_checks(c)
    at_check = check_frames(c, idx)
    cams = config.cameras()
    houses = [k for k in cams if cams[k].get("role", "yard") == "house"]
    yards = [k for k in cams if cams[k].get("role", "yard") != "house"]

    def seen(cams_, ts):
        q = ",".join("?" * len(cams_))
        return bool(cams_) and c.execute(f"SELECT 1 FROM detections WHERE camera IN ({q}) AND ts BETWEEN ? AND ? LIMIT 1",
                                         (*cams_, ts - EVENT_NEAR, ts + EVENT_NEAR)).fetchone() is not None

    def answered(cams_, ts):
        q = ",".join("?" * len(cams_)); h = ts - ts % 3600
        return bool(cams_) and c.execute(f"SELECT 1 FROM cycles WHERE camera IN ({q}) AND hour=? AND n > 0 LIMIT 1", (*cams_, h)).fetchone() is not None
    frames = {}
    for f, dets in owl.items():
        r = idx.get(f)
        if not r:
            continue
        cands = []
        for s, b in sorted(dets, key=lambda d: -d[0]):
            if all(iou(b, x[1]) < 0.5 for x in cands):
                cands.append([s, b, "first pass"])
        obox = cands[0][1] if cands else None
        s = cands[0][0] if cands else 0.0
        m = mbox = None
        if f in pred:
            d = max(pred[f], key=lambda d: d[0], default=None)
            m, mbox = (d[0], d[1]) if d else (0.0, None)
            if d and m >= MODEL_SHOW and all(iou(mbox, x[1]) < 0.5 for x in cands):
                cands.append([m, mbox, "detector"])
        cands.sort(key=lambda x: -x[0])
        snap = r["source"] == "event_snapshot"
        check = None
        if snap:
            check = nearest_check(tss, chk, r["ts"], SNAP_NEAR)
        elif f in at_check:
            check = nearest_check(tss, chk, r["ts"], CHECK_NEAR)
        x = {"file": f, "cam": r["camera"], "ts": r["ts"], "snap": snap, "size": [r["w"], r["h"]],
             "cands": cands[:4], "s": s, "obox": obox, "m": m, "mbox": mbox, "check": check,
             "event": r["event_id"] if r["source"] in ("event", "event_snapshot") else None, "flabel": r["frigate_label"]}
        if x["event"] and not (role(r["camera"]) == "house" and r["frigate_label"] == "cat"):
            x["where"] = ("house" if seen(houses, r["ts"]) else "yard" if seen(yards, r["ts"])
                          else "away" if answered(houses, r["ts"]) else None)
        x["label"], x["info"] = check_label(x)
        frames[f] = x
    return frames, idx


def best_box(x):
    """The box to train on: the detector's when it is sure and the first pass isn't, else the first pass's."""
    if x["m"] is not None and x["m"] >= MODEL_YES and x["mbox"] and (x["s"] < POS or x["m"] >= x["s"]):
        return x["mbox"]
    return x["obox"] if x["s"] >= NEG else (x["mbox"] if x["m"] is not None and x["m"] >= MODEL_SHOW else None)


def check_label(x):
    """What the safety check at this frame's moment says: ('him', box),
    ('not', None), ('ask', reason) or (None, None) (no check, or not usable)."""
    ck = x["check"]
    strong = x["s"] >= POS or (x["m"] is not None and x["m"] >= MODEL_YES)
    some = x["s"] >= NEG or (x["m"] is not None and x["m"] >= MODEL_SHOW)
    if "where" in x:                                 # a Frigate event (not the house camera's "cat": that's him)
        if x["m"] is None:
            return None, None                        # the detector in use hasn't looked at it yet
        house_cam = role(x["cam"]) == "house"
        if x["where"] == ("house" if house_cam else "yard"):
            return None, None                        # he was in this camera's view then
        if x["where"] == "house" or (house_cam and x["where"] == "away"):
            return ("ask", "event_box") if strong else ("not", None)     # he was elsewhere
        return ("ask", "event_box") if some else ("not", None)           # no record of where he was
    if ck is None:
        return ("ask", "snapshot") if x["snap"] and some else (None, None)
    if role(x["cam"]) == "house":
        if ck["house_in"]:
            return ("him", best_box(x)) if some else ("ask", "in_nobox")
        return ("ask", "out_box") if strong else ("not", None)
    if ck["outside"]:
        return ("ask", "outside")
    if ck["house_in"]:
        return ("ask", "in_yardbox") if strong else ("not", None)
    return ("him", best_box(x)) if strong else (None, None)


def why(x):
    return {
        "in_nobox": "The safety check saw him in his house, but there is no box on him here: drag one",
        "out_box": f"The safety check says he's out of his house, but something here looks like him ({max(x['s'], x['m'] or 0):.2f})",
        "in_yardbox": f"The safety check says he's in his house, but something in the yard looks like him ({max(x['s'], x['m'] or 0):.2f})",
        "outside": "The safety check says he's outside the pen walls",
        "snapshot": f"Frigate event snapshot: possible match ({max(x['s'], x['m'] or 0):.2f})",
        "event_box": f"Frigate event ({x.get('flabel') or 'something'}): something here looks like him ({max(x['s'], x['m'] or 0):.2f})",
    }.get(x["info"] if x["label"] == "ask" else "", "")


# ---------------- review, counts, training sets ----------------
def reviews(c):
    return {r["file"]: dict(r) for r in c.execute("SELECT * FROM reviews")}


def context(revs, frames):
    return {}


def final(x, rev, ctx=None):
    """(label, box, source): label pos/neg/uncertain/skip/unused; source you/check."""
    if rev:
        if rev["verdict"] == "tortoise":
            return "pos", json.loads(rev["box"]), "you"
        if rev["verdict"] == "none":
            return "neg", None, "you"
        return "skip", None, "you"
    if x["label"] == "him":
        return "pos", x["info"], "check"
    if x["label"] == "not":
        return "neg", None, "event" if x.get("event") and "where" in x else "check"
    if x["label"] == "ask":
        return "uncertain", None, None
    return "unused", None, None


def hour_key(x):
    return local(x["ts"]).strftime("%Y-%m-%d %H")


def asked(frames):
    """Frames Review asks about: one per camera, hour and reason."""
    keep = set(); seen = set()
    for x in sorted(frames.values(), key=lambda x: x["ts"]):
        if x["label"] != "ask":
            continue
        k = (x["cam"], x["file"] if x["snap"] else hour_key(x), x["info"])
        if k not in seen:
            seen.add(k); keep.add(x["file"])
    return keep


def queue(frames, revs, cam=None):
    keep = asked(frames)
    items = [frames[f] for f in keep if (not cam or frames[f]["cam"] == cam) and f not in revs]
    items.sort(key=lambda x: -x["ts"])               # newest first: today's light and weather
    return items


def counts(frames, revs):
    keep = asked(frames)
    out = {cam: {"boxed": 0, "pos_you": 0, "neg_you": 0, "skipped": 0, "pos_check": 0, "neg_check": 0, "neg_event": 0,
                 "to_review": 0, "unused": 0} for cam in config.cameras()}
    for x in frames.values():
        k = out.setdefault(x["cam"], dict.fromkeys(next(iter(out.values())), 0)); k["boxed"] += 1
        lab, _, src = final(x, revs.get(x["file"]))
        if lab == "uncertain":
            k["to_review" if x["file"] in keep else "unused"] += 1
        elif lab == "skip":
            k["skipped"] += 1
        elif lab == "unused":
            k["unused"] += 1
        else:
            k[f"{lab}_{src}"] += 1
    return out


def teaches(x, box):
    """Whether a labelled frame teaches the detector in use something: a frame
    with him it misses or is less than SURE of (or boxes in the wrong place), a
    frame without him where it puts a box on something. None: it hasn't looked."""
    if x["m"] is None:
        return None
    if box is None:
        return x["m"] >= MODEL_SHOW
    return not (x["mbox"] is not None and x["m"] >= SURE and iou(x["mbox"], box) >= 0.5)


def in_use_items(cam, c=None):
    """(run, files) of the run whose detector is in use for this camera."""
    c = c or db.connect()
    m = db.kv_get(f"model:{cam}", None, c)
    if not m:
        return None, set()
    r = c.execute("SELECT items FROM runs WHERE id=?", (m["run"],)).fetchone()
    return m["run"], set(json.loads(r["items"]) if r and r["items"] else [])


_where = {"latlon": None, "at": 0.0}


def latlon():
    """Home Assistant's location, asked at most once an hour."""
    if time.time() - _where["at"] > 3600:
        _where["at"] = time.time()
        try:
            _where["latlon"] = ha.location()
        except Exception:
            pass
    return _where["latlon"]


def sun_elevation(ts, lat, lon):
    """The sun's elevation in degrees at a time and place (within about a degree)."""
    d = ts / 86400.0 - 10957.5                                  # days since 2000-01-01 12:00 UTC
    g = math.radians((357.529 + 0.98560028 * d) % 360)
    q = (280.459 + 0.98564736 * d) % 360
    L = math.radians((q + 1.915 * math.sin(g) + 0.020 * math.sin(2 * g)) % 360)
    e = math.radians(23.439 - 0.00000036 * d)
    ra = math.atan2(math.cos(e) * math.sin(L), math.cos(L))
    dec = math.asin(math.sin(e) * math.sin(L))
    gmst = (18.697374558 + 24.06570982441908 * d) % 24
    h = math.radians(gmst * 15 + lon) - ra
    la = math.radians(lat)
    return math.degrees(math.asin(math.sin(la) * math.sin(dec) + math.cos(la) * math.cos(dec) * math.cos(h)))


def night(ts):
    """Whether the sun was down (below NIGHT_SUN), or None without a location."""
    ll = latlon()
    return None if not ll else sun_elevation(ts, *ll) < NIGHT_SUN


def day_of(x):
    return local(x["ts"]).strftime("%Y-%m-%d")


def spot(x, box):
    """The part of the picture his feet are in."""
    fx, fy = foot(box, x["size"])
    return min(GRID[0] - 1, int(fx * GRID[0])), min(GRID[1] - 1, int(fy * GRID[1]))


def neg_kind(x):
    """What a frame without him shows: a Frigate event's label, a look-alike, or an empty scene."""
    if x.get("event"):
        return x.get("flabel") or "event"
    lk = max(([x["s"], x["obox"]], [x["m"] or 0, x["mbox"]]), key=lambda t: t[0])
    return "look-alike" if lk[0] >= NEG and lk[1] else "empty"


def diverse(pos, neg):
    """Labelled frames (frame, box, source, from the run in use), made diverse:
    with him, at most one per hour for each part of the picture he's in (the run
    in use's frames first, then yours) and none where he hasn't moved within 2
    hours; without him, a Frigate event once, a look-alike once per hour for
    each spot, an empty scene once per 20 minutes. The run in use's frames always stay."""
    order = lambda t: (not t[3], t[2] != "you", t[0]["snap"], t[0]["ts"])
    kept, keys, recent = [], set(), collections.defaultdict(list)
    for x, b, src, base in sorted(pos, key=order):
        k = (x["file"] if x["snap"] else hour_key(x), *spot(x, b))
        h = int(x["ts"] // 3600)
        if not base:
            if k in keys:
                continue
            if not x["snap"] and any(abs(x["ts"] - t) < SAME_SECS and iou(b, bb) >= SAME_IOU
                                     for hh in range(h - 2, h + 3) for t, bb in recent.get(hh, ())):
                continue
        keys.add(k); recent[h].append((x["ts"], b)); kept.append((x, b, src, base))
    negs, nkeys = [], set()
    for x, b, src, base in sorted(neg, key=order):
        kind = neg_kind(x)
        if x.get("event"):
            k = ("event", x["event"])
        elif x["snap"]:
            k = x["file"]
        elif kind == "look-alike":
            lk = max(([x["s"], x["obox"]], [x["m"] or 0, x["mbox"]]), key=lambda t: t[0])
            k = ("look-alike", hour_key(x), *spot(x, lk[1]))
        else:
            t = local(x["ts"]); k = ("empty", t.strftime("%Y-%m-%d %H"), t.minute // 20)
        if not base and k in nkeys:
            continue
        nkeys.add(k); negs.append((x, None, src, base))
    return kept, negs


def trained_days(run, c):
    """Days a run trained on: the days of its frames, except the days it was
    tested on. Without recorded test days (runs before whole-day testing),
    every day of its frames counts."""
    if run is None:
        return set()
    r = c.execute("SELECT items, counts FROM runs WHERE id=?", (run,)).fetchone()
    if not r or not r["items"]:
        return set()
    k = json.loads(r["counts"] or "{}")
    held = set(k.get("test_days") or k.get("val_days") or [])
    files = json.loads(r["items"]); days = set()
    for i in range(0, len(files), 500):
        part = files[i:i + 500]
        for row in c.execute(f"SELECT ts FROM frames WHERE file IN ({','.join('?' * len(part))})", part):
            days.add(local(row["ts"]).strftime("%Y-%m-%d"))
    return days - held


def choose_test_days(pos, negs):
    """Test days, from full frames only (event snapshots are left out of the
    choice): first, day by day, the day that adds the most of what the test set
    lacks -- him at day and at night in each part of the picture, and without
    him each kind of scene (each Frigate label, look-alikes, empty) at day and
    at night -- preferring days with fewer frames with him; then the smallest
    days until the test set has TEST_MIN of the full frames with him. Never more
    than TEST_MAX of them."""
    cells, npos = collections.defaultdict(set), collections.Counter()
    for x, b, *_ in pos:
        if not x["snap"]:
            d = day_of(x); cells[d].add(("him", night(x["ts"]), *spot(x, b))); npos[d] += 1
    for x, *_ in negs:
        if not x["snap"]:
            cells[day_of(x)].add(("not", night(x["ts"]), neg_kind(x)))
    total = sum(npos.values())
    cand = set(cells)
    chosen, have, n = set(), set(), 0
    fits = lambda d: not total or n + npos[d] <= TEST_MAX * total
    while True:
        best = max((d for d in cand - chosen if fits(d) and cells[d] - have),
                   key=lambda d: (len(cells[d] - have), -npos[d], d), default=None)
        if best is None:
            break
        chosen.add(best); have |= cells[best]; n += npos[best]
    for d in sorted(cand - chosen, key=lambda d: (npos[d], d)):
        if n >= TEST_MIN * total:
            break
        if npos[d] and fits(d):
            chosen.add(d); n += npos[d]
    return chosen


def in_use_score(tpos, tneg):
    """The detector in use on the test frames, from the detector pass's results:
    found him (best box at least MODEL_YES, overlapping his box by at least
    0.3), false alarms (a box at least MODEL_YES on a frame without him), and
    frames it hasn't looked at yet. The training job scores runs the same way."""
    k = {"with_him": 0, "found": 0, "without_him": 0, "false_alarms": 0, "waiting": 0}
    for x, b, *_ in tpos + tneg:
        if x["m"] is None:
            k["waiting"] += 1; continue
        hit = x["m"] >= MODEL_YES and x["mbox"] is not None
        if b:
            k["with_him"] += 1; k["found"] += bool(hit and iou(x["mbox"], b) >= 0.3)
        else:
            k["without_him"] += 1; k["false_alarms"] += bool(hit)
    return k


def dataset(frames, revs, cam, c=None):
    """Training items [(frame, box or None, split)] and their counts. split:
    "train" (the other days) or "test" (the test days).

    The test days are fixed once a run has been approved with them (kv
    testdays:<cam>); until then they are chosen afresh (choose_test_days()).
    Every labelled full frame of a test day is a test frame, made diverse; the
    day's event snapshots are not used; nothing from a test day ever trains.

    On the other days, the run in use's frames stay, and a new labelled frame
    (your answers, the safety checks, Frigate events) joins only if it teaches
    the detector in use something (teaches()); then they are made diverse
    (diverse()), and no training day may give more new frames with him than
    the next-largest day or a third of them, whichever is more."""
    c = c or db.connect()
    run_in_use, base_files = in_use_items(cam, c)
    labelled = []
    for x in frames.values():
        if x["cam"] != cam:
            continue
        lab, box, src = final(x, revs.get(x["file"]))
        if lab == "pos" and box:
            W, H = x["size"]
            box = [max(0, min(W, box[0])), max(0, min(H, box[1])), max(0, min(W, box[2])), max(0, min(H, box[3]))]
        elif lab == "neg":
            box = None
        else:
            continue
        labelled.append((x, box, src, x["file"] in base_files))
    trained = trained_days(run_in_use, c)
    fixed = db.kv_get(f"testdays:{cam}", None, c)
    if fixed:
        tdays = set(fixed["days"])
    else:
        allp, alln = diverse([t for t in labelled if t[1]], [t for t in labelled if not t[1]])
        tdays = choose_test_days(allp, alln)
    tl = [t for t in labelled if day_of(t[0]) in tdays and not t[0]["snap"]]
    test_snaps = sum(1 for t in labelled if day_of(t[0]) in tdays and t[0]["snap"])
    tpos, tneg = diverse([t for t in tl if t[1]], [t for t in tl if not t[1]])

    pos, neg = [], []
    known = waiting = 0
    for x, box, src, base in labelled:
        if day_of(x) in tdays:
            continue
        if run_in_use is not None and not base:
            t = teaches(x, box)
            if t is None:
                waiting += 1; continue
            if not t:
                known += 1; continue
        (pos if box else neg).append((x, box, src, base))
    n_pos_all, n_neg_all = len(pos), len(neg)
    kept, negs = diverse(pos, neg)
    # cap the new frames any one training day gives
    by_day = {}
    for t in kept:
        by_day.setdefault(day_of(t[0]), []).append(t)
    capped = {}
    sizes = sorted((len(v) for v in by_day.values()), reverse=True)
    if len(sizes) >= 2:
        cap = max(sizes[1], sum(min(s, sizes[1]) for s in sizes) // 3)
        for d, v in by_day.items():
            if len(v) > cap:
                old = [t for t in v if t[3]]
                fresh = sorted((t for t in v if not t[3]), key=lambda t: t[0]["ts"])
                room = max(0, cap - len(old))
                step = len(fresh) / room if room else 0
                by_day[d] = old + ([fresh[int(i * step)] for i in range(room)] if room else [])
                capped[d] = [len(v), len(by_day[d])]
    train_pos = [t for v in by_day.values() for t in v]
    items = [(x, b, "train") for x, b, _, _ in train_pos] + [(x, None, "train") for x, _, _, _ in negs] + \
            [(x, b, "test") for x, b, _, _ in tpos] + [(x, None, "test") for x, _, _, _ in tneg]
    nights = [night(x["ts"]) for x, *_ in tpos], [night(x["ts"]) for x, *_ in tneg]
    k = {"pos": len(train_pos), "pos_all": n_pos_all, "pos_you": sum(1 for p in train_pos if p[2] == "you"),
         "neg": len(negs), "neg_all": n_neg_all, "neg_you": sum(1 for n in negs if n[2] == "you"),
         "neg_event": sum(1 for n in negs if n[2] == "event"),
         "in_use_run": run_in_use, "from_in_use": sum(1 for p in train_pos if p[3]) + sum(1 for n in negs if n[3]),
         "new": sum(1 for p in train_pos if not p[3]) + sum(1 for n in negs if not n[3]), "known": known, "waiting": waiting,
         "left_out": sum(1 for x in frames.values() if x["cam"] == cam and x["label"] == "ask" and x["file"] not in revs),
         "split": "test days", "capped": capped, "train_days": len({day_of(t[0]) for t in train_pos}),
         "train": sum(1 for i in items if i[2] == "train"),
         "test_days": sorted(tdays), "test_fixed": bool(fixed), "test_since_run": fixed.get("run") if fixed else None,
         "test": len(tpos) + len(tneg), "test_pos": len(tpos), "test_neg": len(tneg), "test_snaps_unused": test_snaps,
         "test_night_pos": None if None in nights[0] else sum(nights[0]),
         "test_night_neg": None if None in nights[1] else sum(nights[1]),
         "test_spots": len({spot(x, b) for x, b, *_ in tpos}), "spots": len({spot(x, b) for x, b, *_ in labelled if b and not x["snap"]}),
         "test_events": dict(collections.Counter(neg_kind(x) for x, *_ in tneg if x.get("event"))),
         "test_in_use_trained": sum(1 for x, *_ in tpos + tneg if day_of(x) in trained),
         "test_in_use": in_use_score(tpos, tneg) if run_in_use is not None else None,
         "reviews": sum(1 for f in revs if f in frames and frames[f]["cam"] == cam)}
    return items, k


def default_config(cam):
    imgsz = int(config.cameras().get(cam, {}).get("imgsz", 960))
    return {"model": "yolo11n.pt", "imgsz": imgsz, "epochs": 40, "batch": 16, "device": "cpu"}


def protected_files(c):
    """Frames that must never be deleted: everything you answered, and every
    frame a training run used."""
    keep = {r["file"] for r in c.execute("SELECT file FROM reviews")}
    for r in c.execute("SELECT items FROM runs WHERE items IS NOT NULL"):
        keep.update(json.loads(r["items"]))
    return keep
