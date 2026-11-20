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

Your answers in Review always count. Everything that goes into a run is made
diverse first: frames with him, at most one per hour for each part of the
picture he is in, and none where he hasn't moved within 2 hours; frames
without him, an empty scene once per 20 minutes and each look-alike once per
hour. Then whole days are held out for testing and no
day may outweigh the rest (see dataset())."""
import bisect, hashlib, json, threading, time
from datetime import datetime

from . import config, db

POS, NEG, CAND = 0.40, 0.15, 0.10           # first-pass score: clearly him / clearly not / worth drawing
MODEL_YES, MODEL_SHOW = 0.50, 0.25          # detector in use: found / worth drawing
CHECK_NEAR = 40                             # a collected frame within this many seconds of a check is that check's frame
SNAP_NEAR = 120                             # for Frigate's event snapshots
SAME_IOU, SAME_SECS = 0.8, 7200             # him in the same spot within 2 hours: not a new frame
GRID = (4, 3)                               # parts of the picture, for "one per hour for each part"


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
             "cands": cands[:4], "s": s, "obox": obox, "m": m, "mbox": mbox, "check": check}
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
        return "neg", None, "check"
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
    out = {cam: {"boxed": 0, "pos_you": 0, "neg_you": 0, "skipped": 0, "pos_check": 0, "neg_check": 0,
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


def split_of(x):
    key = x["file"] if x["snap"] else x["cam"] + local(x["ts"]).strftime("%Y-%m-%d %H")
    return "val" if int(hashlib.md5(key.encode()).hexdigest(), 16) % 5 == 0 else "train"


def dataset(frames, revs, cam):
    """Training items [(frame, box or None, split)] and their counts.

    Candidates: your answers, and the safety-check frames (see the module).
    Diverse first: frames with him, at most one per hour for each part of the
    picture he's in (your answers first), and none where he hasn't moved
    within 2 hours; frames without him, an empty scene once per 20 minutes and
    a look-alike (something taken for him) once per hour for each spot.
    Then whole days the model never trains on are held out for testing (the
    days with the fewest frames with him, until at least 10 %, never over
    40 %; with only one day so far, hours are held out instead), and no
    training day may give more frames with him than the next-largest day or a
    third of them, whichever is more."""
    pos, neg = [], []
    for x in frames.values():
        if x["cam"] != cam:
            continue
        lab, box, src = final(x, revs.get(x["file"]))
        if lab == "pos" and box:
            W, H = x["size"]
            box = [max(0, min(W, box[0])), max(0, min(H, box[1])), max(0, min(W, box[2])), max(0, min(H, box[3]))]
            pos.append((x, box, src))
        elif lab == "neg":
            neg.append((x, None, src))
    n_pos_all, n_neg_all = len(pos), len(neg)
    # diverse: yours first, then by time
    order = lambda t: (t[2] != "you", t[0]["ts"])
    kept, keys, recent = [], set(), []
    for x, b, src in sorted(pos, key=order):
        fx, fy = foot(b, x["size"])
        k = (x["file"] if x["snap"] else hour_key(x), min(GRID[0] - 1, int(fx * GRID[0])), min(GRID[1] - 1, int(fy * GRID[1])))
        if k in keys:
            continue
        if not x["snap"] and any(abs(x["ts"] - t) < SAME_SECS and iou(b, bb) >= SAME_IOU for t, bb in recent):
            continue
        keys.add(k); recent.append((x["ts"], b)); kept.append((x, b, src))
    # without him: an empty scene once per 20 minutes; a look-alike (something
    # the first pass or the detector took for him) once per hour for each spot
    negs, nkeys = [], set()
    for x, b, src in sorted(neg, key=order):
        if x["snap"]:
            k = x["file"]
        else:
            lk = max(([x["s"], x["obox"]], [x["m"] or 0, x["mbox"]]), key=lambda t: t[0])
            if lk[0] >= NEG and lk[1]:
                fx, fy = foot(lk[1], x["size"])
                k = ("look-alike", hour_key(x), min(GRID[0] - 1, int(fx * GRID[0])), min(GRID[1] - 1, int(fy * GRID[1])))
            else:
                t = local(x["ts"]); k = ("empty", t.strftime("%Y-%m-%d %H"), t.minute // 20)
        if k in nkeys:
            continue
        nkeys.add(k); negs.append((x, None, src))
    # test on whole days it never trains on; cap any one training day
    day = lambda x: local(x["ts"]).strftime("%Y-%m-%d")
    by_day = {}
    for t in kept:
        by_day.setdefault(day(t[0]), []).append(t)
    val_days, n_val = set(), 0
    for d in sorted(by_day, key=lambda d: (len(by_day[d]), d)):
        if n_val >= 0.10 * len(kept) or n_val + len(by_day[d]) > 0.40 * len(kept):
            break
        val_days.add(d); n_val += len(by_day[d])
    capped = {}
    if val_days:
        split_by = "days"
        side = lambda x: "val" if day(x) in val_days else "train"
        train_days = {d: v for d, v in by_day.items() if d not in val_days}
        sizes = sorted((len(v) for v in train_days.values()), reverse=True)
        if len(sizes) >= 2:
            cap = max(sizes[1], sum(min(s, sizes[1]) for s in sizes) // 3)
            for d, v in train_days.items():
                if len(v) > cap:
                    v.sort(key=lambda t: t[0]["ts"]); step = len(v) / cap
                    train_days[d] = [v[int(i * step)] for i in range(cap)]
                    capped[d] = [len(v), cap]
        pos_items = [(t[0], t[1], "train", t[2]) for v in train_days.values() for t in v] + \
                    [(t[0], t[1], "val", t[2]) for d in val_days for t in by_day[d]]
    else:
        split_by = "hours"
        side = split_of
        pos_items = [(x, b, split_of(x), src) for x, b, src in kept]
    items = [(x, b, s) for x, b, s, _ in pos_items] + [(x, None, side(x)) for x, _, _ in negs]
    c = {"pos": len(pos_items), "pos_all": n_pos_all, "pos_you": sum(1 for p in pos_items if p[3] == "you"),
         "neg": len(negs), "neg_all": n_neg_all, "neg_you": sum(1 for n in negs if n[2] == "you"),
         "left_out": sum(1 for x in frames.values() if x["cam"] == cam and x["label"] == "ask" and x["file"] not in revs),
         "split": split_by, "val_days": sorted(val_days), "capped": capped,
         "train_days": len({day(x) for x, b, s, _ in pos_items if s == "train"}),
         "train": sum(1 for i in items if i[2] == "train"), "val": sum(1 for i in items if i[2] == "val"),
         "val_pos": sum(1 for i in items if i[2] == "val" and i[1]),
         "reviews": sum(1 for f in revs if f in frames and frames[f]["cam"] == cam)}
    return items, c


def default_config(cam):
    imgsz = int(config.cameras().get(cam, {}).get("imgsz", 960))
    return {"model": "yolo11n.pt", "imgsz": imgsz, "epochs": 40, "batch": 16, "patience": 10, "device": "cpu"}


def protected_files(c):
    """Frames that must never be deleted: everything you answered, and every
    frame a training run used."""
    keep = {r["file"] for r in c.execute("SELECT file FROM reviews")}
    for r in c.execute("SELECT items FROM runs WHERE items IS NOT NULL"):
        keep.update(json.loads(r["items"]))
    return keep
