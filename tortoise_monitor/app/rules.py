"""Which frames are clear enough to use without review, which are asked, what
a training run uses, and what a detection means against the pen outline.

A frame's first-pass score (OWLv2, on a training server), the detector in use,
the frames around it and Home Assistant's in/out record are compared: when
they agree the frame is used as it is, otherwise it goes to Review."""
import bisect, hashlib, json, threading, time
from datetime import datetime

from . import config, db

POS, NEG, CAND = 0.40, 0.15, 0.10           # first-pass score: clearly him / clearly not / worth drawing
MODEL_YES, MODEL_NO, MODEL_SHOW = 0.50, 0.25, 0.25   # detector in use: found / missed / worth drawing
NEAR = 300          # seconds either side that count as "nearby frames"
SPARSE = 300        # uncertain frames with nothing to point at: one per 5 minutes is asked
MATCH_IOU = 0.6     # a candidate this close to a box you rejected is that same object
SAME_IOU, SAME_SECS = 0.8, 1200   # a box you confirmed covers frames of him in the same spot within 20 min
THIN_SECS = 300     # training: while he sits still, one frame per 5 min


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
    """Frames worth collecting and labelling: all except yard frames at night
    (19:00-06:00), of which one per 5 minutes."""
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
    """Where he is, from a detection: 'house' (house camera), 'pen' / 'outside'
    (yard, against the pen outline), or 'yard' (yard, no outline yet)."""
    if role(cam) == "house":
        return "house"
    if not poly:
        return "yard"
    return "pen" if inside(foot(box, size), poly) else "outside"


PLACE_TEXT = {"house": "in his house", "pen": "inside the pen walls", "outside": "outside the pen walls", "yard": "in the yard"}


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
                " (SELECT group_concat(value) FROM kv WHERE key LIKE 'model:%' OR key = 'record')"))
            if sig != self.sig:
                self.frames, self.index = build(c)
                self.sig = sig


def record_intervals(c):
    return db.kv_get("record", {"covered": [0, 0], "out": []}, c)


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
    rec = record_intervals(c)
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
        out = None
        if not snap and rec["covered"][0] <= r["ts"] <= rec["covered"][1]:
            out = any(a <= r["ts"] < e for a, e in rec["out"])
        frames[f] = {"file": f, "cam": r["camera"], "ts": r["ts"], "snap": snap, "size": [r["w"], r["h"]],
                     "cands": cands[:4], "s": s, "obox": obox, "m": m, "mbox": mbox, "out": out, "n": 0.0, "hint": None}
    for cam in {x["cam"] for x in frames.values()}:
        seq = sorted((x for x in frames.values() if x["cam"] == cam and not x["snap"]), key=lambda x: x["ts"])
        tss = [x["ts"] for x in seq]
        for i, x in enumerate(seq):
            lo = bisect.bisect_left(tss, x["ts"] - NEAR); hi = bisect.bisect_right(tss, x["ts"] + NEAR)
            best = None
            for j in range(lo, hi):
                if j != i and (best is None or seq[j]["s"] > best["s"]):
                    best = seq[j]
            if best:
                x["n"] = best["s"]
                x["hint"] = best["obox"] if best["s"] >= NEG else None
    for x in frames.values():
        x["status"], x["reason"] = classify(x)
    return frames, idx


def classify(x):
    """('pos'|'neg'|'uncertain', reason). Clear = this frame and the frames
    around it agree, and so do the in/out record and the detector in use
    where they exist."""
    s, m = x["s"], x["m"]
    if x["snap"]:
        if s < NEG and not (m is not None and m >= MODEL_YES):
            return "neg", ""
        return "uncertain", "mid" if s >= NEG else "model_yes"
    # does the record say he should be in view? yard: out of his house; house camera: in it
    yard = role(x["cam"]) == "yard"
    expect = None if x["out"] is None else (x["out"] if yard else not x["out"])
    if s >= POS:
        if expect is False:
            return "uncertain", "record_no"
        if x["n"] < POS:
            return "uncertain", "lonely"
        if m is not None and m < MODEL_NO:
            return "uncertain", "model_no"
        return "pos", ""
    if s < NEG:
        if expect is True:
            return "uncertain", "record_yes"
        if x["n"] >= NEG:
            return "uncertain", "nearby"
        if m is not None and m >= MODEL_YES:
            return "uncertain", "model_yes"
        return "neg", ""
    return "uncertain", "mid"


def why(x):
    yard = role(x["cam"]) == "yard"
    return {
        "mid": f"Possible match ({x['s']:.2f})",
        "lonely": f"Found ({x['s']:.2f}), but not in the frames around it",
        "record_no": f"Found ({x['s']:.2f}), but the in/out record says he was " + ("in his house" if yard else "out"),
        "record_yes": "Not found, but the in/out record says he was " + ("out" if yard else "in his house"),
        "nearby": f"Not found here; possible match in nearby frames ({x['n']:.2f})",
        "model_no": f"First pass found him ({x['s']:.2f}); the detector in use did not",
        "model_yes": f"The detector in use found him ({(x['m'] or 0):.2f}); the first pass did not",
    }.get(x["reason"], "")


def reviews(c):
    return {r["file"]: dict(r) for r in c.execute("SELECT * FROM reviews")}


def context(revs, frames):
    rejected, conf = {}, {}
    for f, r in revs.items():
        x = frames.get(f)
        if not x:
            continue
        if r["rejected"]:
            rejected.setdefault(x["cam"], []).extend(json.loads(r["rejected"]))
        if r["verdict"] == "tortoise" and r["box"]:
            conf.setdefault(x["cam"], []).append((x["ts"], json.loads(r["box"])))
    for v in conf.values():
        v.sort(key=lambda t: t[0])
    return {"rejected": rejected, "confirmed": conf, "confirmed_ts": {k: [t[0] for t in v] for k, v in conf.items()}}


def final(x, rev, ctx):
    """(label, box, source): label pos/neg/uncertain/skip; source you/auto/match/same."""
    if rev:
        if rev["verdict"] == "tortoise":
            return "pos", json.loads(rev["box"]), "you"
        if rev["verdict"] == "none":
            return "neg", None, "you"
        return "skip", None, "you"
    rj = ctx["rejected"].get(x["cam"], [])

    def matched(b):
        return any(iou(b, r) >= MATCH_IOU for r in rj)
    shown = [c for c in x["cands"] if c[0] >= NEG]
    if x["status"] != "neg" and shown and all(matched(c[1]) for c in shown):
        return "neg", None, "match"         # only things you already said aren't him
    if x["status"] == "uncertain" and x["reason"] == "nearby" and x["hint"] and matched(x["hint"]):
        return "neg", None, "match"         # the "nearby" find was one of those things
    if x["status"] == "uncertain" and x["reason"] in ("mid", "lonely") and not x["snap"]:
        cf = ctx["confirmed"].get(x["cam"], []); cts = ctx["confirmed_ts"].get(x["cam"], [])
        for ts, b in cf[bisect.bisect_left(cts, x["ts"] - SAME_SECS):bisect.bisect_right(cts, x["ts"] + SAME_SECS)]:
            for cd in x["cands"]:
                if iou(cd[1], b) >= SAME_IOU:
                    return "pos", cd[1], "same"     # him in the same spot as in a frame you confirmed
    if x["status"] == "pos":
        return "pos", x["obox"], "auto"
    if x["status"] == "neg":
        return "neg", None, "auto"
    return "uncertain", None, None


def pointable(x):
    return x["s"] >= NEG or (x["m"] is not None and x["m"] >= MODEL_SHOW)


def asked(frames):
    """Uncertain frames that get asked: all with something to point at; of the
    rest, one per SPARSE seconds per camera (chosen before any review, so
    answering one doesn't pull its neighbour in)."""
    keep = set(); last = {}
    for x in sorted(frames.values(), key=lambda x: (x["cam"], x["ts"])):
        if x["status"] != "uncertain":
            continue
        if pointable(x):
            keep.add(x["file"])
        elif x["ts"] - last.get(x["cam"], -1e18) >= SPARSE:
            keep.add(x["file"]); last[x["cam"]] = x["ts"]
    return keep


def queue(frames, revs, cam=None):
    ctx = context(revs, frames); keep = asked(frames); items = []
    for f in keep:
        x = frames[f]
        if cam and x["cam"] != cam:
            continue
        if final(x, revs.get(f), ctx)[0] == "uncertain":
            items.append(x)
    items.sort(key=lambda x: (not pointable(x), x["ts"]))
    return items


def counts(frames, revs):
    ctx = context(revs, frames); keep = asked(frames)
    out = {cam: {"boxed": 0, "pos_you": 0, "pos_auto": 0, "pos_same": 0, "neg_you": 0, "neg_auto": 0, "neg_match": 0,
                 "to_review": 0, "left_out": 0, "skipped": 0} for cam in config.cameras()}
    for x in frames.values():
        k = out.setdefault(x["cam"], dict.fromkeys(next(iter(out.values())), 0)); k["boxed"] += 1
        lab, _, src = final(x, revs.get(x["file"]), ctx)
        if lab == "uncertain":
            k["to_review" if x["file"] in keep else "left_out"] += 1
        elif lab == "skip":
            k["skipped"] += 1
        else:
            k[f"{lab}_{src}"] += 1
    return out


def split_of(x):
    key = x["file"] if x["snap"] else x["cam"] + local(x["ts"]).strftime("%Y-%m-%d %H")
    return "val" if int(hashlib.md5(key.encode()).hexdigest(), 16) % 5 == 0 else "train"


def dataset(frames, revs, cam):
    """Training items [(frame, box or None, split)] and their counts. Every
    frame with him (thinned while he sits still); frames without him: all you
    rejected, plus automatic ones spread evenly over time up to as many as
    there are frames with him (min 200)."""
    ctx = context(revs, frames)
    pos, neg_you, neg_auto = [], [], []
    left = 0
    for x in frames.values():
        if x["cam"] != cam:
            continue
        lab, box, src = final(x, revs.get(x["file"]), ctx)
        if lab == "pos":
            W, H = x["size"]
            box = [max(0, min(W, box[0])), max(0, min(H, box[1])), max(0, min(W, box[2])), max(0, min(H, box[3]))]
            pos.append((x, box, src))
        elif lab == "neg":
            (neg_auto if src == "auto" else neg_you).append(x)
        elif lab == "uncertain":
            left += 1
    # While he sits still the frames are near copies: group them into sessions
    # (next frame within 10 min, box overlapping), keep one frame per THIN_SECS
    # unless the box moved, and keep each session on one side of the split.
    pos.sort(key=lambda p: p[0]["ts"])
    kept = []; prev = last = None; key = None
    for x, b, src in pos:
        if x["snap"]:
            kept.append((x, b, src, x["file"])); continue
        if prev is None or x["ts"] - prev[0] > 600 or iou(b, prev[1]) < 0.5:
            key = f'{cam}-{x["ts"]:.0f}'; last = None
        if last is None or x["ts"] - last[0] >= THIN_SECS or iou(b, last[1]) < SAME_IOU:
            kept.append((x, b, src, key)); last = (x["ts"], b)
        prev = (x["ts"], b)
    h = lambda k: int(hashlib.md5(k.encode()).hexdigest(), 16)
    sizes = {}
    for k in (t[3] for t in kept):
        sizes[k] = sizes.get(k, 0) + 1
    val_keys = {k for k in sizes if h(k) % 5 == 0}
    for k in sorted(sizes, key=h):            # about 15% of the frames with him held out
        if sum(sizes[v] for v in val_keys) >= 0.15 * len(kept):
            break
        val_keys.add(k)
    want = max(len(kept), 200) - len(neg_you)
    neg_auto.sort(key=lambda x: x["ts"])
    if want <= 0:
        picked = []
    elif want >= len(neg_auto):
        picked = neg_auto
    else:
        step = len(neg_auto) / want
        picked = [neg_auto[int(i * step)] for i in range(want)]
    items = [(x, b, "val" if k in val_keys else "train") for x, b, _, k in kept] + [(x, None, split_of(x)) for x in neg_you + picked]
    c = {"pos": len(kept), "pos_all": len(pos), "pos_you": sum(1 for p in kept if p[2] == "you"), "neg": len(neg_you) + len(picked),
         "neg_you": sum(1 for x in neg_you if revs.get(x["file"])), "left_out": left,
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
