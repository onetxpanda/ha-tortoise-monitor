"""The add-on's page, through Home Assistant's ingress (Home Assistant does
the login). All paths are relative: ingress serves the page under a prefix."""
import io, json, os, re, threading, time, traceback
from datetime import datetime, timedelta
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

from PIL import Image

from . import collect, config, db, passes, retention, rules, servers
from .logbuf import logs


def when(ts):
    return rules.local(ts).strftime("%a %d %b %H:%M")


def item_json(x, revs):
    r = revs.get(x["file"])
    return {"file": x["file"], "cam": x["cam"], "camName": cam_name(x["cam"]), "ts": x["ts"], "when": when(x["ts"]),
            "size": x["size"], "cands": x["cands"], "hint": x.get("hint"), "why": rules.why(x),
            "review": {"verdict": r["verdict"], "box": json.loads(r["box"]) if r["box"] else None} if r else None}


def cam_name(cam):
    return {"yard": "Yard", "house": "House"}.get(rules.role(cam), cam)


def hull(pts):
    pts = sorted(set(pts))
    if len(pts) < 3:
        return pts
    cross = lambda o, a, b: (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, up = [], []
    for p in pts:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0: lo.pop()
        lo.append(p)
    for p in reversed(pts):
        while len(up) >= 2 and cross(up[-2], up[-1], p) <= 0: up.pop()
        up.append(p)
    return lo[:-1] + up[:-1]


class Site:
    def __init__(self, F, detection, sched, log=print):
        self.F, self.detection, self.sched, self.log = F, detection, sched, log
        site = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def send(self, code, body, ctype="application/json", cache=False):
                if isinstance(body, (dict, list)):
                    body = json.dumps(body).encode()
                elif isinstance(body, str):
                    body = body.encode()
                self.send_response(code)
                self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "max-age=86400" if cache else "no-store")
                self.end_headers(); self.wfile.write(body)

            def do_GET(self):
                try:
                    site.get(self)
                except (ValueError, KeyError) as e:
                    self.send(400, {"error": str(e)})
                except (BrokenPipeError, ConnectionResetError):
                    pass
                except (BrokenPipeError, ConnectionResetError):
                    pass
                except Exception as e:
                    traceback.print_exc(); self.send(500, {"error": str(e)})

            def do_POST(self):
                try:
                    n = int(self.headers.get("Content-Length") or 0)
                    site.post(self, urlparse(self.path).path, json.loads(self.rfile.read(n) or b"{}"))
                except (ValueError, KeyError) as e:
                    self.send(400, {"error": str(e)})
                except (BrokenPipeError, ConnectionResetError):
                    pass
                except (BrokenPipeError, ConnectionResetError):
                    pass
                except Exception as e:
                    traceback.print_exc(); self.send(500, {"error": str(e)})

        self.httpd = ThreadingHTTPServer(("0.0.0.0", int(os.environ.get("TM_WEB_PORT", "8099"))), H)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    # ---------------- helpers ----------------
    def frame_path(self, f):
        r = db.connect().execute("SELECT file FROM frames WHERE file=? AND deleted IS NULL", (f,)).fetchone()
        if not r:
            raise ValueError("unknown frame")
        return os.path.join(config.FRAMES, r["file"])

    def review_state(self, cam, first=None):
        F = self.F; F.refresh()
        c = db.connect(); revs = rules.reviews(c)
        q = rules.queue(F.frames, revs, cam or None)
        item = F.frames.get(first) if first else (q[0] if q else None)
        after = None
        if item is not None:
            rest = [x for x in q if x["file"] != item["file"]]
            after = rest[0]["file"] if rest else None
        return {"counts": rules.counts(F.frames, revs), "remaining": len(q), "item": item_json(item, revs) if item else None, "after": after}

    def outline_state(self, cam):
        if not rules.outlined(cam):
            raise ValueError("only the yard camera has an outline")
        F = self.F; F.refresh(); c = db.connect()
        o = db.kv_get(f"outline:{cam}", None, c)
        seen, frames = set(), []
        for r in c.execute("SELECT file, ts, w, h FROM frames WHERE camera=? AND source='continuous' AND deleted IS NULL ORDER BY ts DESC LIMIT 4000", (cam,)):
            t = rules.local(r["ts"]); key = t.strftime("%Y-%m-%d %H")
            if 8 <= t.hour < 17 and key not in seen:
                seen.add(key); frames.append({"file": r["file"], "when": when(r["ts"]), "size": [r["w"], r["h"]]})
            if len(frames) >= 36:
                break
        revs = rules.reviews(c); ctx = rules.context(revs, F.frames)
        feet = [tuple(round(v, 4) for v in rules.foot(b, x["size"])) for x in F.frames.values()
                if x["cam"] == cam and not x["snap"] for lab, b, _ in [rules.final(x, revs.get(x["file"]), ctx)] if lab == "pos"]
        return {"cam": cam, "camName": cam_name(cam), "outline": o, "frames": frames, "size": frames[0]["size"] if frames else None,
                "suggestion": [list(p) for p in hull(feet)] if len(feet) >= 3 else None, "feet": len(feet),
                "feet_pts": feet[::max(1, len(feet) // 1500)]}

    def heatmap_state(self, cam, day=None):
        """Where he was on one day: every sighting's foot point (fractions of the
        frame) with its time, over a frame of that day. The days with sightings
        come from the hourly cycle counts (as many as there are detection rows)."""
        if cam not in config.cameras():
            raise ValueError("unknown camera")
        c = db.connect()
        days = {}
        for r in c.execute("SELECT hour, seen FROM cycles WHERE camera=? AND seen > 0 AND hour >= ? ORDER BY hour DESC",
                           (cam, time.time() - 91 * 86400)):
            d = rules.local(r["hour"]).strftime("%Y-%m-%d")
            days[d] = days.get(d, 0) + r["seen"]
        day_list = [{"day": d, "n": n} for d, n in sorted(days.items(), reverse=True)]
        if day is not None and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", day):
            raise ValueError("bad day")
        day = day or (day_list[0]["day"] if day_list else rules.local(time.time()).strftime("%Y-%m-%d"))
        d0 = datetime.strptime(day, "%Y-%m-%d")
        start, end = d0.timestamp(), (d0 + timedelta(days=1)).timestamp()     # local midnights
        rows = c.execute("SELECT ts, fx, fy, place FROM detections WHERE camera=? AND ts >= ? AND ts < ? ORDER BY ts",
                         (cam, start, end)).fetchall()
        every = float(config.options().get("detect_every_s", 5))
        seen_s, places = 0.0, {}
        for i, r in enumerate(rows):   # time in view: the gap to the next sighting, at most two cycles
            gap = rows[i + 1]["ts"] - r["ts"] if i + 1 < len(rows) else every
            seen_s += min(max(gap, 0.0), 2 * every)
            places[r["place"] or "seen"] = places.get(r["place"] or "seen", 0) + 1
        # The background: that day's frame nearest 1 pm; else the newest daytime
        # frame (8 am to 5 pm), else the newest there is.
        noon = start + 13 * 3600
        fr = c.execute("SELECT file, ts, w, h FROM frames WHERE camera=? AND source='continuous' AND deleted IS NULL AND ts >= ? AND ts < ? "
                       "AND w IS NOT NULL ORDER BY ABS(ts - ?) LIMIT 1", (cam, start, end, noon)).fetchone()
        if not fr:
            newest = c.execute("SELECT file, ts, w, h FROM frames WHERE camera=? AND source='continuous' AND deleted IS NULL AND w IS NOT NULL "
                               "ORDER BY ts DESC LIMIT 2000", (cam,)).fetchall()
            fr = next((r for r in newest if 8 <= rules.local(r["ts"]).hour < 17), newest[0] if newest else None)
        if fr:
            size = [fr["w"], fr["h"]]
        else:      # no frame kept: the size the detector saw
            d = c.execute("SELECT w, h FROM detections WHERE camera=? AND w IS NOT NULL ORDER BY ts DESC LIMIT 1", (cam,)).fetchone()
            size = [d["w"], d["h"]] if d else None
        return {"cam": cam, "camName": cam_name(cam), "role": rules.role(cam), "days": day_list, "day": day,
                "points": [[round(r["fx"], 4), round(r["fy"], 4), round(r["ts"] - start), r["place"]] for r in rows],
                "every": every, "seen_s": round(seen_s), "places": places,
                "first": rows[0]["ts"] if rows else None, "last": rows[-1]["ts"] if rows else None,
                "frame": {"file": fr["file"], "when": when(fr["ts"]), "same_day": start <= fr["ts"] < end} if fr else None,
                "size": size, "outline": rules.outline(cam, c) if rules.outlined(cam) else None}

    def run_json(self, c, r, revs):
        d = dict(r); d.pop("items", None)
        for k in ("config", "counts", "result", "progress"):
            d[k] = json.loads(d[k]) if d[k] else None
        d["camName"] = cam_name(d["camera"])
        if d["status"] == "proposed":
            d["counts"] = rules.dataset(self.F.frames, revs, d["camera"])[1]
        if d["status"] in ("proposed", "approved"):
            d["estimate"] = servers.estimate(c, d, d["config"], d["counts"])
        m = db.kv_get(f"model:{d['camera']}", None, c)
        d["in_use"] = bool(m and m["run"] == d["id"])
        srv = c.execute("SELECT name FROM servers WHERE id=?", (d["server"],)).fetchone() if d.get("server") else None
        d["server_name"] = srv["name"] if srv else None
        return d

    def runs_state(self):
        self.F.refresh(); c = db.connect(); revs = rules.reviews(c)
        ss = servers.servers(c)
        return {"runs": [self.run_json(c, r, revs) for r in c.execute("SELECT * FROM runs ORDER BY id DESC LIMIT 50")],
                "servers": [{"name": s["name"], "online": s["online"], "gpus": len((s["info"] or {}).get("gpus") or [])} for s in ss]}

    def results_state(self, cam):
        c = db.connect(); now = time.time()
        first = now - 86400; first -= first % 3600
        hours = []
        rows = {r["hour"]: r for r in c.execute("SELECT * FROM cycles WHERE camera=? AND hour>=?", (cam, first))}
        for i in range(25):
            h = first + i * 3600; r = rows.get(h)
            hours.append({"start": h, "label": rules.local(h).strftime("%H"), "n": r["n"] if r else 0,
                          "seen": r["seen"] if r else 0, "outside": r["outside"] if r else 0})
        outs = []
        d = os.path.join(config.LIVE, "outside")
        if os.path.isdir(d):
            for name in sorted(os.listdir(d), reverse=True)[:60]:
                if name.startswith(cam + "-"):
                    ts = float(name[len(cam) + 1:-4]); outs.append({"name": name, "when": when(ts)})
        return {"cam": cam, "camName": cam_name(cam), "role": rules.role(cam), "model": db.kv_get(f"model:{cam}", None, c),
                "outline": bool(rules.outline(cam, c)), "hours": hours, "outside": outs,
                "has_snapshot": os.path.exists(os.path.join(config.LIVE, f"{cam}.jpg"))}

    def overview(self):
        self.F.refresh(); c = db.connect(); revs = rules.reviews(c)
        k = rules.counts(self.F.frames, revs)
        cams = {}
        targets = servers.label_targets(c)
        for cam in config.cameras():
            n = c.execute("SELECT COUNT(*), COALESCE(SUM(ts > ?), 0), MAX(CASE WHEN source='continuous' THEN ts END) FROM frames WHERE camera=? AND deleted IS NULL",
                          (time.time() - 86400, cam)).fetchone()
            waiting = sum(1 for r in targets if r["camera"] == cam)
            cams[cam] = {"name": cam_name(cam), "role": rules.role(cam), "collected": n[0], "collected_24h": n[1], "newest": n[2],
                         "waiting_first_pass": waiting, "counts": k.get(cam, {}), "outline": db.kv_get(f"outline:{cam}", None, c) if rules.outlined(cam) else None,
                         "model": db.kv_get(f"model:{cam}", None, c)}
        runs = [dict(r) for r in c.execute("SELECT id, camera, status, config, progress FROM runs WHERE status IN ('proposed','approved','running') ORDER BY id DESC")]
        for r in runs:
            cfg = json.loads(r.pop("config") or "{}"); r["device"] = cfg.get("device", "cpu")
            r["progress"] = json.loads(r["progress"]) if r["progress"] else None
        ss = servers.servers(c)
        label = c.execute("SELECT status, progress, server FROM jobs WHERE kind='label' ORDER BY id DESC LIMIT 1").fetchone()
        return {"version": config.version(), "cams": cams,
                "runs": {"proposed": [r for r in runs if r["status"] == "proposed"], "approved": [r for r in runs if r["status"] == "approved"],
                         "running": [r for r in runs if r["status"] == "running"]},
                "servers": [{"id": s["id"], "name": s["name"], "online": s["online"], "version": (s["info"] or {}).get("version"),
                             "detect": s["settings"].get("detect", "cpu")} for s in ss],
                "jobs": self.sched.status(label), "detection": self.detection.status(), "storage": retention.state}

    # ---------------- GET ----------------
    def get(self, h):
        u = urlparse(h.path); p = u.path; q = {k: v[0] for k, v in parse_qs(u.query).items()}
        if p in ("/", "/index.html"):
            with open(os.path.join(config.WEB, "index.html"), "rb") as f:
                return h.send(200, f.read(), "text/html; charset=utf-8")
        if p == "/app.js":     # the React UI, built from web/src at image build time
            with open(os.path.join(config.WEB, "app.js"), "rb") as f:
                return h.send(200, f.read(), "text/javascript; charset=utf-8")
        if p == "/img":
            im = Image.open(self.frame_path(q["f"])); w = int(q.get("w", 1600))
            if im.width > w:
                im.draft("RGB", (w, w * im.height // im.width))
                im = im.convert("RGB").resize((w, round(w * im.height / im.width)), Image.LANCZOS)
            buf = io.BytesIO(); im.convert("RGB").save(buf, "JPEG", quality=85)
            return h.send(200, buf.getvalue(), "image/jpeg", cache=True)
        if p == "/crop":
            im = Image.open(self.frame_path(q["f"])).convert("RGB")
            x, y, s, w = (int(float(q[k])) for k in ("x", "y", "s", "w"))
            im = im.crop((x, y, x + s, y + s)).resize((w, w), Image.LANCZOS)
            buf = io.BytesIO(); im.save(buf, "JPEG", quality=88)
            return h.send(200, buf.getvalue(), "image/jpeg", cache=True)
        if p == "/live":
            name = q.get("name", "last.jpg")
            if not re.fullmatch(r"(outside/)?[\w.\-]+\.jpg", name):
                raise ValueError("bad name")
            path = os.path.join(config.LIVE, name)
            if not os.path.exists(path):
                return h.send(404, {"error": "no picture yet"})
            with open(path, "rb") as f:
                return h.send(200, f.read(), "image/jpeg")
        routes = {
            "/api/overview": lambda: self.overview(),
            "/api/review": lambda: self.review_state(q.get("cam"), q.get("first")),
            "/api/outline": lambda: self.outline_state(q["cam"]),
            "/api/runs": lambda: self.runs_state(),
            "/api/results": lambda: self.results_state(q["cam"]),
            "/api/heatmap": lambda: self.heatmap_state(q["cam"], q.get("day")),
            "/api/live": lambda: self.detection.status(),
            "/api/servers": lambda: {"servers": servers.servers(), "install": servers.install_command(),
                                     "pair_expires": servers.pair_code()["expires"], "main": (servers.main_detector() or {}).get("id"),
                                     "version": config.version()},
            "/api/storage": lambda: self.storage(),
        }
        if p in routes:
            return h.send(200, routes[p]())
        m = re.fullmatch(r"/api/runs/(\d+)/log", p)
        if m:
            r = db.connect().execute("SELECT progress, result FROM jobs WHERE run=? ORDER BY id DESC LIMIT 1", (int(m.group(1)),)).fetchone()
            pr = json.loads(r["progress"]) if r and r["progress"] else {}
            res = json.loads(r["result"]) if r and r["result"] else {}
            return h.send(200, {"log": res.get("log") or pr.get("log") or ""})
        m = re.fullmatch(r"/api/jobs/(collect|label|pass)/log", p)
        if m:
            return h.send(200, {"log": "\n".join(logs(m.group(1)))})
        return h.send(404, {"error": "not found"})

    def storage(self):
        c = db.connect()
        dels = [dict(r) for r in c.execute("SELECT * FROM deletions ORDER BY at DESC LIMIT 30")]
        return {"state": retention.state, "limit": config.options().get("prune_at_percent", 95), "deletions": dels,
                "frames": c.execute("SELECT COUNT(*) FROM frames WHERE deleted IS NULL").fetchone()[0],
                "protected": len(rules.protected_files(c))}

    # ---------------- POST ----------------
    def post(self, h, p, body):
        F = self.F
        if p == "/api/review":
            F.refresh()
            x = F.frames.get(body.get("file"))
            if not x:
                raise ValueError("unknown frame")
            verdict = body.get("verdict"); shown = [cd[1] for cd in x["cands"] if cd[0] >= rules.NEG]; box = None
            if verdict == "tortoise":
                b = [float(v) for v in body["box"]]; W, H = x["size"]
                b = [max(0, min(W, min(b[0], b[2]))), max(0, min(H, min(b[1], b[3]))), max(0, min(W, max(b[0], b[2]))), max(0, min(H, max(b[1], b[3])))]
                if b[2] - b[0] < 4 or b[3] - b[1] < 4:
                    raise ValueError("box too small")
                box = [round(v) for v in b]; rejected = [s for s in shown if rules.iou(s, box) < 0.3]
            elif verdict == "none":
                rejected = shown
            elif verdict == "skip":
                rejected = []
            else:
                raise ValueError("bad verdict")
            c = db.connect()
            c.execute("INSERT OR REPLACE INTO reviews VALUES (?,?,?,?,?)", (x["file"], verdict, json.dumps(box) if box else None, json.dumps(rejected), time.time()))
            c.commit()
            return h.send(200, self.review_state(body.get("cam")))
        if p == "/api/undo":
            c = db.connect(); cam = body.get("cam")
            for r in c.execute("SELECT file FROM reviews ORDER BY at DESC").fetchall():
                x = F.frames.get(r["file"])
                if x and (not cam or x["cam"] == cam):
                    c.execute("DELETE FROM reviews WHERE file=?", (r["file"],)); c.commit()
                    return h.send(200, self.review_state(cam, first=r["file"]))
            return h.send(200, self.review_state(cam))
        if p == "/api/outline":
            cam = body["cam"]
            if not rules.outlined(cam):
                raise ValueError("only the yard camera has an outline")
            poly = [[round(min(1, max(0, float(a))), 5), round(min(1, max(0, float(b))), 5)] for a, b in body.get("poly") or []]
            if poly and len(poly) < 3:
                raise ValueError("an outline needs at least 3 points")
            c = db.connect()
            db.kv_set(f"outline:{cam}", {"poly": poly, "saved": time.time(), "frame": body.get("frame")} if poly else None, c)
            hist = db.kv_get("outline_history", [], c); hist.append({"cam": cam, "poly": poly, "at": time.time()})
            db.kv_set("outline_history", hist[-50:], c); c.commit()
            return h.send(200, self.outline_state(cam))
        if p == "/api/runs/propose":
            servers.propose(F, body.get("cam")); return h.send(200, self.runs_state())
        m = re.fullmatch(r"/api/runs/(\d+)/(approve|decline|stop|use|unuse)", p)
        if m:
            servers.run_action(F, int(m.group(1)), m.group(2), body); return h.send(200, self.runs_state())
        if p == "/api/servers/pair":
            servers.pair_code(new=True); return h.send(200, {"install": servers.install_command()})
        m = re.fullmatch(r"/api/servers/(\w+)/(settings|remove)", p)
        if m:
            c = db.connect(); sid = m.group(1)
            if m.group(2) == "remove":
                c.execute("DELETE FROM servers WHERE id=?", (sid,))
            else:
                r = c.execute("SELECT settings FROM servers WHERE id=?", (sid,)).fetchone()
                if not r:
                    raise ValueError("no such server")
                s = json.loads(r["settings"] or "{}")
                if "detect" in body:
                    if body["detect"] not in ("off", "cpu", "gpu"):
                        raise ValueError("detect must be off, cpu or gpu")
                    s["detect"] = body["detect"]
                if "auto_update" in body:
                    s["auto_update"] = bool(body["auto_update"])
                c.execute("UPDATE servers SET settings=? WHERE id=?", (json.dumps(s), sid))
                if body.get("name"):
                    c.execute("UPDATE servers SET name=? WHERE id=?", (str(body["name"])[:60], sid))
            c.commit()
            return h.send(200, {"ok": True})
        m = re.fullmatch(r"/api/jobs/(collect|label|pass)/(run|on|off)", p)
        if m:
            self.sched.control(m.group(1), m.group(2)); return h.send(200, self.overview())
        return h.send(404, {"error": "not found"})
