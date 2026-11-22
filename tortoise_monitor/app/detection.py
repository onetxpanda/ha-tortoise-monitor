"""Live detection: where he is now.

The training server is the main detector: it reads Frigate's restream and
posts what it finds. While its results keep arriving, this add-on only
decides what they mean. When they stop (the server is off, busy or gone),
the add-on runs the same detector itself -- OpenVINO on this machine's Intel
GPU (VA-API decodes and scales the keyframes), or its CPU -- and hands back
as soon as the server reports again.

Each result: the best box over MODEL_YES means he is seen; where his feet
are against the pen outline gives the place. The add-on publishes the
entities, keeps a row per sighting (for the heatmap), counts cycles per hour
(for the Results page), and keeps a picture of the last sighting."""
import io, json, os, threading, time

import numpy as np

from . import config, db, rules
from . import paths_worker  # noqa: F401  (puts the shared detector module on the path)
import tmdetect as D

STALE_S = 60            # a camera with no result this long: its entities go to "unknown"
SNAP_EVERY_S = 30       # last-location picture: at most this often while he stays put


def server_fresh_s():
    return max(20, 4 * int(config.options().get("detect_every_s", 5)))


def model_files(run):
    d = os.path.join(config.MODELS, str(run))
    return {"dir": d, "xml": os.path.join(d, "model.xml"), "bin": os.path.join(d, "model.bin"), "pt": os.path.join(d, "best.pt")}


def frame_size(cam):
    c = db.connect()
    r = c.execute("SELECT w, h FROM frames WHERE camera=? AND source='continuous' AND w IS NOT NULL ORDER BY ts DESC LIMIT 1", (cam,)).fetchone()
    return (r["w"], r["h"]) if r else (2688, 1512) if rules.role(cam) == "yard" else (2560, 1440)


def snapshot_jpeg(rgb, cam, box_frame, size, poly):
    """The frame he was last seen in, with the pen outline and his box drawn."""
    from PIL import Image, ImageDraw
    im = Image.fromarray(rgb)
    W, H = im.size; sx, sy = W / size[0], H / size[1]
    d = ImageDraw.Draw(im)
    if poly:
        d.line([(p[0] * W, p[1] * H) for p in poly + poly[:1]], fill=(255, 79, 216), width=2)
    if box_frame:
        b = box_frame
        d.rectangle([b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy], outline=(255, 212, 0), width=3)
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=82)
    return buf.getvalue()


class LocalLoop(threading.Thread):
    """This add-on's own detection for one camera (the fallback)."""

    def __init__(self, mgr, cam, run):
        super().__init__(daemon=True)
        self.mgr, self.cam, self.run_id = mgr, cam, run
        self.stopping = False; self.reader = None; self.device = None; self.error = None

    def build(self, hwaccel):
        import openvino as ov
        cc = config.cameras()[self.cam]
        w, h = frame_size(self.cam)
        self.lb = D.Letterbox(int(cc.get("imgsz", 960)), w, h)
        self.size = (w, h)
        url = f"{config.options()['restream_url'].rstrip('/')}/{self.cam}"
        self.reader = D.KeyframeReader(url, self.lb.sw, self.lb.sh, hwaccel=hwaccel, log=self.mgr.log)
        self.reader.start()
        dev = "GPU" if "GPU" in ov.Core().available_devices else "CPU"
        self.det = D.OpenVinoDetector(model_files(self.run_id)["xml"], dev, self.reader.kind, cache_dir=config.OV_CACHE)
        self.device = f"this add-on, {'Intel GPU' if dev == 'GPU' else 'CPU'}" + (" (VA-API decode)" if hwaccel else "")

    def run(self):
        hw = "vaapi" if os.path.exists("/dev/dri/renderD128") else None
        try:
            self.build(hw)
        except Exception as e:
            self.error = str(e); self.mgr.log(f"local detection {self.cam}: {e}"); return
        started = time.time(); last = 0.0; every = float(config.options().get("detect_every_s", 5))
        while not self.stopping:
            f = self.reader.newest(after=last, timeout=20)
            if f is None:
                if hw and self.reader.frames == 0 and time.time() - started > 30:
                    self.mgr.log(f"local detection {self.cam}: no frames through VA-API; decoding on the CPU instead")
                    self.reader.stop(); hw = None
                    try:
                        self.build(None)
                    except Exception as e:
                        self.error = str(e); return
                continue
            ts, buf, bad = f; last = ts
            if bad:                                      # no answer from this frame
                self.mgr.handle(self.cam, ts, self.run_id, [], self.size, "local", self.device, None, unusable=bad)
                time.sleep(max(0.0, every - (time.time() - ts)))
                continue
            x = self.lb.pad_nv12(buf) if self.reader.kind == "nv12" else self.lb.pad_rgb(np.frombuffer(buf, np.uint8).reshape(self.lb.sh, self.lb.sw, 3))
            t = time.time(); dets = D.decode(self.det(x), self.lb); ms = (time.time() - t) * 1000
            self.mgr.handle(self.cam, ts, self.run_id, dets, self.size, "local", self.device, ms,
                            frame=(self.reader.kind, buf, self.lb.sw, self.lb.sh))
            time.sleep(max(0.0, every - (time.time() - ts)))
        self.reader.stop()

    def stop(self):
        self.stopping = True
        if self.reader:
            self.reader.stop()


class Manager:
    def __init__(self, mqtt, log=print):
        self.mqtt, self.log = mqtt, log
        self.lock = threading.Lock()
        self.cams = {}
        self.local = {}
        self.last_seen = db.kv_get("last_seen", None)
        threading.Thread(target=self._loop, daemon=True).start()

    def cam(self, cam):
        return self.cams.setdefault(cam, {"source": None, "device": None, "ts": None, "ms": None, "seen": None, "place": None,
                                          "score": None, "server_ts": 0, "snap_ts": 0, "snap_place": None, "run": None,
                                          "unusable": None, "unusable_logged": 0})

    def handle(self, cam, ts, run, dets, size, source, device, ms, frame=None, jpeg=None, unusable=None):
        """One detection cycle's result, from a server or from this add-on.
        `unusable` (why the frame can't be used) means no answer: nothing about
        him changes, and the camera's last answer ages towards stale."""
        with self.lock:
            st = self.cam(cam)
            if source != "local":
                st["server_ts"] = time.time()
            elif time.time() - st["server_ts"] < server_fresh_s():
                return                                   # a server is reporting: its results win
            if unusable:
                st["unusable"] = {"ts": ts, "reason": unusable, "source": source}
                log_it = ts - st["unusable_logged"] >= 300
                if log_it:
                    st["unusable_logged"] = ts
        if unusable:
            c = db.connect()
            c.execute("INSERT INTO cycles(camera, hour, n, seen, outside, bad) VALUES (?,?,0,0,0,1) "
                      "ON CONFLICT(camera, hour) DO UPDATE SET bad=bad+1", (cam, ts - ts % 3600))
            c.commit()
            if log_it:
                self.log(f"detection {cam}: frame not used ({unusable}); no answer")
            return
        with self.lock:
            top = dets[0] if dets and dets[0][0] >= rules.MODEL_YES else None
            poly = rules.outline(cam) if rules.outlined(cam) else None
            place = rules.place(cam, top[1], size, poly) if top else None
            st.update(source=source, device=device, ts=ts, ms=ms, seen=bool(top), place=place, score=top[0] if top else None, run=run)
        c = db.connect()
        hour = ts - ts % 3600
        c.execute("INSERT INTO cycles(camera, hour, n, seen, outside) VALUES (?,?,1,?,?) "
                  "ON CONFLICT(camera, hour) DO UPDATE SET n=n+1, seen=seen+excluded.seen, outside=outside+excluded.outside",
                  (cam, hour, 1 if top else 0, 1 if place == "outside" else 0))
        if top:
            fx, fy = rules.foot(top[1], size)
            c.execute("INSERT INTO detections(camera, ts, run, source, score, box, w, h, fx, fy, place) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                      (cam, ts, run, source, top[0], json.dumps(top[1]), size[0], size[1], fx, fy, place))
            self.last_seen = {"ts": ts, "camera": cam, "place": place, "score": top[0], "source": source}
            db.kv_set("last_seen", self.last_seen, c)
        c.commit()
        self.publish()
        if top and (place != st["snap_place"] or ts - st["snap_ts"] >= SNAP_EVERY_S):
            rgb = None
            try:
                if frame:
                    kind, buf, w, h = frame
                    rgb = D.nv12_to_rgb(buf, w, h) if kind == "nv12" else np.frombuffer(buf, np.uint8).reshape(h, w, 3)
                elif jpeg:
                    from PIL import Image
                    rgb = np.asarray(Image.open(io.BytesIO(jpeg)).convert("RGB"))
            except Exception as e:
                self.log(f"snapshot {cam}: {e}")
            if rgb is not None:
                jp = snapshot_jpeg(rgb, cam, top[1], size, poly)
                for name in (f"{cam}.jpg", "last.jpg"):
                    with open(os.path.join(config.LIVE, name + ".part"), "wb") as f:
                        f.write(jp)
                    os.replace(os.path.join(config.LIVE, name + ".part"), os.path.join(config.LIVE, name))
                self.mqtt.image("last_location", jp)
                st["snap_ts"], st["snap_place"] = ts, place
                if place == "outside":                   # kept for the Results page (the newest 200)
                    d = os.path.join(config.LIVE, "outside"); os.makedirs(d, exist_ok=True)
                    with open(os.path.join(d, f"{cam}-{ts:.0f}.jpg"), "wb") as f:
                        f.write(jp)
                    for old in sorted(os.listdir(d))[:-200]:
                        os.remove(os.path.join(d, old))

    def publish(self):
        now = time.time()
        by_role = {}
        for cam, st in self.cams.items():
            fresh = st["ts"] is not None and now - st["ts"] < STALE_S
            by_role.setdefault(rules.role(cam), []).append((st, fresh))

        def any_seen(role, place=None):
            rows = by_role.get(role, [])
            if not rows or not any(f for _, f in rows):
                return None                               # not running: unknown
            return any(f and st["seen"] and (place is None or st["place"] == place) for st, f in rows)
        self.mqtt.state("in_house", any_seen("house"))
        self.mqtt.state("in_yard", any_seen("yard"))
        self.mqtt.state("outside_walls", any_seen("yard", "outside"))
        if self.last_seen:
            from datetime import datetime, timezone
            self.mqtt.state("last_seen", datetime.fromtimestamp(self.last_seen["ts"], timezone.utc).isoformat())
            self.mqtt.state("last_place", rules.PLACE_TEXT.get(self.last_seen["place"], self.last_seen["place"]))
            self.mqtt.attributes("last_seen", {"camera": self.last_seen["camera"], "place": rules.PLACE_TEXT.get(self.last_seen["place"]),
                                               "score": self.last_seen["score"], "detected_by": self.last_seen["source"]})
        srcs = sorted({st["device"] for st in self.cams.values() if st["ts"] and now - st["ts"] < STALE_S and st["device"]})
        self.mqtt.state("detector", "; ".join(srcs) if srcs else "not running")

    def _loop(self):
        """Every 2 s: hand each camera to the server or to this add-on."""
        while True:
            try:
                self.tick()
            except Exception as e:
                self.log(f"detection: {e}")
            time.sleep(2)

    def tick(self):
        now = time.time()
        for cam in config.cameras():
            st = self.cam(cam)
            m = db.kv_get(f"model:{cam}")
            server_ok = now - st["server_ts"] < server_fresh_s()
            loop = self.local.get(cam)
            want_local = bool(m) and not server_ok and os.path.exists(model_files(m["run"])["xml"])
            if loop and (not want_local or loop.run_id != m["run"] or not loop.is_alive()):
                loop.stop(); self.local.pop(cam, None); loop = None
                if want_local and not server_ok:
                    pass
            if want_local and loop is None:
                if st["source"] and st["source"] != "local":
                    self.log(f"detection {cam}: no results from the server for {server_fresh_s()} s; detecting here")
                l = LocalLoop(self, cam, m["run"]); l.start(); self.local[cam] = l
        self.publish()

    def status(self):
        now = time.time(); out = {}
        for cam in config.cameras():
            st = dict(self.cam(cam))
            loop = self.local.get(cam)
            st["local_running"] = bool(loop and loop.is_alive())
            st["local_error"] = loop.error if loop else None
            st["fresh"] = st["ts"] is not None and now - st["ts"] < STALE_S
            st["model"] = db.kv_get(f"model:{cam}")
            out[cam] = st
        return {"cameras": out, "last_seen": self.last_seen}
