#!/usr/bin/env python3
"""Tortoise Monitor training server.

Installed on a machine with the line from the add-on's Servers page; runs as
the systemd service tortoise-worker. It checks in with the add-on every 5 s
and does what the add-on hands it:

  detection   the main detector: reads Frigate's restream, runs the detector
              in use on CPU (OpenVINO, the default) or GPU (PyTorch, only on a
              card nobody else holds, back to the CPU the moment anyone asks)
              and posts every result; the add-on decides what it means
  first pass  OWLv2 boxes for batches of collected frames (own process)
  training    approved runs (own process, own GPU lease)

Updates come only from the add-on: when the add-on is newer, the server
fetches the add-on's copy of this code, checks its SHA-256, installs it next
to the current one and restarts into it (between jobs). If the new version
fails, it fails in the open: systemd keeps restarting it and its log shows
why, and the add-on shows the server silent since it was sent that version.

  tortoise_worker.py register --addon URL --code CODE [--name NAME]
  tortoise_worker.py run
  tortoise_worker.py import DIR      (one-time: an earlier /cpool/TortoiseVision)"""
import argparse, base64, hashlib, io, json, os, shutil, signal, socket, subprocess, sys, tarfile, threading, time, urllib.parse

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, HERE)
import tmclient

BASE = os.environ.get("TM_SERVER_HOME", "/opt/tortoise-worker")
CONF = os.path.join(BASE, "config.json")
VERSION = open(os.path.join(HERE, "VERSION")).read().strip() if os.path.exists(os.path.join(HERE, "VERSION")) else "dev"
JPEG_EVERY_S = 10
UPDATE_ERROR = {}          # the last failed update, reported to the add-on


def log(msg):
    print(time.strftime("%H:%M:%S ") + msg, flush=True)


def conf():
    with open(CONF) as f:
        return json.load(f)


def gpus():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader"], capture_output=True, text=True, timeout=10).stdout
        return [{"index": int(a), "name": b.strip(), "memory": c.strip()} for a, b, c in (l.split(",") for l in out.strip().splitlines())]
    except Exception:
        return []


def host_info(children, det):
    g = gpus()
    gpu_jobs = sum(1 for c in children.values() if c["kind"] == "train" and c["device"] == "gpu")
    return {"version": VERSION, "host": socket.gethostname(), "cpus": os.cpu_count(), "gpus": g,
            "arbiter": os.path.exists("/run/arbiter/arbiter.sock"), "update_error": UPDATE_ERROR.get("error"), "jobs": list(children),
            "free": {"cpu": 0 if any(c["kind"] == "train" and c["device"] == "cpu" for c in children.values()) else 1,
                     "gpu": max(0, len(g) - gpu_jobs), "label": 0 if any(c["kind"] == "label" for c in children.values()) else 1},
            "detect": det.status()}


# ---------------- detection ----------------
class CamLoop(threading.Thread):
    def __init__(self, det, cam):
        super().__init__(daemon=True)
        self.det, self.cam = det, cam
        self.stopping = False; self.device = None; self.ms = None; self.last = None; self.error = None; self.reader = None
        self.unusable = None

    def model(self, name):
        d = os.path.join(BASE, "cache", "models", str(self.cam["run"])); os.makedirs(d, exist_ok=True)
        p = os.path.join(d, name)
        if not os.path.exists(p):
            data = self.det.client.get(f"/api/server/model/{self.cam['run']}/{name}", timeout=300)
            with open(p + ".part", "wb") as f:
                f.write(data)
            os.replace(p + ".part", p)
        return p

    def run(self):
        import numpy as np
        import tmdetect as D
        try:
            w, h = self.cam["frame_size"]
            lb = D.Letterbox(self.cam["imgsz"], w, h)
            self.model("model.bin"); xml = self.model("model.xml")
            cpu = D.OpenVinoDetector(xml, "CPU", "rgb", cache_dir=os.path.join(BASE, "cache", "openvino"))
            self.reader = D.KeyframeReader(self.cam["url"], lb.sw, lb.sh, log=log); self.reader.start()
        except Exception as e:
            self.error = str(e); log(f"detection {self.cam['name']}: {e}"); return
        gpu = None; lease = None; card = None; last_try = 0; last_jpeg = 0; last = 0.0
        while not self.stopping:
            want_gpu = self.det.device == "gpu" and self.cam.get("has_pt")
            if want_gpu and lease is None:
                from gpu_lease import Opportunistic
                lease = Opportunistic(f"tortoise detection ({self.cam['name']})")
            if card and (lease.asked.is_set() or not want_gpu):
                gpu = None; lease.release(); card = None             # someone wants it: back to the CPU
                log(f"detection {self.cam['name']}: gave the GPU back; on the CPU")
            if want_gpu and not card and time.time() - last_try > 60:
                last_try = time.time(); card = lease.try_take()
                if card:
                    try:
                        gpu = D.TorchDetector(self.model("best.pt"), f"cuda:{card[4:]}")
                        log(f"detection {self.cam['name']}: on {card}")
                    except Exception as e:
                        log(f"detection {self.cam['name']}: GPU failed ({e}); on the CPU"); lease.release(); card = None; gpu = None
            f = self.reader.newest(after=last, timeout=20)
            if f is None:
                continue
            ts, buf, bad = f; last = ts
            if bad:                                                  # no answer from this frame
                self.unusable = {"ts": ts, "reason": bad}
                try:
                    self.det.client.post("/api/server/detections", {"camera": self.cam["name"], "ts": ts, "run": self.cam["run"],
                                                                    "unusable": bad, "size": [w, h], "device": self.device or "CPU"},
                                         tries=1, timeout=10)
                except Exception as e:
                    self.error = str(e)
                time.sleep(max(0.0, self.det.every - (time.time() - ts)))
                continue
            rgb = np.frombuffer(buf, np.uint8).reshape(lb.sh, lb.sw, 3)
            x = lb.pad_rgb(rgb)
            t = time.time(); out = (gpu or cpu)(x); dets = D.decode(out, lb); self.ms = (time.time() - t) * 1000
            self.device = f"GPU {card}" if gpu else "CPU"; self.last = ts
            body = {"camera": self.cam["name"], "ts": ts, "run": self.cam["run"], "dets": dets, "size": [w, h], "device": self.device, "ms": round(self.ms, 1)}
            if dets and dets[0][0] >= 0.5 and ts - last_jpeg >= JPEG_EVERY_S:
                from PIL import Image
                b = io.BytesIO(); Image.fromarray(rgb).save(b, "JPEG", quality=80)
                body["jpeg"] = base64.b64encode(b.getvalue()).decode(); last_jpeg = ts
            try:
                self.det.client.post("/api/server/detections", body, tries=1, timeout=10)
            except Exception as e:
                self.error = str(e)
            time.sleep(max(0.0, self.det.every - (time.time() - ts)))
        if lease:
            lease.release()
        self.reader.stop()

    def stop(self):
        self.stopping = True
        if self.reader:
            self.reader.stop()


class Detect:
    def __init__(self, client):
        self.client = client; self.loops = {}; self.device = "cpu"; self.every = 5; self.key = None

    def apply(self, a):
        a = a or {}
        cams = {c["name"]: c for c in a.get("cameras", [])} if a.get("enabled") else {}
        self.device = a.get("device", "cpu"); self.every = a.get("every_s", 5)
        for name in list(self.loops):
            l = self.loops[name]
            if name not in cams or cams[name] != l.cam or not l.is_alive():
                l.stop(); del self.loops[name]
        for name, c in cams.items():
            if name not in self.loops:
                l = CamLoop(self, c); l.start(); self.loops[name] = l
                log(f"detection {name}: run {c['run']}, {self.device}")

    def status(self):
        return {n: {"device": l.device, "ms": l.ms, "last": l.last, "error": l.error, "unusable": l.unusable} for n, l in self.loops.items()}


# ---------------- jobs ----------------
def start_job(job, cfg, children):
    d = os.path.join(BASE, "jobs", str(job["id"])); os.makedirs(d, exist_ok=True)
    jf = os.path.join(d, "job.json")
    with open(os.open(jf, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w") as f:
        json.dump({**job, "addon": cfg["addon"], "key": cfg["key"], "base": BASE}, f)
    script = "tmjob_train.py" if job["kind"] == "train" else "tmjob_label.py"
    device = job["payload"].get("config", {}).get("device", "cpu") if job["kind"] == "train" else "cpu"
    env = dict(os.environ)
    if job["kind"] == "train" and device == "gpu":
        used = {c["card"] for c in children.values() if c.get("card")}
        try:
            from gpu_lease import _Holder
            held = _Holder("probe").holders()
        except Exception:
            held = {}
        cards = sorted((c for c in ("cuda1", "cuda0") if c not in used), key=lambda c: c in held)
        env["TORTOISE_CARD"] = cards[0] if cards else "cuda1"
    logf = open(os.path.join(d, "out.log"), "ab")
    p = subprocess.Popen([sys.executable, os.path.join(HERE, script), jf], stdout=logf, stderr=subprocess.STDOUT, env=env)
    children[job["id"]] = {"proc": p, "kind": job["kind"], "device": device, "card": env.get("TORTOISE_CARD")}
    log(f"job {job['id']} ({job['kind']}{', ' + device if job['kind'] == 'train' else ''}) started")


def reap(children):
    for jid in list(children):
        rc = children[jid]["proc"].poll()
        if rc is not None:
            log(f"job {jid} ended ({rc})"); del children[jid]


# ---------------- updates ----------------
def update(client, u):
    v = u["version"]
    log(f"updating {VERSION} -> {v} from the add-on")
    data = client.get("/" + u["url"].lstrip("/"), timeout=300)
    if hashlib.sha256(data).hexdigest() != u["sha256"]:
        UPDATE_ERROR["error"] = f"{v}: the download's checksum didn't match"
        log("update: checksum mismatch; not installing"); return False
    vd = os.path.join(BASE, "versions", v); tmp = vd + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp)
    with tarfile.open(fileobj=io.BytesIO(data)) as t:
        t.extractall(tmp, filter="data")
    cur = os.path.realpath(os.path.join(BASE, "current"))
    old_req = open(os.path.join(cur, "requirements.txt")).read() if os.path.exists(os.path.join(cur, "requirements.txt")) else ""
    if open(os.path.join(tmp, "requirements.txt")).read() != old_req:
        r = subprocess.run([os.path.join(BASE, "venv", "bin", "pip"), "install", "-q", "-r", os.path.join(tmp, "requirements.txt")], capture_output=True, text=True)
        if r.returncode:
            UPDATE_ERROR["error"] = f"{v}: installing its Python requirements failed: {r.stderr.strip()[-400:]}"
            log("update: installing its requirements failed; staying on " + VERSION + ":\n" + r.stderr[-1500:])
            return False
    shutil.rmtree(vd, ignore_errors=True); os.rename(tmp, vd)
    install_units(vd)
    os.symlink(vd, os.path.join(BASE, "current.tmp")); os.replace(os.path.join(BASE, "current.tmp"), os.path.join(BASE, "current"))
    for old in sorted(os.listdir(os.path.join(BASE, "versions"))):      # only the version now in use stays
        p = os.path.join(BASE, "versions", old)
        if p != vd:
            shutil.rmtree(p, ignore_errors=True)
    log(f"update: {v} installed; restarting into it")
    return True


def install_units(vd):
    """The service unit, from the version's template (when it changed)."""
    src = os.path.join(vd, "tortoise-worker.service")
    if not os.path.exists(src):
        return
    text = open(src).read().replace("__BASE__", BASE).replace("__ADDON__", conf()["addon"])
    dst = "/etc/systemd/system/tortoise-worker.service"
    if not os.path.exists(dst) or open(dst).read() != text:
        with open(dst, "w") as f:
            f.write(text)
        subprocess.run(["systemctl", "daemon-reload"])


# ---------------- commands ----------------
def cmd_register(a):
    info = {"version": VERSION, "host": socket.gethostname(), "gpus": gpus()}
    r = tmclient.Client(a.addon).post("/api/server/register", {"code": a.code, "name": a.name or socket.gethostname(), "info": info})
    os.makedirs(BASE, exist_ok=True)
    with open(os.open(CONF, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w") as f:
        json.dump({"addon": a.addon.rstrip("/"), "id": r["id"], "key": r["key"]}, f)
    print(f"paired with the add-on ({r['addon_version']}) as {a.name or socket.gethostname()}")


def cmd_run(a):
    cfg = conf(); client = tmclient.Client(cfg["addon"], cfg["key"], timeout=30)
    det = Detect(client); children = {}
    log(f"training server {VERSION} for {cfg['addon']}")
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *x: stop.set())
    while not stop.is_set():
        reap(children)
        try:
            r = client.post("/api/server/poll", {"info": host_info(children, det)}, tries=1)
        except Exception as e:
            log(f"add-on unreachable: {e}"); stop.wait(10); continue
        if r.get("update") and not children:
            det.apply({})
            if update(client, r["update"]):
                sys.exit(75)
        det.apply(r.get("detect"))
        for jid in r.get("cancel") or []:
            if jid in children:
                log(f"job {jid}: stopped from the add-on"); children[jid]["proc"].terminate()
        if r.get("job"):
            start_job(r["job"], cfg, children)
        stop.wait(5)
    det.apply({})
    for c in children.values():
        c["proc"].terminate()


def cmd_import(a):
    """Send an earlier setup's frames, labels, answers, outline, runs and
    models (the NAS's /cpool/TortoiseVision layout) to the add-on."""
    import sqlite3
    from concurrent.futures import ThreadPoolExecutor
    cfg = conf(); client = tmclient.Client(cfg["addon"], cfg["key"], timeout=120)
    src = a.dir.rstrip("/")
    strip = lambda f: f[len("frames/"):] if f.startswith("frames/") else f
    idx = {}
    for line in open(f"{src}/index.jsonl"):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        idx[r["file"]] = r
    names = [strip(f) for f in idx]
    have = set()
    for i in range(0, len(names), 2000):
        have |= set(client.post("/api/server/import/have", {"files": names[i:i + 2000]})["have"])
    todo = [r for f, r in idx.items() if strip(f) not in have and os.path.exists(f"{src}/{f}")]
    log(f"frames: {len(idx)} listed, {len(have)} already in the add-on, {len(todo)} to send")

    def send(r):
        meta = {"camera": r["camera"], "ts": r["ts"], "source": r.get("source", "continuous"), "event_id": r.get("event_id"),
                "frigate_label": r.get("frigate_label"), "frigate_score": r.get("frigate_score")}
        with open(f"{src}/{r['file']}", "rb") as f:
            client.put("/api/server/import/frame?" + urllib.parse.urlencode({"file": strip(r["file"]), "meta": json.dumps(meta)}), f.read())
    done = 0
    with ThreadPoolExecutor(4) as ex:
        for _ in ex.map(send, todo):
            done += 1
            if done % 500 == 0:
                log(f"frames: {done}/{len(todo)}")
    rows = []
    if os.path.exists(f"{src}/labels/owl.jsonl"):
        for line in open(f"{src}/labels/owl.jsonl"):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            rows.append({"file": strip(r["file"]), "mode": r["mode"], "secs": r.get("secs"), "dets": r["dets"]})
    for i in range(0, len(rows), 1000):
        client.post("/api/server/import/rows", {"kind": "owl", "rows": rows[i:i + 1000]})
    log(f"first-pass rows: {len(rows)}")
    db = sqlite3.connect(f"{src}/labels/review.db"); db.row_factory = sqlite3.Row
    revs = [{"file": strip(r["file"]), "verdict": r["verdict"], "box": r["box"], "rejected": r["rejected"], "at": r["at"]} for r in db.execute("SELECT * FROM reviews")]
    client.post("/api/server/import/rows", {"kind": "reviews", "rows": revs})
    log(f"answers: {len(revs)}")
    runs = []
    for r in db.execute("SELECT * FROM runs"):
        d = dict(r); items = []
        for split in ("train", "val"):
            p = f"{src}/runs/{d['id']}/data/images/{split}"
            if os.path.isdir(p):
                items += [strip(os.path.relpath(os.path.realpath(os.path.join(p, n)), src)) for n in os.listdir(p)]
        d["items"] = json.dumps(items) if items else None
        d["status"] = {"running": "failed", "approved": "declined"}.get(d["status"], d["status"])
        runs.append(d)
    client.post("/api/server/import/rows", {"kind": "runs", "rows": runs})
    log(f"runs: {len(runs)}")
    kv = []
    outl = json.load(open(f"{src}/labels/outlines.json")) if os.path.exists(f"{src}/labels/outlines.json") else {}
    for cam, o in outl.items():
        kv.append({"key": f"outline:{cam}", "value": {**o, "frame": strip(o.get("frame") or "")}})
    if os.path.exists(f"{src}/labels/record.json"):
        rec = json.load(open(f"{src}/labels/record.json")); kv.append({"key": "record", "value": {"covered": rec["covered"], "out": rec["out"]}})
    use = json.load(open(f"{src}/labels/models.json")) if os.path.exists(f"{src}/labels/models.json") else {}
    import tmdetect as D
    from ultralytics import YOLO
    for r in runs:
        best = f"{src}/runs/{r['id']}/train/weights/best.pt"
        if r["status"] != "done" or not os.path.exists(best):
            continue
        c = json.loads(r["config"]); cam = r["camera"]
        sample = next((v for v in idx.values() if v["camera"] == cam and v.get("source") == "continuous"), None)
        from PIL import Image
        fw, fh = Image.open(f"{src}/{sample['file']}").size if sample else (2688, 1512)
        ih, iw = D.input_hw(c["imgsz"], fw, fh)
        tmp = os.path.join(BASE, "cache", "import", str(r["id"])); shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp)
        shutil.copy(best, f"{tmp}/best.pt")
        xp = YOLO(f"{tmp}/best.pt").export(format="openvino", imgsz=[ih, iw], half=True, verbose=False)
        xml = next(os.path.join(xp, f) for f in os.listdir(xp) if f.endswith(".xml"))
        for name, path in (("model.xml", xml), ("model.bin", xml[:-4] + ".bin"), ("best.pt", best)):
            with open(path, "rb") as f:
                client.put(f"/api/server/import/model?run={r['id']}&name={name}", f.read(), timeout=300)
        log(f"run {r['id']} ({cam}): model sent, OpenVINO at {iw}x{ih}")
    for cam, m in use.items():
        if any(r["id"] == m["run"] and r["status"] == "done" for r in runs):
            kv.append({"key": f"model:{cam}", "value": {"run": m["run"], "imgsz": m.get("imgsz"), "since": m.get("since")}})
    client.post("/api/server/import/rows", {"kind": "kv", "rows": kv})
    miss = []
    for cam in ("tortoise_camera", "yard_camera"):
        p = f"{src}/state/missing-{cam}.txt"
        if os.path.exists(p):
            miss += [{"camera": cam, "ts": float(l)} for l in open(p) if l.strip()]
    for i in range(0, len(miss), 5000):
        client.post("/api/server/import/rows", {"kind": "missing", "rows": miss[i:i + 5000]})
    log(f"import done: outline, in/out record, detectors in use ({', '.join(use) or 'none'}), {len(miss)} known gaps")


def main():
    ap = argparse.ArgumentParser(description="Tortoise Monitor training server")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("register"); r.add_argument("--addon", required=True); r.add_argument("--code", required=True); r.add_argument("--name")
    sub.add_parser("run")
    i = sub.add_parser("import"); i.add_argument("dir")
    sub.add_parser("version")
    a = ap.parse_args()
    if a.cmd == "version":
        print(VERSION)
    else:
        {"register": cmd_register, "run": cmd_run, "import": cmd_import}[a.cmd](a)


if __name__ == "__main__":
    main()
