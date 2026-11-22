"""Training servers and the work handed to them.

A server is installed with one curl line from the Servers page (the add-on
serves the installer and the server's code), pairs with a one-time code, and
then checks in every few seconds. Each check-in carries its version, its
cards and its free slots; the answer carries:

  update   when the add-on is newer than the server: the add-on's own copy of
           the server code and its SHA-256 (servers never fetch from GitHub;
           Home Assistant updates the add-on, the add-on updates the servers)
  detect   whether this server is the main detector, on CPU or GPU, and for
           which cameras with which model
  job      a first-pass batch or an approved training run, when it has room
  cancel   jobs to stop

Runs still wait for your approval; the server only does the work."""
import gzip, hashlib, io, json, os, secrets, tarfile, threading, time

from . import config, db, rules
from .logbuf import logger

ONLINE_S = 45
LOST_S = 300
LABEL_BATCH = 200
TILES_BATCH = 50
DEFAULT_SPE = {"cpu": {640: 0.12, 960: 0.27}, "gpu": {640: 0.004, 960: 0.007}}   # s per image per epoch until measured
_bundle = {"version": None}
_lock = threading.Lock()


def hash_key(k):
    return hashlib.sha256(k.encode()).hexdigest()


def pair_code(new=False):
    c = db.connect()
    p = db.kv_get("pair", None, c)
    if new or not p or p["expires"] < time.time():
        p = {"code": secrets.token_urlsafe(9), "expires": time.time() + 3600}
        db.kv_set("pair", p, c); c.commit()
    return p


def advertise_url():
    o = config.options()
    if o.get("advertise_url"):
        return o["advertise_url"].rstrip("/")
    from . import ha
    return f"http://{ha.host_address() or 'HOME-ASSISTANT-ADDRESS'}:{o['server_port']}"


def install_command():
    return f"curl -fsSL {advertise_url()}/server/install.sh | sudo bash -s -- {pair_code()['code']}"


def register(code, name, info):
    c = db.connect()
    p = db.kv_get("pair", None, c)
    if not p or p["code"] != code or p["expires"] < time.time():
        raise PermissionError("that pairing code is wrong or has expired: copy a fresh install line from the Servers page")
    sid, key = secrets.token_hex(6), secrets.token_urlsafe(32)
    c.execute("INSERT INTO servers VALUES (?,?,?,?,?,?,?)", (sid, (name or "server")[:60], hash_key(key), time.time(), time.time(),
                                                           json.dumps(info or {}), json.dumps({"detect": "cpu", "auto_update": True})))
    db.kv_set("pair", None, c)
    c.commit()
    return sid, key


def auth(key):
    if not key:
        return None
    r = db.connect().execute("SELECT * FROM servers WHERE key_hash=?", (hash_key(key),)).fetchone()
    return dict(r) if r else None


def bundle():
    """The server's code at this add-on's version, as a reproducible tar.gz."""
    v = config.version()
    with _lock:
        if _bundle.get("version") != v:
            raw = io.BytesIO()
            with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz, tarfile.open(fileobj=gz, mode="w") as t:
                def norm(ti):
                    ti.mtime = 0; ti.uid = ti.gid = 0; ti.uname = ti.gname = ""
                    return ti
                for name in sorted(os.listdir(config.WORKER_DIR)):
                    p = os.path.join(config.WORKER_DIR, name)
                    if name.startswith(".") or not os.path.isfile(p) or name.endswith(".pyc"):
                        continue
                    t.add(p, arcname=name, filter=norm)
                vb = v.encode(); ti = norm(tarfile.TarInfo("VERSION")); ti.size = len(vb); t.addfile(ti, io.BytesIO(vb))
            data = raw.getvalue()
            _bundle.update(version=v, data=data, sha256=hashlib.sha256(data).hexdigest())
        return dict(_bundle)


def installer():
    with open(os.path.join(config.WORKER_DIR, "install.sh")) as f:
        return f.read().replace("__ADDON_URL__", advertise_url())


def online(srv, now=None):
    return srv.get("last_seen") and (now or time.time()) - srv["last_seen"] < ONLINE_S


def servers(c=None):
    c = c or db.connect()
    out = []
    for r in c.execute("SELECT * FROM servers ORDER BY created"):
        d = dict(r); d.pop("key_hash")
        d["info"] = json.loads(d["info"] or "{}"); d["settings"] = json.loads(d["settings"] or "{}")
        d["online"] = bool(online(d))
        d["update_offer"] = db.kv_get(f"update_offer:{d['id']}", None, c)
        out.append(d)
    return out


def main_detector(c=None):
    for s in servers(c):
        if s["online"] and s["settings"].get("detect", "cpu") != "off":
            return s
    return None


def detect_assignment(srv_id, c):
    m = main_detector(c)
    if not m or m["id"] != srv_id:
        return {"enabled": False}
    o = config.options()
    base = (o.get("server_restream_url") or "").rstrip("/")
    if not base:
        from . import ha
        base = f"rtsp://{ha.host_address() or 'localhost'}:8554"
    from .detection import frame_size, model_files
    cams = []
    for cam, cc in config.cameras().items():
        mdl = db.kv_get(f"model:{cam}", None, c)
        if mdl and os.path.exists(model_files(mdl["run"])["xml"]):
            cams.append({"name": cam, "url": f"{base}/{cam}", "run": mdl["run"], "imgsz": int(cc.get("imgsz", 960)),
                         "frame_size": list(frame_size(cam)), "has_pt": os.path.exists(model_files(mdl["run"])["pt"])})
    return {"enabled": True, "device": m["settings"].get("detect", "cpu"), "every_s": int(o.get("detect_every_s", 5)), "cameras": cams}


def poll(srv, body):
    c = db.connect(); now = time.time()
    info = body.get("info") or {}
    c.execute("UPDATE servers SET last_seen=?, info=? WHERE id=?", (now, json.dumps(info), srv["id"]))
    settings = json.loads(srv["settings"] or "{}")
    resp = {"addon_version": config.version(), "server_id": srv["id"], "name": srv["name"]}
    b = bundle()
    if info.get("version") != b["version"] and settings.get("auto_update", True):
        resp["update"] = {"version": b["version"], "sha256": b["sha256"], "url": "server/bundle.tar.gz"}
        offer = db.kv_get(f"update_offer:{srv['id']}", None, c)
        if not offer or offer["version"] != b["version"]:
            db.kv_set(f"update_offer:{srv['id']}", {"version": b["version"], "from": info.get("version"), "at": now}, c)
    running = set(info.get("jobs") or [])
    # jobs this server had but no longer reports (it restarted, or the job's process died)
    for j in c.execute("SELECT * FROM jobs WHERE server=? AND status='running' AND started < ?", (srv["id"], now - 60)).fetchall():
        if j["id"] not in running:
            lose(c, j, "the server stopped working on it (it restarted or the job's process ended without a result)")
    resp["cancel"] = [j["id"] for j in c.execute("SELECT id FROM jobs WHERE server=? AND status='cancelled'", (srv["id"],)) if j["id"] in running]
    resp["detect"] = detect_assignment(srv["id"], c)
    if not resp.get("update"):
        job = next_job(srv, info, c)
        if job:
            resp["job"] = job
    c.commit()
    return resp


def lose(c, j, why):
    """A job whose server stopped working on it without a result: failed, not
    retried blindly (a new first-pass batch is made after a pause)."""
    c.execute("UPDATE jobs SET status='failed', finished=?, result=? WHERE id=?", (time.time(), json.dumps({"ok": False, "error": why}), j["id"]))
    if j["run"]:
        c.execute("UPDATE runs SET status='failed', finished=?, result=? WHERE id=? AND status='running'",
                  (time.time(), json.dumps({"ok": False, "error": why}), j["run"]))


def next_job(srv, info, c):
    free = info.get("free") or {}
    for j in c.execute("SELECT * FROM jobs WHERE status='queued' ORDER BY (kind='train') DESC, created").fetchall():
        p = json.loads(j["payload"] or "{}")
        if j["kind"] == "train":
            dev = p.get("config", {}).get("device", "cpu")
            if free.get(dev, 0) < 1:
                continue
        elif free.get("label", 0) < 1:
            continue
        now = time.time()
        c.execute("UPDATE jobs SET status='running', server=?, started=? WHERE id=?", (srv["id"], now, j["id"]))
        if j["kind"] == "label":
            logger("label")(f"batch {j['id']} taken by {srv['name']}")
        if j["run"]:
            c.execute("UPDATE runs SET status='running', started=?, server=? WHERE id=?", (now, srv["id"], j["run"]))
        return {"id": j["id"], "kind": j["kind"], "payload": p}
    return None


# ---------------- results from servers ----------------
def job_of(srv, jid, c):
    j = c.execute("SELECT * FROM jobs WHERE id=? AND server=?", (jid, srv["id"])).fetchone()
    if not j:
        raise KeyError("no such job for this server")
    return j


def progress(srv, jid, body):
    c = db.connect(); j = job_of(srv, jid, c)
    c.execute("UPDATE jobs SET progress=? WHERE id=?", (json.dumps(body), jid))
    if j["run"]:
        c.execute("UPDATE runs SET progress=? WHERE id=?", (json.dumps(body), j["run"]))
    c.commit()


def owl_rows(srv, jid, rows):
    c = db.connect(); job_of(srv, jid, c)
    for r in rows:
        c.execute("INSERT OR REPLACE INTO owl VALUES (?,?,?,?,?)", (r["file"], r["mode"], r.get("secs"), json.dumps(r["dets"]), time.time()))
    c.commit()


def artifact(srv, jid, name, data):
    c = db.connect(); j = job_of(srv, jid, c)
    if name not in ("best.pt", "model.xml", "model.bin") or not j["run"]:
        raise ValueError("unexpected file")
    d = os.path.join(config.MODELS, str(j["run"])); os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, name + ".part"), "wb") as f:
        f.write(data)
    os.replace(os.path.join(d, name + ".part"), os.path.join(d, name))


def done(srv, jid, body):
    c = db.connect(); j = job_of(srv, jid, c)
    ok = bool(body.get("ok"))
    c.execute("UPDATE jobs SET status=?, finished=?, result=? WHERE id=?", ("done" if ok else "failed", time.time(), json.dumps(body), jid))
    if j["kind"] == "label":
        logger("label")(f"batch {jid} on {srv['name']}: " + (f"{body.get('frames')} frames in {body.get('secs')} s" if ok else f"failed: {body.get('error')}"))
    if j["run"]:
        st = c.execute("SELECT status FROM runs WHERE id=?", (j["run"],)).fetchone()["status"]
        if st == "running":
            c.execute("UPDATE runs SET status=?, finished=?, result=? WHERE id=?", ("done" if ok else "failed", time.time(), json.dumps(body), j["run"]))
    c.commit()


# ---------------- scheduling ----------------
def label_targets(c):
    """Frames that need first boxes: each safety check's frame, Frigate's event
    snapshots, and one full-resolution frame from the middle of each Frigate
    event (the rest of the collected frames aren't used)."""
    at_check = rules.check_frames(c)
    mid = {}
    for r in c.execute("SELECT file, event_id FROM frames WHERE source='event' AND deleted IS NULL ORDER BY event_id, ts"):
        mid.setdefault(r["event_id"], []).append(r["file"])
    middles = {v[len(v) // 2] for v in mid.values()}
    return [r for r in c.execute("SELECT f.file, f.camera, f.source, f.ts FROM frames f WHERE f.deleted IS NULL AND NOT EXISTS"
                                 " (SELECT 1 FROM owl o WHERE o.file=f.file AND o.mode='full') ORDER BY f.ts DESC")
            if r["file"] in at_check or r["source"] == "event_snapshot" or r["file"] in middles]


def make_label_jobs(F):
    """Queue first-pass batches for the frames that need them (newest first),
    then the tiled pass for yard frames the first look was unsure about."""
    c = db.connect()
    if c.execute("SELECT 1 FROM jobs WHERE kind='label' AND status IN ('queued','running')").fetchone():
        return
    last = c.execute("SELECT status, finished FROM jobs WHERE kind='label' ORDER BY id DESC LIMIT 1").fetchone()
    if last and last["status"] == "failed" and (last["finished"] or 0) > time.time() - 600:
        return                       # the last batch failed: try again 10 minutes after it
    todo = [{"file": r["file"], "mode": "full"} for r in label_targets(c)][:LABEL_BATCH]
    if not todo:
        best = {}
        for r in c.execute("SELECT file, dets FROM owl WHERE mode='full'"):
            best[r["file"]] = max((d[1] for d in json.loads(r["dets"]) if d[0] in ("tortoise", "turtle")), default=0)
        tiled = {r["file"] for r in c.execute("SELECT file FROM owl WHERE mode='tiles'")}
        at_check = rules.check_frames(c)
        for r in c.execute("SELECT file, camera FROM frames WHERE deleted IS NULL AND source != 'event_snapshot' ORDER BY ts DESC"):
            if r["file"] in at_check and rules.role(r["camera"]) == "yard" and r["file"] not in tiled and 0.08 <= best.get(r["file"], 0) < rules.POS:
                todo.append({"file": r["file"], "mode": "tiles"})
                if len(todo) >= TILES_BATCH:
                    break
    if todo:
        c.execute("INSERT INTO jobs(kind, status, payload, created) VALUES ('label', 'queued', ?, ?)", (json.dumps({"frames": todo}), time.time()))
        c.commit()
        logger("label")(f"batch of {len(todo)} frames queued ({todo[0]['mode']})")


NEW_FOR_RUN = 25           # new frames that teach the detector in use, before a run is proposed


def propose_runs(F):
    """A run is proposed for a camera once its review queue is clear and there
    are NEW_FOR_RUN more frames that teach the detector in use something than
    at the last run (the first run: 150 frames with him)."""
    c = db.connect()
    revs = rules.reviews(c); kc = rules.counts(F.frames, revs)
    for cam in config.cameras():
        if kc.get(cam, {}).get("to_review", 0) > 0:
            continue
        if c.execute("SELECT 1 FROM runs WHERE camera=? AND status IN ('proposed','approved','running')", (cam,)).fetchone():
            continue
        k = rules.dataset(F.frames, revs, cam)[1]
        last = c.execute("SELECT counts FROM runs WHERE camera=? ORDER BY id DESC LIMIT 1", (cam,)).fetchone()
        if last is None or k["in_use_run"] is None:
            due = last is None and k["pos"] >= 150
        else:
            lk = json.loads(last["counts"] or "{}")
            before = lk.get("new", 0) if lk.get("in_use_run") == k["in_use_run"] else 0
            due = k["new"] - before >= NEW_FOR_RUN
        if due:
            c.execute("INSERT INTO runs(camera, status, created, config, counts) VALUES (?,?,?,?,?)",
                      (cam, "proposed", time.time(), json.dumps(rules.default_config(cam)), json.dumps(k)))
            c.commit()


def sweep():
    """Jobs whose server went silent."""
    c = db.connect(); now = time.time()
    for j in c.execute("SELECT j.* FROM jobs j JOIN servers s ON s.id = j.server WHERE j.status='running' AND (s.last_seen IS NULL OR s.last_seen < ?)", (now - LOST_S,)).fetchall():
        lose(c, j, "the training server stopped reporting")
    c.commit()


# ---------------- run actions (from the page) ----------------
def run_action(F, rid, action, body):
    c = db.connect(); r = c.execute("SELECT * FROM runs WHERE id=?", (rid,)).fetchone()
    if not r:
        raise ValueError("no such run")
    now = time.time(); cfg = json.loads(r["config"])
    if action in ("approve", "decline"):
        if r["status"] != "proposed":
            raise ValueError("this run is not waiting for a decision")
        F.refresh(force=True)
        items, k = rules.dataset(F.frames, rules.reviews(c), r["camera"])
        if action == "decline":
            c.execute("UPDATE runs SET status='declined', decided=?, counts=? WHERE id=?", (now, json.dumps(k), rid))
        else:
            dev = body.get("device", "cpu")
            if dev not in ("cpu", "gpu"):
                raise ValueError("device must be cpu or gpu")
            cfg["device"] = dev
            payload = {"run": rid, "camera": r["camera"], "config": cfg,
                       "items": [{"file": x["file"], "box": b, "split": s, "w": x["size"][0], "h": x["size"][1]} for x, b, s in items]}
            c.execute("UPDATE runs SET status='approved', decided=?, counts=?, config=?, items=? WHERE id=?",
                      (now, json.dumps(k), json.dumps(cfg), json.dumps([x["file"] for x, _, _ in items]), rid))
            c.execute("INSERT INTO jobs(kind, status, run, payload, created) VALUES ('train', 'queued', ?, ?, ?)", (rid, json.dumps(payload), now))
    elif action == "stop":
        if r["status"] in ("approved", "running"):
            c.execute("UPDATE jobs SET status=CASE status WHEN 'running' THEN 'cancelled' ELSE 'dropped' END WHERE run=? AND status IN ('queued','running')", (rid,))
            c.execute("UPDATE runs SET status='stopped', finished=? WHERE id=?", (now, rid))
    elif action == "use":
        from .detection import model_files
        if r["status"] != "done" or not os.path.exists(model_files(rid)["xml"]):
            raise ValueError("only a finished run with its model can be used")
        db.kv_set(f"model:{r['camera']}", {"run": rid, "imgsz": cfg["imgsz"], "since": now}, c)
    elif action == "unuse":
        m = db.kv_get(f"model:{r['camera']}", None, c)
        if m and m["run"] == rid:
            db.kv_set(f"model:{r['camera']}", None, c)
    else:
        raise ValueError("bad action")
    c.commit()


def propose(F, cam):
    if cam not in config.cameras():
        raise ValueError("unknown camera")
    c = db.connect()
    if c.execute("SELECT 1 FROM runs WHERE camera=? AND status IN ('proposed','approved','running')", (cam,)).fetchone():
        raise ValueError("a run for this camera is already waiting or running")
    F.refresh(force=True)
    k = rules.dataset(F.frames, rules.reviews(c), cam)[1]
    c.execute("INSERT INTO runs(camera, status, created, config, counts) VALUES (?,?,?,?,?)",
              (cam, "proposed", time.time(), json.dumps(rules.default_config(cam)), json.dumps(k)))
    c.commit()


def estimate(c, r, cfg, counts):
    out = {}
    for dev in ("cpu", "gpu"):
        spe = DEFAULT_SPE[dev].get(cfg["imgsz"], DEFAULT_SPE[dev][960]); measured = False
        for p in c.execute("SELECT config, result FROM runs WHERE status='done' AND camera=? ORDER BY id DESC", (r["camera"],)):
            pc, pr = json.loads(p["config"]), json.loads(p["result"] or "{}")
            if pc.get("imgsz") == cfg["imgsz"] and pc.get("device", "cpu") == dev and pr.get("secs_per_img_epoch"):
                spe = pr["secs_per_img_epoch"]; measured = True; break
        out[dev] = {"secs": spe * (counts or {}).get("train", 0) * cfg["epochs"], "measured": measured}
    return out


# ---------------- import from an earlier setup ----------------
def import_rows(kind, rows):
    c = db.connect()
    if kind == "owl":
        for r in rows:
            c.execute("INSERT OR REPLACE INTO owl VALUES (?,?,?,?,?)", (r["file"], r["mode"], r.get("secs"), json.dumps(r["dets"]), time.time()))
    elif kind == "reviews":
        for r in rows:
            c.execute("INSERT OR REPLACE INTO reviews VALUES (?,?,?,?,?)", (r["file"], r["verdict"], r.get("box"), r.get("rejected"), r["at"]))
    elif kind == "runs":
        for r in rows:
            c.execute("INSERT OR REPLACE INTO runs(id, camera, status, created, decided, started, finished, config, counts, result, items)"
                      " VALUES (?,?,?,?,?,?,?,?,?,?,?)", (r["id"], r["camera"], r["status"], r["created"], r.get("decided"), r.get("started"),
                                                          r.get("finished"), r["config"], r.get("counts"), r.get("result"), r.get("items")))
    elif kind == "kv":
        for r in rows:
            if r["key"].split(":")[0] in ("outline", "model", "record"):
                db.kv_set(r["key"], r["value"], c)
    elif kind == "missing":
        for r in rows:
            c.execute("INSERT OR IGNORE INTO missing VALUES (?,?)", (r["camera"], r["ts"]))
    else:
        raise ValueError("unknown kind")
    c.commit()
    return len(rows)


def import_model(run, name, data):
    if name not in ("best.pt", "model.xml", "model.bin"):
        raise ValueError("unexpected file")
    d = os.path.join(config.MODELS, str(int(run))); os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, name + ".part"), "wb") as f:
        f.write(data)
    os.replace(os.path.join(d, name + ".part"), os.path.join(d, name))
