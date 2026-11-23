"""Detector pass: the detector in use, run over the collected frames it hasn't
seen, here in the add-on (OpenVINO on the Intel GPU, else the CPU). Its
results show on the review page next to the first pass's, and frames where
the two disagree go to Review."""
import json, os, threading, time

from . import config, db
from . import paths_worker  # noqa: F401
import tmdetect as D

state = {"running": False, "last_end": None, "done": 0, "waiting": {}, "error": None, "device": None}
_stop = threading.Event()


def waiting(c=None):
    c = c or db.connect(); out = {}
    for cam in config.cameras():
        m = db.kv_get(f"model:{cam}", None, c)
        if m:
            out[cam] = c.execute("SELECT COUNT(*) FROM frames f WHERE f.camera=? AND f.deleted IS NULL AND NOT EXISTS"
                                 " (SELECT 1 FROM preds p WHERE p.file=f.file AND p.run=?)", (cam, m["run"])).fetchone()[0]
    return out


def write(c, rows):
    if rows:
        c.executemany("INSERT OR REPLACE INTO preds VALUES (?,?,?,?)", rows); c.commit()
        state["done"] += len(rows)


def run_once(log=print, budget_s=600):
    """Work through unchecked frames, newest first, for up to budget_s."""
    import openvino as ov
    from .detection import model_files
    state.update(running=True, error=None)
    t_end = time.time() + budget_s; n0 = state["done"]
    try:
        dev = "GPU" if "GPU" in ov.Core().available_devices else "CPU"
        state["device"] = "Intel GPU" if dev == "GPU" else "CPU"
        c = db.connect()
        for cam in config.cameras():
            m = db.kv_get(f"model:{cam}", None, c)
            if not m or not os.path.exists(model_files(m["run"])["xml"]):
                continue
            det = D.OpenVinoDetector(model_files(m["run"])["xml"], dev, "rgb", cache_dir=config.OV_CACHE)
            rows = c.execute("SELECT f.file, f.w, f.h FROM frames f WHERE f.camera=? AND f.deleted IS NULL AND NOT EXISTS"
                             " (SELECT 1 FROM preds p WHERE p.file=f.file AND p.run=?) ORDER BY f.ts DESC LIMIT 2000", (cam, m["run"])).fetchall()
            imgsz = int(config.cameras()[cam].get("imgsz", 960))
            # Results are written 50 at a time in one short write. A write left open
            # while frames are checked holds the database lock for the whole block,
            # and live detections wait behind it (the server gives up after 10 s).
            done = []
            try:
                for r in rows:
                    if time.time() > t_end or _stop.is_set():
                        return
                    w, h = r["w"] or 2688, r["h"] or 1512
                    lb = D.Letterbox(imgsz, w, h)
                    try:
                        rgb, _ = D.load_rgb(os.path.join(config.FRAMES, r["file"]), lb.sw, lb.sh)
                    except OSError:
                        continue
                    dets = D.decode(det(lb.pad_rgb(rgb)), lb)
                    done.append((r["file"], m["run"], json.dumps(dets), time.time()))
                    if len(done) == 50:
                        write(c, done); done = []
            finally:
                write(c, done)
    except Exception as e:
        state["error"] = str(e); log(f"detector pass: {e}")
    finally:
        state.update(running=False, last_end=time.time(), waiting=waiting())
        log(f"checked {state['done'] - n0} frames on the {state.get('device') or '?'}; not yet checked: "
            + (", ".join(f"{k} {v}" for k, v in state["waiting"].items()) or "nothing"))
