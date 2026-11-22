#!/usr/bin/env python3
"""One approved training run, as its own process on the training server
(so it holds its own GPU lease). Downloads the run's frames from the add-on
(cached between runs), trains the YOLO detector on the training days for
every epoch and keeps the last one, exports it for inference -- OpenVINO,
half precision, at the rectangular input the frames need (960x544 for 16:9
at 960) -- and sends the model back.

Nothing is picked or reported by ultralytics' own validation metrics: an
untrained first epoch that puts a box at confidence 1.00 everywhere scores
near-perfect on them (run 19 was that epoch). Then it scores this
model and the detector in use on the camera's test days, which no run trains
on, with the same OpenVINO code live detection uses, and sends the results.
Progress goes to the add-on after every epoch.

Usage: tmjob_train.py <job.json>"""
import gc, json, os, shutil, statistics, sys, threading, time, traceback
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, HERE)
import tmclient, tmdetect as D


CONF = 0.5          # the detector's "found him", as live detection uses it
HIT_IOU = 0.3       # a found box must overlap the labelled one this much


class Yield(Exception):
    """Forced off the card from the arbiter's page: stop now, keep the last checkpoint."""


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix
    return ix / u if u > 0 else 0.0


def score(xml, imgsz, tests, cache):
    """One detector on the test frames, through tmdetect's OpenVINO path (the
    same letterbox, preprocessing and decoding as live detection)."""
    det = D.OpenVinoDetector(xml, "CPU", "rgb")
    out = {"with_him": 0, "found": 0, "without_him": 0, "false_alarms": 0, "missed": [], "false": []}
    for i in tests:
        lb = D.Letterbox(imgsz, i["w"], i["h"])
        rgb, _ = D.load_rgb(os.path.join(cache, i["file"]), lb.sw, lb.sh)
        dets = D.decode(det(lb.pad_rgb(rgb)), lb)
        top = dets[0] if dets else None            # best first
        hit = top is not None and top[0] >= CONF
        if i["box"]:
            out["with_him"] += 1
            if hit and iou(top[1], i["box"]) >= HIT_IOU:
                out["found"] += 1
            else:
                out["missed"].append(i["file"])
        else:
            out["without_him"] += 1
            if hit:
                out["false_alarms"] += 1; out["false"].append(i["file"])
    return out


def main():
    job = json.load(open(sys.argv[1]))
    base, jid, p = job["base"], job["id"], job["payload"]
    cfg, items = p["config"], p["items"]
    client = tmclient.Client(job["addon"], job["key"])
    out = os.path.join(base, "jobs", str(jid)); os.makedirs(out, exist_ok=True)
    cache = os.path.join(base, "cache", "frames")
    logbuf = []; t_start = time.time()
    P = {"phase": "preparing", "started": t_start, "device": cfg.get("device", "cpu")}
    lock = threading.Lock()

    def log(msg):
        line = time.strftime("%H:%M:%S ") + msg; print(line, flush=True)
        logbuf.append(line); del logbuf[:-60]

    def prog(**kw):
        with lock:
            P.update(kw, at=time.time(), log="\n".join(logbuf))
            try:
                client.post(f"/api/server/jobs/{jid}/progress", P, tries=1, timeout=10)
            except Exception as e:
                print(f"progress: {e}", flush=True)

    gpu = cfg.get("device") == "gpu"
    lease = None
    try:
        # frames, cached by name
        need = [i for i in items if not os.path.exists(os.path.join(cache, i["file"]))]
        log(f"{len(items)} frames, {len(need)} to download")
        prog(phase=f"downloading {len(need)} frames")

        def fetch(i):
            path = os.path.join(cache, i["file"]); os.makedirs(os.path.dirname(path), exist_ok=True)
            data = client.get("/api/server/frame?file=" + __import__("urllib.parse").parse.quote(i["file"]))
            with open(path + ".part", "wb") as f:
                f.write(data)
            os.replace(path + ".part", path)
        with ThreadPoolExecutor(6) as ex:
            list(ex.map(fetch, need))
        data = os.path.join(out, "data"); shutil.rmtree(data, ignore_errors=True)
        for split in ("train", "check"):
            for k in ("images", "labels"):
                os.makedirs(f"{data}/{k}/{split}", exist_ok=True)
        train = [i for i in items if i["split"] != "test"]
        # ultralytics insists on validating once at the end; it gets a few training
        # frames for that, and its numbers are not used
        def link(split, i):
            name = os.path.splitext(os.path.basename(i["file"]))[0]
            os.symlink(os.path.join(cache, i["file"]), f"{data}/images/{split}/{name}.jpg")
            with open(f"{data}/labels/{split}/{name}.txt", "w") as o:
                if i["box"]:
                    x0, y0, x1, y1 = i["box"]; W, H = i["w"], i["h"]
                    o.write(f"0 {(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} {(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}\n")
        for i in train:
            link("train", i)
        for i in train[::max(1, len(train) // 16)][:16]:
            link("check", i)
        with open(f"{data}/data.yaml", "w") as o:
            o.write(f"path: {data}\ntrain: images/train\nval: images/check\nnames:\n  0: tortoise\n")
        n_train = len(train)

        from ultralytics import YOLO, settings
        settings.update({"sync": False})
        if gpu:
            import torch
            from gpu_lease import Lease
            if not torch.cuda.is_available():
                raise RuntimeError("this server's PyTorch has no CUDA")
            lease = Lease(f"tortoise detector training (run {p['run']})", lambda st: prog(lease=st))

        marks = {"start": None}; secs = []

        def on_start(tr):
            marks["start"] = time.time()

        def on_batch(tr):
            if lease and lease.forced.is_set():
                raise Yield()

        def on_end(tr):
            if not marks["start"]:
                return
            secs.append(time.time() - marks["start"]); marks["start"] = None
            log(f"epoch {tr.epoch + 1}/{tr.epochs}: {secs[-1]:.1f} s")
            prog(phase="training", epoch=tr.epoch + 1, epochs=tr.epochs, epoch_secs=secs)
            if lease:
                lease.report(statistics.median(secs[-3:]) * (tr.epochs - tr.epoch - 1) + 60)

        weights = os.path.join(base, "cache", "ultralytics", cfg["model"])
        os.makedirs(os.path.dirname(weights), exist_ok=True)
        last = f"{out}/train/weights/last.pt"
        dev = "cpu"
        while True:
            if lease:
                card = lease.acquire(prefer=os.environ.get("TORTOISE_CARD") or None)
                dev = int(card[4:])
            resume = os.path.exists(last)
            model = YOLO(last if resume else (weights if os.path.exists(weights) else cfg["model"]))
            model.add_callback("on_train_epoch_start", on_start)
            model.add_callback("on_train_batch_end", on_batch)
            model.add_callback("on_fit_epoch_end", on_end)
            prog(phase="loading images")
            try:
                if resume:
                    model.train(resume=True, device=dev)
                else:
                    # every epoch (patience 0: no early stopping), no validation between epochs
                    model.train(data=f"{data}/data.yaml", imgsz=cfg["imgsz"], epochs=cfg["epochs"], batch=cfg["batch"],
                                patience=0, val=False, device=dev, workers=8 if gpu else 4, cache="ram", project=out,
                                name="train", exist_ok=True, seed=0, plots=False, verbose=False, amp=gpu)
                break
            except Yield:
                pass
            del model; gc.collect()
            import torch
            torch.cuda.empty_cache(); marks["start"] = None
            log("forced off the card from the arbiter's page: queueing for it again")
            prog(phase="paused")
            lease.release()

        final = f"{out}/train/weights/last.pt"      # the last epoch (ultralytics' best.pt is picked by its own metrics)

        # for inference: OpenVINO, half precision, at the frames' rectangular input
        prog(phase="exporting for inference")
        fw, fh = max(((i["w"], i["h"]) for i in items), key=lambda s: sum(1 for j in items if (j["w"], j["h"]) == s))
        ih, iw = D.input_hw(cfg["imgsz"], fw, fh)
        xp = YOLO(final).export(format="openvino", imgsz=[ih, iw], half=True, verbose=False)
        xml = next(os.path.join(xp, f) for f in os.listdir(xp) if f.endswith(".xml"))
        if lease:
            lease.release()
        for name, path in (("model.xml", xml), ("model.bin", xml[:-4] + ".bin"), ("best.pt", final)):   # "best.pt": the add-on's name for the PyTorch weights
            with open(path, "rb") as f:
                client.put(f"/api/server/jobs/{jid}/artifact?name={name}", f.read(), timeout=300)

        # this model and the detector in use, on the test days
        tests = [i for i in items if i["split"] == "test"]
        test = None
        if tests:
            prog(phase=f"scoring on {len(tests)} test frames")
            test = {"frames": len(tests)}
            try:
                test["this"] = score(xml, cfg["imgsz"], tests, cache)
                u = p.get("in_use")
                if u:
                    d = os.path.join(out, "in_use"); os.makedirs(d, exist_ok=True)
                    for name in ("model.xml", "model.bin"):
                        with open(os.path.join(d, name), "wb") as f:
                            f.write(client.get(f"/api/server/model/{u['run']}/{name}", timeout=300))
                    test["in_use"] = dict(score(os.path.join(d, "model.xml"), u["imgsz"], tests, cache),
                                          run=u["run"], trained_on_test=u.get("trained_on_test", 0))
                t, w = test["this"], test.get("in_use")
                log(f"test: this run found {t['found']} of {t['with_him']}, false alarms {t['false_alarms']} of {t['without_him']}"
                    + (f"; run {w['run']} (in use) found {w['found']}, false alarms {w['false_alarms']}" if w else ""))
            except Exception as e:
                test["error"] = f"{type(e).__name__}: {e}"
                log("scoring on the test frames failed: " + test["error"])
        t = (test or {}).get("this") or {}
        res = {"ok": True, "with_him": t.get("with_him"), "found": t.get("found"), "without_him": t.get("without_him"),
               "false_alarms": t.get("false_alarms"), "conf": CONF, "test": test,
               "epochs_run": len(secs), "secs": time.time() - t_start, "device": "gpu" if gpu else "cpu", "input": [ih, iw],
               "secs_per_img_epoch": (statistics.median(secs[1:] or secs) / max(1, n_train)) if secs else None,
               "log": "\n".join(logbuf)}
        client.post(f"/api/server/jobs/{jid}/done", res)
        log("done")
    except Exception as e:
        traceback.print_exc()
        if lease:
            lease.release()
        try:
            client.post(f"/api/server/jobs/{jid}/done", {"ok": False, "error": f"{type(e).__name__}: {e}",
                                                        "log": "\n".join(logbuf) + "\n" + traceback.format_exc()[-2500:]})
        except Exception:
            pass
        sys.exit(1)


if __name__ == "__main__":
    main()
