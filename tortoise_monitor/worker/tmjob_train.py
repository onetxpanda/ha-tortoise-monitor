#!/usr/bin/env python3
"""One approved training run, as its own process on the training server
(so it holds its own GPU lease). Downloads the run's frames from the add-on
(cached between runs), trains the YOLO detector, measures it on the held-out
frames, exports it for inference -- OpenVINO, half precision, at the
rectangular input the frames need (960x544 for 16:9 at 960) -- and sends the
model and the results back. Progress goes to the add-on after every epoch.

Usage: tmjob_train.py <job.json>"""
import gc, json, os, shutil, statistics, sys, threading, time, traceback
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, HERE)
import tmclient, tmdetect as D


class Yield(Exception):
    """Forced off the card from the arbiter's page: stop now, keep the last checkpoint."""


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
        for split in ("train", "val"):
            for k in ("images", "labels"):
                os.makedirs(f"{data}/{k}/{split}", exist_ok=True)
        for i in items:
            name = os.path.splitext(os.path.basename(i["file"]))[0]
            os.symlink(os.path.join(cache, i["file"]), f"{data}/images/{i['split']}/{name}.jpg")
            with open(f"{data}/labels/{i['split']}/{name}.txt", "w") as o:
                if i["box"]:
                    x0, y0, x1, y1 = i["box"]; W, H = i["w"], i["h"]
                    o.write(f"0 {(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} {(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}\n")
        with open(f"{data}/data.yaml", "w") as o:
            o.write(f"path: {data}\ntrain: images/train\nval: images/val\nnames:\n  0: tortoise\n")
        n_train = sum(1 for i in items if i["split"] == "train")

        from ultralytics import YOLO, settings
        settings.update({"sync": False})
        if gpu:
            import torch
            from gpu_lease import Lease
            if not torch.cuda.is_available():
                raise RuntimeError("this server's PyTorch has no CUDA")
            lease = Lease(f"tortoise detector training (run {p['run']})", lambda st: prog(lease=st))

        marks = {"start": None}; secs = []; maps = []

        def on_start(tr):
            marks["start"] = time.time()

        def on_batch(tr):
            if lease and lease.forced.is_set():
                raise Yield()

        def on_end(tr):
            if not marks["start"]:
                return
            secs.append(time.time() - marks["start"]); marks["start"] = None
            mt = getattr(tr, "metrics", None) or {}
            maps.append(mt.get("metrics/mAP50(B)"))
            log(f"epoch {tr.epoch + 1}/{tr.epochs}: {secs[-1]:.1f} s, mAP50 {maps[-1] or 0:.3f}")
            prog(phase="training", epoch=tr.epoch + 1, epochs=tr.epochs, epoch_secs=secs, map50=maps[-1])
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
                    model.train(data=f"{data}/data.yaml", imgsz=cfg["imgsz"], epochs=cfg["epochs"], batch=cfg["batch"],
                                patience=cfg["patience"], device=dev, workers=8 if gpu else 4, cache="ram", project=out,
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

        prog(phase="testing on held-out frames")
        best = f"{out}/train/weights/best.pt"
        m = YOLO(best)
        tp = fn = fp = tn = 0
        for i in items:
            if i["split"] != "val":
                continue
            r = m.predict(os.path.join(cache, i["file"]), imgsz=cfg["imgsz"], conf=0.1, verbose=False, device=dev)[0]
            top = None
            if len(r.boxes):
                k = int(r.boxes.conf.argmax()); top = (float(r.boxes.conf[k]), [float(v) for v in r.boxes.xyxy[k]])
            hit = top is not None and top[0] >= 0.5
            if i["box"]:
                ix = lambda a, b: max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
                good = hit and ix(top[1], i["box"]) / ((top[1][2] - top[1][0]) * (top[1][3] - top[1][1]) + (i["box"][2] - i["box"][0]) * (i["box"][3] - i["box"][1]) - ix(top[1], i["box"])) >= 0.3
                tp += good; fn += not good
            elif hit:
                fp += 1
            else:
                tn += 1
        mv = m.val(data=f"{data}/data.yaml", imgsz=cfg["imgsz"], device=dev, plots=False, verbose=False, batch=cfg["batch"])

        # for inference: OpenVINO, half precision, at the frames' rectangular input
        prog(phase="exporting for inference")
        fw, fh = max(((i["w"], i["h"]) for i in items), key=lambda s: sum(1 for j in items if (j["w"], j["h"]) == s))
        ih, iw = D.input_hw(cfg["imgsz"], fw, fh)
        xp = YOLO(best).export(format="openvino", imgsz=[ih, iw], half=True, verbose=False)
        xml = next(os.path.join(xp, f) for f in os.listdir(xp) if f.endswith(".xml"))
        if lease:
            lease.release()
        for name, path in (("model.xml", xml), ("model.bin", xml[:-4] + ".bin"), ("best.pt", best)):
            with open(path, "rb") as f:
                client.put(f"/api/server/jobs/{jid}/artifact?name={name}", f.read(), timeout=300)
        res = {"ok": True, "with_him": tp + fn, "found": tp, "without_him": fp + tn, "false_alarms": fp, "conf": 0.5,
               "map50": float(mv.box.map50), "precision": float(mv.box.mp), "recall": float(mv.box.mr),
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
