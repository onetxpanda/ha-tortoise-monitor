#!/usr/bin/env python3
"""A first-pass batch, as its own process on the training server: OWLv2 boxes
for a list of frames ("full": the whole frame; "tiles": 960 px tiles, which
find him when he's small). On a GPU only while nobody else holds one (it
gives the card back the moment anyone asks and carries on on the CPU).
Results go to the add-on every 20 frames.

Usage: tmjob_label.py <job.json>"""
import json, os, sys, time, traceback, urllib.parse

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, HERE)
import tmclient

QUERIES = ["a tortoise", "a turtle", "a person", "a dog", "a cat", "a bird", "a rabbit", "a squirrel", "a raccoon"]
MODEL = "google/owlv2-base-patch16-ensemble"


def main():
    job = json.load(open(sys.argv[1]))
    base, jid, frames = job["base"], job["id"], job["payload"]["frames"]
    client = tmclient.Client(job["addon"], job["key"])
    cache = os.path.join(base, "cache", "frames")
    try:
        import torch
        from PIL import Image
        from transformers import Owlv2Processor, Owlv2ForObjectDetection
        torch.set_num_threads(int(os.environ.get("THREADS", "8")))
        proc = Owlv2Processor.from_pretrained(MODEL)
        model = Owlv2ForObjectDetection.from_pretrained(MODEL).eval()
    except Exception as e:
        traceback.print_exc()
        client.post(f"/api/server/jobs/{jid}/done", {"ok": False, "error": f"could not load OWLv2: {type(e).__name__}: {e}"})
        sys.exit(1)
    dev = "cpu"; card = None; lease = None
    if torch.cuda.is_available():
        from gpu_lease import Opportunistic
        lease = Opportunistic("tortoise first pass")

    def detect(img, x0=0, y0=0):
        inp = proc(text=[QUERIES], images=img, return_tensors="pt").to(dev)
        with torch.no_grad():
            out = model(**inp)
        S = max(img.size)
        r = proc.post_process_grounded_object_detection(out, threshold=0.05, target_sizes=[(S, S)])[0]
        return [[QUERIES[l][2:], round(float(s), 4), [round(float(b[0])) + x0, round(float(b[1])) + y0, round(float(b[2])) + x0, round(float(b[3])) + y0]]
                for s, l, b in zip(r["scores"].cpu(), r["labels"].cpu(), r["boxes"].cpu())]

    def tiles(img, T=960):
        W, H = img.size; xs = [0, (W - T) // 2, W - T]; ys = [0, H - T] if H > T else [0]
        return [d for y in ys for x in xs for d in detect(img.crop((x, y, x + T, y + T)), x, y)]

    rows = []; done = 0; t0 = time.time(); last_try = 0
    try:
        for fr in frames:
            # a free card if there is one; give it back as soon as anyone asks
            if lease:
                if card and lease.asked.is_set():
                    model.to("cpu"); dev = "cpu"; lease.release(); card = None
                if not card and time.time() - last_try > 60:
                    last_try = time.time(); card = lease.try_take()
                    if card:
                        dev = f"cuda:{card[4:]}"; model.to(dev)
            path = os.path.join(cache, fr["file"])
            if not os.path.exists(path):
                os.makedirs(os.path.dirname(path), exist_ok=True)
                data = client.get("/api/server/frame?file=" + urllib.parse.quote(fr["file"]))
                with open(path + ".part", "wb") as f:
                    f.write(data)
                os.replace(path + ".part", path)
            img = Image.open(path).convert("RGB"); t = time.time()
            dets = detect(img) if fr["mode"] == "full" else tiles(img)
            rows.append({"file": fr["file"], "mode": fr["mode"], "size": list(img.size), "secs": round(time.time() - t, 2), "dets": dets})
            done += 1
            if len(rows) >= 20:
                client.post(f"/api/server/jobs/{jid}/owl", {"rows": rows}); rows = []
                client.post(f"/api/server/jobs/{jid}/progress", {"done": done, "total": len(frames), "device": dev,
                                                                 "per_frame_s": round((time.time() - t0) / done, 2)}, tries=1)
        if rows:
            client.post(f"/api/server/jobs/{jid}/owl", {"rows": rows})
        if lease:
            lease.release()
        client.post(f"/api/server/jobs/{jid}/done", {"ok": True, "frames": done, "secs": round(time.time() - t0)})
    except Exception as e:
        traceback.print_exc()
        if lease:
            lease.release()
        try:
            client.post(f"/api/server/jobs/{jid}/done", {"ok": False, "error": f"{type(e).__name__}: {e}"})
        except Exception:
            pass
        sys.exit(1)


if __name__ == "__main__":
    main()
