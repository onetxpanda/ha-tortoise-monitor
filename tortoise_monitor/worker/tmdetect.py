"""The detector at inference time, shared by the add-on and the training
server: the same trained YOLO weights, run three ways --

  OpenVINO on a CPU            (the server's default; the add-on's fallback)
  OpenVINO on an Intel GPU     (the add-on's fallback on the N100)
  PyTorch on an NVIDIA GPU     (the server's GPU option)

-- with one preprocessing and one postprocessing, so all three give the same
boxes. Frames come from Frigate's restream through ffmpeg, which decodes only
keyframes and scales them to the model's input size (on the GPU with VA-API
where there is one), so no full-size frame ever reaches Python.

Input shape: the long side `imgsz`, the short side scaled to match and padded
to a multiple of 32 (2688x1512 -> 960x540 + 4 rows -> 960x544). That is the
shape ultralytics itself uses for a single image, and the shape the models are
exported at: no square padding to compute."""
import collections, math, subprocess, threading, time

import numpy as np

STRIDE = 32
PAD = 114            # ultralytics' letterbox grey


def input_hw(imgsz, w, h):
    """Model input (height, width) for w x h frames."""
    if w >= h:
        return int(math.ceil(imgsz * h / w / STRIDE) * STRIDE), imgsz
    return imgsz, int(math.ceil(imgsz * w / h / STRIDE) * STRIDE)


def scaled_wh(imgsz, w, h):
    """The frame scaled to fit the input, before padding."""
    r = imgsz / max(w, h)
    return int(round(w * r)), int(round(h * r))


class Letterbox:
    """Where a scaled frame sits in the padded input (centred, as ultralytics does)."""

    def __init__(self, imgsz, w, h):
        self.w, self.h = w, h
        self.ih, self.iw = input_hw(imgsz, w, h)
        self.sw, self.sh = scaled_wh(imgsz, w, h)
        self.r = self.sw / w
        self.left = (self.iw - self.sw) // 2
        self.top = (self.ih - self.sh) // 2

    def pad_rgb(self, rgb):
        """rgb: (sh, sw, 3) uint8 -> (1, ih, iw, 3) uint8."""
        out = np.full((1, self.ih, self.iw, 3), PAD, np.uint8)
        out[0, self.top:self.top + self.sh, self.left:self.left + self.sw] = rgb
        return out

    def pad_nv12(self, nv12):
        """nv12: flat (sw * sh * 3/2) bytes -> (1, ih * 3/2, iw, 1) uint8 (grey padding)."""
        sw, sh, iw, ih = self.sw, self.sh, self.iw, self.ih
        y = np.frombuffer(nv12, np.uint8, sw * sh).reshape(sh, sw)
        uv = np.frombuffer(nv12, np.uint8, sw * sh // 2, sw * sh).reshape(sh // 2, sw)
        out = np.empty((ih * 3 // 2, iw), np.uint8)
        out[:ih] = PAD; out[ih:] = 128
        out[self.top:self.top + sh, self.left:self.left + sw] = y
        t2, l2 = self.top // 2, self.left - self.left % 2
        out[ih + t2:ih + t2 + sh // 2, l2:l2 + sw] = uv
        return out.reshape(1, ih * 3 // 2, iw, 1)

    def to_frame(self, xyxy):
        """Boxes in input pixels -> frame pixels."""
        b = (np.asarray(xyxy, np.float32) - [self.left, self.top, self.left, self.top]) / self.r
        b[:, [0, 2]] = b[:, [0, 2]].clip(0, self.w)
        b[:, [1, 3]] = b[:, [1, 3]].clip(0, self.h)
        return b


def nms(boxes, scores, iou=0.7):
    order = scores.argsort()[::-1]; keep = []
    while order.size:
        i = order[0]; keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0]); yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2]); yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        a = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        b = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (boxes[order[1:], 3] - boxes[order[1:], 1])
        order = order[1:][inter / (a + b - inter + 1e-9) <= iou]
    return keep


def decode(out, lb, conf=0.1, iou=0.7, max_det=20):
    """Raw head output (1, 4 + classes, anchors): centre-x, centre-y, width,
    height in input pixels, then class scores -> [[score, [x0, y0, x1, y1]], ...]
    in frame pixels, best first (one class: the tortoise)."""
    p = np.asarray(out, np.float32)[0]
    scores = p[4:].max(0)
    m = scores > conf
    if not m.any():
        return []
    cx, cy, w, h = p[0, m], p[1, m], p[2, m], p[3, m]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
    s = scores[m]
    keep = nms(boxes, s, iou)[:max_det]
    fb = lb.to_frame(boxes[keep])
    return [[round(float(s[k]), 4), [round(float(v), 1) for v in fb[j]]] for j, k in enumerate(keep)]


class OpenVinoDetector:
    """OpenVINO, on "CPU" or "GPU" (an Intel GPU). Preprocessing (colour
    conversion, uint8 -> float, /255, layout) is built into the compiled model,
    so it runs on the device. kind: "rgb" ((1, h, w, 3) uint8) or "nv12"
    (single-plane NV12, as VA-API hands it over)."""

    def __init__(self, xml, device="CPU", kind="rgb", cache_dir=None, threads=None):
        import openvino as ov
        from openvino.preprocess import PrePostProcessor, ColorFormat
        core = ov.Core()
        if cache_dir:
            core.set_property({"CACHE_DIR": cache_dir})       # compiled kernels are reused across restarts
        model = core.read_model(xml)
        _, _, self.ih, self.iw = [int(d) for d in model.input().get_partial_shape().to_shape()]
        ppp = PrePostProcessor(model)
        t = ppp.input().tensor().set_element_type(ov.Type.u8)
        if kind == "nv12":
            t.set_color_format(ColorFormat.NV12_SINGLE_PLANE)
            ppp.input().preprocess().convert_color(ColorFormat.RGB)
        else:
            t.set_layout(ov.Layout("NHWC")).set_color_format(ColorFormat.RGB)
        ppp.input().preprocess().convert_element_type(ov.Type.f32).scale(255.0)
        ppp.input().model().set_layout(ov.Layout("NCHW"))
        cfg = {"PERFORMANCE_HINT": "LATENCY"}
        if threads and device == "CPU":
            cfg["INFERENCE_NUM_THREADS"] = int(threads)
        self.compiled = core.compile_model(ppp.build(), device, cfg)
        self.req = self.compiled.create_infer_request()
        self.device, self.kind = device, kind

    def __call__(self, x):
        self.req.infer({0: x})
        return self.req.get_output_tensor(0).data


class TorchDetector:
    """PyTorch on an NVIDIA GPU (the server's GPU option): the trained .pt,
    fused, in half precision, fed the same padded RGB input."""

    def __init__(self, pt, device="cuda:0"):
        import torch
        from ultralytics import YOLO
        self.torch = torch
        self.dev = torch.device(device)
        self.model = YOLO(pt).model.fuse().eval().half().to(self.dev)
        self.device, self.kind = device, "rgb"

    def __call__(self, x):
        t = self.torch
        with t.inference_mode():
            inp = t.from_numpy(x).to(self.dev).permute(0, 3, 1, 2).half().div_(255)
            out = self.model(inp)
            out = out[0] if isinstance(out, (list, tuple)) else out
            return out.float().cpu().numpy()


# A frame that can't be used gives no answer (like a camera that is down), never "not seen".
DARK_LUMA = 40.0     # mean brightness (0-255) below which a frame shows nothing. Normal frames,
                     # infrared nights included, are 59 and up; the frame taken as the lamp went
                     # off before the camera switched to infrared was 23.
ERR_GRACE_S = 0.3    # ffmpeg's error lines can trail the frame they belong to


def frame_luma(kind, buf, w, h):
    """Mean brightness (0-255) of a reader frame, from every 4th row and column."""
    if kind == "nv12":
        y = np.frombuffer(buf, np.uint8, w * h).reshape(h, w)[::4, ::4]
        return (float(y.mean()) - 16.0) * 255.0 / 219.0       # video range Y to 0-255
    a = np.frombuffer(buf, np.uint8).reshape(h, w, 3)[::4, ::4]
    m = a.reshape(-1, 3).mean(axis=0)
    return float(0.299 * m[0] + 0.587 * m[1] + 0.114 * m[2])


class KeyframeReader(threading.Thread):
    """One camera's restream through ffmpeg: keyframes only, scaled to the
    model's input (before padding), newest one kept. hwaccel="vaapi" decodes
    and scales on an Intel GPU and hands over NV12; otherwise the CPU decodes
    and hands over RGB. ffmpeg's error output is read as it comes: a frame
    with an error around it is reported as unusable."""

    def __init__(self, url, sw, sh, hwaccel=None, log=print):
        super().__init__(daemon=True)
        self.url, self.sw, self.sh, self.hwaccel, self.log = url, sw, sh, hwaccel, log
        self.kind = "nv12" if hwaccel == "vaapi" else "rgb"
        self.size = sw * sh * 3 // 2 if self.kind == "nv12" else sw * sh * 3
        self.latest = None                     # (time, bytes, time of the frame before it or of ffmpeg's start)
        self.errors = collections.deque(maxlen=200)   # (time, line) from ffmpeg's error output
        self.cond = threading.Condition()
        self.stopping = False
        self.proc = None
        self.frames = 0
        self._logged = 0.0; self._quiet = 0

    def cmd(self):
        c = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-rtsp_transport", "tcp", "-skip_frame", "nokey"]
        if self.hwaccel == "vaapi":
            c += ["-hwaccel", "vaapi", "-hwaccel_device", "/dev/dri/renderD128", "-hwaccel_output_format", "vaapi"]
            vf = f"scale_vaapi=w={self.sw}:h={self.sh}:format=nv12,hwdownload,format=nv12"
            pix = "nv12"
        else:
            vf = f"scale={self.sw}:{self.sh}:flags=bilinear"
            pix = "rgb24"
        return c + ["-i", self.url, "-an", "-fps_mode", "passthrough", "-vf", vf, "-f", "rawvideo", "-pix_fmt", pix, "pipe:1"]

    def _errors(self, proc):
        """Reads ffmpeg's error output while it runs; logs at most one line per 30 s."""
        for raw in iter(proc.stderr.readline, b""):
            line = raw.decode(errors="replace").strip()
            if not line:
                continue
            now = time.time()
            with self.cond:
                self.errors.append((now, line))
            if now - self._logged >= 30:
                more = f" (and {self._quiet} more lines)" if self._quiet else ""
                self.log(f"restream {self.url}: {line[:300]}{more}")
                self._logged = now; self._quiet = 0
            else:
                self._quiet += 1

    def run(self):
        backoff = 2
        while not self.stopping:
            try:
                self.proc = subprocess.Popen(self.cmd(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
                drain = threading.Thread(target=self._errors, args=(self.proc,), daemon=True); drain.start()
                prev = time.time()
                while not self.stopping:
                    buf = bytearray(); need = self.size
                    while need:
                        chunk = self.proc.stdout.read(need)
                        if not chunk:
                            raise EOFError
                        buf += chunk; need -= len(chunk)
                    now = time.time()
                    with self.cond:
                        self.latest = (now, bytes(buf), prev); self.frames += 1
                        self.cond.notify_all()
                    prev = now
                    backoff = 2
            except Exception:
                if not self.stopping:
                    self.log(f"restream {self.url}: ended; retrying in {backoff} s")
            finally:
                if self.proc:
                    self.proc.kill(); self.proc.wait()
            if not self.stopping:
                time.sleep(backoff); backoff = min(backoff * 2, 60)

    def unusable(self, ts, buf, prev):
        """Why a frame can't be used, or None: ffmpeg reported an error between
        the frame before it (or ffmpeg's start) and just after it, or it is too
        dark to show anything."""
        with self.cond:
            errs = [line for t, line in self.errors if prev < t <= ts + ERR_GRACE_S]
        if errs:
            return f"stream error: {errs[-1][:200]}"
        luma = frame_luma(self.kind, buf, self.sw, self.sh)
        if luma < DARK_LUMA:
            return f"too dark to see ({luma:.0f} of 255)"
        return None

    def newest(self, after=0.0, timeout=10.0):
        """The newest frame taken after `after` (wall clock), waiting up to
        timeout: (time, bytes, unusable), where unusable is None or why the
        frame can't be used."""
        end = time.time() + timeout
        with self.cond:
            while not (self.latest and self.latest[0] > after):
                left = end - time.time()
                if left <= 0:
                    return None
                self.cond.wait(left)
            ts, buf, prev = self.latest
        wait = ts + ERR_GRACE_S - time.time()
        if wait > 0:
            time.sleep(wait)
        return ts, buf, self.unusable(ts, buf, prev)

    def stop(self):
        self.stopping = True
        if self.proc:
            self.proc.kill()


def nv12_to_rgb(nv12, w, h):
    """For snapshots only (detection never needs it): NV12 -> (h, w, 3) uint8, BT.601."""
    y = np.frombuffer(nv12, np.uint8, w * h).reshape(h, w).astype(np.float32)
    uv = np.frombuffer(nv12, np.uint8, w * h // 2, w * h).reshape(h // 2, w // 2, 2).astype(np.float32) - 128
    uv = uv.repeat(2, 0).repeat(2, 1)
    c = (y - 16) * 1.164
    r = c + 1.596 * uv[..., 1]; g = c - 0.392 * uv[..., 0] - 0.813 * uv[..., 1]; b = c + 2.017 * uv[..., 0]
    return np.stack([r, g, b], -1).clip(0, 255).astype(np.uint8)


def load_rgb(path, sw, sh):
    """A JPEG file scaled to (sw, sh): libjpeg decodes it at 1/2, 1/4 or 1/8
    size first (draft mode), so the full frame is never decoded."""
    from PIL import Image
    im = Image.open(path)
    full = im.size
    im.draft("RGB", (sw, sh))
    im = im.convert("RGB")
    if im.size != (sw, sh):
        im = im.resize((sw, sh), Image.BILINEAR)
    return np.asarray(im), full
