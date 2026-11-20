"""The port training servers talk to (server_port, 8765 by default).

Open:     GET /server/install.sh, GET /server/bundle.tar.gz (code only, no secrets)
Pairing:  POST /api/server/register {code, name, info} -> {id, key}
Keyed (Authorization: Bearer <key>): check-ins, detections, frames and models
to download, job progress and results, and the one-time import."""
import base64, json, os, re, threading, traceback
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

from . import config, collect, db, servers

MAX_BODY = 200 * 1024 * 1024


class Api:
    def __init__(self, detection, log=print):
        self.detection, self.log = detection, log
        api = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def send(self, code, body, ctype="application/json"):
                if isinstance(body, (dict, list)):
                    body = json.dumps(body).encode()
                elif isinstance(body, str):
                    body = body.encode()
                self.send_response(code)
                self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(body)))
                self.end_headers(); self.wfile.write(body)

            def body(self):
                n = int(self.headers.get("Content-Length") or 0)
                if n > MAX_BODY:
                    raise ValueError("too large")
                return self.rfile.read(n) if n else b""

            def server_(self):
                h = self.headers.get("Authorization", "")
                s = servers.auth(h[7:] if h.startswith("Bearer ") else "")
                if not s:
                    raise PermissionError("unknown server key")
                return s

            def route(self, method):
                u = urlparse(self.path); p = u.path; q = {k: v[0] for k, v in parse_qs(u.query).items()}
                if method == "GET" and p == "/server/install.sh":
                    return self.send(200, servers.installer(), "text/x-shellscript")
                if method == "GET" and p == "/server/bundle.tar.gz":
                    return self.send(200, servers.bundle()["data"], "application/gzip")
                if method == "GET" and p == "/server/version":
                    b = servers.bundle(); return self.send(200, {"version": b["version"], "sha256": b["sha256"]})
                if method == "POST" and p == "/api/server/register":
                    b = json.loads(self.body() or b"{}")
                    sid, key = servers.register(b.get("code"), b.get("name"), b.get("info"))
                    api.log(f"training server '{b.get('name')}' paired")
                    return self.send(200, {"id": sid, "key": key, "addon_version": config.version()})
                srv = self.server_()
                if method == "POST" and p == "/api/server/poll":
                    return self.send(200, servers.poll(srv, json.loads(self.body() or b"{}")))
                if method == "POST" and p == "/api/server/detections":
                    b = json.loads(self.body() or b"{}")
                    jpeg = base64.b64decode(b["jpeg"]) if b.get("jpeg") else None
                    api.detection.handle(b["camera"], float(b["ts"]), b.get("run"), b.get("dets") or [], tuple(b["size"]),
                                         f"server {srv['name']}", f"server {srv['name']}, {b.get('device', '?')}", b.get("ms"), jpeg=jpeg)
                    return self.send(200, {"ok": True})
                if method == "GET" and p == "/api/server/frame":
                    r = db.connect().execute("SELECT file FROM frames WHERE file=? AND deleted IS NULL", (q.get("file"),)).fetchone()
                    if not r:
                        return self.send(404, {"error": "no such frame"})
                    with open(os.path.join(config.FRAMES, r["file"]), "rb") as f:
                        return self.send(200, f.read(), "image/jpeg")
                m = re.fullmatch(r"/api/server/model/(\d+)/(best\.pt|model\.xml|model\.bin)", p)
                if method == "GET" and m:
                    path = os.path.join(config.MODELS, m.group(1), m.group(2))
                    if not os.path.exists(path):
                        return self.send(404, {"error": "no such model file"})
                    with open(path, "rb") as f:
                        return self.send(200, f.read(), "application/octet-stream")
                m = re.fullmatch(r"/api/server/jobs/(\d+)/(progress|owl|done|artifact)", p)
                if m:
                    jid, what = int(m.group(1)), m.group(2)
                    if what == "artifact" and method == "PUT":
                        servers.artifact(srv, jid, q.get("name"), self.body())
                    else:
                        b = json.loads(self.body() or b"{}")
                        {"progress": lambda: servers.progress(srv, jid, b), "owl": lambda: servers.owl_rows(srv, jid, b.get("rows") or []),
                         "done": lambda: servers.done(srv, jid, b)}[what]()
                    return self.send(200, {"ok": True})
                if method == "PUT" and p == "/api/server/import/frame":
                    meta = json.loads(q["meta"]); rel = q["file"]
                    if ".." in rel or rel.startswith("/") or not rel.endswith(".jpg"):
                        raise ValueError("bad file name")
                    have = db.connect().execute("SELECT 1 FROM frames WHERE file=?", (rel,)).fetchone()
                    data = self.body()
                    if not have:
                        collect.store(rel, data, meta)
                    return self.send(200, {"ok": True, "stored": not have})
                if method == "POST" and p == "/api/server/import/have":
                    files = json.loads(self.body() or b"{}").get("files") or []
                    c = db.connect(); have = []
                    for i in range(0, len(files), 500):
                        part = files[i:i + 500]
                        have += [r["file"] for r in c.execute(f"SELECT file FROM frames WHERE file IN ({','.join('?' * len(part))})", part)]
                    return self.send(200, {"have": have})
                if method == "POST" and p == "/api/server/import/rows":
                    b = json.loads(self.body() or b"{}")
                    return self.send(200, {"ok": True, "rows": servers.import_rows(b["kind"], b["rows"])})
                if method == "PUT" and p == "/api/server/import/model":
                    servers.import_model(q["run"], q["name"], self.body())
                    return self.send(200, {"ok": True})
                return self.send(404, {"error": "not found"})

            def handle_any(self, method):
                try:
                    self.route(method)
                except PermissionError as e:
                    self.send(401, {"error": str(e)})
                except (ValueError, KeyError) as e:
                    self.send(400, {"error": str(e)})
                except (BrokenPipeError, ConnectionResetError):
                    pass
                except (BrokenPipeError, ConnectionResetError):
                    pass
                except Exception as e:
                    traceback.print_exc(); self.send(500, {"error": str(e)})

            def do_GET(self):
                self.handle_any("GET")

            def do_POST(self):
                self.handle_any("POST")

            def do_PUT(self):
                self.handle_any("PUT")

        port = int(config.options().get("server_port", 8765))
        self.httpd = ThreadingHTTPServer(("0.0.0.0", port), H)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        log(f"training servers: listening on port {port}")
