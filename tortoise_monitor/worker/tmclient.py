"""Talking to the add-on, for the training server and its jobs."""
import json, time, urllib.error, urllib.request


class Client:
    def __init__(self, addon, key=None, timeout=60):
        self.addon, self.key, self.timeout = addon.rstrip("/"), key, timeout

    def req(self, method, path, body=None, ctype="application/json", timeout=None, tries=3):
        data = body if isinstance(body, (bytes, bytearray)) or body is None else json.dumps(body).encode()
        for i in range(tries):
            r = urllib.request.Request(self.addon + path, data=data, method=method)
            if self.key:
                r.add_header("Authorization", f"Bearer {self.key}")
            if data is not None:
                r.add_header("Content-Type", ctype)
            try:
                with urllib.request.urlopen(r, timeout=timeout or self.timeout) as resp:
                    out = resp.read()
                    return json.loads(out) if resp.headers.get("Content-Type", "").startswith("application/json") else out
            except urllib.error.HTTPError as e:
                msg = e.read().decode(errors="replace")[:300]
                if e.code < 500 or i == tries - 1:
                    raise RuntimeError(f"{method} {path}: {e.code} {msg}")
            except (urllib.error.URLError, OSError) as e:
                if i == tries - 1:
                    raise RuntimeError(f"{method} {path}: {e}")
            time.sleep(2 * (i + 1))

    def get(self, path, **kw):
        return self.req("GET", path, **kw)

    def post(self, path, obj=None, **kw):
        return self.req("POST", path, obj or {}, **kw)

    def put(self, path, data, **kw):
        return self.req("PUT", path, data, "application/octet-stream", **kw)
