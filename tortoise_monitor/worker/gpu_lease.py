"""GPU cards from the machine's lease arbiter (where there is one, e.g. the
NAS's /run/arbiter/arbiter.sock). The holder is a process: training and
first-pass jobs run as their own processes and take their own leases;
detection runs in the server's main process.

Training (Lease): waits for its card (plain request: the holder, e.g. Basalt,
hands it over once its current work is done), keeps it until the run is
done -- minutes -- reporting how long it still needs after every epoch, and
stops only when forced from the arbiter's page (then queues again and
carries on from its last finished epoch).

Detection and first pass (Opportunistic): take a card only when nobody holds
it, and give it back the moment anyone asks (they carry on on the CPU and
try again later).

No arbiter: training uses its card without a lease and says so; the
opportunistic users treat every card as free."""
import json, os, secrets, threading, time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

from arbiter_client import Arbiter, ArbiterUnavailable, LeaseTimeout, ArbiterError

SOCKET = "/run/arbiter/arbiter.sock"
CARDS = ("cuda1", "cuda0")


class _Holder:
    def __init__(self, label, on_change=None):
        self.label, self.on_change = label, on_change or (lambda st: None)
        self.token = secrets.token_hex(16)
        self.held = []; self.expected = None; self.state = {}
        self.arb = Arbiter(SOCKET)
        self.available = os.path.exists(SOCKET)
        self.asked = threading.Event(); self.forced = threading.Event()
        me = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def reply(self, code, obj):
                b = json.dumps(obj).encode()
                self.send_response(code); self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

            def ok(self):
                if self.headers.get("X-Tortoise-Token") != me.token:
                    self.reply(403, {"error": "bad token"}); return False
                n = int(self.headers.get("Content-Length") or 0)
                if n:
                    self.rfile.read(n)
                return True

            def do_GET(self):
                if self.ok():
                    self.reply(200, {"holding": me.held, "expected_s": me.expected, "label": me.label})

            def do_POST(self):
                if not self.ok():
                    return
                try:
                    req = json.loads(self.headers.get("X-Arbiter-Request") or "{}")
                except ValueError:
                    req = {}
                by = req.get("requested_by") or {}
                who = by.get("label") or by.get("comm") or ("administrator" if by.get("admin") else "another program")
                (me.forced if req.get("force") else me.asked).set()
                me.asked.set()
                me.note(asked_by=who, asked_at=time.time(), forced=bool(req.get("force")))
                self.reply(202, {"accepted": True})

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        port = self.httpd.server_address[1]
        cb = lambda path, method="POST": {"url": f"http://127.0.0.1:{port}{path}", "method": method, "headers": {"X-Tortoise-Token": self.token}}
        self.cbs = {"release": cb("/release"), "status": cb("/status", "GET")}
        if self.available:
            try:
                self.arb.register(label=label, **self.cbs)
            except ArbiterUnavailable:
                self.available = False
            except ArbiterError:
                pass

    def note(self, **kw):
        self.state.update(kw)
        self.on_change(dict(self.state))

    def holders(self):
        if not self.available:
            return {}
        try:
            st = self.arb.state()
        except (ArbiterUnavailable, ArbiterError):
            return {}
        return {c: (v.get("label") or v.get("comm")) for c, v in st.get("resources", {}).items()}

    def release(self):
        try:
            if self.available and self.held:
                self.arb.release()
        except (ArbiterUnavailable, ArbiterError):
            pass
        self.held = []; self.asked.clear(); self.forced.clear()


class Lease(_Holder):
    """For a training run: wait for the card, keep it until done."""

    def acquire(self, prefer=None):
        card = self.held[0] if self.held else (prefer or next((c for c in CARDS if c not in self.holders()), CARDS[0]))
        while True:
            if not self.available:
                self.held = [card]; self.state = {}
                self.note(state="no arbiter", card=card, since=time.time())
                return card
            self.state = {}
            self.note(state="waiting", card=card, holder=self.holders().get(card), since=time.time())
            try:
                self.arb.acquire([card], label=self.label, timeout=3600, expected_s=self.expected, report_expected=True, preempt=False, **self.cbs)
                self.held = [card]; self.state = {}; self.asked.clear(); self.forced.clear()
                self.note(state="holding", card=card, since=time.time())
                return card
            except LeaseTimeout:
                continue
            except ArbiterUnavailable:
                self.available = False
            except ArbiterError:
                time.sleep(10)

    def report(self, expected_s):
        self.expected = expected_s
        if self.available and self.held:
            try:
                self.arb.update(expected_s)
            except (ArbiterUnavailable, ArbiterError):
                pass


class Opportunistic(_Holder):
    """For detection and the first pass: a card only while nobody else wants it."""

    def try_take(self, avoid=()):
        if self.held:
            return self.held[0]
        hold = self.holders()
        free = [c for c in CARDS if c not in hold and c not in avoid]
        if not free:
            return None
        if not self.available:
            self.held = [free[0]]; return free[0]
        try:
            self.arb.acquire([free[0]], label=self.label, timeout=5, preempt=False, **self.cbs)
            self.held = [free[0]]; self.asked.clear()
            return free[0]
        except (LeaseTimeout, ArbiterUnavailable, ArbiterError):
            return None
