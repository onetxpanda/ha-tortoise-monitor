"""Logging: each part writes to the add-on's log and keeps its last 200 lines
for its page."""
import collections, threading, time

_bufs = collections.defaultdict(lambda: collections.deque(maxlen=200))
_lock = threading.Lock()


def logger(name):
    def log(msg):
        print(f"[{name}] {msg}", flush=True)
        with _lock:
            _bufs[name].append(time.strftime("%Y-%m-%d %H:%M:%S ") + str(msg))
    return log


def logs(name):
    with _lock:
        return list(_bufs[name])
