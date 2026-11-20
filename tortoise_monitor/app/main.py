"""Tortoise Monitor: starts the page (ingress), the training servers' port,
MQTT, live detection and the scheduler."""
import threading, time

from . import collect, config, db, ha, passes, retention, rules, servers
from .detection import Manager
from .logbuf import logger
from .mqtt import Mqtt
from .serverapi import Api
from .web import Site


class Scheduler:
    """collect   hourly, a minute past the hour (Frigate's recordings)
       label     every minute: first-pass batches queued for the servers
       pass      hourly at half past: the detector in use over collected frames
       and, quietly: the in/out record hourly, disk space every 10 minutes,
       run proposals and lost-job checks every minute."""

    def __init__(self, F, mqtt, log):
        self.F, self.mqtt, self.log = F, mqtt, log
        self.run_now = set(); self.next = {}
        threading.Thread(target=self.loop, daemon=True).start()

    def enabled(self, name):
        return not db.kv_get(f"job_off:{name}", False)

    def control(self, name, action):
        if action in ("on", "off"):
            db.kv_set(f"job_off:{name}", True if action == "off" else None)
        else:
            self.run_now.add(name)

    def due(self, name, every, offset=0):
        now = time.time()
        if name in self.run_now:
            self.run_now.discard(name); return True
        if name not in self.next:
            self.next[name] = now - now % every + offset + (every if now % every > offset else 0)
        if now >= self.next[name]:
            self.next[name] += every * (1 + int((now - self.next[name]) // every)); return True
        return False

    def spawn(self, fn):
        threading.Thread(target=fn, daemon=True).start()

    def loop(self):
        time.sleep(5)
        while True:
            try:
                if self.due("collect", 3600, 60) and self.enabled("collect"):
                    if retention.state["collection_paused"]:
                        logger("collect")("skipped: the disk is over the limit even after deleting what can be lost")
                    else:
                        self.spawn(lambda: collect.run_once(logger("collect")))
                if self.due("pass", 3600, 1800) and self.enabled("pass") and not passes.state["running"]:
                    self.spawn(lambda: passes.run_once(logger("pass"), 900))
                if self.due("label", 60) and self.enabled("label"):
                    servers.make_label_jobs(self.F)
                if self.due("record", 3600, 300):
                    self.spawn(lambda: ha.refresh_record(logger("record")))
                if self.due("retention", 600):
                    retention.check(self.F, logger("storage"))
                if self.due("runs", 60):
                    self.F.refresh(); servers.propose_runs(self.F); servers.sweep()
                    c = db.connect()
                    self.mqtt.state("review_queue", sum(k["to_review"] for k in rules.counts(self.F.frames, rules.reviews(c)).values()))
                    if retention.state["used_percent"] is not None:
                        self.mqtt.state("disk_used", retention.state["used_percent"])
            except Exception as e:
                self.log(f"scheduler: {e}")
            time.sleep(5)

    def status(self, label_row):
        import json
        return {
            "collect": {"on": self.enabled("collect"), "next": self.next.get("collect"), "paused": retention.state["collection_paused"], **collect.state},
            "label": {"on": self.enabled("label"), "job": dict(label_row) if label_row else None,
                      "progress": json.loads(label_row["progress"]) if label_row and label_row["progress"] else None},
            "pass": {"on": self.enabled("pass"), "next": self.next.get("pass"), **passes.state, "waiting": passes.waiting()},
        }


def main():
    log = logger("main")
    log(f"Tortoise Monitor {config.version()} starting")
    db.connect()
    F = rules.Frames()
    mqtt = Mqtt(logger("mqtt"))
    det = Manager(mqtt, logger("detection"))
    sched = Scheduler(F, mqtt, log)
    Api(det, logger("servers"))
    Site(F, det, sched, log)
    log("page ready on port 8099 (ingress)")
    retention.check(F, logger("storage"))
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
