"""Disk space. When the disk holding the media folder reaches prune_at_percent
(95 %), delete what can be lost, oldest first, one tier at a time, until it is
1 % below. Never deleted: frames you answered in Review, frames any training
run used, models, the database.

Tiers, cheapest first:
  1 nothing there      frames clear without review as "not him", unused by any run
  2 never labelled     frames still waiting for the first pass after 7 days
  3 not reviewed       uncertain frames you haven't reviewed (oldest first)
  4 him, unused        frames clear without review as "him", unused by any run
  5 old detections     production detection rows older than 90 days
If all of that is gone and the disk is still over the limit, collection stops
until there is room again (shown on the Storage page)."""
import os, shutil, time

from . import config, db, rules

state = {"used_percent": None, "total": None, "free": None, "ours": None, "collection_paused": False, "last_check": None, "last_pruned": None}


def usage():
    """Disk use as Home Assistant's Storage page shows it: used / total. (The
    filesystem's root reserve, about 5 %, counts as free, so at 95 % only the
    reserve is left.)"""
    st = os.statvfs(config.MEDIA)
    total = st.f_blocks * st.f_frsize; free = st.f_bfree * st.f_frsize
    return 100.0 * (total - free) / total, total, free


def ours():
    c = db.connect()
    return c.execute("SELECT COALESCE(SUM(bytes), 0) FROM frames WHERE deleted IS NULL").fetchone()[0]


def check(frames_cache, log=print):
    limit = float(config.options().get("prune_at_percent", 95))
    used, total, free = usage()
    state.update(used_percent=round(used, 1), total=total, free=free, ours=ours(), last_check=time.time())
    if used < limit:
        state["collection_paused"] = False
        return
    target = limit - 1.0
    c = db.connect()
    frames_cache.refresh(force=True)
    F = frames_cache.frames
    revs = rules.reviews(c); ctx = rules.context(revs, F)
    keep = rules.protected_files(c)
    tiers = {"nothing there": [], "never labelled": [], "not reviewed": [], "him, unused": []}
    week = time.time() - 7 * 86400
    for r in c.execute("SELECT file, ts, added FROM frames WHERE deleted IS NULL ORDER BY ts"):
        f = r["file"]
        if f in keep:
            continue
        x = F.get(f)
        if x is None:
            if r["added"] < week:
                tiers["never labelled"].append(f)
            continue
        lab, _, src = rules.final(x, revs.get(f), ctx)
        if lab == "neg" and src != "you":
            tiers["nothing there"].append(f)
        elif lab == "uncertain":
            tiers["not reviewed"].append(f)
        elif lab == "pos" and src in ("auto", "same"):
            tiers["him, unused"].append(f)
    for tier, files in tiers.items():
        n = b = 0
        for i, f in enumerate(files):
            p = os.path.join(config.FRAMES, f)
            try:
                sz = os.path.getsize(p); os.remove(p); b += sz
            except OSError:
                pass
            c.execute("UPDATE frames SET deleted=?, deleted_why=? WHERE file=?", (time.time(), tier, f)); n += 1
            if i % 100 == 99:
                c.commit()
                if usage()[0] < target:
                    break
        c.commit()
        if n:
            c.execute("INSERT INTO deletions VALUES (?,?,?,?)", (time.time(), tier, n, b)); c.commit()
            log(f"disk over {limit:.0f} %: deleted {n} frames ({b / 1e6:.0f} MB), {tier}")
            state["last_pruned"] = time.time()
        if usage()[0] < target:
            break
    if usage()[0] >= target:
        n = c.execute("DELETE FROM detections WHERE ts < ?", (time.time() - 90 * 86400,)).rowcount; c.commit()
        if n:
            c.execute("INSERT INTO deletions VALUES (?,?,?,?)", (time.time(), "old detections", n, 0)); c.commit()
    used, total, free = usage()
    state.update(used_percent=round(used, 1), free=free, ours=ours(), collection_paused=used >= limit)
    if used >= limit:
        log(f"disk still at {used:.1f} % after deleting everything that can be lost: collection paused")
