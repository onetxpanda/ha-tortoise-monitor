## 1.0.2
- Training servers: the planned exit after installing an update (code 75) is a normal exit for systemd, so the service log no longer reports "Failed" on every update; servers pick up the changed unit with this update

## 1.0.1
- Fix: disk use is measured the way Home Assistant's Storage page shows it (used ÷ total). 1.0.0 also counted the disk's root reserve (about 5 %) as used, saw 96.8 % where Home Assistant shows 91 %, and paused collection

## 1.0.0
- First version: collection from Frigate, review of uncertain frames, the pen outline, training runs that wait for your approval, and live detection with entities in Home Assistant over MQTT (in house, seen in yard, outside pen walls, last seen, last place, last known location picture)
- Training servers: installed with one line from the Servers page, paired with a one-time code; they do the first pass (OWLv2) and training (CPU or GPU, sharing GPUs through the machine's lease arbiter when it has one), and are the main detector (CPU by default, or GPU)
- When no training server reports detections, the add-on detects by itself: keyframes decoded and scaled on the Intel GPU (VA-API), the detector run with OpenVINO on the Intel GPU, or the CPU
- Training servers update from the add-on (never from GitHub): when the add-on is newer, a server installs the add-on's copy of its code between jobs and restarts into it
- Disk: at 95 % full the oldest frames that can be lost are deleted (never frames you answered or that a run used); collection pauses if that isn't enough
