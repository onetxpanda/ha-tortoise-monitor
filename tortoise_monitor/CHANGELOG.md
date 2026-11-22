## 2.0.2
- A frame that can't be used now gives no answer instead of "not seen". That is a frame with an error from ffmpeg around it (a broken or reconnecting stream), or one too dark to show anything (mean brightness under 40 of 255, as when the lamp goes off before the camera switches to infrared). His location stays as it was; a camera with no usable frame for a minute goes to unknown, the same as one that stops
- ffmpeg's error output is read as it comes instead of only when ffmpeg exits
- Results: frames not used in each hour, and when and why the last one was not used

## 2.0.1
- Heatmap: only the day's numbers are shown (how often, how long, first and last sighting); the inside/outside split and the notes about the points and the picture are gone

## 2.0.0
- New Heatmap page: where he was on a day, on the yard (or house) camera's picture with the pen walls. As heat, red where he spent the most time, or as dots coloured by time of day. It says how often and for about how long he was seen, when he was first and last seen, and how his time split between inside and outside the pen walls. Pick any day of the last 90; today's fills in as he's seen
- It draws on the sightings the detector has kept since 1.0.0, so the days since the add-on was installed are there already

## 1.3.0
- The page is now a React app, built into the image the same way as the YouTube Live Manager's UI. Data still refreshes every 10 seconds, but only what changed is updated: no more flicker, open drop-downs stay open and pictures don't reload
- The page says when it has lost contact with the add-on and greys out what it last showed, until it is back
- Fix: Review failed to load ("'hint'") since 1.2.0
- Fix: the Storage page listed the old deletion order; it now lists the 1.2 one
- Undo is available on Review when nothing is left to review

## 1.2.2
- Fix: the safety checks were read from only one day of history (Home Assistant's history API stops a day after the start unless told when to end), so none were found; they are now read up to the present

## 1.2.1
- The safety checks are read once at start-up too, and every read is logged (how many checks, how many saw him, or why reading failed)

## 1.2.0
- Training frames come from the safety checks: every 2 minutes the location automation's LLM says whether he is in his house (and, when he isn't, whether he's outside the pen walls). The add-on reads those answers from Home Assistant's history (it changes nothing in the automation), takes the frame collected at each check's moment, and labels it with the answer and the first pass's boxes: him with his box, or not him. In the yard while he's in his house, look-alikes (such as a garden globe) become hard examples. Where answer and boxes disagree, or the check says he's outside the walls, the frame goes to Review
- A run uses your answers and these frames, made diverse: frames with him at most one per hour for each part of the picture he's in, none where he hasn't moved in 2 hours; frames without him, an empty scene once per 20 minutes and each look-alike once per hour; your answers count first. Whole unseen days for testing and the per-day cap stay
- The first pass only labels the frames that are used (each check's frame and Frigate's event snapshots), about a quarter of the collected frames
- Review asks only about those frames (one per camera, hour and reason), newest first
- At 95 % full, frames nothing uses are deleted first
- Options: `check_entity`, `sighting_entity` and `sighting_camera_entity` (the location automation's helpers) replace `record_entity`

## 1.1.0
- Training runs are tested on whole days they never train on (the days with the fewest frames of him, until at least 10 % of them; never over 40 %), so a run's score says how it does on a new day. With only one day of him so far, whole sitting sessions are held out instead, and the run says so
- No single day may outweigh the rest of a training set: each day gives at most as many frames of him as the next-largest day, or a third of them once there are enough days; a capped day keeps frames spread over its hours
- Frames without him are chosen for training and for testing separately, from each side's own days, so the false-alarm figure is measured on real test days
- The run page shows the test days, how many days it trains on, and which days were capped

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
