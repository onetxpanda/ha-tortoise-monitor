# Tortoise Monitor

## What it does

Finds the tortoise on two Frigate cameras — the one on his house and the one
on the yard — and tells Home Assistant where he is: in his house, inside the
pen walls, or outside them. It also keeps everything needed to make the
detector better: frames from Frigate, a review page for the frames it can't
settle, the pen outline, and training runs that wait for your approval.

Training itself happens on a separate **training server** (any Linux machine,
with an NVIDIA GPU if it has one). The training server is also the main
detector; when it is not reporting, the add-on runs the detector itself.

## How to use it

1. Install the add-on and open it from the sidebar.
2. **Training servers → Add a training server**: copy the line and run it as
   root on the machine that should train. It installs itself, pairs with the
   add-on and shows up on that page.
3. **Pen outline**: click round the pen. Far and side walls: along the line
   where the wall meets the ground on the inside. Near wall (seen from
   outside, at the bottom of the picture): along where it meets the ground on
   the camera's side, so the whole board is inside. He counts as inside when
   the bottom-centre of his box (his feet) is inside the outline.
4. **Review**: answer the frames the add-on is unsure about ("That's him",
   "Not in this frame", or drag a box around him).
5. **Training**: approve a proposed run (CPU or GPU). When it has finished,
   press **Use this detector**.
6. **Heatmap**: where he was on a day, on the yard (or house) camera's
   picture: as heat (red where he spent the most time) or as dots coloured by
   time of day, with the pen walls. Pick the day at the top; today's fills in
   as he's seen.

Entities (device "Tortoise", over MQTT):

| Entity | Meaning |
|---|---|
| `binary_sensor.tortoise_in_house` | the house camera sees him |
| `binary_sensor.tortoise_in_yard` | the yard camera sees him |
| `binary_sensor.tortoise_outside_walls` | the yard camera sees him outside the pen outline |
| `sensor.tortoise_last_seen`, `sensor.tortoise_last_place` | when and where he was last seen |
| `image.tortoise_last_location` | the last frame he was seen in, with the outline and his box |

A camera whose detection isn't running shows its entities as unknown.

## How it works

- **Collection**: hourly, frames from Frigate's recordings through Frigate's
  API, plus frames and snapshots from Frigate's events (other animals and
  people, as examples of "not him").
- **Training frames come from the safety checks**: every 2 minutes the
  location automation asks the local LLM whether he is in his house (and, when
  he isn't, whether he is outside the pen walls). The add-on reads those
  answers from Home Assistant's history, takes the frame collected at each
  check's moment, and a training server puts OWLv2 boxes on it (the first
  pass). Answer and boxes together label the frame: him with his box, or not
  him (in the yard while he's in his house, look-alikes such as a garden globe
  become hard examples). Where they disagree, or the check says he is outside
  the walls, the frame goes to Review. A run uses your answers and these
  frames, made diverse: frames with him at most one per hour for each part of
  the picture he's in (none where he hasn't moved in 2 hours), frames without
  him one per hour; it is tested on whole days it never trains on, and no day
  may outweigh the rest.
- **Training**: YOLO11n, on a training server, after your approval. The model
  is sent back for OpenVINO (half precision, 960×544 for the yard).
- **Detection**: keyframes from Frigate's restream (never the cameras
  directly), every few seconds. In the add-on they are decoded and scaled on
  the Intel GPU and the detector runs there with OpenVINO, else on the CPU.
- **Heatmap**: every time the detector sees him, his foot point (the bottom
  middle of his box) is kept with the time and place, for 90 days. Sightings
  come every few seconds while he's in view, so the heat shows time spent; it
  is scaled to the busiest spots so one long stay doesn't hide the rest. The
  picture behind it is that day's frame nearest 1 pm.
- **Training servers** check in every 5 seconds. They update only from the
  add-on: when Home Assistant has updated the add-on, each server installs the
  add-on's copy of its code between jobs and restarts. A server that stops
  checking in shows as offline; one that fails after an update says so.
- **Disk**: images live in `/media/tortoise_monitor`. At 95 % full (option
  `prune_at_percent`; measured as Home Assistant's Storage page shows it,
  used ÷ total) the oldest frames that can be lost go first: clear
  "not him" frames, frames never labelled after a week, unreviewed uncertain
  frames, then clear "him" frames no run used. Frames you answered and frames
  a run used are never deleted. If that isn't enough, collection pauses.

## Options

| Option | |
|---|---|
| `frigate_url` | Frigate's API |
| `restream_url` | Frigate's restream, for the add-on's own detection |
| `server_restream_url` | the restream as training servers reach it (empty: `rtsp://<this host>:8554`) |
| `advertise_url` | how training servers reach the add-on (empty: `http://<this host>:8765`) |
| `check_entity`, `sighting_entity`, `sighting_camera_entity` | the location automation's helpers: set at every safety check, set when a check saw him, and which camera saw him |
| `detect_every_s` | seconds between detections per camera |
| `prune_at_percent` | disk use at which old frames are deleted |
| `cameras` | Frigate camera names, each with its role (`yard` or `house`), collection interval and model size |
