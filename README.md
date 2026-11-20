# Tortoise Monitor

A Home Assistant add-on that finds the tortoise on Frigate cameras, tells Home
Assistant where he is (in his house, inside the pen walls, outside them), and
manages everything needed to improve the detector: collecting frames from
Frigate, reviewing uncertain ones, the pen outline, and training runs that wait
for approval. Training runs on separate training servers, installed from the
add-on with one line; they are also the main detector, and the add-on detects
by itself (OpenVINO on the Intel GPU) when none is reporting.

Install: Home Assistant → Settings → Add-ons → Add-on store → ⋮ → Repositories,
add `https://github.com/onetxpanda/ha-tortoise-monitor`, then install
**Tortoise Monitor**. Usage and how it works: [tortoise_monitor/DOCS.md](tortoise_monitor/DOCS.md).
