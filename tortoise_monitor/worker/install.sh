#!/bin/bash
# Tortoise Monitor training server: installs (or reinstalls) this machine as
# a training server for the add-on at __ADDON_URL__, which serves this script
# and the server's code. Copy the line from the add-on's Servers page:
#
#   curl -fsSL __ADDON_URL__/server/install.sh | sudo bash -s -- <pairing code>
#
# Needs: Linux with systemd, Python 3.10+, ffmpeg (installed with apt if
# missing). With an NVIDIA driver, PyTorch comes with CUDA; otherwise CPU.
# Everything goes in /opt/tortoise-worker; the service is tortoise-worker.
set -euo pipefail
ADDON="__ADDON_URL__"
CODE="${1:-}"
B=/opt/tortoise-worker

[ "$(id -u)" = 0 ] || { echo "Run it as root: ... | sudo bash -s -- <pairing code>"; exit 1; }
[ -n "$CODE" ] || { echo "Missing the pairing code: copy the whole line from the add-on's Servers page."; exit 1; }
command -v systemctl >/dev/null || { echo "This machine has no systemd."; exit 1; }
PY=$(command -v python3 || true)
[ -n "$PY" ] && "$PY" -c 'import sys; sys.exit(sys.version_info < (3, 10))' || { echo "Needs Python 3.10 or newer (python3)."; exit 1; }
if ! command -v ffmpeg >/dev/null || ! "$PY" -c 'import ensurepip' 2>/dev/null; then
  if command -v apt-get >/dev/null; then
    echo "Installing ffmpeg and python3-venv"
    apt-get update -qq && apt-get install -y -qq ffmpeg python3-venv >/dev/null
  else
    echo "Install ffmpeg and Python's venv module, then run this again."; exit 1
  fi
fi

mkdir -p "$B/versions" "$B/cache"
V=$(curl -fsSL "$ADDON/server/version" | "$PY" -c 'import json, sys; print(json.load(sys.stdin)["version"])')
D="$B/versions/$V"
echo "Fetching the server code ($V) from the add-on"
rm -rf "$D.tmp"; mkdir -p "$D.tmp"
curl -fsSL "$ADDON/server/bundle.tar.gz" | tar xz -C "$D.tmp"
rm -rf "$D"; mv "$D.tmp" "$D"

[ -x "$B/venv/bin/python" ] || "$PY" -m venv "$B/venv"
"$B/venv/bin/pip" install -q --upgrade pip
IDX=https://download.pytorch.org/whl/cpu
if command -v nvidia-smi >/dev/null && DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1) && [ -n "$DRV" ]; then
  M=${DRV%%.*}
  if [ "$M" -ge 580 ]; then IDX=https://download.pytorch.org/whl/cu130
  elif [ "$M" -ge 570 ]; then IDX=https://download.pytorch.org/whl/cu128
  elif [ "$M" -ge 560 ]; then IDX=https://download.pytorch.org/whl/cu126
  fi
  echo "NVIDIA driver $DRV: PyTorch from $IDX"
else
  echo "No NVIDIA driver: CPU PyTorch"
fi
"$B/venv/bin/pip" install -q torch torchvision --index-url "$IDX"
"$B/venv/bin/pip" install -q -r "$D/requirements.txt"

ln -sfn "$D" "$B/current"
TM_SERVER_HOME="$B" "$B/venv/bin/python" "$B/current/tortoise_worker.py" register --addon "$ADDON" --code "$CODE" --name "$(hostname)"

# the service, from the server code's template
sed -e "s|__BASE__|$B|g" -e "s|__ADDON__|$ADDON|g" "$D/tortoise-worker.service" > /etc/systemd/system/tortoise-worker.service
systemctl daemon-reload
systemctl enable tortoise-worker.service >/dev/null
systemctl restart tortoise-worker.service
echo "Done: this machine is a training server for the add-on and shows on its Servers page."
echo "Logs: journalctl -u tortoise-worker -f"
