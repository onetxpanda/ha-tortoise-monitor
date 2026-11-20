# Build the web UI (web/src/app.jsx + React) into one bundle; versions pinned by web/package-lock.json
FROM node:22.12.0-alpine AS webui
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY web/src ./src
RUN npm run build

FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    LIBVA_DRIVER_NAME=iHD

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    # keyframe decoding from Frigate's restream (VA-API on the Intel GPU)
    ffmpeg \
    intel-media-va-driver-non-free \
    vainfo \
    # OpenVINO on the Intel GPU (OpenCL compute runtime)
    intel-opencl-icd \
    ocl-icd-libopencl1 \
    tzdata \
    ca-certificates \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip3 install --break-system-packages --no-cache-dir -r /app/requirements.txt

COPY config.yaml /app/config.yaml
COPY app /app/app
COPY web/index.html /app/app/web/index.html
COPY --from=webui /web/dist/app.js /app/app/web/app.js
COPY worker /app/worker
COPY run.sh /run.sh
RUN chmod +x /run.sh && find /app -name __pycache__ -prune -exec rm -rf {} +

WORKDIR /app
CMD ["/run.sh"]
