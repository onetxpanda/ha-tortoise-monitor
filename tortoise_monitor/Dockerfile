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
COPY worker /app/worker
COPY run.sh /run.sh
RUN chmod +x /run.sh && find /app -name __pycache__ -prune -exec rm -rf {} +

WORKDIR /app
CMD ["/run.sh"]
