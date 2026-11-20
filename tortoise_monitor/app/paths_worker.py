"""The detector module (tmdetect) lives with the training server's code, which
this add-on also serves to the servers; put it on the import path."""
import sys

from . import config

if config.WORKER_DIR not in sys.path:
    sys.path.insert(0, config.WORKER_DIR)
