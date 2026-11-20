"""Home Assistant entities over MQTT discovery (the Mosquitto add-on's broker,
whose login the Supervisor hands over). Without a broker everything here is a
no-op and the add-on runs on regardless."""
import json, threading, time

from . import config, ha

PREFIX = "tortoise_monitor"
DEVICE = {"identifiers": [PREFIX], "name": "Tortoise", "manufacturer": "onetxpanda", "model": "Tortoise Monitor add-on"}
# (component, key, discovery fields). Names are short labels; the device name
# "Tortoise" comes first in Home Assistant ("Tortoise In house").
ENTITIES = [
    ("binary_sensor", "in_house", {"name": "In house", "icon": "mdi:home"}),
    ("binary_sensor", "in_yard", {"name": "Seen in yard", "icon": "mdi:grass"}),
    ("binary_sensor", "outside_walls", {"name": "Outside pen walls", "icon": "mdi:fence"}),
    ("sensor", "last_seen", {"name": "Last seen", "device_class": "timestamp"}),
    ("sensor", "last_place", {"name": "Last place", "icon": "mdi:map-marker"}),
    ("image", "last_location", {"name": "Last known location", "content_type": "image/jpeg"}),
    ("sensor", "detector", {"name": "Detection runs on", "entity_category": "diagnostic", "icon": "mdi:chip"}),
    ("sensor", "review_queue", {"name": "Frames to review", "unit_of_measurement": "frames", "state_class": "measurement", "icon": "mdi:image-search"}),
    ("sensor", "disk_used", {"name": "Disk used", "unit_of_measurement": "%", "entity_category": "diagnostic", "icon": "mdi:harddisk"}),
]


class Mqtt:
    def __init__(self, log=print):
        self.log = log; self.client = None; self.connected = False; self.last = {}
        self.lock = threading.Lock()
        threading.Thread(target=self._connect_loop, daemon=True).start()

    def _connect_loop(self):
        while self.client is None:
            svc = ha.mqtt_service()
            if svc:
                try:
                    import paho.mqtt.client as mqtt
                    c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=PREFIX)
                    if svc.get("username"):
                        c.username_pw_set(svc["username"], svc.get("password"))
                    c.will_set(f"{PREFIX}/status", "offline", retain=True)
                    c.on_connect = self._on_connect
                    c.on_disconnect = lambda *a, **k: setattr(self, "connected", False)
                    c.connect_async(svc["host"], int(svc.get("port", 1883)))
                    c.loop_start()
                    self.client = c
                    return
                except Exception as e:
                    self.log(f"MQTT: {e}")
            time.sleep(60)

    def _on_connect(self, client, userdata, flags, reason, props=None):
        self.connected = True
        dev = dict(DEVICE, sw_version=config.version())
        for comp, key, extra in ENTITIES:
            cfg = {"unique_id": f"{PREFIX}_{key}", "default_entity_id": f"{comp}.tortoise_{key}", "device": dev,
                   "availability_topic": f"{PREFIX}/status", **extra}
            if comp == "image":
                cfg["image_topic"] = f"{PREFIX}/{key}/image"
            else:
                cfg["state_topic"] = f"{PREFIX}/{key}/state"
            if key == "last_seen":
                cfg["json_attributes_topic"] = f"{PREFIX}/{key}/attributes"
            client.publish(f"homeassistant/{comp}/{PREFIX}_{key}/config", json.dumps(cfg), retain=True)
        client.publish(f"{PREFIX}/status", "online", retain=True)
        with self.lock:
            for topic, (payload, retain) in self.last.items():      # re-send current states after a reconnect
                client.publish(topic, payload, retain=retain)
        self.log("MQTT connected; entities announced")

    def _pub(self, topic, payload, retain=True):
        with self.lock:
            if self.last.get(topic, (None,))[0] == payload:
                return
            self.last[topic] = (payload, retain)
        if self.client and self.connected:
            self.client.publish(topic, payload, retain=retain)

    def state(self, key, value):
        """value: True/False for binary sensors, None for unknown, anything else as text."""
        payload = "None" if value is None else ("ON" if value is True else "OFF" if value is False else str(value))
        self._pub(f"{PREFIX}/{key}/state", payload)

    def attributes(self, key, attrs):
        self._pub(f"{PREFIX}/{key}/attributes", json.dumps(attrs))

    def image(self, key, jpeg):
        self._pub(f"{PREFIX}/{key}/image", jpeg)
