"""A reference client for the GPU lease arbiter, standard library only.

Copy it, or import it: a holder acquires before it takes a resource and
releases when it is done. The arbiter identifies the caller by its process,
so this must run in the process that will use the resource.

    from arbiter_client import Arbiter, ArbiterUnavailable

    arbiter = Arbiter()
    try:
        arbiter.acquire(["cuda0", "cuda1"], label="my-service",
                        release={"url": "http://127.0.0.1:8000/release", "method": "POST"},
                        timeout=600)
    except ArbiterUnavailable:
        pass            # no arbiter on this machine: carry on as if granted
    ...
    arbiter.release()

`acquire` blocks until the lease is granted, raises `LeaseTimeout` when the
holder in the way did not give the resource back in time, and raises
`ArbiterUnavailable` when the socket is absent or refuses, which a client
should treat as "granted" so the arbiter is never a single point of failure.
"""
from __future__ import annotations

import http.client
import json
import socket
from typing import Any, Dict, List, Optional

DEFAULT_SOCKET = "/run/arbiter/arbiter.sock"


class ArbiterError(Exception):
    pass


class ArbiterUnavailable(ArbiterError):
    """No arbiter answered: the socket is missing or refused the connection."""


class LeaseTimeout(ArbiterError):
    def __init__(self, holders: List[Dict[str, Any]]):
        super().__init__("timed out waiting for the resource to be released")
        self.holders = holders


class LeaseRefused(ArbiterError):
    pass


class _UnixConnection(http.client.HTTPConnection):
    def __init__(self, path: str, timeout: Optional[float]):
        super().__init__("arbiter", timeout=timeout)
        self._path = path

    def connect(self) -> None:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        if self.timeout is not None:
            sock.settimeout(self.timeout)
        sock.connect(self._path)
        self.sock = sock


class Arbiter:
    def __init__(self, socket_path: str = DEFAULT_SOCKET):
        self.socket_path = socket_path

    def _request(self, method: str, path: str, body: Optional[Dict[str, Any]], timeout: Optional[float]) -> tuple:
        data = json.dumps(body).encode("utf-8") if body is not None else None
        headers = {"Content-Type": "application/json"} if data is not None else {}
        connection = _UnixConnection(self.socket_path, timeout)
        try:
            connection.request(method, path, body=data, headers=headers)
            response = connection.getresponse()
            payload = response.read()
        except (OSError, http.client.HTTPException) as error:
            raise ArbiterUnavailable(str(error)) from error
        finally:
            connection.close()
        try:
            parsed = json.loads(payload.decode("utf-8")) if payload else {}
        except ValueError:
            parsed = {}
        return response.status, parsed

    def health(self) -> Dict[str, Any]:
        status, body = self._request("GET", "/health", None, timeout=5)
        if status != 200:
            raise ArbiterError(body.get("error") or f"health answered {status}")
        return body

    def state(self) -> Dict[str, Any]:
        status, body = self._request("GET", "/state", None, timeout=5)
        if status != 200:
            raise ArbiterError(body.get("error") or f"state answered {status}")
        return body

    def register(
        self, *, label: str, release: Optional[Dict[str, Any]] = None, status: Optional[Dict[str, Any]] = None,
        expected_s: Optional[float] = None, resumable: Optional[bool] = None, resume: Optional[Dict[str, Any]] = None,
        yield_s: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Announce this process before it needs a card: the label it shows
        under, the release callback, and the status URL the arbiter GETs to
        learn what the process holds (`{"holding": [...], "expected_s": N}`),
        which is how a restarted arbiter rebuilds its table.

        A `resumable` process can checkpoint and give its cards up when the
        release callback arrives with `resume: true`, and takes them back
        when the arbiter calls `resume` (the same request shape as
        `release`); `yield_s` is how long it needs from that ask to its
        release (default 120 s). Its status answer then lists the cards it
        expects back under `parked`."""
        body: Dict[str, Any] = {"label": label}
        if release is not None:
            body["release"] = release
        if status is not None:
            body["status"] = status
        if expected_s is not None:
            body["expected_s"] = expected_s
        self._resumable_fields(body, resumable, resume, yield_s)
        code, payload = self._request("POST", "/register", body, timeout=10)
        if code != 200:
            raise LeaseRefused(payload.get("error") or f"register answered {code}")
        return payload

    def update(self, expected_s: Optional[float]) -> bool:
        """Report how long this process expects to keep its cards (None:
        open-ended). True when it holds any."""
        code, payload = self._request("PATCH", "/lease", {"expected_s": expected_s}, timeout=10)
        if code != 200:
            raise LeaseRefused(payload.get("error") or f"update answered {code}")
        return bool(payload.get("holding"))

    def acquire(
        self, resources: List[str], *, label: str = "", release: Optional[Dict[str, Any]] = None, timeout: float = 600,
        status: Optional[Dict[str, Any]] = None, expected_s: Optional[float] = None, report_expected: bool = False,
        resumable: Optional[bool] = None, resume: Optional[Dict[str, Any]] = None, yield_s: Optional[float] = None,
        preempt: bool = True,
    ) -> Dict[str, Any]:
        """Block until `resources` are this process's. `release` is the HTTP
        request the arbiter makes to ask this process to give them back;
        `status` the URL it GETs for what this process holds. `expected_s`
        says how long the cards are kept (None with `report_expected` means
        open-ended; left out means unchanged). `resumable`/`resume`/`yield_s`
        as in `register`. A holder in the way that is resumable is asked to
        yield -- it gets the cards back when this process releases them --
        unless `preempt` is False, which asks plainly and waits."""
        body: Dict[str, Any] = {"resources": list(resources), "label": label, "timeout": timeout}
        if release is not None:
            body["release"] = release
        if status is not None:
            body["status"] = status
        if report_expected or expected_s is not None:
            body["expected_s"] = expected_s
        self._resumable_fields(body, resumable, resume, yield_s)
        if not preempt:
            body["preempt"] = False
        # The HTTP wait outlives the lease timeout by a margin for the round trip.
        status, payload = self._request("POST", "/lease", body, timeout=timeout + 15)
        if status == 200 and payload.get("granted"):
            return payload
        if status == 504:
            raise LeaseTimeout(payload.get("holders") or [])
        raise LeaseRefused(payload.get("error") or f"lease answered {status}")

    @staticmethod
    def _resumable_fields(body: Dict[str, Any], resumable: Optional[bool], resume: Optional[Dict[str, Any]], yield_s: Optional[float]) -> None:
        if resumable is not None:
            body["resumable"] = resumable
        if resume is not None:
            body["resume"] = resume
        if yield_s is not None:
            body["yield_s"] = yield_s

    def release(self, resources: Optional[List[str]] = None) -> List[str]:
        """Give back `resources`, or everything this process holds."""
        body: Dict[str, Any] = {} if resources is None else {"resources": list(resources)}
        status, payload = self._request("DELETE", "/lease", body, timeout=10)
        if status != 200:
            raise ArbiterError(payload.get("error") or f"release answered {status}")
        return list(payload.get("released") or [])
