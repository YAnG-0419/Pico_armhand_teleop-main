"""The teleop operator's control backend: console state + JSON-TCP server.

One unified backend, swappable frontends. `OperatorConsole` is the single
activation/request/status object the input sources and the control loop
share; `OperatorControlServer` exposes it over line-delimited JSON
({"id", "command", "arguments"} answered by {"id", "ok",
"result"|"error"}, gluon's demonstration-server protocol). The PySide6
GUI in apps/operator_gui is a shell over this protocol and is the
operator frontend; the process itself is headless.

The server thread never touches robot state directly. Commands mutate the
console's activation and one-shot request flags; the 100 Hz loop reads
them at its own pace, exactly as it read the old keyboard, and `status`
returns the loop's latest published snapshot.
"""

from __future__ import annotations

import json
import socketserver
import threading
import time

from .types import SIDES


class OperatorConsole:
    """Thread-safe operator state shared by the GUI and coordinator.

    State arrives from the control server, and feedback accumulates in a ring
    the `status` command serves to frontends.
    """

    def __init__(self) -> None:
        self.sides = SIDES
        self.active = {side: False for side in SIDES}
        self.hand_active = {side: False for side in SIDES}
        self.requests = {
            "open_hands": False,
            "open_left_hand": False,
            "open_right_hand": False,
            "reset": False,
            "reset_left": False,
            "reset_right": False,
            "capture_home": False,
            "start_dataset_recording": False,
            "stop_dataset_recording": False,
        }
        self._status_line = "starting..."
        self._dataset_recording = {
            "configured": False,
            "active": False,
            "finalizing": False,
            "path": "",
            "sample_count": 0,
        }
        self._feedback: list[str] = []
        self._lock = threading.Lock()

    # -- operator-state interface ----------------------------------------
    # Every mutation happens under the lock: the server thread writes while
    # the 100 Hz control loop reads, and the keyboard-era code was safe only
    # because both sides ran on one thread. Without the lock a request set
    # between take_requests' read and swap lands in the discarded dict and
    # the operator's click is silently lost.

    def poll(self) -> dict[str, bool]:
        with self._lock:
            return dict(self.active)

    def poll_hands(self) -> dict[str, bool]:
        with self._lock:
            return dict(self.hand_active)

    def take_requests(self) -> dict[str, bool]:
        with self._lock:
            taken = self.requests
            self.requests = {name: False for name in taken}
        return taken

    def set_active(
        self, side: str, engaged: bool, component: str = "both"
    ) -> None:
        if side not in self.sides:
            raise ValueError(f"{side} is not configured for this run")
        if component not in ("arm", "hand", "both"):
            raise ValueError("component must be arm, hand, or both")
        with self._lock:
            if component in ("arm", "both"):
                self.active[side] = engaged
            if component in ("hand", "both"):
                self.hand_active[side] = engaged

    def request(self, name: str) -> None:
        with self._lock:
            if name not in self.requests:
                raise ValueError(f"Unknown request: {name}")
            self.requests[name] = True

    def disable_all(self, reason: str) -> None:
        with self._lock:
            for side in SIDES:
                self.active[side] = False
                self.hand_active[side] = False
        self.show(f"all sides disengaged: {reason}")

    def deny(self, side: str, reason: str) -> None:
        with self._lock:
            self.active[side] = False
        self.show(f"{side}: {reason}")

    def show(self, message: str) -> None:
        with self._lock:
            self._feedback.append(time.strftime("%H:%M:%S ") + str(message))
            del self._feedback[:-50]

    def set_status(self, text: str) -> None:
        with self._lock:
            self._status_line = str(text)

    def set_dataset_recording(self, **status) -> None:
        with self._lock:
            self._dataset_recording.update(status)

    def close(self) -> None:
        pass

    # -- frontend-facing snapshot ----------------------------------------

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "status_line": self._status_line,
                "feedback": list(self._feedback),
                "active": dict(self.active),
                "arm_active": dict(self.active),
                "hand_active": dict(self.hand_active),
                "sides": list(self.sides),
                "dataset_recording": dict(self._dataset_recording),
            }


class _RequestHandler(socketserver.StreamRequestHandler):
    def setup(self) -> None:
        super().setup()
        self.server.client_connected()

    def finish(self) -> None:
        try:
            super().finish()
        finally:
            self.server.client_disconnected()

    def handle(self) -> None:
        while True:
            line = self.rfile.readline()
            if not line:
                return
            request_id = None
            try:
                request = json.loads(line)
                if not isinstance(request, dict):
                    raise ValueError("Request must be a JSON object.")
                request_id = request.get("id")
                command = str(request.get("command", ""))
                arguments = request.get("arguments", {})
                if not isinstance(arguments, dict):
                    raise ValueError("arguments must be an object.")
                result = self.server.dispatch(command, arguments)
                response = {"id": request_id, "ok": True, "result": result}
            except Exception as exception:  # noqa: BLE001 - report to client
                response = {"id": request_id, "ok": False, "error": str(exception)}
            self.wfile.write((json.dumps(response) + "\n").encode("utf-8"))
            self.wfile.flush()


def _require_side(arguments) -> str:
    side = str(arguments.get("side", ""))
    if side not in ("left", "right"):
        raise ValueError("side must be left or right")
    return side


class OperatorControlServer(socketserver.ThreadingTCPServer):
    """Serves engage/disengage, arm-home, and hand-open commands."""

    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, address, console: OperatorConsole):
        super().__init__(address, _RequestHandler)
        self.keyboard = console
        self.snapshot = console.snapshot
        self._thread: threading.Thread | None = None
        self._clients = 0
        self._clients_lock = threading.Lock()

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self.serve_forever, kwargs={"poll_interval": 0.2}, daemon=True
        )
        self._thread.start()

    def close(self) -> None:
        self.shutdown()
        self.server_close()
        self.keyboard.disable_all("operator server stopped")

    def client_connected(self) -> None:
        with self._clients_lock:
            self._clients += 1

    def client_disconnected(self) -> None:
        with self._clients_lock:
            self._clients = max(0, self._clients - 1)
            last_client = self._clients == 0
        if last_client:
            self.keyboard.disable_all("operator frontend disconnected")

    # -- commands ---------------------------------------------------------

    def _scoped_request(
        self,
        arguments,
        *,
        both: str,
        left: str,
        right: str,
    ) -> None:
        scope = str(arguments.get("side", "both"))
        if scope == "both":
            self.keyboard.request(both)
        elif scope == "left":
            self.keyboard.request(left)
        elif scope == "right":
            self.keyboard.request(right)
        else:
            raise ValueError("side must be left, right, or both")

    def dispatch(self, command: str, arguments: dict):
        commands = {
            "status": self.snapshot,
            "engage": lambda: self.keyboard.set_active(
                _require_side(arguments),
                True,
                str(arguments.get("component", "both")),
            ),
            "disengage": lambda: self.keyboard.set_active(
                _require_side(arguments),
                False,
                str(arguments.get("component", "both")),
            ),
            "disengage_all": lambda: self.keyboard.disable_all(
                "operator frontend"
            ),
            "open_hand": lambda: self._scoped_request(
                arguments,
                both="open_hands",
                left="open_left_hand",
                right="open_right_hand",
            ),
            "home_arm": lambda: self._scoped_request(
                arguments,
                both="reset",
                left="reset_left",
                right="reset_right",
            ),
            "capture_home": lambda: self.keyboard.request("capture_home"),
            "start_dataset_recording": lambda: self.keyboard.request(
                "start_dataset_recording"
            ),
            "stop_dataset_recording": lambda: self.keyboard.request(
                "stop_dataset_recording"
            ),
        }
        handler = commands.get(command)
        if handler is None:
            raise ValueError(f"Unknown command: {command}")
        result = handler()
        return {} if result is None else result
