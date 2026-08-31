import multiprocessing
import os
import socket
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(
    0, str(REPO_ROOT / "adapters" / "pico" / "src")
)

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from pico_bimanual_franka_teleop.control_server import (
    OperatorConsole,
    OperatorControlServer,
)
from apps.operator_gui.operator_gui import OperatorWindow


def test_recording_buttons_follow_backend_recording_status(tmp_path):
    QSettings.setPath(
        QSettings.NativeFormat, QSettings.UserScope, str(tmp_path)
    )
    application = QApplication.instance() or QApplication([])
    window = OperatorWindow("127.0.0.1", _unused_port())
    try:
        window.socket.abort()
        window.reconnect_timer.stop()
        window._apply_status(
            {
                "dataset_recording": {
                    "configured": True,
                    "active": False,
                    "finalizing": False,
                    "path": "",
                    "sample_count": 0,
                }
            }
        )
        assert window.start_recording_button.isEnabled()
        assert not window.stop_recording_button.isEnabled()

        window._apply_status(
            {
                "dataset_recording": {
                    "configured": True,
                    "active": True,
                    "finalizing": False,
                    "path": "/data/episode.npz",
                    "sample_count": 42,
                }
            }
        )
        assert not window.start_recording_button.isEnabled()
        assert window.stop_recording_button.isEnabled()
        assert "42 frames" in window.recording_status_label.text()
    finally:
        window.poll_timer.stop()
        window.health_timer.stop()
        window.reconnect_timer.stop()
        window.socket.abort()
        window.close()


def _unused_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _serve(port: int) -> None:
    server = OperatorControlServer(("127.0.0.1", port), OperatorConsole())
    server.serve_forever(poll_interval=0.05)


def _wait(application, predicate, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        application.processEvents()
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("GUI state did not converge before timeout")


def test_backend_crash_displays_disconnect_and_reconnects_safely(tmp_path):
    QSettings.setPath(
        QSettings.NativeFormat, QSettings.UserScope, str(tmp_path)
    )
    application = QApplication.instance() or QApplication([])
    port = _unused_port()
    context = multiprocessing.get_context("spawn")
    backend = context.Process(target=_serve, args=(port,), daemon=True)
    backend.start()
    window = OperatorWindow("127.0.0.1", port)
    try:
        _wait(application, lambda: window.connection_state == "connected")
        hand_button = window.activation_buttons["left"]["hand"]
        arm_button = window.activation_buttons["left"]["arm"]
        combined_button = window.activation_buttons["left"]["both"]
        assert hand_button.text() == "Hand Enable"
        assert arm_button.text() == "Arm Enable"
        assert combined_button.text() == "Hand and Arm Enable (L)"

        hand_button.click()
        _wait(
            application,
            lambda: (
                hand_button.isChecked()
                and not arm_button.isChecked()
                and not combined_button.isChecked()
            ),
        )
        hand_button.click()
        _wait(application, lambda: not hand_button.isChecked())

        status_before_engage = window.last_status_at
        window.engage_buttons["left"].click()
        _wait(
            application,
            lambda: (
                window.last_status_at is not None
                and window.last_status_at != status_before_engage
                and window.engage_buttons["left"].isChecked()
            ),
        )

        backend.terminate()
        backend.join(timeout=2.0)
        _wait(application, lambda: window.connection_state == "disconnected")
        assert window.connection_indicator.text().startswith("DISCONNECTED")
        assert "backend status unavailable" in window.status_label.toPlainText()
        assert window.disconnect_message is not None
        assert window.disconnect_message.isVisible()
        assert window.disconnect_message.text() == (
            "The teleoperation backend connection was lost."
        )
        assert window.connect_action.isEnabled()
        assert not window.engage_buttons["left"].isChecked()
        assert not window.engage_buttons["left"].isEnabled()

        reconnect_port = _unused_port()
        backend = context.Process(
            target=_serve, args=(reconnect_port,), daemon=True
        )
        backend.start()
        window.disconnect_message.accept()
        window.connect_action.trigger()
        _wait(application, lambda: window.connection_dialog is not None)
        assert window.connection_dialog.host_field.text() == "127.0.0.1"
        assert window.connection_dialog.port_field.value() == port
        window.connection_dialog.port_field.setValue(reconnect_port)
        window.connection_dialog._accept_if_valid()
        _wait(application, lambda: window.connection_state == "connected")
        assert window.port == reconnect_port
        assert window.engage_buttons["left"].isEnabled()
        assert not window.engage_buttons["left"].isChecked()
        assert window.disconnect_message is None
    finally:
        window.poll_timer.stop()
        window.health_timer.stop()
        window.reconnect_timer.stop()
        window.socket.abort()
        window.close()
        if backend.is_alive():
            backend.terminate()
        backend.join(timeout=2.0)
