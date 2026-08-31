from types import SimpleNamespace

from pico_teleop_bridge.node import PicoTeleopBridge


def make_bridge():
    bridge = PicoTeleopBridge.__new__(PicoTeleopBridge)
    bridge.source_id = "pico"
    bridge.command_stream_id = "current-session"
    bridge.gateway_status_sequence = -1
    bridge.gateway_active_sides = ()
    bridge.gateway_faults = ()
    return bridge


def status(session_id, sequence, accepted_sides=(), faults=(), source="pico"):
    return SimpleNamespace(
        source=source,
        session_id=session_id,
        sequence=sequence,
        accepted_sides=accepted_sides,
        faults=faults,
    )


def test_gateway_status_is_scoped_and_monotonic():
    bridge = make_bridge()
    bridge._gateway_status(
        status("previous-session", 50, faults=("stale fault",))
    )
    assert bridge.gateway_status_sequence == -1

    bridge._gateway_status(
        status(
            "current-session",
            3,
            accepted_sides=("left",),
            faults=("right first target is too far from measured state.",),
        )
    )
    assert bridge.gateway_status_sequence == 3
    assert bridge.gateway_active_sides == ("left",)
    assert bridge.gateway_faults == (
        "right first target is too far from measured state.",
    )

    bridge._gateway_status(
        status("current-session", 4, source="vive", faults=("wrong source",))
    )
    assert bridge.gateway_status_sequence == 3

    bridge._gateway_status(
        status("current-session", 2, accepted_sides=("left", "right"))
    )
    assert bridge.gateway_status_sequence == 3
    assert bridge.gateway_active_sides == ("left",)
