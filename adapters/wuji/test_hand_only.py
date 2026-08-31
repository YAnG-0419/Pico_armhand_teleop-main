import pytest

from adapters.wuji.hand_only import parse_args


def test_hand_only_defaults_to_explicit_right_hand2():
    args = parse_args(["--wuji-right-address", "192.168.2.111:7447"])

    assert args.selected_sides == ("right",)
    assert args.wuji_right_model == "wuji_hand_2"
    assert args.wuji_right_address == "192.168.2.111:7447"


def test_hand_only_accepts_full_entry_side_flags():
    args = parse_args(
        [
            "--wuji-sides",
            "left",
            "--wuji-left-model",
            "wuji_hand",
            "--wuji-left-serial",
            "TEST-SERIAL",
        ]
    )

    assert args.selected_sides == ("left",)
    assert args.wuji_left_serial == "TEST-SERIAL"


def test_hand2_requires_selected_side_address():
    with pytest.raises(SystemExit):
        parse_args(["--wuji-sides", "right"])
