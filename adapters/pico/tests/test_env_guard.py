import os
import subprocess
import sys
from pathlib import Path

from pico_bimanual_franka_teleop.env_guard import reexec_argv

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_reexec_argv_keeps_module_flag(monkeypatch):
    monkeypatch.setattr(
        sys,
        "orig_argv",
        [
            "/opt/conda/bin/python",
            "-m",
            "teleop_runtime.cli",
            "--config",
            "config/modes/pico.yaml",
        ],
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [str(REPO_ROOT / "teleop_runtime" / "cli.py"), "--config", "unused"],
    )
    assert reexec_argv() == [
        sys.executable,
        "-m",
        "teleop_runtime.cli",
        "--config",
        "config/modes/pico.yaml",
    ]


def test_reexec_argv_keeps_script_invocation(monkeypatch):
    script = str(REPO_ROOT / "teleop_runtime" / "cli.py")
    monkeypatch.setattr(sys, "orig_argv", [sys.executable, script, "--help"])
    monkeypatch.setattr(sys, "argv", [script, "--help"])
    assert reexec_argv() == [sys.executable, script, "--help"]


def test_module_reexec_keeps_package_name():
    environment = os.environ.copy()
    environment["AMENT_PREFIX_PATH"] = "/opt/ros/humble"
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [sys.executable, "-m", "pico_bimanual_franka_teleop.env_guard"],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "re-executing without it" in completed.stderr
    assert completed.stdout.strip() == "PACKAGE=pico_bimanual_franka_teleop"
