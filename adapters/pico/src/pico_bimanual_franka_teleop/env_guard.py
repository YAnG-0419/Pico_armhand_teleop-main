"""Re-exec the process with a ROS-free environment when a shell sourced one.

The operator process runs in the Conda environment and must never load ROS
python packages or ROS native libraries: a sourced ROS setup leaves its
site-packages on PYTHONPATH, where a pinocchio built for the distro's python
shadows the Conda one, and its lib directory on LD_LIBRARY_PATH, where the
distro's libeigenpy is resolved by the dynamic linker ahead of the Conda one and
fails with undefined symbols.

PYTHONPATH could be neutralized by editing sys.path, but LD_LIBRARY_PATH cannot:
the dynamic linker captures it at process start, and changing os.environ later
has no effect on dlopen. The only reliable repair is to scrub the environment
and exec the same interpreter again, which is what this does. After the exec the
trigger condition is false, so it runs at most once.

Re-exec must preserve ``python -m package.module``.  ``sys.argv[0]`` is the
module file path, so restarting as ``[executable, *sys.argv]`` turns a module
invocation into a script invocation: ``sys.path[0]`` becomes the file's
directory instead of the cwd, and repository-root imports such as ``adapters``
fail.

Since the hand controllers moved into the container, no host terminal needs ROS
sourced at all; this guard exists so that a terminal which sourced it anyway,
out of habit or a stale runbook, works instead of failing with a bewildering
import error.

Import and call this before anything that could import pinocchio.
"""

from __future__ import annotations

import os
import sys

_SCRUB_MARKER = "/opt/ros/"
_PATH_VARIABLES = ("PYTHONPATH", "LD_LIBRARY_PATH", "PATH")
_DROP_VARIABLES = (
    "AMENT_PREFIX_PATH",
    "CMAKE_PREFIX_PATH",
    "COLCON_PREFIX_PATH",
    "ROS_DISTRO",
    "ROS_VERSION",
    "ROS_PYTHON_VERSION",
)


def _scrubbed(value: str) -> str:
    return os.pathsep.join(
        entry
        for entry in value.split(os.pathsep)
        if entry and _SCRUB_MARKER not in entry
    )


def reexec_argv() -> list[str]:
    """Rebuild argv so a ``python -m`` launch still uses ``-m`` after exec.

    ``sys.orig_argv`` is the interpreter command before Python rewrote
    ``sys.argv[0]`` to the module file.  Script launches keep the existing
    ``[executable, *sys.argv]`` form.
    """
    original = getattr(sys, "orig_argv", None) or []
    try:
        dash_m = original.index("-m")
    except ValueError:
        dash_m = -1
    if dash_m >= 0 and dash_m + 1 < len(original):
        return [sys.executable, *original[dash_m:]]
    return [sys.executable, *sys.argv]


def ensure_ros_free_process() -> None:
    """Exec into a clean copy of this process if ROS paths pollute it."""
    polluted = any(
        _SCRUB_MARKER in os.environ.get(name, "")
        for name in (*_PATH_VARIABLES, *_DROP_VARIABLES)
    )
    if not polluted:
        return
    environment = dict(os.environ)
    for name in _PATH_VARIABLES:
        if name in environment:
            cleaned = _scrubbed(environment[name])
            if cleaned:
                environment[name] = cleaned
            else:
                environment.pop(name)
    for name in _DROP_VARIABLES:
        environment.pop(name, None)
    sys.stderr.write(
        "note: a sourced ROS environment was detected and removed; "
        "re-executing without it\n"
    )
    sys.stderr.flush()
    os.execve(sys.executable, reexec_argv(), environment)


if __name__ == "__main__":
    ensure_ros_free_process()
    sys.stdout.write(f"PACKAGE={__package__}\n")
    sys.stdout.flush()
