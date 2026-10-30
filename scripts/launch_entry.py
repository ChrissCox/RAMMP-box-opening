#!/usr/bin/env python3
"""Container entry for `ros2 launch`: turn SIGTERM into the SIGINT launch honours.

Humble's `ros2 launch` shuts its child processes down cleanly on SIGINT but NOT
on SIGTERM: on SIGTERM it cancels its own event loop and exits, orphaning every
node it started (its own log says so). `docker stop` sends SIGTERM, so this runs
as PID 1 and forwards it as SIGINT. It is Python rather than bash because a
non-interactive bash starts background children with SIGINT ignored, which
silently defeats the forwarding.

Identical in role and implementation to rammp-teleop's launch_entry.py.
"""

import signal
import subprocess
import sys


def main(argv) -> int:
    child = subprocess.Popen(["ros2", "launch", *argv])

    def forward(_signum, _frame):
        try:
            child.send_signal(signal.SIGINT)
        except ProcessLookupError:
            pass

    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGINT, forward)
    return child.wait()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
