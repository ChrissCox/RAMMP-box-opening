#!/usr/bin/env python3
"""Campaign 1's readiness, before the metric: the Jetson is free for OWLv2.

The evaluator loads a third OWLv2 onto the GPU the live bench shares (its two
OWL detectors hold theirs). While the arm is running — a mission, a
recording, a homing, a calibration — that slows the live detectors and skews
the evaluator's own timing guardrail. So this WAITS, up to WAIT_S, for:

  - no attended arm command running (press_demo, record_detection_set,
    home_arm, calibrate_scene_camera);
  - at least MIN_AVAIL_GIB of memory available (the GPU's memory is the
    system's on a Jetson).

Reads /proc only. Never joins ROS, never moves anything.
"""

import os
import sys
import time

WAIT_S = 600.0
POLL_S = 5.0
MIN_AVAIL_GIB = 3.0  # OWLv2 on the GPU takes ~1.5 GiB, and the bench keeps running
ARM_COMMANDS = ("press_demo", "record_detection_set", "home_arm", "calibrate_scene_camera.py")


def arm_commands_running():
    """Names of the attended arm commands running now."""
    found = set()
    for pid in os.listdir("/proc"):
        if not pid.isdigit() or int(pid) == os.getpid():
            continue
        try:
            with open("/proc/%s/cmdline" % pid, "rb") as f:
                argv = f.read().split(b"\0")
        except OSError:
            continue
        for a in argv[:3]:  # the interpreter, then the script
            name = os.path.basename(a.decode(errors="replace"))
            if name in ARM_COMMANDS:
                found.add(name)
    return sorted(found)


def mem_available_gib():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / (1 << 20)
    return 0.0


def blockers():
    out = ["the arm is running (%s)" % ", ".join(cmds) for cmds in [arm_commands_running()] if cmds]
    avail = mem_available_gib()
    if avail < MIN_AVAIL_GIB:
        out.append("%.1f GiB memory available, need %.1f" % (avail, MIN_AVAIL_GIB))
    return out


def main():
    t0 = time.monotonic()
    said = None
    while True:
        why = blockers()
        if not why:
            print("readiness ok: the bench is idle, %.1f GiB available" % mem_available_gib())
            return
        if time.monotonic() - t0 > WAIT_S:
            print("FAIL not ready after %.0f s: %s" % (WAIT_S, "; ".join(why)))
            sys.exit(1)
        if why != said:
            print("waiting: %s" % "; ".join(why), flush=True)
            said = why
        time.sleep(POLL_S)


if __name__ == "__main__":
    main()
