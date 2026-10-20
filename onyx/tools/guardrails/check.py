#!/usr/bin/env python3
"""Campaign 1's guardrails, after the metric: a found_rate bought with false
detections or with a detector too slow for the mission does not count.

  - no box answered on an EMPTY scene (scene camera or wrist search);
  - the scene camera's frame (OWL + lift) within SCENE_S on this Jetson,
    the wrist's (box-top detector) within WRIST_S — the live search reads
    frames while the arm pans.
"""

import json
import sys
from pathlib import Path

REPORT = Path.cwd() / ".onyx_eval" / "report.json"
SCENE_S = 2.0  # the baseline: ~1.1 s
WRIST_S = 0.15  # the baseline: ~0.01 s


def main():
    if not REPORT.exists():
        print("FAIL no evaluation report (%s)" % REPORT)
        sys.exit(1)
    r = json.loads(REPORT.read_text())
    fails = []
    if r["false_positives"] > 0:
        fails.append("%d of %d empty scenes answered with a box" % (r["false_positives"], r["empty_scenes"]))
    if r["empty_scenes"] == 0:
        fails.append("the set has no empty scenes: false detections cannot be judged")
    if r["sec_per_scene_frame"] > SCENE_S:
        fails.append("scene frame %.2f s > %.1f s" % (r["sec_per_scene_frame"], SCENE_S))
    if r["sec_per_wrist_frame"] > WRIST_S:
        fails.append("wrist frame %.3f s > %.2f s" % (r["sec_per_wrist_frame"], WRIST_S))
    for f in fails:
        print("FAIL %s" % f)
    print("guardrails: %s" % ("ok" if not fails else "%d failed" % len(fails)))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
