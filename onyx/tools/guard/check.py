#!/usr/bin/env python3
"""Campaign 1's guard, before the metric. No motion, no ROS graph.

  1. The container configs: only their `vlm:` section (the detector's
     prompts, model and floors) may differ from the pins — every press,
     grip, geometry and safety setting is fixed.
  2. Candidate perception code never reads the recorded set or its labels.
  3. The unit suite passes (the mission's own tests, offline).
"""

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path.cwd()
PINS = ROOT / "onyx" / "tools" / "guard" / "pins.json"
CONTAINERS = ROOT / "src" / "rammp_box_opening" / "config" / "containers"
PERCEPTION = ROOT / "src" / "rammp_box_opening" / "rammp_box_opening" / "perception"
LABEL_READS = re.compile(r"detection_sets|detection_set\.json|truth\.json|onyx_eval|report\.json")


def outside_vlm(path):
    """sha256 of a container config with its vlm section taken out."""
    doc = yaml.safe_load(path.read_text())
    doc.pop("vlm", None)
    return hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()


def main():
    fails = []
    pins = json.loads(PINS.read_text())
    have = {p.name: outside_vlm(p) for p in sorted(CONTAINERS.glob("*.yaml"))}
    for name, h in pins.items():
        if have.get(name) != h:
            fails.append("config/containers/%s changed outside its vlm: section (or is gone)" % name)
    for name in set(have) - set(pins):
        fails.append("config/containers/%s is new: a container is not a detection change" % name)
    for py in sorted(PERCEPTION.glob("*.py")):
        for n, line in enumerate(py.read_text().splitlines(), 1):
            if LABEL_READS.search(line):
                fails.append("%s:%d reads the recorded set or its labels" % (py.relative_to(ROOT), n))
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "src/rammp_box_opening/test", "-q", "-x",
         "-p", "no:launch_testing", "-p", "no:launch_ros", "-p", "no:cacheprovider"],
        capture_output=True, text=True,
    )
    tail = (r.stdout.strip().splitlines() or ["(no output)"])[-1]
    if r.returncode != 0:
        fails.append("unit tests: %s" % tail)
    print("guard: unit tests — %s" % tail)
    for f in fails:
        print("FAIL %s" % f)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
