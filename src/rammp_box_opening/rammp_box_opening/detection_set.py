"""A recorded detection set, the parts with no robot in them: where the box
is placed (the grid), and the manifest that pins the set — every file's
sha256 and every placement's truth. The recorder (tasks/
record_detection_set.py) writes it; the campaign's evaluator (onyx/tools/
evaluation) checks the set against it. Importable with no ROS at all."""

import hashlib
import json
from pathlib import Path

# The grid, over the reachable table (the scene camera sees all of it: 187/187
# points of a finer grid, 2026-09-23). Snake order keeps each hover short.
GRID_X = (0.30, 0.40, 0.50, 0.60, 0.70)
GRID_Y = (-0.30, -0.15, 0.0, 0.15, 0.30)


def grid_points(xs=GRID_X, ys=GRID_Y):
    """[(x, y)] in snake order: along y, reversing every other row of x."""
    out = []
    for i, x in enumerate(xs):
        row = list(ys) if i % 2 == 0 else list(reversed(ys))
        out += [(float(x), float(y)) for y in row]
    return out


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(set_dir):
    """manifest.json: every file's sha256 and every placement's truth — what
    the evaluator checks the set against, and what the campaign commits."""
    set_dir = Path(set_dir)
    files = {}
    placements = []
    for p in sorted(set_dir.glob("p*")):
        if not p.is_dir():
            continue
        truth = json.loads((p / "truth.json").read_text())
        placements.append(dict(truth, id=p.name))
        for f in sorted(p.rglob("*")):
            if f.is_file():
                files[str(f.relative_to(set_dir))] = sha256(f)
    doc = {"set": set_dir.name, "placements": placements, "files": files}
    (set_dir / "manifest.json").write_text(json.dumps(doc, indent=1, sort_keys=True))
    return doc
