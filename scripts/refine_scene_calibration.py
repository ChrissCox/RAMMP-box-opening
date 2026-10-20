#!/usr/bin/env python3
"""Refine config/camera_scene.yaml from the scene-vs-wrist pairs the
mission records (~/.ros/rammp_box_opening/scene_calib/residuals.jsonl).

    python3 scripts/refine_scene_calibration.py            # report only
    python3 scripts/refine_scene_calibration.py --apply    # write the yaml (backup kept)

A translation with few pairs or pairs at one spot; a full rigid fit once
4+ pairs span 0.15 m of table — put the box at different spots over the
next runs to get there. Nothing here moves the arm; the launch's static TF
picks the new yaml up on its next start, the mission on its next run.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "rammp_box_opening"))
from rammp_box_opening.perception.d405 import quat_to_mat  # noqa: E402
from rammp_box_opening.perception.scene_calib import transform  # noqa: E402
from rammp_box_opening.perception.scene_refine import (  # noqa: E402
    MIN_PAIRS_ROTATION, MIN_SPREAD_M, apply, calibrated_at, fit, load_pairs, write_refinement,
)

RESIDUALS = Path.home() / ".ros" / "rammp_box_opening" / "scene_calib" / "residuals.jsonl"
CALIB = Path(__file__).resolve().parents[1] / "src" / "rammp_box_opening" / "config" / "camera_scene.yaml"
STUB_BOX = [0.4595, -0.0497, 0.085]  # the e2e harness's box: pairs it wrote before it had its own file


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--residuals", default=str(RESIDUALS))
    ap.add_argument("--calib", default=str(CALIB))
    ap.add_argument("--apply", action="store_true", help="write the refined calibration")
    a = ap.parse_args()
    # a pair measures the calibration it was made under: those from before
    # this file was last corrected describe a camera that no longer exists
    scene, wrist = load_pairs(a.residuals, synthetic=STUB_BOX, since=calibrated_at(a.calib))
    retired = len(load_pairs(a.residuals, synthetic=STUB_BOX)[0]) - len(scene)
    if retired:
        print("%d older pair(s) were made under an earlier calibration and are left out" % retired)
    if len(scene) == 0:
        sys.exit("no real pairs in %s made under this calibration" % a.residuals)
    r = fit(scene, wrist)
    print("%d pair(s), spread %.2f m -> %s fit: rms %.1f mm -> %.1f mm" % (r.n, r.spread_m, r.mode, r.rms_before_mm, r.rms_after_mm))
    print("correction (horizontal): shift [%+.1f %+.1f] mm, turn %.2f deg" % (
        *(1000 * r.D[:2, 3]), np.degrees(np.arctan2(r.D[1, 0], r.D[0, 0]))))
    for s_, w in zip(scene, wrist):
        f = r.D[:3, :3] @ s_ + r.D[:3, 3]
        print("  scene %s -> fitted %s vs wrist %s (left %.1f mm in xy)" % (
            np.round(s_[:2], 4), np.round(f[:2], 4), np.round(w[:2], 4), 1000 * np.linalg.norm(f[:2] - w[:2])))
    if r.mode == "translation":
        print("a rotation needs >= %d pairs spanning >= %.2f m: place the box at other spots on the next runs" % (MIN_PAIRS_ROTATION, MIN_SPREAD_M))
    doc = yaml.safe_load(open(a.calib))
    T2 = apply(transform(quat_to_mat(*doc["quat_xyzw"]), doc["xyz"]), r)
    print("camera: %s -> %s" % ([round(v, 4) for v in doc["xyz"]], [round(float(v), 4) for v in T2[:3, 3]]))
    if not a.apply:
        print("(report only — add --apply to write %s)" % a.calib)
        return
    # backups beside the residuals, not in the package's config dir
    backup = write_refinement(a.calib, r, Path(a.residuals).parent, "by hand")
    print("written %s (backup %s)" % (a.calib, backup))


if __name__ == "__main__":
    main()
