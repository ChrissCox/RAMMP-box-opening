#!/usr/bin/env python3
"""Campaign 1's evaluator: can the mission find the box, wherever it stands?

Replays a recorded, labelled detection set (tasks/record_detection_set.py)
through THIS worktree's perception code, exactly as the mission runs it:

  scene camera  OWLv2 (owl_source.owl_detect, the container's prompts and
                scene floor) on each recorded scene frame, the depth lifted
                (scene.color_cloud) through the committed calibration, the
                first OWL box whose top is this container (scene_source.
                scene_fix_from_boxes) — the first scene frame with a fix
                is the scene's answer;
  wrist search  the box-top detector (depth_source.top_face_from_depth) on
                the still wrist frames of each search pose, in the order
                the mission flies them; the first pose with a sighting
                (whole or partial) is the search's answer, its median.

A placement is FOUND when either answer lands within FOUND_M of the truth
(the mission's own close-up aim, recorded) — close enough that the arm, at
staging over it, has the button in its close-up view, and re-centres onto
it. The mission falls back from the scene to the search, so either counts.

    METRIC found_rate=<fraction of box placements found>     (primary)

plus secondaries: scene_found_rate, wrist_found_rate, scene_median_err_mm,
wrist_median_err_mm, false_positives (answers on EMPTY scenes),
sec_per_scene_frame, sec_per_wrist_frame. The full report goes to
.onyx_eval/report.json for the guardrails.

Never joins ROS, never moves anything: it reads files. The set must match
onyx/data/detection_set.json byte for byte (sha256), or there is no metric.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
SRC = ROOT / "src" / "rammp_box_opening"
sys.path.insert(0, str(SRC))  # THIS worktree's code, never the installed copy

MANIFEST = ROOT / "onyx" / "data" / "detection_set.json"
CALIB = SRC / "config" / "camera_scene.yaml"
CONTAINERS = SRC / "config" / "containers"
REPORT = ROOT / ".onyx_eval" / "report.json"
FOUND_M = 0.04  # the close-up view at staging holds the button this far off, in its worst direction
WRIST_POSES = ("look", "sweep:left", "sweep:right")  # the mission's search order (press_demo.search_targets)
SCENE_STRIDE = 2  # as the live locate lifts it


def die(msg):
    print("evaluation: %s" % msg, file=sys.stderr)
    sys.exit(2)


def sha256(path):
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_set(manifest, set_dir):
    if not manifest.exists():
        die("no %s — record a set (record_detection_set) and commit its manifest" % manifest)
    man = json.loads(manifest.read_text())
    base = Path(set_dir) if set_dir else Path.home() / ".ros/rammp_box_opening/detection_sets" / man["set"]
    bad = [rel for rel, h in man["files"].items() if not (base / rel).is_file() or sha256(base / rel) != h]
    if bad:
        die("the recorded set at %s does not match the manifest (%d file(s), e.g. %s)" % (base, len(bad), bad[0]))
    return man, base


def err(xy, truth):
    return None if xy is None else float(np.hypot(xy[0] - truth[0], xy[1] - truth[1]))


def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    # development only: the campaign's tool runs with neither (onyx/setup.json)
    ap.add_argument("--manifest", default=str(MANIFEST))
    ap.add_argument("--set-dir", default=None)
    args = ap.parse_args()
    import rammp_box_opening

    if not Path(rammp_box_opening.__file__).resolve().is_relative_to(SRC.resolve()):
        die("imported rammp_box_opening from %s, not this worktree" % rammp_box_opening.__file__)
    from rammp_box_opening.models.container import ContainerModel, load_press_demo
    from rammp_box_opening.perception.depth_source import top_face_from_depth
    from rammp_box_opening.perception.owl_source import owl_detect, owl_floor
    from rammp_box_opening.perception.scene import color_cloud
    from rammp_box_opening.perception.scene_calib import load_scene_yaml
    from rammp_box_opening.perception.scene_source import scene_fix_from_boxes

    man, base = load_set(Path(args.manifest), args.set_dir)
    _doc, T_base_link = load_scene_yaml(str(CALIB))
    containers = [p["container"] for p in man["placements"] if p.get("container")]
    if not containers:
        die("the set has no box placements")
    default_container = containers[0]
    models, cfgs = {}, {}

    def model_cfg(name):
        if name not in models:
            path = CONTAINERS / name
            models[name], cfgs[name] = ContainerModel.load(str(path)), load_press_demo(str(path))
        return models[name], cfgs[name]

    import torch
    from transformers import Owlv2ForObjectDetection, Owlv2Processor

    _m, cfg0 = model_cfg(default_container)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    proc = Owlv2Processor.from_pretrained(cfg0.owl_model)
    owl = Owlv2ForObjectDetection.from_pretrained(cfg0.owl_model).eval().to(device)

    rows, t_scene, t_wrist = [], [], []
    for p in man["placements"]:
        if p.get("truth_suspect"):
            continue
        name = p.get("container") or default_container
        model, cfg = model_cfg(name)
        table_z = float(p["table_z"])
        pdir = base / p["id"]
        # the scene camera: the first recorded frame with a fix answers
        scene_xy = None
        for fdir in sorted(pdir.glob("scene_*")):
            z = np.load(fdir / "frame_000.npz")
            t0 = time.monotonic()
            boxes = owl_detect(proc, owl, z["color"][:, :, ::-1], cfg.owl_queries, owl_floor(cfg, "scene"), device=device)
            pc, uv = color_cloud(z["depth"], z["kd"], z["k"], z["dist"], z["T_color_depth"], stride=SCENE_STRIDE)
            T = np.asarray(T_base_link, float) @ z["T_link_color"]
            got = scene_fix_from_boxes(boxes, pc @ T[:3, :3].T + T[:3, 3], uv, table_z, model)
            t_scene.append(time.monotonic() - t0)
            if got.pose is not None:
                scene_xy = (float(got.top[0]), float(got.top[1]))
                break
        # the wrist search: the first pose with a sighting answers
        wrist_xy = None
        for pose in WRIST_POSES:
            seen = []
            for fdir in sorted(pdir.glob("wrist_%s_*" % pose)):
                z = np.load(fdir / "frame_000.npz")
                t0 = time.monotonic()
                fix, _why = top_face_from_depth(z["depth"], z["k"], z["rot_cam"], z["trans_cam"], table_z, model)
                t_wrist.append(time.monotonic() - t0)
                if fix is not None:
                    seen.append(fix.center[:2])
            if seen:
                wrist_xy = tuple(float(v) for v in np.median(np.asarray(seen), axis=0))
                break
        truth = p.get("truth_xyz")
        rows.append({
            "id": p["id"], "box": truth is not None, "truth": truth,
            "scene_xy": scene_xy, "wrist_xy": wrist_xy,
            "scene_err_m": err(scene_xy, truth) if truth else None,
            "wrist_err_m": err(wrist_xy, truth) if truth else None,
        })

    box = [r for r in rows if r["box"]]
    empty = [r for r in rows if not r["box"]]
    if not box:
        die("no usable box placements (all flagged suspect?)")
    ok = lambda e: e is not None and e <= FOUND_M  # noqa: E731
    for r in box:
        r["scene_found"], r["wrist_found"] = ok(r["scene_err_m"]), ok(r["wrist_err_m"])
        r["found"] = r["scene_found"] or r["wrist_found"]
    for r in empty:
        r["false_positive"] = r["scene_xy"] is not None or r["wrist_xy"] is not None
    se = [r["scene_err_m"] for r in box if r["scene_err_m"] is not None]
    we = [r["wrist_err_m"] for r in box if r["wrist_err_m"] is not None]
    report = {
        "found_rate": sum(r["found"] for r in box) / len(box),
        "scene_found_rate": sum(r["scene_found"] for r in box) / len(box),
        "wrist_found_rate": sum(r["wrist_found"] for r in box) / len(box),
        "scene_median_err_mm": 1000 * float(np.median(se)) if se else -1.0,
        "wrist_median_err_mm": 1000 * float(np.median(we)) if we else -1.0,
        "false_positives": sum(r["false_positive"] for r in empty),
        "empty_scenes": len(empty),
        "box_placements": len(box),
        "sec_per_scene_frame": float(np.mean(t_scene)) if t_scene else 0.0,
        "sec_per_wrist_frame": float(np.mean(t_wrist)) if t_wrist else 0.0,
        "found_m": FOUND_M,
        "placements": rows,
    }
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=1))
    missed = [r["id"] for r in box if not r["found"]]
    print("placements %d (+%d empty); missed: %s" % (len(box), len(empty), ", ".join(missed) or "none"))
    print("METRIC found_rate=%.4f" % report["found_rate"])
    for k in ("scene_found_rate", "wrist_found_rate", "scene_median_err_mm", "wrist_median_err_mm",
              "false_positives", "sec_per_scene_frame", "sec_per_wrist_frame"):
        print("METRIC %s=%.4f" % (k, report[k]))


if __name__ == "__main__":
    main()
