#!/usr/bin/env python3
"""Measure the bench surface from the wrist D405 and update world_bench.yaml.

Move the arm to the scan pose first (tool pointing straight down):

    ros2 launch rammp_box_opening press_demo.launch.py
    # move arm to the look/scan pose (look_joints(HOME)) — e.g. via rviz
    python3 scripts/measure_bench.py            # measure and write
    python3 scripts/measure_bench.py --check    # report without writing

The script fits a plane through the table points seen in the aligned depth
stream, transforms the fit to base_link via TF, and writes:

  table_normal_base   — measured unit normal (re-uses the wrist's own TF tree,
                        the same reference as the scene calibration)
  obstacles[table].position[2] — adjusted so that table_top_z equals the
                        measured surface intercept (x/y extents unchanged)

Run this whenever the arm base, the table, or the bench moves; then rebuild
and re-run calibrate_scene_camera.py (which uses table_normal_base as its
reference plane).
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src" / "rammp_box_opening"))

from rammp_box_opening.perception.d405 import D405Grabber, camera_config, quat_to_mat  # noqa: E402
from rammp_box_opening.perception.planes import angle_deg, cloud_from_depth, fit_plane  # noqa: E402
from rammp_box_opening.perception.scene_calib import transform  # noqa: E402

CONFIG_DIR = REPO / "src" / "rammp_box_opening" / "config"
BENCH_FILE = CONFIG_DIR / "world_bench.yaml"

MIN_INLIERS = 0.30   # RANSAC table-plane quality gate
MAX_TILT_DEG = 10.0  # refuse a normal more than this far from +z
FRAMES = 20          # depth frames averaged


def spin_until(node, ready, timeout_s, what):
    t0 = time.monotonic()
    while not ready():
        import rclpy
        rclpy.spin_once(node, timeout_sec=0.05)
        if time.monotonic() - t0 > timeout_s:
            sys.exit("timed out waiting for %s" % what)


def tf_4x4(buf, parent, child):
    import rclpy.time as rt
    tf = buf.lookup_transform(parent, child, rt.Time())
    t, r = tf.transform.translation, tf.transform.rotation
    return transform(quat_to_mat(r.x, r.y, r.z, r.w), [t.x, t.y, t.z])


def measure(node, wrist, frames):
    """Collect `frames` depth captures from the wrist pointing down,
    fit the table plane in base_link.  Returns (normal_base, top_z, n_pts)."""
    cfg = camera_config()
    clouds = []
    seen, last = 0, None
    print("collecting %d depth frames …" % frames)
    while seen < frames:
        spin_until(
            node,
            lambda: wrist.depth is not None and wrist.k is not None
            and wrist.color_stamp is not None
            and (wrist.color_stamp.sec, wrist.color_stamp.nanosec) != last,
            15.0,
            "wrist depth + camera_info",
        )
        last = (wrist.color_stamp.sec, wrist.color_stamp.nanosec)
        T_base_ee = tf_4x4(wrist.tf_buffer, "base_link", cfg["parent_frame"])
        T_base_cam = T_base_ee @ transform(
            quat_to_mat(*cfg["mount_quat_xyzw"]), cfg["mount_xyz"]
        )
        pts_cam, _, _ = cloud_from_depth(wrist.depth, wrist.k, stride=2)
        pts_base = pts_cam @ T_base_cam[:3, :3].T + T_base_cam[:3, 3]
        # keep only the table band: z in [−0.20, +0.10] base_link, well
        # above the floor, below the pedestal base, and x > 0.10
        ok = (pts_base[:, 2] > -0.20) & (pts_base[:, 2] < 0.10) & (pts_base[:, 0] > 0.10)
        clouds.append(pts_base[ok])
        seen += 1

    pts = np.concatenate(clouds, axis=0)
    print("fitting plane through %d table-band points …" % len(pts))
    fit = fit_plane(pts, tol_m=0.006, iters=300)
    if fit is None:
        sys.exit("plane fit failed — too few points in the table band")
    n, centroid, frac = fit
    if frac < MIN_INLIERS:
        sys.exit(
            "table plane fit too weak (%.0f%% inliers, need %.0f%%)"
            % (100 * frac, 100 * MIN_INLIERS)
        )
    if n[2] < 0:
        n = -n  # orient upward
    tilt = angle_deg(n, [0.0, 0.0, 1.0])
    if tilt > MAX_TILT_DEG:
        sys.exit(
            "measured normal %.3f deg from +z — arm is not at the scan pose, "
            "or the bench is unusually tilted" % tilt
        )
    # table top z: median z of inlier points (robust to outliers)
    inliers = pts[np.abs((pts - centroid) @ n) < 0.006]
    top_z = float(np.median(inliers[:, 2]))
    print(
        "table: normal %s (%.2f deg off +z), top z %.4f m, %.0f%% inliers over %d pts"
        % (np.round(n, 4), tilt, top_z, 100 * frac, len(pts))
    )
    return n.tolist(), top_z, len(inliers)


def update_bench(bench_path, normal, top_z, dry_run=False):
    text = Path(bench_path).read_text()
    cfg = yaml.safe_load(text)

    old_normal = cfg.get("table_normal_base", [0.0, 0.0, 1.0])
    old_top_z = None
    for obs in cfg.get("obstacles", []):
        if obs.get("name") == "table":
            old_top_z = obs["position"][2] + obs["dims"][2] / 2
            break

    print("  table_normal_base: %s  ->  %s" % (old_normal, [round(v, 4) for v in normal]))
    if old_top_z is not None:
        print("  table_top_z:       %.4f  ->  %.4f" % (old_top_z, top_z))

    if dry_run:
        print("(--check mode: nothing written)")
        return

    # update table_normal_base
    cfg["table_normal_base"] = [round(v, 6) for v in normal]

    # update the table obstacle's z position so top_z is correct
    for obs in cfg.get("obstacles", []):
        if obs.get("name") == "table":
            half = obs["dims"][2] / 2
            obs["position"][2] = round(top_z - half, 6)

    Path(bench_path).write_text(yaml.dump(cfg, default_flow_style=False, sort_keys=False))
    print("written: %s" % bench_path)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--frames", type=int, default=FRAMES, help="depth frames to average")
    ap.add_argument("--check", action="store_true", help="report measurement without writing")
    ap.add_argument("--out", default=str(BENCH_FILE), help="bench yaml to update")
    args = ap.parse_args()

    import rclpy
    rclpy.init()
    node = rclpy.create_node("measure_bench")
    wrist = D405Grabber(node, need_depth=True)

    print("waiting for wrist TF …")
    spin_until(
        node,
        lambda: wrist.tf_buffer.can_transform(
            "base_link", camera_config()["parent_frame"],
            __import__("rclpy.time", fromlist=["Time"]).Time(),
        ),
        10.0,
        "base_link <- %s TF" % camera_config()["parent_frame"],
    )

    normal, top_z, n_pts = measure(node, wrist, args.frames)
    update_bench(args.out, normal, top_z, dry_run=args.check)


if __name__ == "__main__":
    main()
