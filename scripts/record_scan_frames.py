#!/usr/bin/env python3
"""Record wrist-camera frames (+ camera pose) for offline perception work.

    # terminal A (this): record for 40 s
    python3 scripts/record_scan_frames.py
    # terminal B: park the camera over the bench
    ros2 run rammp_box_opening press_demo --execute --detect-only

Nothing here moves the arm — it only subscribes. Frames land in
~/.ros/rammp_box_opening/captures/<stamp>/frame_*.npz with color, depth,
K, and the frame-stamp camera pose (base_link <- camera), i.e. exactly
the inputs top_face_from_depth takes, so the depth detector can be
developed and tuned entirely offline.

    # replay a capture through the CURRENT detector:
    python3 scripts/record_scan_frames.py --analyze <capture-dir>
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "src" / "rammp_box_opening")
)

CAPTURES = Path.home() / ".ros" / "rammp_box_opening" / "captures"


def record(seconds, period, lag_sweep=False):
    import rclpy
    from rclpy.node import Node

    from rammp_box_opening.perception.depth_source import camera_pose_at
    from rammp_box_opening.perception.lag import OFFSETS_S

    rclpy.init()
    node = Node("scan_recorder")
    from rammp_box_opening.perception.d405 import D405Grabber

    grab = D405Grabber(node, need_depth=True)
    out = CAPTURES / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    print(
        "recording to %s for %.0f s — run --detect-only in another shell"
        % (out, seconds)
    )

    n, last_stamp = 0, None
    t_end = time.monotonic() + seconds
    t_next = 0.0
    while time.monotonic() < t_end:
        rclpy.spin_once(node, timeout_sec=0.05)
        now = time.monotonic()
        if now < t_next:
            continue
        if grab.depth is None or grab.color is None or grab.k is None:
            continue
        stamp = (grab.color_stamp.sec, grab.color_stamp.nanosec)
        if stamp == last_stamp:
            continue
        cam = camera_pose_at(grab)
        if cam is None:
            continue  # no TF yet (bringup still coming up)
        rot, trans = cam
        extra = {}
        if lag_sweep:
            # the camera pose at stamp + every candidate lag, from the TF
            # buffer's history: --analyze-lag picks the one at which frames
            # shot while moving agree with the still ones
            rots, transes, ok = [], [], []
            for off in OFFSETS_S:
                c = camera_pose_at(grab, offset_s=float(off))
                ok.append(c is not None)
                rots.append(c[0] if c is not None else np.full((3, 3), np.nan))
                transes.append(c[1] if c is not None else np.full(3, np.nan))
            extra = {"lag_offsets": np.asarray(OFFSETS_S), "lag_rots": np.array(rots),
                     "lag_trans": np.array(transes), "lag_ok": np.array(ok)}
        np.savez_compressed(
            out / ("frame_%03d.npz" % n),
            color=grab.color,
            depth=grab.depth,
            k=grab.k,
            rot_cam=rot,
            trans_cam=trans,
            stamp=np.array(stamp),
            **extra,
        )
        last_stamp = stamp
        n += 1
        t_next = now + period
        print(
            "  frame %3d  cam_t [%.3f %.3f %.3f]" % (n, trans[0], trans[1], trans[2]),
            flush=True,
        )
    print("done: %d frames in %s" % (n, out))
    rclpy.shutdown()


def analyze(cap_dir):
    from rammp_box_opening.models.container import ContainerModel
    from rammp_box_opening.perception.depth_source import top_face_from_depth
    from rammp_box_opening.worlds import WorldStore

    repo = Path(__file__).resolve().parents[1] / "src" / "rammp_box_opening"
    model = ContainerModel.load(str(repo / "config/containers/oxo_pop.yaml"))
    table_z = WorldStore(str(repo / "config/world_bench.yaml")).table_top_z
    frames = sorted(Path(cap_dir).glob("frame_*.npz"))
    print("%d frames | table_z %.3f" % (len(frames), table_z))
    hits = 0
    for f in frames:
        d = np.load(f)
        fix, why = top_face_from_depth(
            d["depth"], d["k"], d["rot_cam"], d["trans_cam"], table_z, model
        )
        if fix is None:
            print("  %-14s -                     (%s)" % (f.stem, why))
            continue
        hits += 1
        print(
            "  %-14s top [%.3f %.3f %.3f] yaw %5.1f  %.3fx%.3f m  %4d px"
            % (
                f.stem,
                *fix.center,
                np.degrees(fix.yaw),
                *fix.footprint,
                fix.n_px,
            )
        )
    print("hit rate: %d/%d" % (hits, len(frames)))


def analyze_lag(cap_dir):
    """Which timestamp lag places the MOVING frames' box where the STILL
    frames put it. Needs a --lag-sweep capture taken while the arm moved
    over the box and rested on it too — any ordinary mission run does."""
    from rammp_box_opening.models.container import ContainerModel
    from rammp_box_opening.perception.depth_source import top_face_from_depth
    from rammp_box_opening.perception.lag import best_offset, moving_flags
    from rammp_box_opening.worlds import WorldStore

    repo = Path(__file__).resolve().parents[1] / "src" / "rammp_box_opening"
    model = ContainerModel.load(str(repo / "config/containers/oxo_pop.yaml"))
    table_z = WorldStore(str(repo / "config/world_bench.yaml")).table_top_z
    frames = sorted(Path(cap_dir).glob("frame_*.npz"))
    fixes, trans, offsets = [], [], None
    for f in frames:
        d = np.load(f)
        if "lag_offsets" not in d:
            sys.exit("%s has no lag sweep — record with --lag-sweep" % f)
        offsets = d["lag_offsets"]
        trans.append(d["trans_cam"])
        row = []
        for k in range(len(offsets)):
            if not d["lag_ok"][k]:
                row.append([np.nan] * 3)
                continue
            fix, _why = top_face_from_depth(d["depth"], d["k"], d["lag_rots"][k], d["lag_trans"][k], table_z, model)
            row.append(list(fix.center) if fix is not None else [np.nan] * 3)
        fixes.append(row)
    moving = moving_flags(trans)
    print("%d frames, %d moving, %d still" % (len(frames), int(moving.sum()), int((~moving).sum())))
    off, table = best_offset(offsets, np.array(fixes), moving)
    for o, rms, n in table:
        print("  offset %+.3f s: moving-vs-still rms %s mm over %d fixes%s"
              % (o, "  -  " if not np.isfinite(rms) else "%5.1f" % rms, n, "   <-- best" if off is not None and abs(o - off) < 1e-9 else ""))
    if off is None:
        print("not enough fixes on both sides to decide — record a run that moves over the box AND rests on it")
    else:
        print("\nset stamp_offset_s: %+.3f in config/camera_d405_wrist.yaml" % off)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=40.0)
    ap.add_argument("--period", type=float, default=0.4)
    ap.add_argument("--lag-sweep", action="store_true", help="also record the camera pose at every candidate timestamp lag")
    ap.add_argument("--analyze", metavar="DIR", help="replay a capture offline")
    ap.add_argument("--analyze-lag", metavar="DIR", help="pick the timestamp lag from a --lag-sweep capture")
    args = ap.parse_args()
    if args.analyze:
        analyze(args.analyze)
    elif args.analyze_lag:
        analyze_lag(args.analyze_lag)
    else:
        record(args.seconds, args.period, lag_sweep=args.lag_sweep)


if __name__ == "__main__":
    main()
