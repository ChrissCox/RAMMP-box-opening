#!/usr/bin/env python3
"""Calibrate the scene camera to the arm — no motion, no marker to attach.

    ros2 launch rammp_box_opening press_demo.launch.py     # TF for the arm
    python3 scripts/calibrate_scene_camera.py             # arm at HOME
    python3 scripts/calibrate_scene_camera.py --check     # later: still good?

From HOME the wrist camera sees the ArUco tag on the cabinet door at 30 cm.
The wrist camera is calibrated to the arm, so the tag's centre and the
door's plane are known in base_link. The scene camera sees the same tag,
the same door and the table. Rotation comes from the two planes (thousands
of depth points each), translation from the tag centre on both sides — see
perception/scene_calib.py for why the tag alone is not enough.

Nothing here commands the arm; it only subscribes. Every frame is
averaged over --frames captures, and the result is refused when the
door-to-table angle disagrees between the two frames by more than
PLANE_ANGLE_TOL_DEG, when either plane fit is weak, or when the tag is
not seen steadily. The calibration lands in config/camera_scene.yaml of
this source tree (rebuild to install it); press_demo.launch.py then
publishes it, and the scene camera joins the arm's TF tree.
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src" / "rammp_box_opening"))

from rammp_box_opening.constants import FINGERTIP_FRAMES  # noqa: E402
from rammp_box_opening.perception.d405 import D405Grabber, camera_config, quat_to_mat  # noqa: E402
from rammp_box_opening.perception.planes import angle_deg, cloud_from_depth, fit_plane, ray_plane  # noqa: E402
from rammp_box_opening.perception.scene import COLOR_FRAME, LINK_FRAME, SceneGrabber  # noqa: E402
from rammp_box_opening.perception.scene_calib import (  # noqa: E402
    PLANE_ANGLE_TOL_DEG,
    SCENE_CALIB_FILE,
    detect_tag,
    load_scene_yaml,
    optical_to_link,
    scene_pose_from_planes,
    transform,
    write_scene_yaml,
)
from rammp_box_opening.worlds import WorldStore  # noqa: E402

CONFIG_DIR = REPO / "src" / "rammp_box_opening" / "config"
REPORT_DIR = Path.home() / ".ros" / "rammp_box_opening" / "scene_calib"
WRIST_PAPER_PX = 70  # the tag's paper, left OUT of the wrist's door fit
WRIST_DOOR_PX = 220  # ... and the door patch around it that is fitted
SCENE_DOOR_WINDOW_PX = 110  # ... and in the scene image
TABLE_ROW_FRAC = 0.60  # the table is in the lower part of the scene image
MIN_DOOR_INLIERS = 0.5
MIN_TABLE_INLIERS = 0.3
MIN_TAG_SEEN = 0.8


def pixel_ray(uv, k, dist):
    und = cv2.undistortPoints(np.asarray(uv, np.float64).reshape(1, 1, 2), k, dist).reshape(2)
    return np.array([und[0], und[1], 1.0])


def spin_until(node, ready, timeout_s, what, missing=None):
    t0 = time.monotonic()
    while not ready():
        import rclpy

        rclpy.spin_once(node, timeout_sec=0.05)
        if time.monotonic() - t0 > timeout_s:
            sys.exit("timed out waiting for %s%s" % (what, (" — missing: %s" % missing()) if missing else ""))


def tf_4x4(buf, parent, child):
    import rclpy.time as rt

    tf = buf.lookup_transform(parent, child, rt.Time())
    t, r = tf.transform.translation, tf.transform.rotation
    return transform(quat_to_mat(r.x, r.y, r.z, r.w), [t.x, t.y, t.z])


def wrist_side(node, wrist, frames):
    """(p_tag_base, (n_door_base, c_door_base), n_seen) from `frames` wrist
    captures, each giving the tag centre via the door plane in depth."""
    cfg = camera_config()
    p_tags, normals, centres, seen = [], [], [], 0
    last = None
    while seen < frames:
        spin_until(
            node,
            lambda: wrist.color is not None and wrist.depth is not None and wrist.k is not None
            and wrist.color_stamp is not None
            and (wrist.color_stamp.sec, wrist.color_stamp.nanosec) != last,
            15.0,
            "wrist frames",
            missing=wrist.missing,
        )
        last = (wrist.color_stamp.sec, wrist.color_stamp.nanosec)
        c = detect_tag(wrist.color)
        if c is None:
            print("wrist: frame without the tag")
            continue
        pts, u, v = cloud_from_depth(wrist.depth, wrist.k)
        ctr = c.mean(axis=0)
        # a generous patch of DOOR around the tag (30 cm of it at this
        # range), with the PAPER the tag is printed on left out: it need not
        # lie flat, and depth on the printed pattern is the noisiest in the
        # frame. The whole frame would also take in the wall, the cabinet
        # frame and the fingers, and no longer be one plane.
        du, dv = np.abs(u - ctr[0]), np.abs(v - ctr[1])
        paper = (du < WRIST_PAPER_PX) & (dv < WRIST_PAPER_PX)
        patch = (du < WRIST_DOOR_PX) & (dv < WRIST_DOOR_PX) & ~paper
        fit = fit_plane(pts[patch], tol_m=0.004, iters=150)
        if fit is None or fit[2] < MIN_DOOR_INLIERS:
            print("wrist: door fit too weak (%s)" % (fit and "%.0f%% inliers" % (100 * fit[2])))
            continue
        n, cen, _frac = fit
        if n[2] > 0:
            n = -n  # toward the camera
        p = ray_plane(pixel_ray(ctr, wrist.k, wrist.dist), n, cen)
        if p is None:
            continue
        p_tags.append(p)
        normals.append(n)
        centres.append(cen)
        seen += 1
    tf_base_ee = tf_4x4(wrist.tf_buffer, "base_link", cfg["parent_frame"])
    T_base_wcam = tf_base_ee @ transform(quat_to_mat(*cfg["mount_quat_xyzw"]), cfg["mount_xyz"])
    p_w = np.median(np.array(p_tags), axis=0)
    n_w = np.mean(np.array(normals), axis=0)
    n_w /= np.linalg.norm(n_w)
    c_w = np.median(np.array(centres), axis=0)
    R = T_base_wcam[:3, :3]
    p_tag_base = R @ p_w + T_base_wcam[:3, 3]
    door_base = (R @ n_w, R @ c_w + T_base_wcam[:3, 3])
    scatter = 1000 * np.std(np.array(p_tags), axis=0).max()
    normal_scatter = max(angle_deg(n, n_w) for n in normals)
    print(
        "wrist: tag at %.3f m, centre in base [%.3f %.3f %.3f] (frame scatter %.1f mm), "
        "door normal in base %s (frame scatter %.1f deg)"
        % (np.linalg.norm(p_w), *p_tag_base, scatter, np.round(door_base[0], 3), normal_scatter)
    )
    return p_tag_base, door_base


def scene_side(node, scene, frames):
    """(p_tag_scene, door_scene, table_scene, table_inliers, info) from the
    scene camera: the tag centre over `frames` colour captures, the planes
    from the median depth."""
    scene.keep = frames
    spin_until(
        node,
        lambda: len(scene.colors) >= frames and len(scene.depths) >= frames
        and scene.k is not None and scene.kd is not None and scene.depth_to_color() is not None,
        30.0,
        "scene frames, intrinsics and the depth<-colour transform",
    )
    ctrs = [c.mean(axis=0) for c in (detect_tag(im) for im in scene.colors) if c is not None]
    if len(ctrs) < MIN_TAG_SEEN * frames:
        sys.exit("scene: tag seen in only %d/%d frames — is it in view and lit?" % (len(ctrs), frames))
    ctr = np.median(np.array(ctrs), axis=0)
    pc, uv = scene.cloud_in_color_frame(depth=scene.depth_median())
    near = (np.abs(uv[:, 0] - ctr[0]) < SCENE_DOOR_WINDOW_PX) & (np.abs(uv[:, 1] - ctr[1]) < SCENE_DOOR_WINDOW_PX)
    door = fit_plane(pc[near], tol_m=0.006)
    if door is None or door[2] < MIN_DOOR_INLIERS:
        sys.exit("scene: the door around the tag is not a clean plane (%s)" % (door and "%.0f%% inliers" % (100 * door[2])))
    h = scene.color.shape[0]
    low = (uv[:, 1] > TABLE_ROW_FRAC * h) & (pc[:, 2] > 0.3) & (pc[:, 2] < 1.5)
    table = fit_plane(pc[low], tol_m=0.006)
    if table is None or table[2] < MIN_TABLE_INLIERS:
        sys.exit("scene: no table plane in the lower image (%s)" % (table and "%.0f%% inliers" % (100 * table[2])))
    n_d, c_d, f_d = door
    n_t, c_t, f_t = table
    p_tag_scene = ray_plane(pixel_ray(ctr, scene.k, scene.dist), n_d, c_d)
    # the table's inlier points, for the height check afterwards
    inl = np.abs((pc[low] - c_t) @ n_t) < 0.006
    info = "scene: tag px (%.1f, %.1f) over %d frames (scatter %.2f px), %.3f m away; door %.0f%% inliers, table %.0f%%" % (
        *ctr, len(ctrs), np.std(np.array(ctrs), axis=0).max(), np.linalg.norm(p_tag_scene), 100 * f_d, 100 * f_t)
    print(info)
    return p_tag_scene, (n_d, c_d), (n_t, c_t), pc[low][inl], (float(ctr[0]), float(ctr[1]))


def report(scene, wrist, T_base_scene, table_pts, table_survey, stamp):
    """What the calibration implies for things it was not fitted to."""
    zs = (table_pts @ T_base_scene[:3, :3].T + T_base_scene[:3, 3])[:, 2]
    print(
        "check: %d table points -> base z median %.3f (bench survey %.3f), spread %.1f mm"
        % (len(zs), np.median(zs), table_survey, 1000 * np.std(zs))
    )
    T_sb = np.linalg.inv(T_base_scene)
    img = scene.color.copy()
    for f in (*FINGERTIP_FRAMES, "end_effector_link"):
        try:
            p = tf_4x4(wrist.tf_buffer, "base_link", f)[:3, 3]
        except Exception:
            print("check: no TF for %s" % f)
            continue
        pc = T_sb[:3, :3] @ p + T_sb[:3, 3]
        uvp, _ = cv2.projectPoints(pc.reshape(1, 3), np.zeros(3), np.zeros(3), scene.k, scene.dist)
        u, v = [int(round(x)) for x in uvp.ravel()]
        cv2.circle(img, (u, v), 8, (0, 255, 0), 2)
        cv2.putText(img, f.replace("robotiq_85_", ""), (u + 10, v - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        print("check: %-34s projects to scene pixel (%d, %d)" % (f, u, v))
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / ("%s.jpg" % stamp)
    cv2.imwrite(str(out), img)
    print("check: the arm's fingertips drawn on the scene image -> %s (they should sit on the fingers)" % out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=int, default=20, help="captures averaged per camera")
    ap.add_argument("--check", action="store_true", help="report against the saved calibration; write nothing")
    ap.add_argument("--out", default=str(CONFIG_DIR / SCENE_CALIB_FILE), help="where the yaml goes")
    args = ap.parse_args()

    import rclpy

    rclpy.init()
    node = rclpy.create_node("calibrate_scene_camera")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    bench = WorldStore(str(CONFIG_DIR / "world_bench.yaml"))
    table_survey = bench.table_top_z
    up_base = bench.table_normal
    try:
        # The wrist FIRST, alone on the node. The scene camera's 30 Hz of
        # 2.7 MB colour frames over best-effort loopback starves the wrist
        # streams once both are subscribed (2026-09-16: the same script got
        # wrist frames on one run and none for 15 s on the next).
        wrist = D405Grabber(node, need_depth=True)
        p_tag_base, door_base = (None, None) if args.check else wrist_side(node, wrist, args.frames)
        scene = SceneGrabber(node, keep=args.frames)
        p_tag_scene, door_scene, table_scene, table_pts, tag_px = scene_side(node, scene, args.frames)
        T_link_color = scene.link_to_color()
        if T_link_color is None:
            sys.exit("no TF %s -> %s from the scene camera driver" % (LINK_FRAME, COLOR_FRAME))
        if args.check:
            p = Path(args.out)
            if not p.exists():
                sys.exit("no calibration at %s" % p)
            _doc, T_base_link = load_scene_yaml(str(p))
            report(scene, wrist, T_base_link @ T_link_color, table_pts, table_survey, stamp)
            return
        cal = scene_pose_from_planes(p_tag_base, door_base, p_tag_scene, door_scene, table_scene,
                                     up_base=up_base, table_point_base=(0.0, 0.0, table_survey))
        print("planes: 'up' in base is the measured table normal %s (%.2f deg off +z)"
              % (np.round(up_base, 4), angle_deg(up_base, [0, 0, 1])))
        print(
            "planes: door-to-table angle %.2f deg in base, %.2f deg in the scene camera (mismatch %.2f, limit %.1f)"
            % (cal.door_table_deg_base, cal.door_table_deg_scene, cal.plane_angle_mismatch_deg, PLANE_ANGLE_TOL_DEG)
        )
        if cal.plane_angle_mismatch_deg > PLANE_ANGLE_TOL_DEG:
            sys.exit("REFUSED: the two cameras do not agree on the scene's geometry — nothing written")
        T = cal.T_base_scene
        print("SCENE CAMERA in base_link: [%.3f %.3f %.3f], looking along %s"
              % (*T[:3, 3], np.round(T[:3, 2], 3)))
        report(scene, wrist, T, table_pts, table_survey, stamp)
        T_base_link = optical_to_link(T, T_link_color)
        note = ("tag position via the wrist camera + door and table planes; %s; mismatch %.2f deg; "
                "%d frames per camera" % (stamp, cal.plane_angle_mismatch_deg, args.frames))
        doc = write_scene_yaml(args.out, T_base_link, LINK_FRAME, note=note, tag_px=tag_px)
        print("wrote %s: xyz %s quat %s" % (args.out, doc["xyz"], doc["quat_xyzw"]))
        print("rebuild (colcon) to install it; press_demo.launch.py then publishes base_link -> %s" % LINK_FRAME)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
