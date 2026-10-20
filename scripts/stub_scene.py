#!/usr/bin/env python3
"""Synthetic scene camera + scene OWL stub for stub-isolated e2e.

The fixed scene camera's topic surface (colour 1280x720, depth 848x480 with
its own intrinsics, both camera_infos, and the driver's own TF tree from
scene_camera_link to its optical frames), with the depth RAY-CAST from the
same table-plus-box-top renderer the wrist stub and the unit tests use,
seen from the pose a calibration yaml gives — exactly the file
scripts/calibrate_scene_camera.py writes and press_demo --scene-calib reads.
So the mission's SceneLocator runs the REAL path end to end: bbox, depth
cloud, the driver's depth<-colour extrinsic, the calibration, the lid slab,
the pose.

It also stands in for the scene instance of owl_detector: a bbox around
the box's top face on the scene bbox topic, or a heartbeat with --no-box.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float32MultiArray
from tf2_ros import StaticTransformBroadcaster

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src/rammp_box_opening/test"))
from test_depth_source import H, K, W, render_depth  # noqa: E402  (848x480 depth)

from rammp_box_opening.models.container import ContainerModel  # noqa: E402
from rammp_box_opening.perception.d405 import mat_to_quat_xyzw, quat_to_mat  # noqa: E402
from rammp_box_opening.perception.owl_source import topics_for  # noqa: E402
from rammp_box_opening.perception.scene import (  # noqa: E402
    COLOR_FRAME,
    COLOR_INFO,
    COLOR_TOPIC,
    DEPTH_INFO,
    DEPTH_TOPIC,
    LINK_FRAME,
)
from rammp_box_opening.perception.scene_calib import load_scene_yaml, transform  # noqa: E402

DEPTH_FRAME = "scene_camera_depth_optical_frame"
CW, CH = 1280, 720
KC = np.array([[612.7, 0.0, 635.4], [0.0, 612.7, 361.0], [0.0, 0.0, 1.0]])  # the Gemini's colour K
# the driver's link -> optical: the usual optical rotation, lens 24 mm off the link
T_LINK_OPTICAL = transform(quat_to_mat(0.5, -0.5, 0.5, -0.5), [0.0, -0.024, 0.0])
OWL_SCORE = 0.21  # what the real scene instance reports for the bench box


def tf_msg(node, parent, child, T):
    m = TransformStamped()
    m.header.stamp = node.get_clock().now().to_msg()
    m.header.frame_id = parent
    m.child_frame_id = child
    m.transform.translation.x, m.transform.translation.y, m.transform.translation.z = [float(v) for v in T[:3, 3]]
    q = mat_to_quat_xyzw(T[:3, :3])
    m.transform.rotation.x, m.transform.rotation.y, m.transform.rotation.z, m.transform.rotation.w = [float(v) for v in q]
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calib", required=True, help="camera_scene.yaml: base_link -> scene_camera_link")
    ap.add_argument("--container", default=str(REPO / "src/rammp_box_opening/config/containers/oxo_pop.yaml"))
    ap.add_argument("--box-x", type=float, default=0.45)
    ap.add_argument("--box-y", type=float, default=0.0)
    ap.add_argument("--box-yaw-deg", type=float, default=0.0)
    ap.add_argument("--table-z", type=float, default=-0.027)
    ap.add_argument("--no-box", action="store_true", help="an empty table")
    a = ap.parse_args()

    model = ContainerModel.load(a.container)
    _doc, T_base_link = load_scene_yaml(a.calib)
    T_base_color = T_base_link @ T_LINK_OPTICAL
    R_cam, t_cam = T_base_color[:3, :3], T_base_color[:3, 3]
    top_z = a.table_z + model.dims[2]
    yaw = np.radians(a.box_yaw_deg)
    boxes = [] if a.no_box else [(a.box_x, a.box_y, top_z, model.dims[0], model.dims[1], yaw)]
    depth_m = render_depth(boxes, rot_cam=R_cam, t_cam=t_cam, table_z=a.table_z)
    depth = np.where(np.isfinite(depth_m), depth_m * 1000.0, 0.0).round().astype(np.uint16)
    # the colour frame: flat grey; the scene OWL is stubbed, nothing reads pixels
    canvas = np.full((CH, CW, 3), 200, np.uint8)
    bbox = None
    if boxes:
        c, s = np.cos(yaw), np.sin(yaw)
        hx, hy = model.dims[0] / 2, model.dims[1] / 2
        px = []
        for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            p = np.array([a.box_x + c * sx * hx - s * sy * hy, a.box_y + s * sx * hx + c * sy * hy, top_z])
            pc = R_cam.T @ (p - t_cam)
            px.append([KC[0, 0] * pc[0] / pc[2] + KC[0, 2], KC[1, 1] * pc[1] / pc[2] + KC[1, 2]])
        px = np.array(px)
        pad = 12
        bbox = [float(px[:, 0].min() - pad), float(px[:, 1].min() - pad), float(px[:, 0].max() + pad), float(px[:, 1].max() + pad)]

    rclpy.init()
    node = rclpy.create_node("stub_scene")
    pub_c = node.create_publisher(Image, COLOR_TOPIC, 10)
    pub_ci = node.create_publisher(CameraInfo, COLOR_INFO, 10)
    pub_d = node.create_publisher(Image, DEPTH_TOPIC, 10)
    pub_di = node.create_publisher(CameraInfo, DEPTH_INFO, 10)
    bbox_topic, _enable = topics_for("scene")
    pub_owl = node.create_publisher(Float32MultiArray, bbox_topic, 1)
    # the driver's own tree: link -> colour optical, link -> depth optical
    # (coincident here). base_link -> link is the launch's job, from the
    # same calibration file this stub renders from.
    StaticTransformBroadcaster(node).sendTransform([
        tf_msg(node, LINK_FRAME, COLOR_FRAME, T_LINK_OPTICAL),
        tf_msg(node, LINK_FRAME, DEPTH_FRAME, T_LINK_OPTICAL),
    ])
    last = {"t": None}

    def publish():
        stamp = node.get_clock().now().to_msg()
        last["t"] = stamp.sec + stamp.nanosec * 1e-9
        im = Image()
        im.header.stamp = stamp
        im.header.frame_id = COLOR_FRAME
        im.height, im.width, im.encoding, im.step = CH, CW, "bgr8", CW * 3
        im.data = canvas.tobytes()
        pub_c.publish(im)
        ci = CameraInfo()
        ci.header.stamp = stamp
        ci.header.frame_id = COLOR_FRAME
        ci.height, ci.width = CH, CW
        ci.k = [float(v) for v in KC.ravel()]
        ci.d = [0.0] * 5
        pub_ci.publish(ci)
        dm = Image()
        dm.header.stamp = stamp
        dm.header.frame_id = DEPTH_FRAME
        dm.height, dm.width, dm.encoding, dm.step = H, W, "16UC1", W * 2
        dm.data = depth.tobytes()
        pub_d.publish(dm)
        di = CameraInfo()
        di.header.stamp = stamp
        di.header.frame_id = DEPTH_FRAME
        di.height, di.width = H, W
        di.k = [float(v) for v in K.ravel()]
        di.d = [0.0] * 5
        pub_di.publish(di)

    def owl():
        msg = Float32MultiArray()
        if bbox is None or last["t"] is None:
            msg.data = [0.0, 0.0, 0.0, 0.0, -1.0, 0.0]
        else:
            now = node.get_clock().now().nanoseconds * 1e-9
            msg.data = [*bbox, OWL_SCORE, max(0.0, now - last["t"])]
        pub_owl.publish(msg)

    node.create_timer(1.0 / 15.0, publish)
    node.create_timer(0.5, owl)
    print("STUB SCENE READY (%s; camera at [%.3f %.3f %.3f]%s)" % (
        "empty table" if a.no_box else "box at [%.2f %.2f] yaw %.0f" % (a.box_x, a.box_y, a.box_yaw_deg),
        *t_cam, "" if bbox is None else "; bbox (%.0f,%.0f)-(%.0f,%.0f)" % tuple(bbox)), flush=True)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
