"""The wrist D405: its streams, where it sits on the arm, and the fix gate.

Box-opening owns these since RAMMP-CuRobo v1.0.0 became a pure planner and
deleted its perception (seek_core, cameras, perception.py and the D405
mount YAML) — they live nowhere else now, so there is no upstream copy to
drift from. Only what this repo uses came along: the grabber's three
streams and its mount, the n-agreeing-sightings commit gate, and the
quaternion math the mount composition needs.
"""

import os

import numpy as np
import yaml

CAMERA_CONFIG = "camera_d405_wrist.yaml"


def camera_config(name=CAMERA_CONFIG):
    """config/<name> from the installed package, else from the source tree."""
    candidates = []
    try:
        from ament_index_python.packages import get_package_share_directory

        share = get_package_share_directory("rammp_box_opening")
        candidates.append(os.path.join(share, "config", name))
    except Exception:
        pass
    package_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    candidates.append(os.path.join(package_root, "config", name))
    for path in candidates:
        if os.path.isfile(path):
            with open(path) as f:
                return yaml.safe_load(f)
    raise FileNotFoundError("camera config %r not found (tried %s)" % (name, candidates))


def quat_to_mat(x, y, z, w):
    """xyzw quaternion (normalized here) -> 3x3 rotation matrix."""
    n = (x * x + y * y + z * z + w * w) ** 0.5 or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def mat_to_quat_xyzw(m):
    """3x3 rotation -> xyzw quaternion (the inverse of quat_to_mat)."""
    m = np.asarray(m, dtype=float)
    t = float(np.trace(m))
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        return np.array(
            [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s]
        )
    i = int(np.argmax(np.diag(m)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = np.sqrt(1.0 + m[i, i] - m[j, j] - m[k, k]) * 2.0
    q = np.zeros(4)
    q[i] = 0.25 * s
    q[j] = (m[j, i] + m[i, j]) / s
    q[k] = (m[k, i] + m[i, k]) / s
    q[3] = (m[k, j] - m[j, k]) / s
    return q / np.linalg.norm(q)


def stable_fix(samples, tol, n):
    """Median position once the newest `n` sightings agree pairwise within
    `tol` metres, else None. `samples` = [(position, t), ...], newest last.

    The acquisition gate: one frame never moves the arm — n agreeing frames
    from a still camera do."""
    if len(samples) < n:
        return None
    pts = np.array([np.asarray(p, dtype=float) for p, _t in samples[-n:]])
    spread = np.linalg.norm(pts[:, None] - pts[None, :], axis=2)
    if float(spread.max()) > tol:
        return None
    return np.median(pts, axis=0)


class D405Grabber:
    """The newest colour frame (BGR), aligned depth (metres) and intrinsics
    of the wrist D405, with what composing the camera's pose needs: a TF
    buffer, the parent frame and the mount (config/camera_d405_wrist.yaml).
    It only stores what arrives; callers spin their own node.

    need_depth=False (the OWL node): colour and intrinsics only."""

    def __init__(self, node, need_depth=True):
        from tf2_ros import Buffer, TransformListener

        self.node = node
        cfg = camera_config()
        self.parent = cfg["parent_frame"]
        self.mount_xyz = np.asarray(cfg["mount_xyz"], dtype=float)
        self.mount_quat = list(cfg["mount_quat_xyzw"])
        # the stamp lag against the arm (config/camera_d405_wrist.yaml);
        # camera_pose_at looks the pose up at stamp + this
        self.stamp_offset_s = float(cfg.get("stamp_offset_s", 0.0) or 0.0)
        ns = cfg["depth_topic"].rsplit("/depth/", 1)[0]
        self.tf_buffer = Buffer()
        # spin_thread=False: the listener's own executor thread would spin
        # this node alongside the one that already spins it (the OWL node's
        # rclpy.spin, the mission's perception executor) — two executors
        # on one node, which rclpy does not guard. The OWL scene instance
        # went deaf after its first inferences three times on 2026-09-17
        # (ticks and params alive, image callbacks never delivered again,
        # re-subscribing did nothing). Whoever spins the node feeds TF.
        self.tf_listener = TransformListener(self.tf_buffer, node, spin_thread=False)
        self.need_depth = bool(need_depth)
        self.color = self.depth = self.info = None
        self.k = self.dist = None
        self.color_stamp = None
        self._ns = ns
        self._subs = []
        self.resubscribes = 0
        self._subscribe()

    def _subscribe(self):
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import CameraInfo, Image

        node, ns = self.node, self._ns
        self._subs = [
            node.create_subscription(
                Image, ns + "/color/image_raw", self._color_cb, qos_profile_sensor_data
            ),
            node.create_subscription(
                CameraInfo, ns + "/color/camera_info", self._info_cb, qos_profile_sensor_data
            ),
        ]
        if self.need_depth:
            self._subs.append(
                node.create_subscription(
                    Image,
                    ns + "/aligned_depth_to_color/image_raw",
                    self._depth_cb,
                    qos_profile_sensor_data,
                )
            )

    def resubscribe(self):
        """Fresh image subscriptions (see SceneGrabber.resubscribe)."""
        for sub in self._subs:
            try:
                self.node.destroy_subscription(sub)
            except Exception:
                pass
        self._subs = []
        self.resubscribes += 1
        self._subscribe()

    def _color_cb(self, msg):
        if msg.encoding in ("rgb8", "bgr8"):
            a = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
            self.color = a[:, :, ::-1].copy() if msg.encoding == "rgb8" else a.copy()
            self.color_stamp = msg.header.stamp

    def _depth_cb(self, msg):
        if msg.encoding == "16UC1":
            self.depth = (
                np.frombuffer(msg.data, dtype=np.uint16)
                .reshape(msg.height, msg.width)
                .astype(np.float32)
                / 1000.0
            )

    def _info_cb(self, msg):
        k = np.array(msg.k).reshape(3, 3)
        self.k = k
        self.dist = np.array(msg.d, dtype=float).ravel()
        self.info = dict(fx=k[0, 0], fy=k[1, 1], cx=k[0, 2], cy=k[1, 2])

    def missing(self):
        """Which streams have not arrived yet (diagnosis)."""
        out = []
        if self.color is None:
            out.append("color")
        if self.need_depth and self.depth is None:
            out.append("ALIGNED depth (align_depth.enable:=true?)")
        if self.info is None:
            out.append("camera_info")
        return out
