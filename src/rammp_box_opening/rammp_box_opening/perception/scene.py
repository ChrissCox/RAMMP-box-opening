"""The scene camera: the Orbbec Gemini on the fixed mount, seeing the whole
workspace. Colour at 1280x720, depth at 848x480 with its own intrinsics and
its own optical frame; the driver publishes the depth<-colour extrinsic in
its TF tree, and `cloud_in_color_frame` uses it so every depth point can be
placed in the colour image and, through config/camera_scene.yaml, in
base_link.

Stores only what arrives; callers spin their own node.
"""

from pathlib import Path

import numpy as np

from rammp_box_opening.perception.d405 import quat_to_mat
from rammp_box_opening.perception.planes import cloud_from_depth
from rammp_box_opening.perception.scene_calib import SCENE_CALIB_FILE, load_scene_yaml, transform

SCENE_NS = "/scene_camera"
COLOR_TOPIC = SCENE_NS + "/color/image_raw"
COLOR_INFO = SCENE_NS + "/color/camera_info"
DEPTH_TOPIC = SCENE_NS + "/depth/image_raw"
DEPTH_INFO = SCENE_NS + "/depth/camera_info"
LINK_FRAME = "scene_camera_link"
COLOR_FRAME = "scene_camera_color_optical_frame"


def scene_calib_path():
    """The installed calibration, or None when the camera is uncalibrated."""
    from ament_index_python.packages import get_package_share_directory

    p = Path(get_package_share_directory("rammp_box_opening"), "config", SCENE_CALIB_FILE)
    return p if p.exists() else None


class SceneGrabber:
    def __init__(self, node, keep=1, need_depth=True):
        from tf2_ros import Buffer, TransformListener

        self.node = node
        self.keep = int(keep)
        self.need_depth = bool(need_depth)
        self.colors = []  # newest last, up to `keep`
        self.depths = []
        self.color_stamp = None
        self.k = self.dist = None  # colour intrinsics
        self.kd = None  # depth intrinsics
        self.depth_frame = None
        self.tf_buffer = Buffer()
        # spin_thread=False: the listener's own executor thread would spin
        # this node alongside the one that already spins it (the OWL node's
        # rclpy.spin, the mission's perception executor) — two executors
        # on one node, which rclpy does not guard. The OWL scene instance
        # went deaf after its first inferences three times on 2026-09-17
        # (ticks and params alive, image callbacks never delivered again,
        # re-subscribing did nothing). Whoever spins the node feeds TF.
        self.tf_listener = TransformListener(self.tf_buffer, node, spin_thread=False)
        self._subs = []
        self.resubscribes = 0
        self._subscribe()

    def _subscribe(self):
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import CameraInfo, Image

        node = self.node
        self._subs = [
            node.create_subscription(Image, COLOR_TOPIC, self._color_cb, qos_profile_sensor_data),
            node.create_subscription(CameraInfo, COLOR_INFO, self._info_cb, qos_profile_sensor_data),
        ]
        if self.need_depth:
            self._subs += [
                node.create_subscription(Image, DEPTH_TOPIC, self._depth_cb, qos_profile_sensor_data),
                node.create_subscription(CameraInfo, DEPTH_INFO, self._dinfo_cb, qos_profile_sensor_data),
            ]

    def resubscribe(self):
        """Tear the image subscriptions down and create them afresh. A
        long-lived OWL instance stopped receiving scene frames minutes
        after start while a fresh subscriber in another process got them
        at 30 Hz (2026-09-16, twice); a fresh subscription is the repair
        that needs no launch restart."""
        for sub in self._subs:
            try:
                self.node.destroy_subscription(sub)
            except Exception:
                pass
        self._subs = []
        self.resubscribes += 1
        self._subscribe()

    # -- callbacks ---------------------------------------------------------
    def _push(self, lst, item):
        lst.append(item)
        del lst[: -self.keep]

    def _color_cb(self, msg):
        if msg.encoding not in ("rgb8", "bgr8"):
            return
        a = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        self._push(self.colors, a[:, :, ::-1].copy() if msg.encoding == "rgb8" else a.copy())
        self.color_stamp = msg.header.stamp

    def _depth_cb(self, msg):
        if msg.encoding != "16UC1":
            return
        d = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
        self._push(self.depths, d.astype(np.float32) / 1000.0)

    def _info_cb(self, msg):
        self.k = np.array(msg.k, dtype=float).reshape(3, 3)
        self.dist = np.array(msg.d, dtype=float)

    def _dinfo_cb(self, msg):
        self.kd = np.array(msg.k, dtype=float).reshape(3, 3)
        self.depth_frame = msg.header.frame_id

    # -- state -------------------------------------------------------------
    @property
    def color(self):
        return self.colors[-1] if self.colors else None

    @property
    def depth(self):
        return self.depths[-1] if self.depths else None

    def depth_median(self):
        """The median of the kept depth frames, zeros ignored — a steadier
        surface than any single frame."""
        if not self.depths:
            return None
        import warnings

        stack = np.stack(self.depths)
        stack = np.where(stack > 0, stack, np.nan)
        # pixels with no depth in any frame are all-NaN columns: nanmedian
        # warns once per such column (83k warnings, 50 ms, and the noise at
        # every mission start) — they are simply holes
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(stack, axis=0)
        return np.nan_to_num(med, nan=0.0)

    def missing(self):
        out = []
        if not self.colors:
            out.append("colour")
        if self.k is None:
            out.append("colour camera_info")
        if self.need_depth and not self.depths:
            out.append("depth")
        if self.need_depth and self.kd is None:
            out.append("depth camera_info")
        return out

    def lookup(self, parent, child):
        """base_link-style 4x4 from TF, or None."""
        import rclpy.time as rt

        try:
            tf = self.tf_buffer.lookup_transform(parent, child, rt.Time())
        except Exception:
            return None
        t, r = tf.transform.translation, tf.transform.rotation
        return transform(quat_to_mat(r.x, r.y, r.z, r.w), [t.x, t.y, t.z])

    def depth_to_color(self):
        """colour optical <- depth optical, from the driver's own tree."""
        if self.depth_frame is None:
            return None
        return self.lookup(COLOR_FRAME, self.depth_frame)

    def link_to_color(self):
        return self.lookup(LINK_FRAME, COLOR_FRAME)

    def cloud_in_color_frame(self, depth=None, stride=1):
        """(points (N,3) in the colour optical frame, their colour pixels
        (N,2)) for the valid depth pixels. None until depth, intrinsics and
        the depth<-colour extrinsic have all arrived."""
        d = self.depth if depth is None else depth
        T_c_d = self.depth_to_color()
        if d is None or self.kd is None or self.k is None or T_c_d is None:
            return None
        return color_cloud(d, self.kd, self.k, self.dist, T_c_d, stride=stride)


def color_cloud(depth, kd, k, dist, T_color_depth, stride=1):
    """(points (N,3) in the colour optical frame, their colour pixels (N,2))
    for the valid pixels of a depth image with intrinsics `kd`, moved into
    the colour camera by `T_color_depth` and projected with its `k` and
    `dist`. The live grabber and an offline replay of recorded arrays both
    compute it here."""
    import cv2

    pts, _u, _v = cloud_from_depth(depth, kd, stride=stride)
    T = np.asarray(T_color_depth, float)
    pc = pts @ T[:3, :3].T + T[:3, 3]
    uv, _ = cv2.projectPoints(
        pc.reshape(-1, 1, 3).astype(np.float64), np.zeros(3), np.zeros(3), np.asarray(k, float),
        None if dist is None or not len(np.atleast_1d(dist)) else np.asarray(dist, float),
    )
    return pc, uv.reshape(-1, 2)


def load_scene_calibration(path=None):
    """base_link <- scene_camera_link (4x4) from the calibration yaml — the
    installed one unless `path` names another; None when uncalibrated."""
    p = scene_calib_path() if path is None else Path(path)
    if p is None or not p.exists():
        return None
    _doc, T_base_link = load_scene_yaml(str(p))
    return T_base_link
