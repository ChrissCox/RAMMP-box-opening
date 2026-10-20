"""Every-session preflight (spec §4): the checks a mission needs to pass.

FAIL (nonzero exit): this shell not on Cyclone DDS; /joint_states missing
or without effort fields; the driver's trajectory action or the planner's
actions unreachable; the planner unable to load a world this repo writes
(it runs in a container, so the worlds directory must be mounted into it
at the same path); no TF for the finger tips; no gripper reported; a
wrist camera stream the mission reads silent (colour, camera_info, and the
ALIGNED depth, which only exists with align_depth.enable:=true).
WARN (exit unaffected — the mission has a fallback): a silent scene camera
stream (it searches with the wrist instead), an owl_detector not running,
or running but deaf (no frame reaching it).
Report-only: the gripper's position.

Preflight never moves anything: its client is not armed.
"""

import os
import time

import rclpy

from rammp_box_opening.constants import FINGERTIP_FRAMES, GRIPPER_STATE_TOPIC
from rammp_box_opening.runtime.client import PlannerClient
from rammp_box_opening.runtime.driver import rmw_refusal
from rammp_box_opening.tasks import cli_common
from rammp_box_opening.worlds import WorldStore


STREAM_WINDOW_S = 3.0  # how long every camera and detector topic is listened to


def camera_topics():
    """(topic, required, fix) for every camera stream the mission reads:
    required ones FAIL preflight when silent, the rest WARN. The wrist
    namespace comes from the same config the mission's grabber uses."""
    from rammp_box_opening.perception.d405 import camera_config
    from rammp_box_opening.perception.scene import COLOR_INFO, COLOR_TOPIC, DEPTH_INFO, DEPTH_TOPIC

    ns = camera_config()["depth_topic"].rsplit("/depth/", 1)[0]
    wrist = "is sheppy's wrist_camera node up, and the D405 on USB (lsusb | grep 8086)?"
    scene = "the mission will search with the wrist instead — is sheppy's scene_camera node up?"
    return [
        (ns + "/color/image_raw", True, wrist),
        (ns + "/color/camera_info", True, wrist),
        (
            ns + "/aligned_depth_to_color/image_raw",
            True,
            "the wrist camera must launch with align_depth.enable:=true (sheppy manifest, "
            "wrist_camera command) — without it every wrist fix fails",
        ),
        (COLOR_TOPIC, False, scene),
        (COLOR_INFO, False, scene),
        (DEPTH_TOPIC, False, scene),
        (DEPTH_INFO, False, scene),
    ]


def stream_verdicts(topics, counts, window_s=STREAM_WINDOW_S):
    """[(topic, "PASS" | "FAIL" | "WARN", text)] from how many messages each
    topic delivered in the window. Pure."""
    out = []
    for topic, required, fix in topics:
        n = int(counts.get(topic, 0))
        if n:
            out.append((topic, "PASS", "%s at %.1f Hz" % (topic, n / float(window_s))))
        else:
            out.append((topic, "FAIL" if required else "WARN", "%s silent for %.0f s — %s" % (topic, window_s, fix)))
    return out


def owl_verdict(scores, camera):
    """("PASS" | "WARN", text) for one owl_detector from the score slots of
    the messages it sent in the window: none means it is not running; only
    BLIND heartbeats mean it is up but no camera frame reaches it — the
    "deaf" state a restart has always cured (2026-09-16, -17, -22). Pure."""
    from rammp_box_opening.perception.owl_source import BLIND

    if not scores:
        return "WARN", (
            "owl_detector (%s camera) not running — just restarted? it takes ~9 s to load; run "
            "preflight again. Otherwise: is sheppy's box_opening node up? (the mission falls back "
            "to the cloud rung, then plain depth)" % camera
        )
    if all(s <= BLIND for s in scores):
        return "WARN", (
            "owl_detector (%s camera) is up but has had no %s frame — `sheppy restart box_opening` "
            "(the camera itself is checked above)" % (camera, camera)
        )
    return "PASS", "owl_detector (%s camera) up and receiving frames" % camera


def _listen(node, topics, window_s=STREAM_WINDOW_S):
    """{topic: message count} and {camera: [owl scores]} over one window."""
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CameraInfo, Image
    from std_msgs.msg import Float32MultiArray

    from rammp_box_opening.perception.owl_source import topics_for

    counts = {t: 0 for t, _r, _f in topics}
    owl = {"wrist": [], "scene": []}
    subs = []

    def counter(topic):
        def cb(_msg):
            counts[topic] += 1

        return cb

    for topic, _r, _f in topics:
        kind = CameraInfo if topic.endswith("camera_info") else Image
        subs.append(node.create_subscription(kind, topic, counter(topic), qos_profile_sensor_data))
    for cam in owl:
        subs.append(
            node.create_subscription(
                Float32MultiArray, topics_for(cam)[0], lambda m, cam=cam: owl[cam].append(float(m.data[4])) if len(m.data) >= 5 else None, 1
            )
        )
    t0 = time.monotonic()
    while time.monotonic() - t0 < window_s:
        rclpy.spin_once(node, timeout_sec=0.05)
    for sub in subs:
        node.destroy_subscription(sub)
    return counts, owl


def _gripper_state(node, timeout_s=3.0):
    """One /gripper_state message, or None."""
    from rclpy.qos import qos_profile_sensor_data

    from rammp_arm_interfaces.msg import GripperState

    got = []
    sub = node.create_subscription(
        GripperState, GRIPPER_STATE_TOPIC, got.append, qos_profile_sensor_data
    )
    t0 = time.monotonic()
    while not got and time.monotonic() - t0 < timeout_s:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_subscription(sub)
    return got[0] if got else None


def main():
    args = cli_common.make_parser(__doc__).parse_args()
    why = rmw_refusal(os.environ)
    if why:
        print("FAIL  " + why)
        raise SystemExit(1)

    rclpy.init()
    node = rclpy.create_node("rammp_box_opening_preflight")
    client = PlannerClient(node)  # not armed: nothing here can move
    failures = 0

    # 1. /joint_states fresh, with efforts (guarded primitives require them)
    t0 = time.monotonic()
    try:
        client.joints()
        fresh = time.monotonic() - t0
        efforts = client.efforts_present()
        print(
            "PASS  /joint_states fresh (%.1f s) — efforts %s"
            % (fresh, "present" if efforts else "MISSING")
        )
        if not efforts:
            print("FAIL  effort fields absent — guarded primitives will refuse")
            failures += 1
    except SystemExit as exc:
        print("FAIL  %s" % exc)
        raise SystemExit(1) from exc

    # 2. the driver's trajectory action
    if client.driver_reachable(timeout_s=5.0):
        print("PASS  driver /execute_joint_trajectory reachable")
    else:
        print("FAIL  /execute_joint_trajectory unreachable — is sheppy's `arm` node up?")
        failures += 1

    # 3. the planner's actions
    if client.planner_reachable(timeout_s=5.0):
        print("PASS  planner actions reachable")
    else:
        print("FAIL  planner actions unreachable — is sheppy's `planner` node up?")
        failures += 1

    # 4. the planner can read the worlds this repo writes. It runs in a
    #    container: a path that exists here is invisible there unless its
    #    directory is mounted at the same path.
    store = WorldStore(args.bench_world or cli_common.default_bench_yaml())
    _name, path = store.push_name("bench")
    ok, msg = client.set_world(path)
    if ok:
        print("PASS  planner loaded %s" % path)
    else:
        print(
            "FAIL  planner could not load %s (%s) — mount %s into the planner "
            "container at the same path" % (path, msg, path.parent)
        )
        failures += 1

    # 5. TF for the finger tips (robot_state_publisher from the launch)
    tips = client.contact_xyz(timeout_s=3.0)
    if tips is not None:
        print("PASS  TF base_link -> %s" % ", ".join(FINGERTIP_FRAMES))
    else:
        print(
            "FAIL  no TF for the finger tips — run press_demo.launch.py "
            "(tf:=true starts robot_state_publisher)"
        )
        failures += 1

    # 6. the gripper
    state = _gripper_state(node)
    if state is None:
        print("FAIL  no %s — is sheppy's `arm` node up?" % GRIPPER_STATE_TOPIC)
        failures += 1
    elif not state.present:
        print("FAIL  the driver reports no gripper")
        failures += 1
    else:
        print("PASS  gripper present, at %.3f (0 open .. 1 closed)" % state.position)

    # 7. the camera streams the mission reads, and the two OWL detectors
    #    (a mission used to discover a missing stream mid-run: 2026-09-22)
    topics = camera_topics()
    counts, owl = _listen(node, topics)
    for _topic, level, text in stream_verdicts(topics, counts):
        print("%-4s  %s" % (level, text))
        failures += level == "FAIL"
    for cam in ("wrist", "scene"):
        level, text = owl_verdict(owl[cam], cam)
        print("%-4s  %s" % (level, text))

    raise SystemExit(1 if failures else 0)
