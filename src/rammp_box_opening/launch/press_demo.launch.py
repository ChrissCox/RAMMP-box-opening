"""Robot TF, the wrist camera and the OWL detector for the press demo.

    ros2 launch rammp_box_opening press_demo.launch.py

sheppy runs exactly this as its `box_opening` node (rammp-deployments,
december_2026 manifest, `bench` profile), so `sheppy up bench` is the whole
bringup; the line above is the no-sheppy bench. Then, in its own shell (the
CLI is human-run; --execute alone arms it):

    export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
    ros2 run rammp_box_opening press_demo --execute

The arm driver and the planner are NOT started here: they are sheppy's `arm`
and `planner` nodes (rammp-deployments, december_2026 manifest), containers
speaking Cyclone DDS. This launch adds what the mission needs on the host,
every node on Cyclone too:

  robot_state_publisher  kinova_gen3_description's description.launch.py —
                         the driver publishes /joint_states but no TF, and
                         the mission reads the finger-tip and camera-mount
                         frames from TF
  realsense2_camera      the wrist D405, aligned depth on — OFF by default:
                         sheppy's `wrist_camera` node owns that camera now.
                         camera:=true starts a host driver on the same topic
                         names instead, for a bench without sheppy.
  owl_detector           the persistent OWL bbox node (the model loads once)

A leftover camera driver keeps the D405 claimed ("Device or resource busy",
field 2026-08-26) and a leftover OWL node doubles GPU load, so strays are
stopped first — HOST processes only (runtime/strays.py): pgrep also sees
sheppy's containers, and those are not this launch's to stop.
"""

import os
import signal
import subprocess
from pathlib import Path
import time

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    GroupAction,
    IncludeLaunchDescription,
    OpaqueFunction,
    SetEnvironmentVariable,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

from rammp_box_opening.runtime.driver import REQUIRED_RMW
from rammp_box_opening.perception.scene_calib import SCENE_CALIB_FILE, load_scene_yaml
from rammp_box_opening.runtime.strays import host_pids

ARGS = [
    ("tf", "true", "start robot_state_publisher for the arm (false: TF comes from elsewhere)"),
    ("camera", "false", "start a host D405 driver (default: sheppy's wrist_camera node runs it)"),
    ("owl", "true", "start the persistent OWL bbox detector (loads once)"),
]


def _pgrep(pattern):
    return subprocess.run(["pgrep", "-af", pattern], capture_output=True, text=True).stdout


def _strays():
    found = [(pid, "realsense2_camera_node") for pid in host_pids(_pgrep("realsense2_camera_node"))]
    ours = "\n".join(
        ln for ln in _pgrep("owl_detector").splitlines() if "rammp_box_opening" in ln
    )
    found += [(pid, "owl_detector") for pid in host_pids(ours)]
    return found


def _sweep_strays():
    found = _strays()
    for pid, name in found:
        print("[press_demo.launch] stopping stray %s (pid %d)" % (name, pid))
        try:
            os.kill(pid, signal.SIGINT)
        except (ProcessLookupError, PermissionError):
            pass
    if found:
        time.sleep(1.5)
        for pid, _name in _strays():
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass


def _nodes(context, *_args, **_kwargs):
    _sweep_strays()

    def flag(name):
        return LaunchConfiguration(name).perform(context).strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )

    nodes = []
    if flag("tf"):
        nodes.append(
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    [FindPackageShare("kinova_gen3_description"), "/launch/description.launch.py"]
                )
            )
        )
    scene_calib = Path(
        FindPackageShare("rammp_box_opening").perform(context), "config", SCENE_CALIB_FILE
    )
    if scene_calib.exists():
        # the scene camera joins the arm's TF tree: base_link -> its link
        # frame, from scripts/calibrate_scene_camera.py. The driver hangs its
        # optical frames off that link itself.
        doc, _T = load_scene_yaml(str(scene_calib))
        x, y, z = doc["xyz"]
        qx, qy, qz, qw = doc["quat_xyzw"]
        nodes.append(
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name="scene_camera_tf",
                arguments=[
                    "--x", str(x), "--y", str(y), "--z", str(z),
                    "--qx", str(qx), "--qy", str(qy), "--qz", str(qz), "--qw", str(qw),
                    "--frame-id", doc["parent_frame"], "--child-frame-id", doc["child_frame"],
                ],
            )
        )
    if flag("owl"):
        # persistent semantic gate: the OWL model loads ONCE here instead of
        # once per CLI run (field 2026-09-01: per-run loading made every
        # detect wait)
        nodes.append(
            Node(
                package="rammp_box_opening",
                executable="owl_detector",
                name="owl_detector",
                output="screen",
            )
        )
        if scene_calib.exists():
            # ... and a second instance watching the scene camera, which finds
            # the box before the arm moves. Each infers only inside its own
            # enable window, so the two never load the GPU together.
            nodes.append(
                Node(
                    package="rammp_box_opening",
                    executable="owl_detector",
                    name="owl_scene",
                    output="screen",
                    # a lower floor than the wrist's: from a metre the box
                    # scores 0.18-0.23, and the lid-slab geometry behind
                    # this box rejects anything that is not a box top
                    parameters=[{"camera": "scene", "min_score": 0.12}],
                )
            )
    if flag("camera"):
        camera = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                [FindPackageShare("realsense2_camera"), "/launch/rs_launch.py"]
            ),
            launch_arguments={
                # the names sheppy's wrist_camera node publishes, so the
                # mission's config is the same either way
                "camera_namespace": "/",
                "camera_name": "wrist_camera",
                # the watcher's depth refinement reads depth at the
                # box's COLOR pixel — alignment is required
                "align_depth.enable": "true",
            }.items(),
        )
        # forwarding=False: rs_launch.py warns (80 names each) about every
        # launch configuration it does not declare — ours leaked into it
        # and buried real camera warnings
        nodes.append(GroupAction([camera], forwarding=False))
    return nodes


def generate_launch_description():
    return LaunchDescription(
        # the driver and planner containers speak Cyclone; so must every
        # node started here (Fast DDS discovers them, then loses the data)
        [SetEnvironmentVariable("RMW_IMPLEMENTATION", REQUIRED_RMW)]
        + [DeclareLaunchArgument(n, default_value=d, description=h) for n, d, h in ARGS]
        + [OpaqueFunction(function=_nodes)]
    )
