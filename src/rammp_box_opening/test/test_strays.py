"""The launch's stray sweep may only ever touch host processes: sheppy runs
the driver, the planner and (later) the cameras in containers, and those
processes are visible — and killable — from the host."""

from rammp_box_opening.runtime.strays import host_pids, in_container

# read on abra 2026-09-11: this shell, sheppy-arm's kinova_gen3_node,
# sheppy-planner's planner_node
HOST = "0::/user.slice/user-1000.slice/session-2.scope\n"
ARM = "0::/system.slice/docker-4615a3a9e913910c608129c39f857e2b1f9137fbee1657b2b87d078af0465105.scope\n"
PLANNER = "0::/system.slice/docker-585cc5f96c36b9714f2c264f01dc8050880ff5c4e29f936a55f1d560612ac410.scope\n"


def test_container_processes_are_recognised():
    assert in_container(ARM)
    assert in_container(PLANNER)
    assert in_container("12:pids:/docker/4615a3a9e913\n11:memory:/docker/4615a3a9e913\n")


def test_a_host_process_is_not_in_a_container():
    assert not in_container(HOST)


def test_only_host_processes_are_strays():
    pgrep = (
        "812 /opt/ros/humble/lib/realsense2_camera/realsense2_camera_node --ros-args\n"
        "913 /opt/ros/humble/lib/realsense2_camera/realsense2_camera_node --ros-args\n"
    )
    cgroups = {812: HOST, 913: ARM}
    assert host_pids(pgrep, cgroups.get) == [812]


def test_a_process_that_vanished_is_not_a_stray():
    pgrep = "812 /opt/ros/humble/lib/realsense2_camera/realsense2_camera_node\n"
    assert host_pids(pgrep, lambda pid: None) == []
