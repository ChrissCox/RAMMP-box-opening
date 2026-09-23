"""Shared constants: arm facts and runner defaults (spec §3, §6)."""

import os
from pathlib import Path


def state_dir(env=None):
    """Where run logs, captures, generated worlds and calibration residuals
    are kept: ~/.ros/rammp_box_opening, or $RAMMP_BOX_OPENING_STATE. The
    stub harnesses point it at their own folder — their synthetic runs once
    filled the real residuals file (2026-09-17), and the every-run captures
    are pruned to the newest few, real or not."""
    env = os.environ if env is None else env
    return Path(env.get("RAMMP_BOX_OPENING_STATE") or Path.home() / ".ros" / "rammp_box_opening")


# Gen3 home joints in controller order — FK-verified in RAMMP-CuRobo's
# tour_demo.py; joint_3 sits AT +pi (every comparison uses ang_diff).
HOME = [0.0, 0.262, 3.142, -2.269, 0.0, 0.960, 1.571]
JOINTS = ["joint_%d" % i for i in range(1, 8)]

NODE_NAMESPACE = "/rammp_curobo"  # the planner container's actions and service
# The kinova-gen3-ros2 driver (sheppy's `arm` node): trajectories and the gripper.
EXECUTE_ACTION = "/execute_joint_trajectory"
GRIPPER_SETPOINT_TOPIC = "/setpoint/gripper"
GRIPPER_STATE_TOPIC = "/gripper_state"
# Sent with every gripper setpoint — the driver keeps neither between commands.
# speed: fraction of the maximum closing speed. force: a CURRENT CEILING
# (fraction of 1.0 A) the fingers stall at, not a force setpoint. Both are the
# driver's defaults; tune at the bench against open_box.grip_band.
GRIPPER_SPEED = 1.0
GRIPPER_FORCE = 0.5

# Cruise fraction for free-air motion, NOT a time dilation: a leg's speed
# scales the per-joint velocity cap it cruises at (0.9 x the joint limit,
# RetimeParams.vmax_margin), and the re-timed profile eases out of rest and
# into the arrival on its own — which is what gives a full-speed cruise the
# settled arrival a uniform 0.75 was once bought for (owner, 2026-09-16).
# The press stroke has its own speed (press_demo.speed).
TRANSIT_SPEED = 1.0
# The base sweep that looks for the box. Slower than a transit because it
# exists to SEE, not to arrive: the coarse detector needs frames it can lift
# through TF while the arm is moving, and the leg is cancelled the instant
# one of them commits (runtime/legs.py stop_when).
SEARCH_SPEED = 0.35
CONTACT_SPEED = 0.15
DRIFT_REPLAN_RAD = 0.04  # < START_GATE_RAD: replan before the gate would refuse
# The driver commands a trajectory's first waypoint at once, wherever the arm
# is; runtime/driver.py refuses a goal whose start is farther than this from
# the live joints (the old planner-side executor's own start gate).
START_GATE_RAD = 0.05
SANITY_MARGIN_RAD = 0.35  # per-joint excursion allowance beyond |start->end|
# A mission must START near HOME: every legit run begins there, so a
# distant start means the last run ended badly. Refuse before any motion
# instead of planning something dramatic from wreckage (field
# 2026-09-01: a plan from a failure-held pose swung the arm half upside
# down). This is the ONLY start-shape guard: a per-leg joint-sweep cap
# was tried and measured wrong the same day — wrap-aware endpoint sweep
# saturates at pi, and six live HOME->scan plans legitimately swept
# 2.4-2.9 rad on wrist/elbow joints for the tool-down reorientation.
HOME_START_TOL_RAD = 1.2

GRIPPER_CMD_CLOSED = 0.8  # knuckle rad at full close (the driver's setpoint 1.0)
GRIPPER_CMD_OPEN = 0.0  # ~85 mm aperture

REST_TOL_RAD = 0.05  # "already there": the start gate's tolerance

# Joint velocity limits the planner plans against (cuRobo gen3_real.yaml, the
# URDF's): the re-timer caps every joint below these and runtime/driver.py
# refuses a goal above them — two independent gates on the same numbers.
JOINT_VMAX = [1.396, 1.396, 1.396, 1.396, 1.222, 1.222, 1.222]

# Reflex recoil after a press trip: how much of the descent's own path to
# reverse, and at what cruise fraction. Measured with the real planner and
# FK at two bench placements (2026-09-03): 0.085 rad of joint arc lifts
# the tool 20.7-21.0 mm, 0.05 rad lifts 13.3-13.9 mm — the mapping barely
# moves with the box, so a joint-arc budget is a fair stand-in for the
# Cartesian lift this TF tree cannot measure. 0.09 rad ~ 22 mm: enough to
# unload the contact while the considered retreat is planned.
# A reflex, not a considered move: brisk, short, no planning.
RECOIL_ARC_RAD = 0.09
RECOIL_SPEED = 0.5

# The planner is commanded in tool_frame, but the FINGERTIPS reach past it.
# Measured two independent ways (2026-09-03), agreeing to 0.7 mm:
#   - cuRobo's own model: the *_inner_finger_pad link sits 10.3 mm beyond
#     tool_frame (sphere index API on robot_gen3_2f85.yaml);
#   - the arm's own contact event: replaying run-20260903-123629's press,
#     tool_frame was at z 0.0941 when the guard tripped on a button top the
#     depth had measured at 0.0831 -> 11.0 mm.
# Nothing accounted for it, so every fingertip-referenced target was that
# much too deep: grip:down, commanded to button + 5 mm, put the pads at
# button - 6 mm — INSIDE the lid (its logged peak torque was 1.5-1.7 Nm
# where free air reads ~0), so the fingers could not close on the knob.
# Note the real robot's description does not even define tool_frame once a
# gripper is attached; it is cuRobo's own frame. Command tool_frame this
# much HIGHER than where the fingertips should land.
TCP_OFFSET_M = 0.011

# The fingertip links, for measuring a contact with the arm's own
# kinematics. These EXIST in the live TF tree (kinova_gen3_description:
# kortex_description's arm macro plus robotiq_description); tool_frame does
# not once a gripper is attached.
FINGERTIP_FRAMES = (
    "robotiq_85_left_finger_tip_link",
    "robotiq_85_right_finger_tip_link",
)

# The finger-tip LINK origin sits this far above tool_frame along the tool
# axis (real URDF: tip link at EE + 0.1118 m closed, tool_frame at
# EE + 0.120; cross-checked on two bench runs where the tip TF at the
# trip and the replayed tool_frame differed by 8.0-8.3 mm). Converts a
# fingertip TF reading into the frame the planner is commanded in.
TIP_TO_TOOL_M = 0.008

# Joint-space arc per metre of tool travel along a vertical descent at the
# press pose: 0.085 rad lifted the tool 20.7-21.0 mm at two placements
# (recoil calibration, 2026-09-03). Lets a short vertical move be cut
# from an already-planned trajectory without FK.
JOINT_ARC_PER_M = 4.05
