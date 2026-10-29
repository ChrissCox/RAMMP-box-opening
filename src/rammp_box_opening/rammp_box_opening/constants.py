"""Shared constants: arm facts and runner defaults (spec §3, §6)."""

import os
from collections import namedtuple
from pathlib import Path


def state_dir(env=None):
    """Where run logs, captures, generated worlds and calibration residuals
    are kept: ~/.ros/rammp_box_opening, or $RAMMP_BOX_OPENING_STATE. The
    stub harnesses point it at their own folder — their synthetic runs once
    filled the real residuals file (2026-09-17), and the every-run captures
    are pruned to the newest few, real or not."""
    env = os.environ if env is None else env
    return Path(env.get("RAMMP_BOX_OPENING_STATE") or Path.home() / ".ros" / "rammp_box_opening")


# Gen3 home joints in controller order (flatbot, new_adapter_plate_v3, 2026-10-21).
# joint_3 sits AT -pi (every comparison uses ang_diff; -pi and +pi are equivalent).
HOME = [0.067, -1.897, -3.137, -2.269, -0.211, -0.683, 1.725]
JOINTS = ["joint_%d" % i for i in range(1, 8)]

NODE_NAMESPACE = "/rammp_curobo"  # the planner container's actions and service
REQUIRED_RMW = "rmw_cyclonedds_cpp"  # driver and planner containers speak Cyclone
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
# A drag-guarded straight line in free air that moves up or away from
# anything it could touch, flown only after an at-rest check (or in place,
# pivoting about the fingertip): the reach-out's rise, slide out and back
# off, the depth press's turn, rise and centring. Every line toward the lid,
# or loaded, stays at CONTACT_SPEED in a guarded run; a compliant run, which
# nothing it reads stops, flies its free-air lines here too. Run 2026-10-22
# 20:04 flew its turn, rise and back off at 0.15: 18 s of a 45 s press; the
# compliant 2026-10-27 16:23 its re-aim, slide in and clear: 10.5 s.
FREE_LINE_SPEED = 0.3
DRIFT_REPLAN_RAD = 0.04  # < START_GATE_RAD: replan before the gate would refuse
# The driver commands a trajectory's first waypoint at once, wherever the arm
# is; runtime/driver.py refuses a goal whose start is farther than this from
# the live joints (the old planner-side executor's own start gate).
START_GATE_RAD = 0.05
SANITY_MARGIN_RAD = 0.35  # per-joint excursion allowance beyond |start->end|
# rammp-curobo plans inside the URDF joint limits narrowed by this much
# (its configs/gen3.yaml joint_limit_margin_rad) and refuses a start or a
# goal inside it: START_IN_LIMIT_MARGIN / GOAL_IN_LIMIT_MARGIN
PLANNER_LIMIT_MARGIN_RAD = 0.10
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

# A compliant run re-times every motion to what the arm follows in the driver's
# joint impedance (runtime/runner.py Runner.retime_params): the robot's, not
# the scene's. Measured under RIGID at zeta 0.7 on the 2026-10-28 15:45
# /joint_states bag (49 Hz, two runs), against a reference rebuilt with
# retime_group.
# The wrist's speed ceiling, rad/s, ABSOLUTE (RetimeParams.qd_ceiling): it
# caps what a leg's cruise fraction leaves and no fraction scales it, so a 0.3
# line whose wrist runs under it is not slowed. On both 15:45 way homes
# arm_joint_6 held a median 0.536 / 0.541 rad/s for 1.75 s, its command at the
# 9 N·m clamp and its link torque at its rest value, against the 0.9 x 1.222 =
# 1.10 the re-timer allows: it stood 0.25 rad short when the goal ended (the
# tool 86-88 mm off), overshot 56 mrad and was within 2 mm only 1.15 s later.
# 15:41:58's POSITION home drove it to 0.738 and arrived 0.7 mm off.
# arm_joint_5 was seen at 0.51; arm_joint_7 never above 0.32, so its own
# ceiling is unmeasured: it is capped like its siblings until a flight log shows
# it can go faster. 0.45 keeps the re-timer's 0.9 margin under the ~0.5 the
# wrist makes. Joints 1-4 have none of their own (0.9 x JOINT_VMAX).
COMPLIANT_QD_CEILING = (float("inf"),) * 4 + (0.45,) * 3
# Out of rest, rad/s^2 along the path (RetimeParams.accel 6). The 15:45 look's
# reference asked arm_joint_2 for ~4.2 rad/s^2; it made ~3, fell 101-123 mrad
# behind, caught up at 1.50-1.62 rad/s — over the 1.257 cruise and the URDF's
# 1.396 — and stood 45-47 mrad PAST its end when the goal ended (the tool 14 mm
# off), creeping back for 0.55 s. At 3 the look asks it 2.2-2.4: inside what
# it made.
COMPLIANT_ACCEL = 3.0
# Into rest (RetimeParams.decel 3), still half the ease out, so the arrival
# stays the long half. Every fast stop of those runs ended with the arm past its
# end: the look 45-47 mrad, the way home 56 mrad after its short arrival. How
# much the two cut that is unmeasured (no model of the arm gets the look's sign
# right): the first logged run tells. Offline the three cost the look 0.4-0.5 s
# (16:23: 1.36 -> 1.82 s), a 0.3 line 0.0-0.2 s, 15:45's way out 1.0 s.
COMPLIANT_DECEL = 1.5

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

# The reach-out (press_demo's run) runs COMPLIANT by default (owner 2026-10-27:
# "with impedance mode, lets remove the safety check for now that stops the
# arm when there's a bad reading or it thinks it hit or is going to hit
# something"): every leg in the driver's joint impedance, nothing the run
# reads stops it. press_demo --guarded flies it guarded, as before.
COMPLIANT_DEFAULT = True

# The driver's named joint-impedance tunings (ImpedanceGains.profile;
# kinova-gen3-driver interface/gains.cpp kSoft/kMedium/kStiff): the robot's,
# not the scene's. A goal that names none (PROFILE_SESSION_DEFAULT, 0) flies
# the driver's session default, MEDIUM, since nothing calls /set_gains.
GAINS_SOFT, GAINS_MEDIUM, GAINS_STIFF = 1, 2, 3
# Their joint stiffness Kq, N·m/rad on joints 1-4 / 5-7. Damping follows from
# zeta (0.4 / 0.5 / 0.7), and each spring's leash is 0.45 / 0.35 / 0.25 rad.
IMPEDANCE_KQ = {
    GAINS_SOFT: (40.0,) * 4 + (15.0,) * 3,
    GAINS_MEDIUM: (80.0,) * 4 + (30.0,) * 3,
    GAINS_STIFF: (160.0,) * 4 + (60.0,) * 3,
}
IMPEDANCE_NAMES = {GAINS_SOFT: "SOFT", GAINS_MEDIUM: "MEDIUM", GAINS_STIFF: "STIFF"}
# ImpedanceGains.PROFILE_CUSTOM: the goal carries its own kq, zeta and
# torque_limit, bounds-checked by the driver when it accepts the goal (Kq
# 1-400 N·m/rad on every joint, zeta 0.05-2.0, each torque limit between the
# floor that holds gravity and 39/39/39/39/9/9/9 N·m: kinova-gen3-driver
# interface/gains.cpp validate_custom), refused with a reason, never clamped.
# Its leash is the session default's, MEDIUM's 0.35 rad.
GAINS_CUSTOM = 4


class CustomGains(namedtuple("CustomGains", "name kq zeta torque_limit")):
    """A PROFILE_CUSTOM gains spec: tuples, so legs compare it (legs.can_merge)
    and IMPEDANCE_KQ / IMPEDANCE_NAMES key it as they key a profile."""

    __slots__ = ()


# Owner 2026-10-28, after 13:43's press under STIFF held its pads 6.4 mm off
# the aimed point: "impedance mode is still too loose. can we ramp up the gain
# when we go to press the button by a lot so it's extreamly rigid and doesn't
# miss the button". The driver's ceiling on joints 1-4 (2.5x STIFF) and 200 on
# the wrist (3.3x STIFF). On flatbot's URDF at 15:19's, 16:23's and 13:43's
# buttons the pads stand 560-890 N/m stiff sideways (STIFF: 210-320) and
# 1050-1160 N/m straight down (STIFF: 420-460), so 1.5 N·m of friction or
# model error on each of joints 1-4 and 0.5 on 5-7 puts them 3-4 mm off
# (STIFF: 8-11). 400 on the wrist too buys only 0.3-0.5 mm more, on the
# joints with the least inertia, where nothing above 60 has flown. zeta and
# the torque limits are STIFF's: the driver derives damping from zeta and each
# joint's inertia, so the damping ratio stays STIFF's.
GAINS_RIGID = CustomGains("RIGID", (400.0,) * 4 + (200.0,) * 3, 0.7, (39.0,) * 4 + (9.0,) * 3)
IMPEDANCE_KQ[GAINS_RIGID] = GAINS_RIGID.kq
IMPEDANCE_NAMES[GAINS_RIGID] = GAINS_RIGID.name
# Every compliant goal that names no gains flies this (PlannerClient
# .compliant_gains), and so does every leg of the compliant reach-out
# (depth_press.COMPLIANT_GAINS): one spec, so the one-motion way in and way
# home still merge, and the press model (depth_press.COMPLIANT_KQ) is the
# spring that flies. Owner 2026-10-27, after 16:23 flew all six of its goals
# at MEDIUM: "impedance mode needs to be firmer" (STIFF, first flown 13:43);
# 2026-10-28: RIGID (above). Nothing above STIFF has flown on this arm: the
# first attended run is its test.
COMPLIANT_RUN_GAINS = GAINS_RIGID
