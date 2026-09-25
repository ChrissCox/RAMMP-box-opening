"""The single ROS surface: the arm driver, the planner, the gripper, TF.

Two containers answer it, sheppy's arm module. The kinova-gen3-ros2 driver
executes (/execute_joint_trajectory), reports (/joint_states, the gripper
knuckle included) and takes gripper commands on /setpoint/gripper. The
RAMMP-CuRobo v1.0.0 planner plans (/rammp_curobo/plan_to_pose,
plan_to_joints, set_world) and never moves anything. TF comes from
kinova_gen3_description's robot_state_publisher, which the launch starts.

The spec §6 hardening stands: the guard is ARMED by execution feedback
progress > 0 (never at goal-accept) and efforts come from the /joint_states
stream. What the driver does not check before it moves — the start state,
velocity limits, timing — is checked here (runtime/driver.py), and nothing
moves at all unless this client was armed by --execute.
"""

import sys
import time
from collections import namedtuple

import rclpy
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformListener

from rammp_arm_interfaces.action import ExecuteJointTrajectory
from rammp_arm_interfaces.msg import GripperSetpoint
from rammp_curobo_interfaces.action import PlanToJoints, PlanToPose
from rammp_curobo_interfaces.srv import SetWorld

from rammp_box_opening.constants import (
    EXECUTE_ACTION,
    FINGERTIP_FRAMES,
    GRIPPER_FORCE,
    GRIPPER_SETPOINT_TOPIC,
    GRIPPER_SPEED,
    JOINT_VMAX,
    JOINTS,
    NODE_NAMESPACE,
)
from rammp_box_opening.runtime import driver
from rammp_box_opening.runtime.approach import plan_with_vertical_approach
from rammp_box_opening.runtime.branches import joint_time, nearest_branch, planner_branch

_GRIPPER_JOINT_HINTS = ("robotiq", "knuckle", "finger")

# ExecuteJointTrajectory.Goal.control_mode / .preemption (rammp_arm_interfaces)
POSITION = 0
QUEUE = 0

# a gripper command in flight: knuckle-radian target, the knuckle at send, when
GripperHandle = namedtuple("GripperHandle", "target start t0")


def spin_until_done(node, future, timeout_s, abort=None):
    """Spin `node` until `future` resolves; None on timeout — or at once
    when `abort` has been requested, so a plan hosted inside an execution
    (see PlannerClient.execute while_running) cannot delay the cancel."""
    t0 = time.monotonic()
    while not future.done():
        if abort is not None and abort.requested:
            return None
        rclpy.spin_once(node, timeout_sec=0.1)
        if time.monotonic() - t0 > timeout_s:
            return None
    return future.result()


class PlannerClient:
    # a pending gripper setpoint is re-sent this often: the topic is
    # best-effort and latest-wins, so one lost message must not strand a
    # command, and re-sending the same absolute setpoint changes nothing
    GRIPPER_RESEND_S = 0.1
    GRIPPER_TIMEOUT_S = 10.0
    # a knuckle reading older than this is a stream gap, never "settled"
    GRIPPER_STREAM_GAP_S = 0.2

    def __init__(self, node, abort=None, motion_enabled=False):
        self.node = node
        self._abort = abort  # AbortFlag when the CLI owns SIGINT (abort.py)
        # The motion latch, set only by --execute. The driver executes
        # whatever it receives and has no dry-run parameter, so this is the
        # software gate on arm AND gripper motion.
        self._armed = bool(motion_enabled)
        self._q = None
        self._eff = None
        self._eff_at = 0.0  # monotonic stamp of the last joint_states message
        self._gripper_pos = None
        self._gripper_target = None  # driver setpoint (0..1) re-sent while pending
        self.world_held = None  # path of the world the planner holds
        # the driver publishes /joint_states best-effort (SensorDataQoS): a
        # reliable subscription would match nothing and hear nothing
        node.create_subscription(
            JointState, "/joint_states", self._js_cb, qos_profile_sensor_data
        )
        self._plan_pose = ActionClient(
            node, PlanToPose, NODE_NAMESPACE + "/plan_to_pose"
        )
        self._plan_joints = ActionClient(
            node, PlanToJoints, NODE_NAMESPACE + "/plan_to_joints"
        )
        self._execute = ActionClient(node, ExecuteJointTrajectory, EXECUTE_ACTION)
        self._set_world = node.create_client(SetWorld, NODE_NAMESPACE + "/set_world")
        # latest-wins, best-effort depth 1: the driver subscribes exactly so
        self._gripper_pub = node.create_publisher(
            GripperSetpoint,
            GRIPPER_SETPOINT_TOPIC,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT),
        )
        node.create_timer(self.GRIPPER_RESEND_S, self._publish_gripper)
        self._tf = Buffer()
        self._tf_listener = TransformListener(self._tf, node)

    # -- state streams -----------------------------------------------------
    def _js_cb(self, msg):
        idx = {n: i for i, n in enumerate(msg.name)}
        try:
            q = [float(msg.position[idx[n]]) for n in JOINTS]
            eff = (
                [float(msg.effort[idx[n]]) for n in JOINTS]
                if len(msg.effort) == len(msg.name)
                else None
            )
        except (KeyError, IndexError):
            return
        self._q = q
        self._eff = eff
        self._eff_at = time.monotonic()
        for name in msg.name:
            if any(h in name for h in _GRIPPER_JOINT_HINTS):
                self._gripper_pos = float(msg.position[idx[name]])
                break

    def motion_enabled(self):
        """Whether this client may move the arm or the gripper (--execute)."""
        return self._armed

    def joints(self):
        t0 = time.monotonic()
        while self._q is None:
            rclpy.spin_once(self.node, timeout_sec=0.2)
            if time.monotonic() - t0 > 10:
                sys.exit(
                    "no /joint_states — is sheppy's `arm` node up, and does this "
                    "shell export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp?"
                )
        return list(self._q)

    # A guarded leg polls wrist_efforts() and trips on deviation from a
    # baseline. If the /joint_states subscription STALLS, self._eff keeps
    # its last value forever: the deviation stays flat, the guard can never
    # trip, and the existing "efforts lost" watchdog never fires either —
    # it only catches messages that arrive WITHOUT effort fields. Stale
    # readings must therefore read as no readings (review 2026-08-28).
    EFFORT_STALE_S = 0.5

    def wrist_efforts(self):
        if self._eff is None:
            return None
        if time.monotonic() - self._eff_at > self.EFFORT_STALE_S:
            return None  # frozen stream: the guard has lost its senses
        return list(self._eff[3:])

    def efforts_present(self):
        self.joints()  # ensure at least one message arrived
        return self._eff is not None

    def _fresh_joints(self, timeout_s=1.0):
        """The live joints while /joint_states is flowing, else None: a
        start gate judged against a frozen sample is no gate."""
        t0 = time.monotonic()
        while True:
            if (
                self._q is not None
                and time.monotonic() - self._eff_at <= self.EFFORT_STALE_S
            ):
                return list(self._q)
            if time.monotonic() - t0 > timeout_s:
                return None
            rclpy.spin_once(self.node, timeout_sec=0.05)

    def contact_xyz(self, timeout_s=0.25):
        """Where the FINGERTIPS are right now, from TF: the midpoint of the
        two finger-tip links, or None.

        Called the instant a guard trip lands, this is the arm measuring the
        surface it just touched with its own kinematics — no camera, no
        model constant. Unlike tool_frame these frames exist in the live
        tree, so the lookup resolves instead of stalling."""
        if getattr(self, "_tip_frames_missing", False):
            return None
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout_s:
            try:
                pts = []
                for frame in FINGERTIP_FRAMES:
                    tf = self._tf.lookup_transform("base_link", frame, Time())
                    tr = tf.transform.translation
                    pts.append([tr.x, tr.y, tr.z])
                return [sum(c) / len(pts) for c in zip(*pts)]
            except Exception:
                rclpy.spin_once(self.node, timeout_sec=0.05)
        self._tip_frames_missing = True
        print("[client] no %s in TF — contact heights unavailable" % (FINGERTIP_FRAMES[0],))
        return None

    def planner_reachable(self, timeout_s=5.0):
        """True when both planner actions AND its set_world service respond.

        set_world is the first thing every plan calls, and it is advertised
        only once cuRobo has finished loading, so an action that answers
        before it does is not a planner that can plan yet."""
        return (
            self._plan_pose.wait_for_server(timeout_sec=timeout_s)
            and self._plan_joints.wait_for_server(timeout_sec=2.0)
            and self._set_world.wait_for_service(timeout_sec=2.0)
        )

    def driver_reachable(self, timeout_s=5.0):
        """True when the driver's trajectory action responds."""
        return self._execute.wait_for_server(timeout_sec=timeout_s)

    def tf_ready(self, target, source, timeout_s=2.0):
        """True once TF can give `target` <- `source` (the latest): the arm's
        TF comes from robot_state_publisher in sheppy's box_opening node."""
        from rclpy.time import Time

        t0 = time.monotonic()
        while True:
            if self._tf.can_transform(target, source, Time()):
                return True
            if time.monotonic() - t0 > timeout_s:
                return False
            rclpy.spin_once(self.node, timeout_sec=0.05)

    # Programs that move this arm on their own: one of these running beside a
    # mission is two controllers on one arm.
    OTHER_ARM_CLIENTS = ("rammp_adl_runtime",)

    def other_arm_clients(self):
        """The OTHER_ARM_CLIENTS on the ROS graph now."""
        return sorted({n for n, _ns in self.node.get_node_names_and_namespaces() if n in self.OTHER_ARM_CLIENTS})

    # -- planning ----------------------------------------------------------
    def _call(self, client, goal, timeout_s=120.0):
        if not client.wait_for_server(timeout_sec=5.0):
            sys.exit("planner not reachable — is sheppy's `planner` node up?")
        send = spin_until_done(
            self.node, client.send_goal_async(goal), 10.0, abort=self._abort
        )
        if send is None or not send.accepted:
            return None
        wrapped = spin_until_done(
            self.node, send.get_result_async(), timeout_s, abort=self._abort
        )
        return None if wrapped is None else wrapped.result

    def plan_to_pose(self, xyz, quat_xyzw, start_joints, approach_offset_m=0.0):
        """>0 approach_offset_m: arrive from a waypoint that far straight above
        the goal — a diagonal descent touches a surface before its lateral
        convergence finishes (edge presses, field 2026-09-01). The v1.0.0
        planner cannot constrain that, so it is two plans flown as one
        trajectory (runtime/approach.py)."""
        return plan_with_vertical_approach(
            self._plan_pose_near, xyz, quat_xyzw, start_joints, float(approach_offset_m)
        )

    def _plan_pose_near(self, xyz, quat_xyzw, start_joints):
        """A pose goal planned as a JOINT goal: the IK solution quickest to
        fly to from the start (kinematics.ArmChain.nearest_solution), so the
        planner cannot pick a far one of the redundant arm's solutions —
        handed the pose, it turned the wrist 3.2-3.6 rad on some plans (a
        4.7 s approach to a box on the left, bench 2026-09-25). The planner
        still plans and collision-checks the path; when no solution is found
        or its plan fails, the pose goes to it as before."""
        if start_joints:
            try:
                from rammp_box_opening.constants import HOME
                from rammp_box_opening.primitives.look import look_joints
                from rammp_box_opening.runtime.approach import _chain

                start = planner_branch(start_joints)
                q_goal = _chain().nearest_solution(start, xyz, quat_xyzw, seeds=(look_joints(HOME),))
            except Exception:  # a kinematics problem must never cost the plan
                q_goal = None
            if q_goal is not None:
                res = self.plan_to_joints(q_goal, start_joints)
                if res is not None and res.success:
                    return res
        return self._no_long_way(self._plan_pose_once(xyz, quat_xyzw, start_joints), start_joints)

    # A plan is re-planned when an equivalent end is at least this much quicker
    # to fly to: a continuous joint the long way round is up to ~5 s of it.
    QUICKER_BY_S = 0.3

    def _no_long_way(self, res, start_joints):
        """The planner's own pose plan — flown when no quicker solution was
        found up front (the table's far edge) — unless it ends somewhere the
        same tool pose is quicker to reach: a continuous joint turned the long
        way round, or the wrist folded the far way (kinematics.equivalents).
        Then it is re-planned to that end (plan_to_joints); the original stands
        if that plan fails. The arm does not turn a joint round when it does
        not need to (owner, 2026-09-25)."""
        if res is None or not res.success or not start_joints or res.trajectory is None or not res.trajectory.points:
            return res
        try:
            from rammp_box_opening.runtime.approach import _chain

            start = planner_branch(start_joints)
            end = [float(v) for v in res.trajectory.points[-1].positions]
            alts = _chain().equivalents(start, end)
        except Exception:  # a kinematics problem must never cost the plan
            return res
        if not alts:
            return res
        best = min(alts, key=lambda q: joint_time(start, q))
        if joint_time(start, best) > joint_time(start, end) - self.QUICKER_BY_S:
            return res
        better = self.plan_to_joints(best, start_joints)
        return better if better is not None and better.success else res

    def _plan_pose_once(self, xyz, quat_xyzw, start_joints):
        g = PlanToPose.Goal()
        g.target.position.x, g.target.position.y, g.target.position.z = (
            float(v) for v in xyz
        )
        (
            g.target.orientation.x,
            g.target.orientation.y,
            g.target.orientation.z,
            g.target.orientation.w,
        ) = (float(v) for v in quat_xyzw)
        # on HOME's side of the wrap: the planner plans between the numbers
        # it is given, and a start written the other way round is a full turn
        g.start_joints = planner_branch(start_joints) if start_joints else []
        return self._call(self._plan_pose, g)

    def plan_to_joints(self, q7, start_joints):
        """A joint goal, each continuous joint taken the short way round from
        the start (branches.nearest_branch): the same arm configuration,
        never a winding. The planner does this for itself since 1.0.0
        (_nearest_branch); done here too, it does not depend on that."""
        start = planner_branch(start_joints) if start_joints else None
        goal = nearest_branch(q7, start) if start is not None else [float(v) for v in q7]
        g = PlanToJoints.Goal(target_joints=goal)
        g.start_joints = start or []  # on HOME's side of the wrap (plan_to_pose)
        return self._call(self._plan_joints, g)

    # -- execution ---------------------------------------------------------
    def _cancel_confirm(self, send, result_future):
        """Ctrl+C path: cancel on a live context and report what the driver
        actually confirmed — never claim a stop that wasn't answered."""
        spin_until_done(self.node, send.cancel_goal_async(), 5.0)
        wrapped = spin_until_done(self.node, result_future, 10.0)
        if wrapped is not None:
            print("\nCtrl+C — cancel delivered; the driver stops and holds")
        else:
            print(
                "\nCtrl+C — cancel sent; no result confirmation in 10 s — "
                "check the arm"
            )

    def execute(self, traj, speed, guard=None, while_running=None, stop_when=None):
        """Run one trajectory on the driver; outcome 'arrived' | 'touch' | 'failed'.

        `speed` dilates the trajectory's time base (the driver has no speed
        scale of its own); 1.0 flies it as planned or re-timed. Nothing is
        sent unless the client is armed and the trajectory passes the gates
        the driver lacks (runtime/driver.py): fresh live joints at its start,
        velocities under the limits, time running forward.

        With a guard: feedback progress arms it, /joint_states efforts feed
        it; a trip cancels the goal — the driver stops and holds its last
        reference, so the contact load stays on.

        `stop_when(progress)`: polled while the motion flies with the
        driver's progress fraction; the first True cancels the goal and
        returns 'stopped'. This is how a search ends the moment it finds
        what it was looking for, leaving the arm where it saw it — and how
        a reach that carries its own descent stops at the junction when the
        box has not been confirmed by then.

        `while_running` (UNGUARDED legs only): a callable run once right
        after the goal is accepted — the next phase's planning, hidden under
        this motion instead of after it (the planner is its own container).
        Its blocking plans spin this same node, so feedback keeps flowing;
        its result lands in info["while_running"], an exception in
        info["while_running_error"] — never past the cancel path."""
        info = {"message": "", "progress": 0.0, "torque_peak": None}
        if self._abort is not None and self._abort.requested:
            raise KeyboardInterrupt  # aborted before this leg ever started
        if not self._armed:
            info["message"] = "motion disabled: the client was not armed (--execute)"
            return "failed", info
        if not self._execute.wait_for_server(timeout_sec=5.0):
            info["message"] = "execute_joint_trajectory not available — is the arm node up?"
            return "failed", info
        flown = traj if float(speed) == 1.0 else driver.dilate(traj, speed)
        live = self._fresh_joints()
        if live is None:
            info["message"] = "no fresh /joint_states — refusing to send a trajectory"
            return "failed", info
        why = driver.refusal(flown, live, JOINT_VMAX)
        if why:
            info["message"] = "refused before sending: " + why
            return "failed", info
        goal = ExecuteJointTrajectory.Goal()
        goal.trajectory = flown
        goal.control_mode = POSITION
        goal.preemption = QUEUE  # never displace a goal that is still settling
        goal.sender_id = "rammp_box_opening"

        def _fb(msg):
            info["progress"] = float(msg.feedback.fraction_complete)
            if guard is not None:
                guard.on_progress(info["progress"])

        # goal_in_flight makes SIGINT set the flag instead of raising: the
        # loop below then delivers the cancel on a LIVE context (abort.py)
        if self._abort is not None:
            self._abort.goal_in_flight = True
        try:
            send = spin_until_done(
                self.node,
                self._execute.send_goal_async(goal, feedback_callback=_fb),
                10.0,
            )
            if send is None or not send.accepted:
                info["message"] = (
                    "goal rejected by the driver (a stream session open, or a "
                    "mode change while moving)"
                )
                return "failed", info
            result_future = send.get_result_async()
            contact = False
            stopped = False
            aborted = False
            t0 = time.monotonic()
            efforts_ok_at = time.monotonic()
            try:
                if while_running is not None and guard is None:
                    # INSIDE the cancel backstop: a second Ctrl+C raised from
                    # the hosted plan must still reach the cancel below, and
                    # a first Ctrl+C that only set the flag while the motion
                    # finished under the plan must not be swallowed (review
                    # 2026-09-02)
                    try:
                        info["while_running"] = while_running()
                    except KeyboardInterrupt:
                        raise
                    except Exception as exc:  # a failed lookahead is not a failed leg
                        info["while_running_error"] = str(exc)
                    if self._abort is not None and self._abort.requested:
                        self._cancel_confirm(send, result_future)
                        aborted = True
                while not aborted and not result_future.done():
                    if self._abort is not None and self._abort.requested:
                        self._cancel_confirm(send, result_future)
                        aborted = True
                        break
                    rclpy.spin_once(self.node, timeout_sec=0.05)
                    if stop_when is not None and stop_when(info["progress"]):
                        spin_until_done(self.node, send.cancel_goal_async(), 3.0)
                        spin_until_done(self.node, result_future, 10.0)
                        stopped = True
                        break
                    if guard is not None:
                        eff = self.wrist_efforts()
                        if eff is not None:
                            efforts_ok_at = time.monotonic()
                        elif time.monotonic() - efforts_ok_at > 1.0:
                            # a guard that has lost its senses is no guard:
                            # stop the stroke instead of pressing blind
                            spin_until_done(self.node, send.cancel_goal_async(), 3.0)
                            spin_until_done(self.node, result_future, 10.0)
                            info["message"] = (
                                "effort stream lost during guarded leg — "
                                "cancelled (guard blind > 1 s)"
                            )
                            return "failed", info
                        if guard.on_efforts(eff):
                            contact = True
                            spin_until_done(self.node, send.cancel_goal_async(), 3.0)
                            spin_until_done(self.node, result_future, 10.0)
                            break
                    if time.monotonic() - t0 > 240:
                        spin_until_done(self.node, send.cancel_goal_async(), 3.0)
                        info["message"] = "execution watchdog timeout (240 s)"
                        return "failed", info
            except KeyboardInterrupt:
                # backstop: default-handler CLIs and the second-Ctrl+C
                # escalation land here; try the cancel, promise nothing
                spin_until_done(self.node, send.cancel_goal_async(), 3.0)
                print("\nCtrl+C — cancel sent (unconfirmed; check the arm)")
                raise
            if aborted:
                raise KeyboardInterrupt  # unwind AFTER the confirmed cancel
        finally:
            if self._abort is not None:
                self._abort.goal_in_flight = False
        if guard is not None:
            info["torque_peak"] = guard.peak
        if contact:
            info["message"] = "torque guard trip"
            return "touch", info
        if stopped:
            info["message"] = "stopped part-way: what it was watching for happened"
            return "stopped", info
        wrapped = result_future.result()
        if wrapped is None:
            info["message"] = "no result"
            return "failed", info
        result = wrapped.result
        if result.error_code == driver.SUCCESSFUL:
            info["message"] = "arrived"
            return "arrived", info
        info["message"] = driver.result_message(result.error_code, result.error_string)
        return "failed", info

    # -- services / gripper ------------------------------------------------
    def set_world(self, path_or_name):
        """Push a world; a no-op when the planner already holds it. This is
        the ONE record of what the planner holds (plan-time pushes and
        replans both route here; two trackers drifted, review 2026-09-02).
        Worlds are content-hashed paths, so equal path == equal world. The
        planner runs in a container: the path must exist inside it too."""
        key = str(path_or_name)
        if key == self.world_held:
            return True, "held"
        if not self._set_world.wait_for_service(timeout_sec=5.0):
            return False, "set_world service unavailable"
        req = SetWorld.Request(world=key)
        resp = spin_until_done(self.node, self._set_world.call_async(req), 10.0)
        if resp is None or not resp.success:
            self.world_held = None
            return False, "set_world timed out" if resp is None else resp.message
        self.world_held = key
        return True, resp.message

    def gripper_cmd(self, position):
        """Command the gripper and wait (knuckle rad: 0.0 open .. 0.8 closed);
        position None = query only (the live knuckle, no motion).
        Returns (ok, position, stalled)."""
        if position is None:
            return self._gripper_pos is not None, self._gripper_pos or 0.0, False
        handle = self.gripper_send(position)
        if handle is None:
            return False, 0.0, False
        return self.gripper_join(handle)

    def gripper_send(self, position):
        """Start a gripper command (knuckle rad) and return a handle WITHOUT
        waiting, or None when it cannot be sent.

        Lets a close overlap the motion that follows it — the fingers shut
        while the arm transits instead of the arm standing still for the
        whole close; the driver's gripper is not a control mode, so it rides
        alongside a trajectory by design. The caller MUST join before
        anything that depends on the fingers having arrived."""
        if not self._armed:
            return None
        if self._gripper_pos is None:
            t0 = time.monotonic()
            while self._gripper_pos is None and time.monotonic() - t0 < 2.0:
                rclpy.spin_once(self.node, timeout_sec=0.1)
            if self._gripper_pos is None:
                return None  # no knuckle reading: nothing could confirm the command
        self._gripper_target = driver.setpoint_position(position)
        self._publish_gripper()
        return GripperHandle(float(position), float(self._gripper_pos), time.monotonic())

    def _publish_gripper(self):
        """Send the pending setpoint; also the re-send timer's callback."""
        if self._gripper_target is None:
            return
        msg = GripperSetpoint()
        msg.position = float(self._gripper_target)
        # sent on every message: the driver keeps neither speed nor force
        msg.speed = GRIPPER_SPEED
        msg.force = GRIPPER_FORCE
        self._gripper_pub.publish(msg)

    def gripper_join(self, handle):
        """Wait out a gripper_send: (ok, position, stalled), the same verdict
        as gripper_cmd — the overlap must not change what callers verify.

        The driver's gripper reports no result, so completion is read off the
        knuckle in /joint_states (driver.GripperWait): at the target; moved
        and then still (closed on the knob, a stall the grip band judges);
        or never moved at all (a failure). Re-sending stops once it is in.

        The wait is timed from HERE, not from the dispatch: a deferred command
        is meant to sit pending through the motion it overlaps, and that motion
        can outlast the timeout on its own — in slow mode the scan flight did,
        and the join reported a closed gripper as a failure without ever
        reading it (field 2026-09-14). GripperWait still counts "never moved"
        from the dispatch, where that question belongs."""
        wait = driver.GripperWait(handle.target, handle.start, handle.t0)
        deadline = time.monotonic() + self.GRIPPER_TIMEOUT_S
        try:
            while time.monotonic() < deadline:
                rclpy.spin_once(self.node, timeout_sec=0.02)
                now = time.monotonic()
                pos = self._gripper_pos
                fresh = pos is not None and now - self._eff_at <= self.GRIPPER_STREAM_GAP_S
                done = wait.update(handle.start if pos is None else pos, now, fresh)
                if done is not None:
                    return done
            return False, float(self._gripper_pos or 0.0), False
        finally:
            self._gripper_target = None
