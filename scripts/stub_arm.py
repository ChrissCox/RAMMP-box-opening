"""Stand-in for the kinova-gen3-ros2 driver container, for the stub-isolated
e2e harnesses.

Serves what box-opening uses of the driver, minus the physics:

  /execute_joint_trajectory  plays the trajectory on its own time base (no
                             speed scale: the client dilates), feedback
                             fraction_complete, a cancel -> PREEMPTED with
                             the arm holding where it stopped
  /joint_states              best-effort (SensorDataQoS, like the driver),
                             20 Hz: seven arm joints with efforts plus
                             robotiq_85_left_knuckle_joint
  /setpoint/gripper          best-effort depth 1; the knuckle travels to
                             position x 0.8 over up to GRIP_TRAVEL_S
  /gripper_state             normalized position, present

Like the real driver it checks only that every point carries seven
positions: whatever the client sends, it flies. Run ONLY on an isolated
ROS_DOMAIN_ID (the harnesses enforce this) — it serves the real names.

STUB_TRIP_EXEC_N=<n>[,<m>...] spikes the wrist efforts partway through exec
goal n. The 9 Nm load PERSISTS while the arm holds against what it hit (a
real arm on a button keeps its load through the cancel) and releases when a
later goal runs to completion, i.e. the arm moved away: the two-stage press
baselines the push on the in-contact load.
STUB_GRIP_POS: the knuckle a CLOSE stalls at (the knob between the fingers).
STUB_POP_ON_TRIP=<k>: the stub button pops on the k-th spike (default 1, the
first press). 2 makes the first press fail to pop it — the mission's pop
check must notice and press again (the re-press scenario).
"""

import os
import time

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool

from rammp_arm_interfaces.action import ExecuteJointTrajectory
from rammp_arm_interfaces.msg import GripperSetpoint, GripperState

# HOME: every mission run now starts there, flying there first when it is not
# (press_demo.home_first). STUB_START_OFF_HOME=1 starts the stub 0.4 rad off
# it (joint_1) — the scenario that proves that first move.
START = (
    [0.4, 0.262, 3.142, -2.269, 0.0, 0.960, 1.571]
    if os.environ.get("STUB_START_OFF_HOME") == "1"
    else [0.0, 0.262, 3.142, -2.269, 0.0, 0.960, 1.571]
)
NAMES = ["joint_%d" % i for i in range(1, 8)]
KNUCKLE = "robotiq_85_left_knuckle_joint"
KNUCKLE_CLOSED = 0.8
GRIP_TRAVEL_S = 0.3  # a full open-to-closed stroke
from rclpy.qos import DurabilityPolicy

KNOB_QOS = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)

TRIP_EXEC_NS = {
    int(v) for v in os.environ.get("STUB_TRIP_EXEC_N", "").split(",") if v.strip()
}
GRIP_POS = os.environ.get("STUB_GRIP_POS")
POP_ON_TRIP = int(os.environ.get("STUB_POP_ON_TRIP", "1"))


class StubArm(Node):
    """The kinova_gen3_node surface box-opening uses, minus the physics."""

    def __init__(self):
        super().__init__("kinova_gen3_node")
        cb = ReentrantCallbackGroup()
        self.q = list(START)
        self.exec_goals = 0
        self.level = 0.0  # wrist effort load (STUB_TRIP_EXEC_N)
        self.injected = False  # one injection per goal
        self.knuckle = 0.0
        self.grip = (0.0, 0.0, 0.0, 0.0)  # (from, to, t0, duration)
        self.grip_command = None
        self.pub = self.create_publisher(JointState, "/joint_states", qos_profile_sensor_data)
        # the stub bench's own contract with stub_d405: latched, so a
        # camera stub that starts later still learns the knob is up
        self.knob_up = False
        self.spikes = 0
        self.knob_pub = self.create_publisher(Bool, "/stub_bench/knob_up", KNOB_QOS)
        self.state_pub = self.create_publisher(
            GripperState, "/gripper_state", qos_profile_sensor_data
        )
        self.create_subscription(
            GripperSetpoint,
            "/setpoint/gripper",
            self._on_gripper,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT),
            callback_group=cb,
        )
        self.create_timer(0.05, self._tick, callback_group=cb)
        ActionServer(
            self,
            ExecuteJointTrajectory,
            "/execute_joint_trajectory",
            execute_callback=self._execute,
            goal_callback=self._check,
            cancel_callback=self._cancel,
            callback_group=cb,
        )
        print("STUB ARM READY", flush=True)

    def _knuckle_now(self):
        start, end, t0, dur = self.grip
        if dur <= 0.0:
            return end
        a = min(1.0, (time.monotonic() - t0) / dur)
        return start + a * (end - start)

    def _on_gripper(self, msg):
        target = max(0.0, min(1.0, float(msg.position))) * KNUCKLE_CLOSED
        if target != self.grip_command:
            # latest-wins re-sends of the same setpoint are one command
            self.grip_command = target
            print("GRIPPER GOAL pos=%.2f" % (target / KNUCKLE_CLOSED), flush=True)
            now = self._knuckle_now()
            end = target
            if GRIP_POS and target > now:
                end = min(target, float(GRIP_POS))  # closes on the knob
            dur = GRIP_TRAVEL_S * abs(end - now) / KNUCKLE_CLOSED
            self.grip = (now, end, time.monotonic(), dur)

    def _tick(self):
        self.knuckle = self._knuckle_now()
        m = JointState()
        m.header.stamp = self.get_clock().now().to_msg()
        m.name = NAMES + [KNUCKLE]
        m.position = list(self.q) + [self.knuckle]
        m.velocity = [0.0] * 7 + [float("nan")]
        m.effort = [self.level] * 7 + [float("nan")]
        self.pub.publish(m)
        s = GripperState()
        s.header.stamp = m.header.stamp
        s.position = float(self.knuckle / KNUCKLE_CLOSED)
        s.effort = 0.05
        s.current = 0.05
        s.present = True
        self.state_pub.publish(s)

    def _check(self, goal):
        pts = goal.trajectory.points
        if not pts or any(len(p.positions) != 7 for p in pts):
            print("EXEC GOAL REJECTED (malformed)", flush=True)
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _cancel(self, _goal):
        print("CANCEL RECEIVED", flush=True)
        return CancelResponse.ACCEPT

    def _execute(self, gh):
        self.exec_goals += 1
        goal_n = self.exec_goals
        pts = gh.request.trajectory.points
        times = [p.time_from_start.sec + p.time_from_start.nanosec * 1e-9 for p in pts]
        dur = max(times[-1], 1e-6)
        print(
            "EXEC GOAL ACCEPTED #%d: %d points over %.1f s" % (goal_n, len(pts), dur),
            flush=True,
        )
        t0 = time.monotonic()
        k = 0
        self.injected = False
        while True:
            el = time.monotonic() - t0
            while k < len(times) - 1 and times[k] < el:
                k += 1
            self.q = list(pts[k].positions)
            # late in a long goal (a stroke's contact comes near its end);
            # early in a short one — the 4 mm push is ~0.2 s, and a spike at
            # 88 % of that lands after the goal has already completed
            spike_at = 0.88 if dur >= 1.0 else 0.3
            if goal_n in TRIP_EXEC_NS and el / dur > spike_at and not self.injected:
                print("EFFORT SPIKE injected (goal #%d)" % goal_n, flush=True)
                self.level += 9.0
                self.injected = True
                self.spikes += 1
                if not self.knob_up and self.spikes >= POP_ON_TRIP:
                    # the first trip is the press (POP_ON_TRIP: a later
                    # one): the stub button pops, and the stub camera
                    # (stub_d405) renders the knob standing up from here
                    # on — the mission's pop check reads it
                    self.knob_up = True
                    self.knob_pub.publish(Bool(data=True))
                    print("KNOB UP (stub button popped)", flush=True)
            fb = ExecuteJointTrajectory.Feedback()
            fb.fraction_complete = float(min(1.0, el / dur))
            gh.publish_feedback(fb)
            if gh.is_cancel_requested:
                # the load stays: the arm holds against what it hit
                print("STOPPED at %.1f of %.1f s" % (el, dur), flush=True)
                gh.canceled()
                return ExecuteJointTrajectory.Result(error_code=-6)
            if el >= dur:
                break
            time.sleep(0.05)
        self.level = 0.0  # the arm moved away: the load releases
        self.q = list(pts[-1].positions)
        print("RAN TO COMPLETION", flush=True)
        gh.succeed()
        return ExecuteJointTrajectory.Result(error_code=0)


def main():
    rclpy.init()
    ex = MultiThreadedExecutor()
    ex.add_node(StubArm())
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
