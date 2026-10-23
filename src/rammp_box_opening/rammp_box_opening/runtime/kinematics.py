"""The arm's own kinematics, for the one motion the planner cannot shape.

A cuRobo plan between two poses is a joint-space trajectory: exact at
both ends and free in between. Between a waypoint 60 mm above the
button and the press target that freedom is a sideways bow of 5-7 mm,
widest 26-29 mm above the target — which is exactly where the guard
trips (travel_m + TCP_OFFSET_M above it). Measured on the live planner
against three attended runs' fixes (2026-09-16): predicted 7.4/4.9/6.8 mm
radially outward, missed by 6.1/5.6/6.6. The pre-migration planner's
approach_offset_m constrained that stretch; v1.0.0 has no such field.

So the final stretch is built here instead: forward kinematics from the
driver's own URDF (the same chain cuRobo plans with — the FK agrees with
the planner's returned joints to 0.01 mm), and a damped-least-squares IK
walked down the vertical line in 2 mm steps from the waypoint's joints,
holding the goal attitude. The waypoint itself stays a planner result:
collision-checked, and exact.
"""

import math
import xml.etree.ElementTree as ET

import numpy as np

from rammp_box_opening.perception.d405 import mat_to_quat_xyzw as _mat_to_quat_xyzw
from rammp_box_opening.perception.d405 import quat_to_mat

# cuRobo's ee_link: 0.120 m beyond end_effector_link along its z
# (robot_gen3_2f85.yaml); the driver's URDF stops at end_effector_link
TOOL_FROM_EE_M = 0.120
STEP_M = 0.002  # line sampling; the retimer times it
LIMIT_MARGIN_RAD = 0.02
CONVERGE_M = 5e-5
CONVERGE_RAD = 1e-4
MAX_ITERS = 12
MAX_STEP_RAD = 0.15  # a bigger joint jump per 2 mm is a singularity, refuse
REACH_STEP_M = 0.02  # reach(): the pose walked in steps this long ...
REACH_STEP_RAD = math.radians(6.0)  # ... and this much turn
MAX_REACH_STEP_RAD = 0.6  # a bigger joint jump over one such step is a branch change, refuse


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return Rz @ Ry @ Rx


def _axis_rot(axis, th):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    K = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    return np.eye(3) + math.sin(th) * K + (1.0 - math.cos(th)) * (K @ K)


def mat_to_quat_xyzw(R):
    """Rotation matrix -> unit quaternion (x, y, z, w) as a list of floats."""
    return [float(v) for v in _mat_to_quat_xyzw(R)]


def rotvec(R):
    """The rotation vector of R (axis * angle)."""
    c = (np.trace(R) - 1.0) / 2.0
    c = min(1.0, max(-1.0, c))
    ang = math.acos(c)
    if ang < 1e-9:
        return np.zeros(3)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return v / (2.0 * math.sin(ang)) * ang


def urdf_path():
    from ament_index_python.packages import get_package_share_directory

    return "%s/urdf/kinova_gen3_7dof.urdf" % get_package_share_directory("kinova_gen3_description")


class ArmChain:
    """base_link -> tool_frame of the Gen3, from its URDF."""

    def __init__(self, path=None, tip="end_effector_link", tool_from_tip_m=TOOL_FROM_EE_M, root="base_link"):
        xml_root = ET.parse(path or urdf_path()).getroot()
        by_child = {}
        for j in xml_root.findall("joint"):
            o = j.find("origin")
            xyz = [float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()]
            rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
            ax = j.find("axis")
            axis = [float(v) for v in (ax.get("xyz") if ax is not None else "0 0 1").split()]
            lim = j.find("limit")
            lo = float(lim.get("lower")) if lim is not None and lim.get("lower") else None
            hi = float(lim.get("upper")) if lim is not None and lim.get("upper") else None
            by_child[j.find("child").get("link")] = (
                j.get("name"), j.get("type"), j.find("parent").get("link"), np.array(xyz), _rpy(*rpy), axis, (lo, hi),
            )
        # The nightly arm URDF (flatbot) prefixes all link names with "arm_":
        # e.g. "arm_end_effector_link" instead of "end_effector_link".
        if tip not in by_child and ("arm_" + tip) in by_child:
            tip = "arm_" + tip
        chain = []
        link = tip
        while link != root:
            if link not in by_child:  # walked off the top: this URDF has no `root`
                raise RuntimeError("URDF %s has no chain from %s to root %s" % (path or urdf_path(), tip, root))
            name, typ, parent, t, R, axis, lim = by_child[link]
            chain.append((name, typ, t, R, axis, lim))
            link = parent
        self.chain = chain[::-1]
        self.joint_names = [c[0] for c in self.chain if c[1] in ("revolute", "continuous")]
        self.limits = [c[5] for c in self.chain if c[1] in ("revolute", "continuous")]
        self.tool_from_tip = np.array([0.0, 0.0, float(tool_from_tip_m)])

    def fk(self, q):
        """(R, t) of tool_frame in base_link for the 7 joint values."""
        T = np.eye(4)
        k = 0
        for _name, typ, t, R, axis, _lim in self.chain:
            A = np.eye(4)
            A[:3, :3] = R
            A[:3, 3] = t
            T = T @ A
            if typ in ("revolute", "continuous"):
                J = np.eye(4)
                J[:3, :3] = _axis_rot(axis, q[k])
                k += 1
                T = T @ J
        Rt = T[:3, :3]
        return Rt, T[:3, 3] + Rt @ self.tool_from_tip

    def within_limits(self, q, margin=LIMIT_MARGIN_RAD):
        for v, (lo, hi) in zip(q, self.limits):
            if lo is not None and v < lo + margin:
                return False
            if hi is not None and v > hi - margin:
                return False
        return True

    def _jacobian(self, q, eps=1e-6):
        R0, t0 = self.fk(q)
        J = np.zeros((6, len(q)))
        for i in range(len(q)):
            dq = list(q)
            dq[i] += eps
            R1, t1 = self.fk(dq)
            J[:3, i] = (t1 - t0) / eps
            J[3:, i] = rotvec(R1 @ R0.T) / eps
        return J

    def solve_pose(self, q_seed, xyz, R_goal, damping=0.01):
        """Joints near q_seed putting tool_frame at (xyz, R_goal), or None."""
        q = np.array(q_seed, float)
        xyz = np.asarray(xyz, float)
        for _ in range(MAX_ITERS):
            R, t = self.fk(q)
            err = np.r_[xyz - t, rotvec(R_goal @ R.T)]
            if np.linalg.norm(err[:3]) < CONVERGE_M and np.linalg.norm(err[3:]) < CONVERGE_RAD:
                return q
            J = self._jacobian(q)
            JJt = J @ J.T + (damping ** 2) * np.eye(6)
            q = q + J.T @ np.linalg.solve(JJt, err)
        R, t = self.fk(q)
        err = np.r_[xyz - t, rotvec(R_goal @ R.T)]
        if np.linalg.norm(err[:3]) < CONVERGE_M and np.linalg.norm(err[3:]) < CONVERGE_RAD:
            return q
        return None

    def reach(self, q_start, xyz, quat_xyzw, step_m=REACH_STEP_M, step_rad=REACH_STEP_RAD):
        """Joints putting tool_frame at (xyz, quat_xyzw), found by
        CONTINUATION from q_start: the pose is walked from where the tool is
        to the goal — position in a straight line, orientation about one
        axis — each step solved from the last. The solution is the one on
        q_start's own branch: the elbow and the wrist the arm already has,
        not whichever of the redundant arm's solutions an IK seed lands on
        (the planner's, handed a pose, turned the wrist 3.2-3.6 rad on some
        plans to a box on the left, bench 2026-09-25). None when a step
        does not converge, jumps (a singularity) or hits a limit.

        The walk is only a way to FIND the goal joints; it is not flown. The
        planner still plans, and collision-checks, the path to them."""
        pts, _why = self.walk(q_start, xyz, quat_xyzw, step_m=step_m, step_rad=step_rad)
        return None if pts is None else pts[-1]

    def walk(self, q_start, xyz, quat_xyzw, step_m=REACH_STEP_M, step_rad=REACH_STEP_RAD, max_step_rad=MAX_REACH_STEP_RAD):
        """reach()'s continuation with every step kept: joint waypoints from
        q_start to tool_frame at (xyz, quat_xyzw) — position in a straight
        line, orientation about one axis, each step solved from the last.
        With small steps and max_step_rad it is a path to FLY (a turn in
        place, tasks/depth_press). (points, None) with points[0] == q_start,
        or (None, why)."""
        q = np.array(q_start, float)
        R0, t0 = self.fk(q)
        R_goal = quat_to_mat(*quat_xyzw)
        xyz = np.asarray(xyz, float)
        w = rotvec(R_goal @ R0.T)
        ang = float(np.linalg.norm(w))
        axis = w / ang if ang > 1e-9 else np.array([0.0, 0.0, 1.0])
        n = max(1, int(math.ceil(max(float(np.linalg.norm(xyz - t0)) / step_m, ang / step_rad))))
        pts = [q]
        for k in range(1, n + 1):
            f = k / n
            nxt = self.solve_pose(q, t0 + (xyz - t0) * f, _axis_rot(axis, ang * f) @ R0)
            if nxt is None:
                return None, "IK did not converge %.0f%% of the way" % (100.0 * f)
            if float(np.abs(nxt - q).max()) > max_step_rad:
                return None, "a joint jump of %.2f rad %.0f%% of the way — near a singularity" % (float(np.abs(nxt - q).max()), 100.0 * f)
            if not self.within_limits(nxt):
                return None, "a joint limit %.0f%% of the way" % (100.0 * f)
            pts.append(nxt)
            q = nxt
        return pts, None

    def nearest_solution(self, q_start, xyz, quat_xyzw, seeds=()):
        """Of the redundant arm's IK solutions for tool_frame at (xyz,
        quat_xyzw), the one QUICKEST to fly to from q_start: the slowest
        joint's travel over its velocity limit, the travel counted as the
        planner flies it (the literal difference; runtime/branches). The
        candidates: reach() from q_start and from each of `seeds`, and each
        of those with its wrist folded the other way (joint_5 and joint_7 a
        half turn round, joint_6 mirrored — the same tool pose, re-solved).
        None when there is none inside every limit.

        Measured from HOME to staging over boxes across the table (2026-09-25):
        the planner's own pick took 2.4-4.7 s and turned the wrist 3.2-3.6
        rad; the walk from HOME 2.4-2.5 s; the folded wrist 1.6 s."""
        from rammp_box_opening.runtime.branches import joint_time

        start = np.array(q_start, float)
        cands = []
        for seed in [start, *[np.array(s, float) for s in seeds]]:
            if cands:
                break  # the seeds are only for when the walk from where the arm is found nothing
            q = self.reach(seed, xyz, quat_xyzw)
            if q is not None:
                cands = self.equivalents(start, q)
        if not cands:
            return None
        return min(cands, key=lambda q: joint_time(start, q))

    def equivalents(self, q_start, q):
        """Every way this arm can hold tool_frame where `q` holds it, written
        to be flown from q_start: `q` itself and `q` with its wrist folded
        the other way (joint_5 and joint_7 a half turn round, joint_6
        mirrored, re-solved to the same tool pose), each continuous joint
        taken the short way round from q_start (branches.nearest_branch) —
        those inside every limit."""
        from rammp_box_opening.runtime.branches import (
            CONTINUOUS_JOINTS, PLANNER_CONTINUOUS_LIMIT_RAD, nearest_branch,
        )

        R, t = self.fk(q)
        out = [np.array(q, float)]
        for d5 in (math.pi, -math.pi):
            for d7 in (math.pi, -math.pi):
                flip = np.array(q, float)
                flip[4] += d5
                flip[5] = -flip[5]
                flip[6] += d7
                qf = self.solve_pose(flip, t, R)
                if qf is not None:
                    out.append(qf)
        out = [np.array(nearest_branch(c, q_start), float) for c in out]
        return [
            c for c in out
            if self.within_limits(c) and all(abs(c[i]) <= PLANNER_CONTINUOUS_LIMIT_RAD for i in CONTINUOUS_JOINTS)
        ]

    def straight_line(self, q_start, xyz_end, quat_xyzw, step_m=STEP_M):
        """Joint waypoints from q_start along the straight line from
        tool_frame's current position to xyz_end, holding quat_xyzw.
        Returns (points, None) with points[0] == q_start, or (None, why)."""
        q0 = np.array(q_start, float)
        R_goal = quat_to_mat(*quat_xyzw)
        _R, t0 = self.fk(q0)
        xyz_end = np.asarray(xyz_end, float)
        length = float(np.linalg.norm(xyz_end - t0))
        n = max(1, int(math.ceil(length / step_m)))
        pts = [q0]
        q = q0
        for k in range(1, n + 1):
            target = t0 + (xyz_end - t0) * (k / n)
            nxt = self.solve_pose(q, target, R_goal)
            if nxt is None:
                return None, "IK did not converge %.0f mm down the line" % (1000.0 * length * k / n)
            if float(np.abs(nxt - q).max()) > MAX_STEP_RAD:
                return None, "joint jump %.2f rad over one %.0f mm step — near a singularity" % (float(np.abs(nxt - q).max()), 1000 * step_m)
            if not self.within_limits(nxt):
                return None, "a joint limit %.0f mm down the line" % (1000.0 * length * k / n)
            pts.append(nxt)
            q = nxt
        return pts, None


G_MPS2 = 9.81
GRIPPER_CLOSED_RAD = 0.8  # the 2F-85's knuckle shut: the reach-out presses with the fingers closed


class GravityModel:
    """What the arm's own weight puts on each joint, from the URDF's
    <inertial> masses: the torque each joint HOLDS against gravity at q.

    The driver's /joint_states effort reads this plus a steady offset per
    joint. Flatbot, recorded 2026-10-21 (vsp_boxer_test: 345 samples at
    rest across 0.46 rad of joint_4 and 0.48 of joint_6; every 4th sample is
    kept in test/data/joint_states_flatbot_20261021.npz): effort =
    holding(q) + const to 0.11 / 0.13 / 0.07 Nm rms on joints 4 / 5 / 6. Over
    that recording joint_4's effort drifted 1.47 Nm; with this model's
    share taken out, 0.76. The wrist camera and its mount are not in the
    generated URDF (the driver README's known gaps); fitted in, they moved
    those residuals by under 0.03 Nm, so they stay out. The driver turns
    KORTEX's reaction-signed torque into this sign itself (its
    kortex_transport.cpp fill_feedback: "~ +g(q) at free hold").

    A guard judged on what this does NOT explain feels the same push
    wherever along a reach it comes (runtime/guards.CompensatedGuard)."""

    def __init__(self, path=None, root="base_link", gripper_rad=GRIPPER_CLOSED_RAD):
        path = path or urdf_path()
        xml_root = ET.parse(path).getroot()
        self._joints = {}  # child link -> (name, type, parent, xyz, R, unit axis, mimic)
        for j in xml_root.findall("joint"):
            o = j.find("origin")
            xyz = [float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()]
            rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
            ax = j.find("axis")
            axis = np.array([float(v) for v in (ax.get("xyz") if ax is not None else "0 0 1").split()])
            n = float(np.linalg.norm(axis))
            mim = j.find("mimic")
            mimic = None if mim is None else (mim.get("joint"), float(mim.get("multiplier", "1")), float(mim.get("offset", "0")))
            self._joints[j.find("child").get("link")] = (
                j.get("name"), j.get("type"), j.find("parent").get("link"), np.array(xyz), _rpy(*rpy),
                axis / n if n > 0 else np.array([0.0, 0.0, 1.0]), mimic,
            )
        self._mass = {}  # link -> (kg, centre of mass in the link)
        for link in xml_root.findall("link"):
            i = link.find("inertial")
            if i is None or i.find("mass") is None:
                continue
            o = i.find("origin")
            com = [float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()]
            self._mass[link.get("name")] = (float(i.find("mass").get("value")), np.array(com))
        arm = ArmChain(path, root=root)
        self.joint_names = list(arm.joint_names)
        children = {}
        for child, jt in self._joints.items():
            children.setdefault(jt[2], []).append(child)
        order, frontier = [], [root]
        while frontier:  # parents before children
            link = frontier.pop(0)
            order.append(link)
            frontier += children.get(link, [])
        self._root = root
        self._order = order[1:]
        child_of = {jt[0]: child for child, jt in self._joints.items()}
        self._below = {}  # arm joint -> the links with mass it carries
        for name in self.joint_names:
            out, stack = set(), [child_of[name]]
            while stack:
                link = stack.pop()
                out.add(link)
                stack += children.get(link, [])
            self._below[name] = [link for link in out if link in self._mass]
        # tool_frame: TOOL_FROM_EE_M along the end-effector link's z, as ArmChain
        self._ee = next(child for child, jt in self._joints.items() if jt[0] == arm.chain[-1][0])
        self._tool = np.array(arm.tool_from_tip, float)
        self.gripper_rad = float(gripper_rad)

    def _frames(self, q):
        qmap = dict(zip(self.joint_names, (float(v) for v in q)))
        T = {self._root: np.eye(4)}
        axes = {}
        for link in self._order:
            name, typ, parent, xyz, R, axis, mimic = self._joints[link]
            A = np.eye(4)
            A[:3, :3] = R
            A[:3, 3] = xyz
            Tj = T[parent] @ A
            if typ in ("revolute", "continuous"):
                if name in qmap:
                    th = qmap[name]
                    axes[name] = (Tj[:3, 3].copy(), Tj[:3, :3] @ axis)
                elif mimic is not None:
                    th = mimic[1] * self.gripper_rad + mimic[2]  # the gripper's mimic joints follow its knuckle
                else:
                    th = self.gripper_rad
                J = np.eye(4)
                J[:3, :3] = _axis_rot(axis, th)
                Tj = Tj @ J
            T[link] = Tj
        return T, axes

    def holding(self, q, extra=(), first=0):
        """The torque each arm joint holds against gravity at q (Nm), in the
        sign the driver reports — joints `first` onward. `extra`: (kg,
        (x, y, z) in tool_frame) point masses the URDF does not carry."""
        T, axes = self._frames(q)
        g = np.array([0.0, 0.0, -G_MPS2])
        ee = T[self._ee]
        com = {link: T[link][:3, :3] @ c + T[link][:3, 3] for link, (_m, c) in self._mass.items() if link in T}
        pts = [(float(m), ee[:3, :3] @ (np.asarray(p, float) + self._tool) + ee[:3, 3]) for m, p in extra]
        out = []
        for name in self.joint_names[first:]:
            p, a = axes[name]
            f = np.zeros(3)  # the weight's moment about the joint, summed: sum m (r - p) x g
            for link in self._below[name]:
                f += self._mass[link][0] * (com[link] - p)
            for m, r in pts:
                f += m * (r - p)
            out.append(-float(a @ np.cross(f, g)))
        return np.array(out)

    def wrist_holding(self, q):
        """holding(q) for the joints the guards are fed (client.wrist_efforts: joints 4-7)."""
        return self.holding(q, first=3)

    def tip_load(self, q, point_tool):
        """The wrist joints' holding torque per newton pulling straight DOWN
        at `point_tool` (tool_frame): a weight hanging there — or, negated, a
        surface pushing up on it."""
        return self.holding(q, extra=[(1.0 / G_MPS2, point_tool)], first=3) - self.wrist_holding(q)
