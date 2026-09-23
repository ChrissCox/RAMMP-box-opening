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

    def __init__(self, path=None, tip="end_effector_link", tool_from_tip_m=TOOL_FROM_EE_M):
        root = ET.parse(path or urdf_path()).getroot()
        by_child = {}
        for j in root.findall("joint"):
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
        chain = []
        link = tip
        while link != "base_link":
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
