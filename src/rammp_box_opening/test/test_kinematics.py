"""The arm's own FK/IK: the straight final descent the planner cannot shape."""

import math
import os

import numpy as np
import pytest

from rammp_box_opening.constants import HOME
from rammp_box_opening.perception.d405 import quat_to_mat
from rammp_box_opening.runtime import kinematics as kin
from rammp_box_opening.runtime.kinematics import ArmChain, _axis_rot, mat_to_quat_xyzw, rotvec

FLATBOT_URDF = os.path.join(os.path.dirname(__file__), "data", "arm_flatbot_20261021.urdf")


@pytest.fixture(scope="module")
def chain():
    return ArmChain()


def test_fk_matches_the_planner_at_home(chain):
    # flatbot HOME (new_adapter_plate_v3, 2026-10-21): arm tilted 30° forward,
    # tool z at 30° above horizontal, frame ~0.04 m out and 0.586 m up in arm_mount.
    R, t = chain.fk(HOME)
    assert np.allclose(t, [-0.03551, 0.04043, 0.58633], atol=2e-4), t
    assert np.allclose(R[:, 2], [0.86353, 0.07361, 0.49889], atol=2e-3)


def test_fk_matches_the_planner_at_a_press_staging(chain):
    """A staging pose the live planner solved (probe 2026-09-16): its joints
    must land on the pose it was asked for, or the line below is built
    from the wrong place."""
    q = [0.0978, 0.441, 3.3815, -1.7197, -0.1212, -0.9954, 0.1181]
    _R, t = chain.fk(q)
    # bearing -15 deg box: staging = button top + 0.12
    assert np.allclose(t, [0.4264, -0.1151, 0.2055], atol=1.5e-3), t
    assert np.allclose(_R[:, 2], [0.0, 0.0, -1.0], atol=2e-3)  # tool down


def test_the_line_is_straight_and_holds_the_attitude(chain):
    # the planner's waypoint joints 60 mm above the press target for the
    # bearing -15 deg box (live probe 2026-09-16)
    q_start = [0.1208, 0.5375, 3.3522, -1.8057, -0.1478, -0.8136, 0.1407]
    _R0, t0 = chain.fk(q_start)
    from rammp_box_opening.models.container import attitude_quat

    quat = attitude_quat([180.0, 0.0, 0.0], math.atan2(-0.1151, 0.4264))
    # 60 mm straight down from wherever the FK says the start is
    end = [t0[0], t0[1], t0[2] - 0.06]
    pts, why = chain.straight_line(q_start, end, quat)
    assert why is None and pts is not None and len(pts) >= 30
    assert np.allclose(pts[0], q_start)
    R_goal = quat_to_mat(*quat)
    worst_lat = 0.0
    for q in pts[1:]:
        R, t = chain.fk(q)
        worst_lat = max(worst_lat, float(np.hypot(t[0] - t0[0], t[1] - t0[1])))
        ang = math.degrees(math.acos(min(1.0, (np.trace(R_goal @ R.T) - 1) / 2)))
        assert ang < 0.05
    assert worst_lat < 1e-4  # 0.1 mm: the bow was 5-7 mm
    _R, t_end = chain.fk(pts[-1])
    assert np.allclose(t_end, end, atol=1e-4)


def test_an_unreachable_line_is_refused_with_a_reason(chain):
    q_start = [0.1208, 0.5375, 3.3522, -1.8057, -0.1478, -0.8136, 0.1407]
    _R0, t0 = chain.fk(q_start)
    pts, why = chain.straight_line(q_start, [t0[0] + 1.5, t0[1], t0[2]], [1.0, 0.0, 0.0, 0.0])
    assert pts is None and why


# fk's precomputed chain and the analytic Jacobian: the same FK as the chain
# multiplied out link by link, the exact derivative of it, and the same
# lines (and refusals) the finite-difference Jacobian built.


def _chains():
    return [ArmChain(), ArmChain(FLATBOT_URDF, root="arm_mount")]


def _fk_multiplied_out(chain, q):
    """The chain multiplied out link by link in 4x4s (fk as it was)."""
    T = np.eye(4)
    k = 0
    for _name, typ, t, R, axis, _lim in chain.chain:
        A = np.eye(4)
        A[:3, :3] = R
        A[:3, 3] = t
        T = T @ A
        if typ in ("revolute", "continuous"):
            J = np.eye(4)
            J[:3, :3] = _axis_rot(axis, q[k])
            k += 1
            T = T @ J
    return T[:3, :3], T[:3, 3] + T[:3, :3] @ chain.tool_from_tip


def _jacobian_fd(chain, q, eps=1e-6):
    """The forward difference of fk solve_pose used to take."""
    R0, t0 = chain.fk(q)
    J = np.zeros((6, len(q)))
    for i in range(len(q)):
        dq = np.array(q, float)
        dq[i] += eps
        R1, t1 = chain.fk(dq)
        J[:3, i] = (t1 - t0) / eps
        J[3:, i] = rotvec(R1 @ R0.T) / eps
    return J


def _random_q(chain, rng):
    return np.array([rng.uniform(-math.pi if lo is None else lo, math.pi if hi is None else hi) for lo, hi in chain.limits])


def test_the_precomputed_fk_is_the_multiplied_out_chain():
    rng = np.random.default_rng(0)
    for chain in _chains():
        for _ in range(200):
            q = _random_q(chain, rng)
            R, t = chain.fk(q)
            R_ref, t_ref = _fk_multiplied_out(chain, q)
            assert np.abs(R - R_ref).max() < 1e-12
            assert np.abs(t - t_ref).max() < 1e-12


def test_a_short_q_still_raises(chain):
    with pytest.raises(IndexError):
        chain.fk([0.0] * 6)


def test_the_analytic_jacobian_is_the_derivative_of_fk():
    """Against a central difference: of the tool position, and the rotvec
    of R(q+) R(q-)^T, both in the root frame. Exact to 1e-8 (measured
    2.5e-11); the forward difference it replaced was 5e-7 off."""
    rng = np.random.default_rng(1)
    eps = 1e-5
    for chain in _chains():
        for _ in range(50):
            q = _random_q(chain, rng)
            J = chain._jacobian(q)
            assert J.shape == (6, 7)
            for i in range(7):
                qp, qm = q.copy(), q.copy()
                qp[i] += eps
                qm[i] -= eps
                Rp, tp = chain.fk(qp)
                Rm, tm = chain.fk(qm)
                assert np.abs(J[:3, i] - (tp - tm) / (2 * eps)).max() < 1e-8
                assert np.abs(J[3:, i] - rotvec(Rp @ Rm.T) / (2 * eps)).max() < 1e-8


def test_a_line_takes_one_fk_pass_per_ik_iteration(chain, monkeypatch):
    """solve_pose takes R, t and J from one pass, not fk plus 8 more for a
    finite difference: the IK is the dead time before every plan-free leg."""
    q_start = [0.1208, 0.5375, 3.3522, -1.8057, -0.1478, -0.8136, 0.1407]
    R0, t0 = chain.fk(q_start)
    iters, fks = [0], [0]
    fk_jac, fk = chain._fk_jac, chain.fk

    def counting_fk_jac(q):
        iters[0] += 1
        return fk_jac(q)

    def counting_fk(q):
        fks[0] += 1
        return fk(q)

    monkeypatch.setattr(chain, "_fk_jac", counting_fk_jac)
    monkeypatch.setattr(chain, "fk", counting_fk)
    pts, why = chain.straight_line(q_start, t0 - [0.0, 0.0, 0.06], mat_to_quat_xyzw(R0))
    assert why is None and len(pts) == 31
    assert iters[0] >= 30  # at least one per step
    assert fks[0] <= len(pts)  # the line's start, and a final check per step at most


def _line_both_ways(chain, monkeypatch, q_start, end, quat):
    """straight_line as built now, and as the finite-difference solve_pose
    (fk and the Jacobian taken separately) built it."""
    now = chain.straight_line(q_start, end, quat)

    def solve_fd(q_seed, xyz, R_goal, damping=0.01):
        q = np.array(q_seed, float)
        xyz = np.asarray(xyz, float)
        for _ in range(kin.MAX_ITERS):
            R, t = chain.fk(q)
            err = np.r_[xyz - t, rotvec(R_goal @ R.T)]
            if np.linalg.norm(err[:3]) < kin.CONVERGE_M and np.linalg.norm(err[3:]) < kin.CONVERGE_RAD:
                return q
            J = _jacobian_fd(chain, q)
            q = q + J.T @ np.linalg.solve(J @ J.T + (damping ** 2) * np.eye(6), err)
        R, t = chain.fk(q)
        err = np.r_[xyz - t, rotvec(R_goal @ R.T)]
        if np.linalg.norm(err[:3]) < kin.CONVERGE_M and np.linalg.norm(err[3:]) < kin.CONVERGE_RAD:
            return q
        return None

    with monkeypatch.context() as m:
        m.setattr(chain, "solve_pose", solve_fd)
        ref = chain.straight_line(q_start, end, quat)
    return now, ref


def _assert_same_line(now, ref):
    (pts, why), (pts_ref, why_ref) = now, ref
    assert why == why_ref
    if pts_ref is None:
        assert pts is None
        return
    assert len(pts) == len(pts_ref)
    for a, b in zip(pts, pts_ref):
        assert np.abs(np.asarray(a) - np.asarray(b)).max() < 1e-6


def test_the_lines_are_the_ones_the_finite_difference_built(chain, monkeypatch):
    from rammp_box_opening.models.container import attitude_quat

    # the 60 mm descent of test_the_line_is_straight_and_holds_the_attitude
    q_start = [0.1208, 0.5375, 3.3522, -1.8057, -0.1478, -0.8136, 0.1407]
    _R0, t0 = chain.fk(q_start)
    quat = attitude_quat([180.0, 0.0, 0.0], math.atan2(-0.1151, 0.4264))
    now, ref = _line_both_ways(chain, monkeypatch, q_start, [t0[0], t0[1], t0[2] - 0.06], quat)
    assert now[1] is None and len(now[0]) == 31
    _assert_same_line(now, ref)

    # a 290 mm stroke along the tool's z from HOME on flatbot's chain (a depth:touch)
    fb = ArmChain(FLATBOT_URDF, root="arm_mount")
    R, t = fb.fk(HOME)
    now, ref = _line_both_ways(fb, monkeypatch, HOME, t + 0.29 * R[:, 2], mat_to_quat_xyzw(R))
    assert now[1] is None and len(now[0]) == 147
    _assert_same_line(now, ref)


def test_a_refusal_is_the_one_the_finite_difference_gave(chain, monkeypatch):
    q_start = [0.1208, 0.5375, 3.3522, -1.8057, -0.1478, -0.8136, 0.1407]
    _R0, t0 = chain.fk(q_start)
    now, ref = _line_both_ways(chain, monkeypatch, q_start, [t0[0] + 1.5, t0[1], t0[2]], [1.0, 0.0, 0.0, 0.0])
    assert now[0] is None and now[1]
    _assert_same_line(now, ref)




def _four_solve_equivalents(chain, q_start, q):
    """equivalents() as it was: the folded wrist re-solved from all four
    half-turn seeds on joint_5 and joint_7."""
    from rammp_box_opening.runtime.branches import (
        CONTINUOUS_JOINTS, PLANNER_CONTINUOUS_LIMIT_RAD, nearest_branch,
    )

    R, t = chain.fk(q)
    out = [np.array(q, float)]
    for d5 in (math.pi, -math.pi):
        for d7 in (math.pi, -math.pi):
            flip = np.array(q, float)
            flip[4] += d5
            flip[5] = -flip[5]
            flip[6] += d7
            qf = chain.solve_pose(flip, t, R)
            if qf is not None:
                out.append(qf)
    out = [np.array(nearest_branch(c, q_start), float) for c in out]
    return [
        c for c in out
        if chain.within_limits(c) and all(abs(c[i]) <= PLANNER_CONTINUOUS_LIMIT_RAD for i in CONTINUOUS_JOINTS)
    ]


def test_the_folded_wrist_is_solved_once(chain, monkeypatch):
    # joint_5 and joint_7 are continuous: the four half-turn seeds differ by
    # whole turns and converge together, so one solve is all equivalents pays
    calls = []
    solve = chain.solve_pose

    def counting(*a, **kw):
        calls.append(1)
        return solve(*a, **kw)

    # on the instance: a test that patched the shared chain's instance
    # attribute leaves it behind, and that would hide a class patch
    monkeypatch.setattr(chain, "solve_pose", counting)
    q = np.array(HOME, float) + np.array([0.05, -0.05, 0.03, 0.05, -0.04, 0.06, 0.02])
    cands = chain.equivalents(HOME, q)
    assert len(calls) == 1
    assert len(cands) >= 2  # q and its fold


def test_one_fold_solve_gives_the_four_solves_candidates(chain):
    from rammp_box_opening.runtime.branches import CONTINUOUS_JOINTS, joint_time

    rng = np.random.default_rng(20261022)
    lo = np.array([-math.pi if a is None else a for a, _b in chain.limits])
    hi = np.array([math.pi if b is None else b for _a, b in chain.limits])
    checked = 0
    for k in range(30):
        q = rng.uniform(lo * 0.8, hi * 0.8)
        for i in CONTINUOUS_JOINTS:
            q[i] += 2.0 * math.pi * rng.integers(-1, 2)
        q_start = rng.uniform(lo * 0.8, hi * 0.8)
        if k % 2:  # continuous joints right at the planner's +/-6 rad bound
            for i in CONTINUOUS_JOINTS:
                q_start[i] = rng.choice([-1.0, 1.0]) * rng.uniform(5.0, 5.99)
        new = chain.equivalents(q_start, q)
        old = _four_solve_equivalents(chain, q_start, q)
        assert len(new) == len(old)
        unmatched = list(old)
        for c in new:
            j = next((j for j, o in enumerate(unmatched) if np.allclose(c, o, atol=1e-6)), None)
            assert j is not None, (q, q_start, c)
            unmatched.pop(j)
        if old:
            pick_new = min(new, key=lambda c: joint_time(q_start, c))
            pick_old = min(old, key=lambda c: joint_time(q_start, c))
            assert np.allclose(pick_new, pick_old, atol=1e-6)
            checked += 1
    assert checked >= 10




def test_the_same_reach_question_twice_is_walked_once(monkeypatch):
    """The staging's reach proof and the planner client ask nearest_solution
    the same thing back to back (2026-10-22): the second is answered from
    the first — and only for the exact same inputs."""
    from rammp_box_opening.runtime import approach
    from rammp_box_opening.runtime.kinematics import mat_to_quat_xyzw

    arm = ArmChain()
    q_goal = list(HOME)
    q_goal[1] += 0.3
    R, t = arm.fk(q_goal)
    xyz, quat = [float(v) for v in t], [float(v) for v in mat_to_quat_xyzw(R)]
    walks = []
    real = arm.reach

    def counted(*a, **k):
        walks.append(1)
        return real(*a, **k)

    monkeypatch.setattr(arm, "reach", counted)
    first = arm.nearest_solution(HOME, xyz, quat)
    assert first is not None and len(walks) == 1
    kept = float(first[0])
    first[0] += 1.0  # the caller's copy: the memo keeps its own
    again = arm.nearest_solution(tuple(HOME), np.array(xyz), quat)  # the same numbers, other types
    assert len(walks) == 1 and again[0] == pytest.approx(kept)
    nudged = list(HOME)
    nudged[2] += 1e-6
    assert arm.nearest_solution(nudged, xyz, quat) is not None  # another start: walked again
    assert len(walks) == 2
    # a new root (set_chain_root) is a new chain, with nothing remembered
    for name in ("_ROOT", "_URDF_PATH", "_CHAIN", "_GRAVITY"):
        monkeypatch.setattr(approach, name, getattr(approach, name))
    monkeypatch.setattr(approach, "_CHAIN", arm)
    approach.set_chain_root("base_link")
    fresh = approach._chain()
    assert fresh is not arm and fresh._nearest_memo is None
