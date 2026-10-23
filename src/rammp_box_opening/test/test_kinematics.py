"""The arm's own FK/IK: the straight final descent the planner cannot shape."""

import math

import numpy as np
import pytest

from rammp_box_opening.constants import HOME
from rammp_box_opening.perception.d405 import quat_to_mat
from rammp_box_opening.runtime.kinematics import ArmChain


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
    solve = ArmChain.solve_pose

    def counting(self, *a, **kw):
        calls.append(1)
        return solve(self, *a, **kw)

    monkeypatch.setattr(ArmChain, "solve_pose", counting)
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
