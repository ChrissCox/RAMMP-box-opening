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
