"""The look phase: how the camera gets pointed at the table before anything
about the scene is known."""

import math

import pytest

from rammp_box_opening.constants import HOME
from rammp_box_opening.primitives.look import look_joints, sweep_targets


def test_the_look_turns_the_wrist_down_and_lifts_the_hand_clear():
    """Measured on the real planner from HOME: flying to the old scan pose
    3.89 s, re-orienting the tool in place 4.01 s — both swing ~3 rad through
    the wrist-flat -> tool-down IK family change — against 3.42 s for this.

    Two joints, not one. Turning the wrist alone leaves the fingertips 49 mm
    over the pre-detection keep-out band, inside cuRobo's activation
    distance, and the goal is refused (IK_FAIL, live planner 2026-09-15);
    past that, lift buys VIEW, which is what the first field run ran out of.
    The elbow opens to lift the hand and the wrist takes the same angle back,
    so the camera still points straight down."""
    from rammp_box_opening.primitives.look import LIFT_RAD

    q = look_joints(HOME)
    assert q[3] == pytest.approx(HOME[3] + LIFT_RAD)
    assert q[5] == pytest.approx(HOME[5] - math.pi / 2 - LIFT_RAD)
    # the tool's pitch is what the elbow and the wrist pitch add up to: the
    # lift must leave it exactly where the bare tilt would have
    assert q[3] + q[5] == pytest.approx(HOME[3] + HOME[5] - math.pi / 2)
    assert q[:3] == pytest.approx(HOME[:3])
    assert q[4] == pytest.approx(HOME[4])
    assert q[6] == pytest.approx(HOME[6])


def test_the_look_follows_whatever_rest_pose_it_is_given():
    # a chair's rest pose is not this bench's HOME; the look is relative
    from rammp_box_opening.primitives.look import LIFT_RAD

    other = [0.2, 0.3, 3.0, -2.0, 0.1, 0.9, 1.5]
    q = look_joints(other)
    assert q[3] == pytest.approx(other[3] + LIFT_RAD)
    assert q[5] == pytest.approx(other[5] - math.pi / 2 - LIFT_RAD)
    assert q[:3] == pytest.approx(other[:3])
    assert q[4] == pytest.approx(other[4])


def test_the_sweep_pans_the_base_each_way_from_the_look():
    """The nadir view is 0.77 x 0.44 m at LID height; a box outside it is
    found by panning the base, which keeps the view nadir while moving it
    across the table."""
    look = look_joints(HOME)
    first, second = sweep_targets(look, 0.6)
    assert first[0] == pytest.approx(look[0] + 0.6)
    assert second[0] == pytest.approx(look[0] - 0.6)
    assert first[1:] == pytest.approx(look[1:])
    assert second[1:] == pytest.approx(look[1:])
