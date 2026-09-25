"""The gripper's empty-close reading, learned (runtime/fingers).

Bench 2026-09-25: after the arm driver restarted, a close on nothing read
0.636 instead of 0.793 — the fingertips visibly shut in the wrist frames
either way. The mission's close found them already shut, "never moved", and
it refused to press three runs running."""

import pytest

from rammp_box_opening.runtime import fingers


@pytest.fixture
def state(tmp_path, monkeypatch):
    monkeypatch.setenv("RAMMP_BOX_OPENING_STATE", str(tmp_path))
    return tmp_path


def test_fingers_shut_at_the_new_reading_count_as_closed(state):
    assert fingers.learned_closed() == pytest.approx(0.793)  # nothing learned: the reading the band was measured at
    assert not fingers.already_closed(0.636)
    assert fingers.learn_closed(0.636)  # the 2026-09-25 bench
    assert fingers.already_closed(0.636) and fingers.already_closed(0.793)  # either reading of a shut hand
    assert not fingers.already_closed(0.41)  # on the knob
    assert not fingers.already_closed(0.05)  # open
    assert not fingers.already_closed(None)


def test_only_an_empty_close_is_ever_learned(state):
    assert fingers.learn_closed(0.79) and fingers.learned_closed() == pytest.approx(0.79)
    assert not fingers.already_closed(0.636)  # the gripper reads full again: 0.636 is not shut then
    assert not fingers.learn_closed(0.41)  # the knob in the fingers: never passes for shut
    assert not fingers.learn_closed(0.09)  # the box
    assert not fingers.learn_closed(None)
    assert fingers.learned_closed() == pytest.approx(0.79)


def test_the_grip_band_follows_the_reading(state):
    """The band [0.3, 0.55] was measured when an empty close read 0.793: a
    knob grip read 0.41 then. At 0.636 every reading is 0.8 of what it was."""
    fingers.learn_closed(0.636)
    lo, hi = fingers.scaled_band((0.3, 0.55))
    assert lo == pytest.approx(0.3 * 0.636 / 0.793) and hi == pytest.approx(0.55 * 0.636 / 0.793)
    assert lo < 0.41 * 0.636 / 0.793 < hi  # the knob, as it now reads
    assert hi < 0.636  # a close on air still fails the band
    assert fingers.scaled_band((0.3, 0.55), closed=0.793) == pytest.approx((0.3, 0.55))
