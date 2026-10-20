"""Measuring the wrist camera's timestamp lag from a recorded motion."""

import numpy as np

from rammp_box_opening.perception.lag import OFFSETS_S, best_offset, moving_flags


def _synthetic(true_lag_s, rng):
    """Frames over a static box: still ones place it at the truth; moving
    ones place it off by velocity * (candidate - true lag), plus noise."""
    truth = np.array([0.45, -0.10, 0.085])
    n_still, n_move = 10, 30
    fixes = np.full((n_still + n_move, len(OFFSETS_S), 3), np.nan)
    moving = np.zeros(n_still + n_move, bool)
    for i in range(n_still):
        for k in range(len(OFFSETS_S)):
            fixes[i, k] = truth + rng.normal(0, 0.002, 3)
    for j in range(n_move):
        i = n_still + j
        moving[i] = True
        v = rng.uniform(-0.5, 0.5, 3)  # m/s, this frame's camera velocity
        for k, off in enumerate(OFFSETS_S):
            fixes[i, k] = truth + v * (off - true_lag_s) + rng.normal(0, 0.003, 3)
    return fixes, moving


def test_the_offset_that_makes_moving_frames_agree_is_the_lag():
    rng = np.random.default_rng(0)
    for lag in (-0.03, 0.0, 0.04):
        fixes, moving = _synthetic(lag, rng)
        off, table = best_offset(OFFSETS_S, fixes, moving)
        assert off is not None and abs(off - lag) < 0.011, (lag, off)
        # ... and the table is honest about the residual at the pick
        rms_at_pick = next(r for o, r, n in table if abs(o - off) < 1e-9)
        assert rms_at_pick < 8.0  # mm: noise only


def test_no_decision_without_enough_frames_on_both_sides():
    fixes = np.full((4, len(OFFSETS_S), 3), np.nan)
    moving = np.array([True, True, False, False])
    off, table = best_offset(OFFSETS_S, fixes, moving)
    assert off is None and all(np.isnan(r) for _o, r, _n in table)


def test_moving_frames_are_those_where_the_camera_travelled():
    t = np.array([[0, 0, 0.5], [0, 0, 0.5], [0.01, 0, 0.5], [0.02, 0, 0.5], [0.02, 0, 0.5]])
    assert moving_flags(t).tolist() == [False, False, True, True, False]
