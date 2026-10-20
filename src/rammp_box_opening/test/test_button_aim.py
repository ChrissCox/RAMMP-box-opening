"""The circle-only aim from staging, on the real frame that the plateau
gate refused (2026-09-17, the 4.1-inch OXO, wrist 25 cm above the lid)."""

from pathlib import Path

import numpy as np
import pytest

from rammp_box_opening.models.container import ContainerModel
from rammp_box_opening.perception.button_aim import (
    MIN_HITS,
    agreeing_median,
    circle_gates,
    find_button,
)

CFG = "src/rammp_box_opening/config/containers/oxo_pop.yaml"
FRAME = Path(__file__).parent / "data" / "wrist_staging_big_box.npz"
TABLE_Z = -0.027


@pytest.fixture(scope="module")
def frame():
    z = np.load(FRAME)
    return {k: z[k] for k in ("color", "depth", "k", "rot_cam", "trans_cam")}


@pytest.fixture(scope="module")
def model():
    return ContainerModel.load(CFG)


def test_the_button_is_found_on_the_refused_staging_frame(frame, model):
    s, why = find_button(frame["color"], frame["depth"], frame["k"], frame["rot_cam"], frame["trans_cam"], TABLE_Z, model)
    assert why is None and s is not None
    # where the wrist's own search found it a minute later: [0.573, -0.022]
    assert s.xyz[0] == pytest.approx(0.574, abs=0.004)
    assert s.xyz[1] == pytest.approx(-0.026, abs=0.004)
    assert s.xyz[2] == pytest.approx(TABLE_Z + model.dims[2], abs=0.012)
    assert s.diameter_m == pytest.approx(model.button_diameter_m, rel=0.15)


def test_a_lid_corner_at_lid_height_is_not_the_button(frame, model):
    """Hough also returns the lid's rounded corners: flat, at lid height,
    and half their surround is table. The annulus gate refuses them."""
    lid_z = TABLE_Z + model.dims[2]
    why = circle_gates(frame["depth"], frame["k"], frame["rot_cam"], frame["trans_cam"], 508.0, 200.0, 33.5, lid_z)
    assert why is not None and "corner" in why
    assert circle_gates(frame["depth"], frame["k"], frame["rot_cam"], frame["trans_cam"], 462.0, 274.0, 41.0, lid_z) is None


def test_a_frame_without_the_lid_finds_nothing(frame, model):
    depth = frame["depth"].copy()
    depth[:] = 0.30  # a bare table at that range: nothing at lid height
    s, why = find_button(frame["color"], depth, frame["k"], frame["rot_cam"], frame["trans_cam"], TABLE_Z, model)
    assert s is None and "none the button" in why


def test_too_close_refuses_rather_than_aims(frame, model):
    t = np.array(frame["trans_cam"], float)
    t[2] = TABLE_Z + model.dims[2] + 0.05
    s, why = find_button(frame["color"], frame["depth"], frame["k"], frame["rot_cam"], t, TABLE_Z, model)
    assert s is None and "too close" in why


def test_hits_must_agree_before_they_aim():
    pts = [(0.5, 0.0, 0.08), (0.5, 0.01, 0.08), (0.5, 0.0, 0.08)]  # one 10 mm off
    assert agreeing_median(pts) is None
    pts = [(0.5, 0.0, 0.08)] * (MIN_HITS - 1)
    assert agreeing_median(pts) is None
    pts = [(0.500, 0.000, 0.080), (0.502, 0.001, 0.081), (0.501, -0.001, 0.080)]
    m = agreeing_median(pts)
    assert m is not None and m[0] == pytest.approx(0.501) and m[1] == pytest.approx(0.0)


MISSED = Path(__file__).parent / "data" / "wrist_aim_pressed_right_20260921.npz"


def test_the_press_point_is_the_centre_of_the_circle(model):
    """Bench 2026-09-21 13:35, the owner: "it's still pressing too far to
    the right of the button". This is that frame. Hough had the seam exactly
    (centre at pixel 454, 375), and the press went 13 mm up and to the right
    of it, onto the raised ring — the pads met a surface 10.8 mm above the
    button. A "dimple" detector (2026-09-17) had taken the lowest patch of
    depth inside the disc for the place a finger presses, and here that
    patch was stereo noise beside the seam. It fired on one real aim out of
    four, and that one missed; every press at the centre landed on the
    button. The owner had already said it: the gripper goes to the centre
    of the circle OpenCV segments. Nothing overrides that now."""
    z = np.load(MISSED)
    k, rot, trans = z["k"], z["rot_cam"], z["trans_cam"]

    def centre_of(s):
        pc = np.array([(s.uv[0] - k[0, 2]) / k[0, 0] * s.range_m, (s.uv[1] - k[1, 2]) / k[1, 1] * s.range_m, s.range_m])
        return tuple((rot @ pc + trans)[:2])

    s, why = find_button(z["color"], z["depth"], k, rot, trans, TABLE_Z, model)
    assert why is None and s is not None
    assert s.uv == pytest.approx((454.5, 375.5), abs=3.0)  # the seam, as logged that day (another of its 16 frames)
    assert s.xyz[:2] == pytest.approx(centre_of(s), abs=1e-4)  # the press point IS that pixel, lifted at the disc's range
    # that day's press point was [0.4463, -0.0542]: 11-13 mm from the centre
    assert np.hypot(s.xyz[0] - 0.4463, s.xyz[1] + 0.0542) > 0.010

    # The patch that fooled it came and went with the depth noise (it is not
    # in this frame of that run's 21). Put it back where it was — 4 mm low,
    # up and to the right — and the press point must not move.
    depth = z["depth"].copy()
    yy, xx = np.mgrid[0 : depth.shape[0], 0 : depth.shape[1]]
    depth[(xx - 470.7) ** 2 + (yy - 357.6) ** 2 < 9.0 ** 2] += 0.004
    s2, why = find_button(z["color"], depth, k, rot, trans, TABLE_Z, model)
    assert why is None and s2.uv == pytest.approx(s.uv, abs=0.5)
    assert s2.xyz[:2] == pytest.approx(centre_of(s2), abs=1e-4)
    assert not hasattr(s2, "dimple")


def test_the_top_height_is_the_disc_median(frame, model):
    """The height reported for the button's top is the median over the
    disc, not the one pixel at its centre: what the closed pads meet, and
    the honest "top" for the calibration line."""
    s, why = find_button(frame["color"], frame["depth"], frame["k"], frame["rot_cam"], frame["trans_cam"], TABLE_Z, model)
    assert s.xyz[2] == pytest.approx(0.086, abs=0.003)


def test_the_aim_says_whether_the_button_already_stands_proud(model):
    """Two real aim frames of the same box, 2026-09-21. At 14:16 it was
    closed: the button's disc level with the lid ring around it. At 13:35 it
    had been left open by the run before, and the whole disc stood 8-10 mm
    above its own lid — a difference taken inside one frame, so neither the
    camera's mount nor the table's height enters it. That run pressed the
    open box shut; so did two more at 14:14. The touch cannot be trusted to
    notice (the raised knob is soft: strokes have ridden it all the way down
    under a 3 Nm guard), so the aim reports it and the mission does not
    press."""
    closed = np.load(Path(__file__).parent / "data" / "wrist_aim_closed_20260921.npz")
    s, why = find_button(closed["color"], closed["depth"], closed["k"], closed["rot_cam"], closed["trans_cam"], TABLE_Z, model)
    assert why is None and abs(s.above_lid_m) < 0.003
    opened = np.load(MISSED)
    s, why = find_button(opened["color"], opened["depth"], opened["k"], opened["rot_cam"], opened["trans_cam"], TABLE_Z, model)
    assert why is None and 0.006 < s.above_lid_m < 0.013


def test_a_knob_standing_high_is_still_the_button(model):
    """This knob has been met at +17.7 mm (2026-09-17), not only at ~10. The
    lid-height gate allowed the centre 12 mm either way — so a knob that
    high was "not the button", the aim found nothing, the mission fell back
    to the wrist search, and THAT path presses with no open-box check at
    all. Above the lid is where a popped button is: the gate lets the centre
    stand up to a knob's height above the lid, and still nothing below it."""
    from rammp_box_opening.perception.button_aim import KNOB_MAX_UP_M

    z = np.load(Path(__file__).parent / "data" / "wrist_aim_closed_20260921.npz")
    s0, _ = find_button(z["color"], z["depth"], z["k"], z["rot_cam"], z["trans_cam"], TABLE_Z, model)
    yy, xx = np.mgrid[0 : z["depth"].shape[0], 0 : z["depth"].shape[1]]
    disc = (xx - s0.uv[0]) ** 2 + (yy - s0.uv[1]) ** 2 < (0.97 * s0.r_px) ** 2
    for up in (0.017, KNOB_MAX_UP_M - 0.003):
        depth = z["depth"].copy()
        depth[disc] -= up  # the whole button that much nearer the camera
        s, why = find_button(z["color"], depth, z["k"], z["rot_cam"], z["trans_cam"], TABLE_Z, model)
        assert why is None, why
        assert s.above_lid_m == pytest.approx(up, abs=0.003)
    depth = z["depth"].copy()
    depth[disc] += 0.017  # a hole that deep is no button
    s, why = find_button(z["color"], depth, z["k"], z["rot_cam"], z["trans_cam"], TABLE_Z, model)
    assert s is None and "lid height" in why


def test_the_pink_canister_is_aimed_at_staging():
    """Bench 2026-09-23, the owner: "it isn't finding the box even though
    the box is right underneath it". This is that frame: the button in plain
    view, and the aim refused it — "ring around it is not lid (0% at lid
    height)" — because the model said the lid stood 95 mm above the table (a
    phone-photo estimate) and this frame measures 121. The aim looks for the
    lid within 12 mm of that height, so it was searching 26 mm too low; the
    wrist search's height band with it. Measured here: top 121 mm, lid
    ~105 mm across, button 44.3 mm, flush."""
    from rammp_box_opening.models.container import ContainerModel

    m = ContainerModel.load("src/rammp_box_opening/config/containers/ankou_pink.yaml")
    z = np.load(Path(__file__).parent / "data" / "wrist_staging_pink_20260923.npz")
    s, why = find_button(z["color"], z["depth"], z["k"], z["rot_cam"], z["trans_cam"], TABLE_Z, m)
    assert why is None, why
    assert s.uv == pytest.approx((238.0, 390.0), abs=4.0)
    assert s.diameter_m == pytest.approx(0.044, abs=0.003)
    assert abs(s.above_lid_m) < 0.003  # closed: flush with its lid
    assert s.xyz[2] == pytest.approx(TABLE_Z + m.dims[2], abs=0.006)


# the live D405 colour stream's plumb_bob coefficients (camera_info, 2026-09-23)
D405_DIST = [-0.05412, 0.06031, 0.00018, 0.00016, -0.02035]


def test_the_press_point_is_corrected_for_lens_distortion():
    """The aim lifted the circle's RAW pixel through a pinhole K, but the
    colour stream is not rectified: near the image edge that is 1-2 mm of
    press error (1.4 mm at the pixel the 2026-09-23 miss was aimed from).
    The centre is undistorted before it becomes a ray; its depth is still
    read at the raw pixel, where the aligned depth lives."""
    import cv2

    from rammp_box_opening.models.container import ContainerModel

    m = ContainerModel.load("src/rammp_box_opening/config/containers/ankou_pink.yaml")
    z = np.load(Path(__file__).parent / "data" / "wrist_staging_pink_20260923.npz")
    raw, _ = find_button(z["color"], z["depth"], z["k"], z["rot_cam"], z["trans_cam"], TABLE_Z, m)
    s, why = find_button(z["color"], z["depth"], z["k"], z["rot_cam"], z["trans_cam"], TABLE_Z, m, dist=D405_DIST)
    assert why is None and s.uv == raw.uv  # the same circle, the same pixel reported
    k = np.asarray(z["k"], float)
    u, v = cv2.undistortPoints(np.array([[raw.uv]], np.float64), k, np.array(D405_DIST), P=k)[0, 0]
    pc = np.array([(u - k[0, 2]) / k[0, 0] * s.range_m, (v - k[1, 2]) / k[1, 1] * s.range_m, s.range_m])
    want = z["rot_cam"] @ pc + z["trans_cam"]
    assert s.xyz[:2] == pytest.approx(tuple(want[:2]), abs=1e-5)
    moved = np.hypot(s.xyz[0] - raw.xyz[0], s.xyz[1] - raw.xyz[1])
    assert 0.0005 < moved < 0.003  # a millimetre or two, at this pixel
