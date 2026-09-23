"""Finding the box from the scene camera: depth inside the OWL box, at lid
height, in base_link."""

import numpy as np
import pytest

from rammp_box_opening.models.container import ContainerModel
from rammp_box_opening.perception.scene_source import (
    LID_BAND_M,
    MIN_LID_POINTS,
    box_from_scene_points,
    table_from_scene_points,
)

CFG = "src/rammp_box_opening/config/containers/oxo_pop.yaml"
TABLE_Z = -0.027


@pytest.fixture(scope="module")
def model():
    return ContainerModel.load(CFG)


def _scene(model, box_xy=(0.45, -0.12), yaw=0.3, n_table=4000, n_top=400, rng=None):
    """Points in base_link with pixels: a table, a box top at lid height,
    a second container elsewhere, and speckle above the table. The pixel
    layout is a simple map of x -> u and y -> v so a bbox can be drawn."""
    rng = np.random.default_rng(0) if rng is None else rng
    lid_z = TABLE_Z + model.dims[2]
    hx, hy = model.dims[0] / 2, model.dims[1] / 2
    c, s = np.cos(yaw), np.sin(yaw)
    local = rng.uniform([-hx, -hy], [hx, hy], (n_top, 2))
    top = np.c_[
        box_xy[0] + c * local[:, 0] - s * local[:, 1],
        box_xy[1] + s * local[:, 0] + c * local[:, 1],
        lid_z + rng.normal(0, 0.004, n_top),
    ]
    # the near face, as an oblique scene camera sees it: a vertical wall of
    # points on the box's own -x side, running from the table up to the lid
    fl = np.c_[np.full(300, -hx) + rng.normal(0, 0.003, 300), rng.uniform(-hy, hy, 300)]
    face = np.c_[
        box_xy[0] + c * fl[:, 0] - s * fl[:, 1],
        box_xy[1] + s * fl[:, 0] + c * fl[:, 1],
        rng.uniform(TABLE_Z, lid_z, 300),
    ]
    table = np.c_[rng.uniform(0.2, 0.8, n_table), rng.uniform(-0.5, 0.5, n_table), TABLE_Z + rng.normal(0, 0.006, n_table)]
    other = np.c_[rng.uniform(0.55, 0.65, 200), rng.uniform(0.2, 0.3, 200), lid_z + rng.normal(0, 0.004, 200)]
    speckle = np.c_[rng.uniform(0.2, 0.8, 300), rng.uniform(-0.5, 0.5, 300), rng.uniform(-0.05, 0.3, 300)]
    pts = np.vstack([table, top, face, other, speckle])
    uv = np.c_[(pts[:, 1] + 0.5) * 1000, (0.8 - pts[:, 0]) * 1000]  # y -> u, x -> v
    return pts, uv


def _bbox_around(pts, uv, box_xy, half=0.08):  # the 4.1-inch box is 0.052 to a side
    sel = (np.abs(pts[:, 0] - box_xy[0]) < half) & (np.abs(pts[:, 1] - box_xy[1]) < half)
    return uv[sel, 0].min(), uv[sel, 1].min(), uv[sel, 0].max(), uv[sel, 1].max()


def test_the_box_is_where_the_lid_points_inside_the_pixel_box_are(model):
    pts, uv = _scene(model)
    bbox = _bbox_around(pts, uv, (0.45, -0.12))
    pose, top, n = box_from_scene_points(pts, uv, bbox, TABLE_Z, model)
    assert pose is not None
    assert pose.xyz[0] == pytest.approx(0.45, abs=0.006)
    assert pose.xyz[1] == pytest.approx(-0.12, abs=0.006)
    assert pose.xyz[2] == pytest.approx(TABLE_Z)  # origin pinned to the table
    assert top[2] == pytest.approx(TABLE_Z + model.dims[2], abs=0.005)
    assert n >= MIN_LID_POINTS
    d = abs((pose.yaw - 0.3 + np.pi / 4) % (np.pi / 2) - np.pi / 4)
    assert d < np.radians(6)  # yaw mod 90 from the rect


def test_the_near_face_does_not_pull_the_box_toward_the_camera(model):
    """The face's upper part lies inside the lid band; only the top slab
    may vote, or the centre walks toward the camera."""
    pts, uv = _scene(model)
    bbox = _bbox_around(pts, uv, (0.45, -0.12))
    pose, top, _n = box_from_scene_points(pts, uv, bbox, TABLE_Z, model)
    assert pose.xyz[0] == pytest.approx(0.45, abs=0.006)  # not pulled to -x
    assert top[2] == pytest.approx(TABLE_Z + model.dims[2], abs=0.005)


def test_the_other_container_does_not_pull_the_box(model):
    """A second container in the scene is outside the OWL box, so it is not
    counted, however box-like it is."""
    pts, uv = _scene(model)
    bbox = _bbox_around(pts, uv, (0.45, -0.12))
    pose, _top, _n = box_from_scene_points(pts, uv, bbox, TABLE_Z, model)
    assert abs(pose.xyz[1] + 0.12) < 0.01


def test_a_pixel_box_with_nothing_at_lid_height_is_refused(model):
    pts, uv = _scene(model)
    bbox = _bbox_around(pts, uv, (0.3, 0.3), half=0.03)  # bare table
    pose, why, n = box_from_scene_points(pts, uv, bbox, TABLE_Z, model)
    assert pose is None and "lid height" in why and n < MIN_LID_POINTS


def test_a_taller_things_face_crossing_lid_height_is_not_a_box(model):
    """With the real top out of the band, only the near face's strip is left
    at lid height — a strip, and refused as one."""
    rng = np.random.default_rng(3)
    pts, uv = _scene(model, rng=rng)
    pts = pts.copy()
    top = np.abs(pts[:, 2] - (TABLE_Z + model.dims[2])) < 0.02
    pts[top, 2] += 0.08  # a taller container: its top is well above the band
    bbox = _bbox_around(pts, uv, (0.45, -0.12))
    pose, why, _n = box_from_scene_points(pts, uv, bbox, TABLE_Z, model)
    assert pose is None and "strip" in why


def test_the_lid_band_is_the_detectors_own(model):
    """A box whose top sits further from nominal than the band is not this
    box (or the table height is wrong) — refused, as the wrist path does."""
    rng = np.random.default_rng(3)
    pts, uv = _scene(model, rng=rng)
    pts = pts.copy()
    top = np.abs(pts[:, 2] - (TABLE_Z + model.dims[2])) < 0.02
    pts[top, 2] += LID_BAND_M + 0.02
    bbox = _bbox_around(pts, uv, (0.45, -0.12))
    pose, _why, _n = box_from_scene_points(pts, uv, bbox, TABLE_Z, model)
    assert pose is None


def test_the_table_height_is_read_off_the_scene(model):
    pts, uv = _scene(model)
    z = table_from_scene_points(pts, uv, image_h=1000, row_frac=0.0)
    assert z == pytest.approx(TABLE_Z, abs=0.004)


def test_another_larger_container_in_the_pixel_box_is_not_this_one(model):
    """A 19x9 cm tray at the same height as the OXO (2026-09-17: the OWL's
    best box flipped between the two): its top is the wrong size."""
    rng = np.random.default_rng(5)
    pts, uv = _scene(model, rng=rng)
    lid_z = TABLE_Z + model.dims[2]
    tray = np.c_[rng.uniform(0.25, 0.44, 1200), rng.uniform(0.25, 0.34, 1200), lid_z + rng.normal(0, 0.004, 1200)]
    tray_uv = np.c_[(tray[:, 1] + 0.5) * 1000, (0.8 - tray[:, 0]) * 1000]
    pts = np.vstack([pts, tray]); uv = np.vstack([uv, tray_uv])
    bbox = (tray_uv[:, 0].min() - 5, tray_uv[:, 1].min() - 5, tray_uv[:, 0].max() + 5, tray_uv[:, 1].max() + 5)
    pose, why, _n = box_from_scene_points(pts, uv, bbox, TABLE_Z, model)
    assert pose is None and "another container" in why
    # the real box in its own pixel box still passes
    pose, _top, _n = box_from_scene_points(pts, uv, _bbox_around(pts, uv, (0.45, -0.12)), TABLE_Z, model)
    assert pose is not None

