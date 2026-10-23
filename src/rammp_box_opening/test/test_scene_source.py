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



def test_the_scene_fix_is_the_first_owl_box_whose_top_is_this_container():
    """scene_fix_from_boxes: the loop the live locate ran inline, as one
    function an offline replay can call — the OWL's boxes best first, each
    lifted to the lid slab, the first that is this container's size wins."""
    import numpy as np

    from rammp_box_opening.models.container import ContainerModel
    from rammp_box_opening.perception.scene_source import scene_fix_from_boxes

    model = ContainerModel.load("src/rammp_box_opening/config/containers/ankou_pink.yaml")
    table_z = -0.027
    top = table_z + model.dims[2]
    # a lid of the model's size at [0.45, 0.05], seen as points with pixels
    xs, ys = np.meshgrid(np.linspace(-0.05, 0.05, 30), np.linspace(-0.05, 0.05, 30))
    lid = np.stack([0.45 + xs.ravel(), 0.05 + ys.ravel(), np.full(xs.size, top)], 1)
    table = np.stack([0.20 + xs.ravel(), -0.20 + ys.ravel(), np.full(xs.size, table_z)], 1)
    pts = np.vstack([lid, table])
    uv = np.vstack([np.stack([100 + 30 * (xs.ravel() + 0.05) / 0.1, 100 + 30 * (ys.ravel() + 0.05) / 0.1], 1),
                    np.stack([400 + 30 * (xs.ravel() + 0.05) / 0.1, 400 + 30 * (ys.ravel() + 0.05) / 0.1], 1)])
    boxes = [(0.60, [390, 390, 440, 440]), (0.40, [90, 90, 140, 140])]  # the best box is the bare table
    got = scene_fix_from_boxes(boxes, pts, uv, table_z, model)
    assert got.pose is not None and got.box == boxes[1][1]
    assert abs(got.top[0] - 0.45) < 0.005 and abs(got.top[1] - 0.05) < 0.005
    assert len(got.tried) == 1  # the table box, refused and said why
    none = scene_fix_from_boxes(boxes[:1], pts, uv, table_z, model)
    assert none.pose is None and none.tried


def _table_low_wall_high(rng, n=3000, h=1000):
    """A table filling the view's lower rows, a wall its upper ones."""
    table = np.c_[rng.uniform(0.3, 0.9, n), rng.uniform(-0.4, 0.4, n), TABLE_Z + rng.normal(0, 0.003, n)]
    wall = np.c_[1.2 + rng.normal(0, 0.003, n), rng.uniform(-0.4, 0.4, n), rng.uniform(0.0, 0.6, n)]
    uv = np.c_[rng.uniform(0, 1000, 2 * n), np.r_[rng.uniform(0.65 * h, h, n), rng.uniform(0, 0.35 * h, n)]]
    return np.vstack([table, wall]), uv


def test_the_table_is_read_off_the_top_rows_of_a_camera_hung_upside_down():
    """Flatbot's scene OAK hangs upside down: the table in front of it is in
    the image's TOP rows, and its bottom rows are the wall behind the box."""
    pts, uv = _table_low_wall_high(np.random.default_rng(1))
    assert table_from_scene_points(pts, uv, image_h=1000) == pytest.approx(TABLE_Z, abs=0.004)
    over = np.c_[uv[:, 0], 1000 - uv[:, 1]]  # the same view, the camera turned over
    assert table_from_scene_points(pts, over, image_h=1000, upside_down=True) == pytest.approx(TABLE_Z, abs=0.004)
    assert np.isnan(table_from_scene_points(pts, over, image_h=1000))  # the wall is not a table


def test_which_way_up_the_scene_camera_hangs_is_read_off_its_pose():
    from rammp_box_opening.perception.scene_source import image_upside_down

    def pose(x, y, z):  # the colour optical frame's axes in the base frame
        T = np.eye(4)
        T[:3, 0], T[:3, 1], T[:3, 2] = x, y, z
        return T

    assert image_upside_down(pose((0, 1, 0), (0, 0, 1), (1, 0, 0))) is True  # flatbot's OAK, live TF 2026-10-22
    assert image_upside_down(pose((0, -1, 0), (0, 0, -1), (1, 0, 0))) is False
    assert image_upside_down(pose((0, 1, 0), (1, 0, 0), (0, 0, -1))) is None  # straight down: no up to speak of


def test_a_robot_whose_tf_carries_the_scene_camera_is_placed_by_tf_not_the_calibration_file():
    """Flatbot 2026-10-22: the installed camera_scene.yaml calibrates the
    dojo bench's camera to base_link; flatbot's mount is on TF (sheppy's
    static_tf). A fix through the wrong one would be flown to."""
    from rammp_box_opening.perception.scene_source import SceneLocator

    T_tf, T_cal = np.eye(4), np.eye(4)
    T_tf[:3, 3], T_cal[:3, 3] = (0.2, -0.5, 0.4), (-0.07, 0.64, 0.35)
    asked = []

    class Grab:
        link_frame = "scene_camera_color_optical_frame"

        def lookup(self, parent, child):
            asked.append((parent, child))
            return T_tf

        def link_to_color(self):
            return np.eye(4)

    loc = SceneLocator.__new__(SceneLocator)
    loc.grab, loc.T_base_link, loc.base_frame = Grab(), None, "arm_mount"
    assert np.allclose(loc.base_to_color(), T_tf) and asked == [("arm_mount", "scene_camera_color_optical_frame")]
    asked.clear()
    loc.T_base_link = T_cal  # a calibration file in force: TF is not asked
    assert np.allclose(loc.base_to_color(), T_cal) and asked == []


# A camera level with the lid, as flatbot's scene OAK sits: 0.2 m behind the
# table's near edge, 0.1 m above the table, looking along +x.
SIDE_CAM = np.array([0.2, -0.5, 0.40])
SIDE_TABLE = 0.30
PINK = "src/rammp_box_opening/config/containers/ankou_pink.yaml"


def _project(pts, cam=SIDE_CAM, f=284.0, cx=320.0, cy=200.0):
    """Pixels of base-frame points in a level camera at `cam` looking along +x."""
    rel = pts - cam
    return np.c_[cx + f * (-rel[:, 1]) / rel[:, 0], cy + f * (-rel[:, 2]) / rel[:, 0]]


def _cylinder_front(centre, r, z0, z1, n, rng, cam=SIDE_CAM):
    """The half of a standing cylinder's side the camera sees."""
    face = np.arctan2(cam[1] - centre[1], cam[0] - centre[0])
    a = face + rng.uniform(-1.3, 1.3, n)
    return np.c_[centre[0] + r * np.cos(a), centre[1] + r * np.sin(a), rng.uniform(z0, z1, n)] + rng.normal(0, 0.002, (n, 3))


def _side_scene(rng, box=(0.70, -0.55), r=0.0525, height=0.121):
    """Table, the canister's visible side, the wall behind it, a taller thing
    beside it and the streaks a level camera sees under the table."""
    table = np.c_[rng.uniform(0.45, 1.2, 8000), rng.uniform(-0.95, -0.1, 8000), SIDE_TABLE + rng.normal(0, 0.003, 8000)]
    table = table[np.hypot(table[:, 0] - box[0], table[:, 1] - box[1]) > r + 0.01]
    canister = _cylinder_front(box, r, SIDE_TABLE, SIDE_TABLE + height, 1500, rng)
    tall = _cylinder_front((0.70, -0.85), 0.05, SIDE_TABLE, SIDE_TABLE + 0.26, 1500, rng)
    wall = np.c_[np.full(4000, 1.30), rng.uniform(-0.95, -0.1, 4000), rng.uniform(0.30, 0.90, 4000)]
    under = np.c_[rng.uniform(0.45, 0.9, 4000), rng.uniform(-0.95, -0.1, 4000), rng.uniform(0.0, 0.28, 4000)]
    pts = np.vstack([table, canister, tall, wall, under])
    return pts, _project(pts), canister, tall


def _pixel_box(uv_part, pad=(2, 6)):
    """The OWL box around some points' pixels, padded the way a lid's box
    takes in some wall above it."""
    lo, hi = uv_part.min(axis=0), uv_part.max(axis=0)
    return lo[0] - pad[0], lo[1] - pad[1], hi[0] + pad[0], hi[1] + pad[0]


def test_the_table_around_a_box_is_its_commonest_height_not_what_is_under_it():
    from rammp_box_opening.perception.scene_source import table_around

    pts, _uv, _c, _t = _side_scene(np.random.default_rng(3))
    assert table_around(pts, (0.70, -0.55), below_z=0.39) == pytest.approx(SIDE_TABLE, abs=0.004)
    # a TF 3 deg off in pitch (the table rises away from the camera), and the
    # table behind the box hidden by it: the visible ring is all in front, so
    # its median sits low; the plane is read AT the box
    tilted = pts[~((pts[:, 0] > 0.70) & (np.abs(pts[:, 2] - SIDE_TABLE) < 0.02))].copy()
    tilted[:, 2] += 0.05 * (tilted[:, 0] - 0.70)
    assert table_around(tilted, (0.70, -0.55), below_z=0.39) == pytest.approx(SIDE_TABLE, abs=0.004)


def test_a_level_camera_finds_the_canister_from_its_side():
    """Flatbot 2026-10-22: its scene OAK sits level with the lid and cannot
    move. The pixel box takes in the wall behind the lid; the side lift
    keeps the nearest object, finds the table around it, and puts the
    centre half the canister's width behind the face it sees."""
    from rammp_box_opening.perception.scene_source import box_from_side_points

    model = ContainerModel.load(PINK)
    pts, uv, canister, tall = _side_scene(np.random.default_rng(4))
    cap = canister[:, 2] > SIDE_TABLE + model.dims[2] - 0.025
    bbox = _pixel_box(_project(canister[cap]))
    pose, top, n, table_z = box_from_side_points(pts, uv, bbox, None, model, SIDE_CAM)
    assert pose is not None, top
    assert np.hypot(pose.xyz[0] - 0.70, pose.xyz[1] + 0.55) < 0.012
    assert table_z == pytest.approx(SIDE_TABLE, abs=0.004) and pose.xyz[2] == pytest.approx(table_z)
    # the thing beside it is too tall to be it, whatever the OWL says
    pose, why, _n, _tz = box_from_side_points(pts, uv, _pixel_box(_project(tall[tall[:, 2] > SIDE_TABLE + 0.23])), None, model, SIDE_CAM)
    assert pose is None and "from lid height" in why
    # and a canister twice as wide is another container
    pts2, uv2, wide, _t = _side_scene(np.random.default_rng(5), r=0.11)
    bbox = _pixel_box(_project(wide[wide[:, 2] > SIDE_TABLE + model.dims[2] - 0.025]))
    pose, why, _n, _tz = box_from_side_points(pts2, uv2, bbox, None, model, SIDE_CAM)
    assert pose is None and "another container" in why


def test_flatbot_scene_frame_finds_the_pink_canister_from_the_side():
    """The recorded flatbot frame (2026-10-22, camera level with the lid,
    the canister in a blue holder, a black speaker beside it): the top-face
    lift refuses the canister — the wall behind it falls in its box — and
    the side lift finds it, refusing the speaker on height."""
    from pathlib import Path

    from rammp_box_opening.perception.scene import color_cloud
    from rammp_box_opening.perception.scene_source import scene_fix_from_boxes

    z = np.load(Path(__file__).parent / "data" / "scene_side_flatbot_20261022.npz")
    T = z["T_base_color"]
    pc, uv = color_cloud(z["depth_mm"].astype(float) / 1000.0, z["kd"], z["k"], z["dist"], z["T_color_depth"], stride=2)
    pts = pc @ T[:3, :3].T + T[:3, 3]
    model = ContainerModel.load(PINK)
    boxes = [(float(b[4]), [float(v) for v in b[:4]]) for b in z["owl_boxes"]]
    lid = boxes[0][1]
    top_face, why, _n = box_from_scene_points(pts, uv, lid, 0.30, model)
    assert top_face is None and "another container" in why
    got = scene_fix_from_boxes(boxes, pts, uv, None, model, cam_xyz=T[:3, 3])
    assert got.box == lid and not got.tried
    # where this frame puts it under the hand-measured TF: a regression pin, not a tape truth
    assert got.pose.xyz[0] == pytest.approx(0.675, abs=0.01) and got.pose.xyz[1] == pytest.approx(-0.578, abs=0.01)
    assert got.table_z == pytest.approx(0.30, abs=0.01)
    speaker = scene_fix_from_boxes(boxes[2:], pts, uv, None, model, cam_xyz=T[:3, 3])
    assert speaker.pose is None and "from lid height" in speaker.tried[0]


def test_the_table_is_measured_where_the_camera_saw_it():
    from rammp_box_opening.perception.scene_source import table_extent

    pts, _uv, _c, _t = _side_scene(np.random.default_rng(6))
    ext = table_extent(pts, SIDE_TABLE, (0.70, -0.55))
    assert ext is not None
    x0, x1, y0, y1 = ext
    assert 0.44 <= x0 <= 0.50 and 1.10 <= x1 <= 1.21  # the table spans x 0.45-1.2 in the scene
    assert -0.96 <= y0 <= -0.90 and -0.20 <= y1 <= -0.10  # ... and only 0.6 m around the box is read
    assert table_extent(pts, SIDE_TABLE + 0.2, (0.70, -0.55)) is None  # nothing at a height with no table


def test_a_measured_table_height_always_comes_with_somewhere_to_plan_it():
    """Review 2026-10-22: a height with too few points to measure its
    extent reached the planner as no table at all. It falls back to the
    ring square the height was read from, and says it assumed it."""
    from rammp_box_opening.perception.scene_source import MIN_TABLE_POINTS, TABLE_RING_M, seen_table_extent, table_extent

    pts, _uv, _c, _t = _side_scene(np.random.default_rng(6))
    assert seen_table_extent(pts, SIDE_TABLE, (0.70, -0.55)) == (table_extent(pts, SIDE_TABLE, (0.70, -0.55)), False)
    rng = np.random.default_rng(7)
    a, d = rng.uniform(0, 2 * np.pi, MIN_TABLE_POINTS - 1), rng.uniform(*TABLE_RING_M, MIN_TABLE_POINTS - 1)
    sparse = np.c_[0.70 + d * np.cos(a), -0.55 + d * np.sin(a), np.full(len(a), SIDE_TABLE)]
    assert table_extent(sparse, SIDE_TABLE, (0.70, -0.55)) is None
    ext, assumed = seen_table_extent(sparse, SIDE_TABLE, (0.70, -0.55))
    assert assumed and ext == pytest.approx((0.40, 1.00, -0.85, -0.25))


def test_a_camera_that_sees_the_top_face_keeps_the_top_face_lift(model):
    """Which view holds depends on how high the table stands against the
    camera, so both are tried: the top-face lift first."""
    from rammp_box_opening.perception.scene_source import scene_fix_from_boxes

    pts, uv = _scene(model)
    boxes = [(0.5, list(_bbox_around(pts, uv, (0.45, -0.12))))]
    got = scene_fix_from_boxes(boxes, pts, uv, TABLE_Z, model, cam_xyz=np.array([0.0, 0.0, 0.6]))
    assert got.pose is not None and got.lift == "top"


def test_a_tray_ranked_first_does_not_beat_the_box_from_the_side(model):
    """Every candidate's top-face lift before any side lift: the side lift
    never measures how far a container runs along the view, so the 19x9 cm
    tray the top-face lift refuses would pass it ahead of the real box."""
    from rammp_box_opening.perception.scene_source import scene_fix_from_boxes

    rng = np.random.default_rng(5)
    pts, uv = _scene(model, rng=rng)
    lid_z = TABLE_Z + model.dims[2]
    tray = np.c_[rng.uniform(0.25, 0.44, 1200), rng.uniform(0.25, 0.34, 1200), lid_z + rng.normal(0, 0.004, 1200)]
    tray_uv = np.c_[(tray[:, 1] + 0.5) * 1000, (0.8 - tray[:, 0]) * 1000]
    pts = np.vstack([pts, tray]); uv = np.vstack([uv, tray_uv])
    tray_box = [tray_uv[:, 0].min() - 5, tray_uv[:, 1].min() - 5, tray_uv[:, 0].max() + 5, tray_uv[:, 1].max() + 5]
    true_box = list(_bbox_around(pts, uv, (0.45, -0.12)))
    got = scene_fix_from_boxes([(0.26, tray_box), (0.255, true_box)], pts, uv, TABLE_Z, model,
                               cam_xyz=np.array([0.0, 0.0, 0.6]))
    assert got.box == true_box and got.lift == "top", got.tried
    assert got.pose.xyz[0] == pytest.approx(0.45, abs=0.01) and got.pose.xyz[1] == pytest.approx(-0.12, abs=0.01)
    assert "another container" in got.tried[0]


def test_a_camera_level_with_the_lid_takes_the_side_lift():
    """The recorded flatbot frame of the 20:04 run (2026-10-22, camera z
    0.400, the pink lid's top ~0.397, a table height read off the scene):
    the top-face 'slab' is the lid's side band seen edge-on, and whether
    it passed decided the fix on a few pixels of the OWL box — 13 mm in xy,
    7.5 mm in lid height between runs of the same placement."""
    import itertools
    from pathlib import Path

    from rammp_box_opening.perception.scene import color_cloud
    from rammp_box_opening.perception.scene_source import scene_fix_from_boxes

    z = np.load(Path(__file__).parent / "data" / "scene_level_flatbot_20261022_2004.npz")
    T = z["T_base_color"]
    pc, uv = color_cloud(z["depth_mm"].astype(float) / 1000.0, z["kd"], z["k"], z["dist"], z["T_color_depth"], stride=2)
    pts = pc @ T[:3, :3].T + T[:3, 3]
    model = ContainerModel.load(PINK)
    table_z = table_from_scene_points(pts, uv, z["color"].shape[0], upside_down=True)  # what locate() does unsurveyed
    assert not np.isnan(table_z)
    x0, y0, x1, y1 = (float(v) for v in z["lid_box"])
    tops = []
    for dx, dy in itertools.product((0, 2, 4), (0, 3, 6)):  # an OWL box a few pixels looser
        got = scene_fix_from_boxes([(0.6, [x0 - dx, y0 - dy, x1 + dx, y1 + dy])], pts, uv, table_z, model,
                                   cam_xyz=T[:3, 3])
        assert got.pose is not None and got.lift == "side", (dx, dy, got.lift, got.tried)
        tops.append(got.top)
    tops = np.asarray(tops)
    assert np.all(tops.max(0) - tops.min(0) < 0.010)


def test_the_top_face_is_seen_only_from_above_the_lid():
    import itertools

    from rammp_box_opening.perception.scene_source import sees_top_face

    lid_z = 0.094
    xs, ys = np.meshgrid(np.linspace(-0.05, 0.05, 11), np.linspace(-0.05, 0.05, 11))

    def lid_at(x, y):
        pts = np.c_[x + xs.ravel(), y + ys.ravel(), np.full(xs.size, lid_z)]
        uv = np.c_[100 + 100 * xs.ravel(), 100 + 100 * ys.ravel()]
        return pts, uv

    bbox = (90, 90, 110, 110)
    pts, uv = lid_at(0.5, 0.0)
    assert not sees_top_face(pts, uv, bbox, lid_z, (0.0, 0.0, lid_z + 0.003))  # level with it, 0.5 m away
    bench_cam = (-0.06506, 0.64157, 0.35206)  # config/camera_scene.yaml
    for x, y in itertools.product((0.3, 0.75), (-0.45, 0.45)):
        pts, uv = lid_at(x, y)
        assert sees_top_face(pts, uv, bbox, lid_z, bench_cam), (x, y)
    assert sees_top_face(pts, uv, (500, 500, 510, 510), lid_z, (0.0, 0.0, lid_z))  # nothing in the box: its lift says why
