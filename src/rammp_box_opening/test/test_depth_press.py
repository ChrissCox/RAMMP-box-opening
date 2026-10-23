"""The depth press (tasks/depth_press): once the hand is over the box, the
height from the wrist camera's own depth, the press from the arm's torques."""

from types import SimpleNamespace

import numpy as np
import pytest

from rammp_box_opening.models.container import ContainerModel, load_press_demo
from rammp_box_opening.perception.d405 import quat_to_mat
from rammp_box_opening.tasks import depth_press as dp
from rammp_box_opening.tasks import reach_press as rp

PINK = "src/rammp_box_opening/config/containers/ankou_pink.yaml"
BUTTON = (0.495, -0.16, 0.24)


@pytest.fixture(scope="module")
def pink():
    return ContainerModel.load(PINK)


# ---- what the wrist camera measures, in its own frame -------------------------

def test_the_button_is_found_in_the_lens_frame_with_no_transform_at_all(pink):
    from test_button_aim import BOX_H, BUTTON_XY, OAK_K, ROT_DOWN, _h0, render

    h0 = _h0(pink)
    b = np.r_[BUTTON_XY, h0]
    for off in ((0.0, 0.0), (0.03, -0.02), (-0.04, 0.025)):
        cam = b + np.r_[off[0], off[1], 0.26]
        color, depth = render(cam, ROT_DOWN, [(BUTTON_XY, h0, 0.0)], h0 - BOX_H)
        s, why = dp.find_in_camera(color, depth, OAK_K, pink)
        assert why is None, why
        want = ROT_DOWN.T @ (b - cam)  # where the button is in the camera's frame
        assert (s.x, s.y) == pytest.approx((want[0], want[1]), abs=0.002)
        assert s.lid_depth == pytest.approx(0.26, abs=0.003) and abs(s.up_mm) < 2.0
    cam = b + np.r_[0.0, 0.0, 0.26]
    color, depth = render(cam, ROT_DOWN, [(BUTTON_XY, h0, 0.012)], h0 - BOX_H)
    s, why = dp.find_in_camera(color, depth, OAK_K, pink)
    assert why is None, why
    assert s.up_mm == pytest.approx(12.0, abs=3.0) and s.button_depth < s.lid_depth  # the knob up


def test_a_look_that_knows_where_the_button_must_be_takes_no_circle_anywhere_else(pink):
    """Flatbot 2026-10-23 17:30: after a 21 mm centring move the look took a
    circle on the cabinet 288 mm away and 99 mm deeper, in the frames Hough
    missed the button in. Told where the button must be, the finder takes
    nothing else — and still finds the button there."""
    from test_button_aim import BOX_H, BUTTON_XY, OAK_K, ROT_DOWN, _h0, render

    h0 = _h0(pink)
    b = np.r_[BUTTON_XY, h0]
    cam = b + np.r_[0.0, 0.0, 0.26]
    want = ROT_DOWN.T @ (b - cam)
    here = (want[0], want[1], 0.26)
    decoy = (BUTTON_XY[0] + 0.06, BUTTON_XY[1] - 0.12)  # another round lid's button, 10 cm lower, 13 cm off
    # only the decoy in view, as in the frames Hough missed the button in
    color, depth = render(cam, ROT_DOWN, [(decoy, h0 - 0.10, 0.0)], h0 - BOX_H)
    s, why = dp.find_in_camera(color, depth, OAK_K, pink)
    assert why is None and s.lid_depth == pytest.approx(0.36, abs=0.005)  # what the look used to take
    s, why = dp.find_in_camera(color, depth, OAK_K, pink, expect=here)
    assert s is None and why
    # both in view: the button, where it must be
    color, depth = render(cam, ROT_DOWN, [(BUTTON_XY, h0, 0.0), (decoy, h0 - 0.10, 0.0)], h0 - BOX_H)
    s, why = dp.find_in_camera(color, depth, OAK_K, pink, expect=(want[0] + 0.01, want[1] - 0.01, 0.265))
    assert why is None and (s.x, s.y) == pytest.approx((want[0], want[1]), abs=0.002)
    # ... and nothing on a lid at another depth, even right where the button should be
    s, why = dp.find_in_camera(color, depth, OAK_K, pink, expect=(want[0], want[1], 0.26 + 2 * dp.LOOK_EXPECT_DEPTH_M))
    assert s is None


def test_the_depth_under_the_fingers_is_read_where_the_fingers_line_meets_the_surface(pink):
    from test_button_aim import BOX_H, BUTTON_XY, OAK_K, ROT_DOWN, _h0, render

    # ROT_DOWN: the camera's +x is base -y, its +y base -x. A lens at
    # (lx, ly) from the fingers' line, in the camera's axes, stands at
    # fingers - ly * base_x... — here the fingers' line runs through the button
    h0 = _h0(pink)
    for lens_xy in ((0.0, 0.10), (0.0, 0.065), (0.02, 0.06)):
        for height in (0.24, 0.30, 0.40):
            off = ROT_DOWN @ np.r_[lens_xy, 0.0]  # the lens from the fingers' line, in base
            cam = np.r_[BUTTON_XY[0] + off[0], BUTTON_XY[1] + off[1], h0 + height]
            color, depth = render(cam, ROT_DOWN, [(BUTTON_XY, h0, 0.0)], h0 - BOX_H)
            assert dp.depth_under_fingers(depth, OAK_K, lens_xy) == pytest.approx(height, abs=0.004)
            # and the button itself is in view where the fingers' line meets it
            s, why = dp.find_in_camera(color, depth, OAK_K, pink)
            assert why is None, why
            assert dp.under_fingers(s, lens_xy) == pytest.approx((0.0, 0.0), abs=0.003)
            assert s.lid_depth == pytest.approx(height, abs=0.004)
    # nearer than the frame's edge can see on that line: no reading
    flat = np.full((400, 640), 0.15)
    assert dp.depth_under_fingers(flat, OAK_K, (0.0, 0.10)) is None
    # on the lens's own axis it is the depth straight ahead
    assert dp.depth_under_fingers(np.full((400, 640), 0.33), OAK_K, (0.0, 0.0)) == pytest.approx(0.33, abs=0.003)


def test_a_camera_turned_in_the_hand_reads_under_the_fingers_through_its_own_axes(pink):
    """The bench D405 sits 180 deg about its lens axis: the sighting, in the
    camera's axes, is turned into the hand's before the lens offset is added,
    and the fingers' line is projected through the camera's axes."""
    from test_button_aim import BOX_H, BUTTON_XY, OAK_K, ROT_DOWN, _h0, render

    h0 = _h0(pink)
    for lens_xy in ((0.0, 0.075), (0.02, 0.06)):
        for height in (0.24, 0.30):
            off = ROT_DOWN @ np.r_[lens_xy, 0.0]  # the hand's axes are ROT_DOWN's, the camera turned in it
            cam = np.r_[BUTTON_XY[0] + off[0], BUTTON_XY[1] + off[1], h0 + height]
            color, depth = render(cam, ROT_DOWN @ RZ180, [(BUTTON_XY, h0, 0.0)], h0 - BOX_H)
            assert dp.depth_under_fingers(depth, OAK_K, lens_xy, RZ180) == pytest.approx(height, abs=0.004)
            s, why = dp.find_in_camera(color, depth, OAK_K, pink)
            assert why is None, why
            assert dp.under_fingers(s, lens_xy, RZ180) == pytest.approx((0.0, 0.0), abs=0.003)
            # the camera's axes dropped: twice the lens offset, the wrong way
            assert np.hypot(*dp.under_fingers(s, lens_xy)) == pytest.approx(2 * np.hypot(*lens_xy), abs=0.004)


def test_the_lens_in_the_hand_keeps_the_cameras_axes(ctx, monkeypatch):
    from rammp_box_opening.perception import depth_source
    from rammp_box_opening.runtime import approach

    arm = _Arm()
    monkeypatch.setattr(approach, "_chain", lambda: arm)
    quat = rp.reach_quat(ContainerModel.load(PINK), BUTTON[:2], 0)
    ctx.client.live = [0.5, -0.1, 0.6] + list(quat)
    R, t = arm.fk(ctx.client.live)
    monkeypatch.setattr(depth_source, "camera_pose_at",
                        lambda g: (R @ RZ180, t + R @ np.r_[0.0, 0.075, -0.125]))
    lens, r_hc = dp.lens_in_hand(ctx, None)
    assert lens == pytest.approx([0.0, 0.075, -0.125]) and r_hc == pytest.approx(RZ180)


def test_the_lid_is_the_nearest_flat_surface():
    depth = np.full((400, 640), 0.42)  # the table
    depth[150:250, 240:400] = 0.31  # the lid, nearer
    depth[::7, ::5] = 0.0  # holes
    g = dp.lid_depth_guesses(depth)
    assert g[0] == pytest.approx(0.31, abs=0.003) and g[1] == pytest.approx(0.42, abs=0.003)
    assert dp.lid_depth_guesses(np.zeros((400, 640))) == []


# ---- the press, the arm faked ---------------------------------------------------

class _Arm:
    """A chain whose joints ARE the tool pose — position then quaternion — so
    every line and turn flies and fk follows the arm."""

    def __init__(self, refuse_turn=False, refuse_first=0):
        self.lines, self.turns, self.refuse_turn = [], [], refuse_turn
        self.refuse_first = refuse_first  # the first this many turn attempts (exactly down, 2 deg short ...) refused
        self.trace = []  # legs, looks and at-rest reads, in order (_press)

    def fk(self, q):
        return np.asarray(quat_to_mat(*q[3:7]), float), np.asarray(q[:3], float)

    def straight_line(self, q0, xyz, quat):
        self.lines.append(list(xyz))
        return [list(q0), [float(v) for v in xyz] + [float(v) for v in quat]], None

    def walk(self, q0, xyz, quat, step_m=None, step_rad=None, max_step_rad=None):
        if self.refuse_turn:
            return None, "a joint limit 40% of the way"
        if self.refuse_first > 0:
            self.refuse_first -= 1
            return None, "a 0.26 rad jump in the last step"
        self.turns.append((list(xyz), list(quat)))
        return [list(q0), [float(v) for v in xyz] + [float(v) for v in quat]], None


class _Gravity:
    def wrist_holding(self, q):
        return [0.0] * 4

    def tip_load(self, q, p):
        return [0.48, 0.0, 0.2, 0.0]


LENS = (0.0, 0.065, -0.105)  # the lens from the tool frame, in the hand: 65 mm beside the fingers, 105 mm behind
EYE = np.eye(3)
RZ180 = np.diag([-1.0, -1.0, 1.0])  # the bench D405: 180 deg about its lens axis
# where TF's lens puts the fingertips in front of it with LENS (116 mm): the lid
# 270 mm below the lens is met 154 mm down
TOUCH_AT = 0.27 - (dp.DOWN_CONTACT_TOOL[2] - LENS[2])


def _sighting(x=0.0, y=-LENS[1], lid=0.27, up=0.4):
    """The button seen at (x, y) from the lens axis: by default under the fingers."""
    return dp.LensSighting(x=x, y=y, lid_depth=lid, button_depth=lid - up / 1000.0, up_mm=up, raw=None)


def _seen(hand_off=(0.0, 0.0), depth=0.27, lens=LENS, r_hc=EYE, up=0.4):
    """A look that sees one fixed button wherever the fake arm stands: its
    lid `hand_off` from the fingers' line, in the hand's axes, `depth` below
    the lens along the tool, from the hand at the first look. The sighting is in
    the camera's own axes (r_hc: the camera's axes in the hand's)."""
    world = {}

    def at(ctx, up=up):
        q = ctx.client.joints()
        R, t = np.asarray(quat_to_mat(*q[3:7]), float), np.asarray(q[:3], float)
        if "b" not in world:
            world["b"] = t + R @ np.r_[hand_off[0], hand_off[1], lens[2] + depth]
        c = np.asarray(r_hc).T @ (R.T @ (world["b"] - t) - np.asarray(lens))
        s = dp.LensSighting(x=float(c[0]), y=float(c[1]), lid_depth=float(c[2]),
                            button_depth=float(c[2]) - up / 1000.0, up_mm=up, raw=None)
        return s, None, "2/2"

    at.world = world
    at.after = lambda ctx: at(ctx, up=12.0)  # the same button, the knob up
    return at


def _press(ctx, monkeypatch, tmp_path, looks, tilt=30, touch_at=TOUCH_AT, push="touch", refuse_turn=False,
           lens=LENS, r_hc=EYE, measured=LENS, refuse_first=0, here=None, turned=None, knob_up_mm=None):
    """press_down_from_here with the hand `tilt` deg over BUTTON (at the
    reach-out's stroke start, 6 cm above TF's lid at 0.40), the wrist camera
    scripted by `looks` (a (sighting, z, status) or a _seen) with TF's lens
    at `lens`, turned `r_hc`, against the `measured` one in the camera
    config, and the fingertips meeting the lid `touch_at` down the stroke.
    Every at-rest read is a distinct reading, ("rest", n, where) in
    arm.trace. Returns (DepthPress, events, arm, legs, start tool pose)."""
    from rammp_box_opening.runtime import approach
    from rammp_box_opening.runtime.runner import Runner

    cfg = load_press_demo(PINK)
    ctx.model = ContainerModel.load(PINK)
    ctx.press_cfg = cfg
    arm = _Arm(refuse_turn, refuse_first)
    monkeypatch.setattr(approach, "_chain", lambda: arm)
    monkeypatch.setattr(approach, "_gravity", lambda: _Gravity())
    quat = rp.reach_quat(ctx.model, BUTTON[:2], tilt)
    start = rp.tool_for_contact((BUTTON[0], BUTTON[1], 0.46), quat)  # TF's stroke start: 6 cm over its lid at 0.40
    ctx.client.live = list(start) + list(quat)
    events, legs = [], {}
    seq = list(looks)
    arm.reads = []

    def settled(*a, **k):
        got = ([0.001 * (len(arm.reads) + 1), 0.0, 0.0, 0.0], list(ctx.client.live))
        arm.reads.append(got)
        return got

    ctx.client.settled_efforts = settled
    real_settled = rp._settled
    monkeypatch.setattr(rp, "_settled", lambda ctx_, where: (arm.trace.append("rest:" + where), real_settled(ctx_, where))[1])

    arm.expects = []  # each look's `expect`, in order

    def look(node, watcher, model, tag, runner=None, ctx=None, settled=False, expect=None):
        events.append("look:" + tag)
        arm.trace.append("look:%s%s" % (tag, " (settled)" if settled else ""))
        arm.expects.append(expect)
        item = seq.pop(0)
        s, z, status = item(ctx) if callable(item) else item[:3]
        return s, z, (list(lens), np.asarray(r_hc, float)), status

    monkeypatch.setattr(dp, "look", look)

    class R(Runner):
        def run(self, legs_, execute, lookahead=None):
            for lg in legs_:
                events.append(lg.name)
                arm.trace.append(lg.name)
                legs[lg.name] = lg
            name = legs_[-1].name
            top = list(ctx.client.live)
            # the stroke flies fast, then slowly (depth:approach, depth:touch):
            # `touch_at` is down from where the stroke began, and a lid higher
            # than the fast stretch reaches is met on it
            if name == "depth:approach":
                self.stroke_top = top
            stroke_top = getattr(self, "stroke_top", None) or top
            meets = touch_at is not None and (
                name == "depth:touch"
                or (name == "depth:approach" and stroke_top[2] - legs_[-1].traj.points[-1].positions[2] >= touch_at))
            if meets:
                ctx.client.exec_script = [("touch", {"message": "torque guard trip", "progress": 0.7, "torque_peak": 3.2})]
            if name == "press:push":
                ctx.client.exec_script = [(push, {"message": "x", "progress": 0.6 if push == "touch" else 1.0, "torque_peak": 7.4})]
            out = super().run(legs_, execute, lookahead)
            if meets:
                ctx.client.live = [stroke_top[0], stroke_top[1], stroke_top[2] - touch_at] + stroke_top[3:]  # stopped on the lid
            if name == "depth:touch" or meets:
                self.stroke_top = None
            return out

    runner = R(ctx.client, log_dir=tmp_path)
    contact = rp.CONTACT_TOOL if tilt else dp.DOWN_CONTACT_TOOL
    watcher = SimpleNamespace(grab=SimpleNamespace(wrist_lens_in_hand_m=None if measured is None else list(measured)))
    result = dp.press_down_from_here(None, ctx, cfg, runner, watcher, contact, here=here, turned=turned,
                                     knob_up_mm=knob_up_mm)
    return result, events, arm, legs, start


def test_over_the_box_it_turns_down_measures_the_depth_and_presses_by_feel(ctx, monkeypatch, tmp_path):
    looks = [(_sighting(), 0.27, "2/2"), (_sighting(up=12.0), 0.255, "2/2")]
    result, events, arm, legs, start = _press(ctx, monkeypatch, tmp_path, looks)
    assert dp.exit_code(result) == 0 and result.free and result.felt and result.popped
    assert events == [
        "depth:turn", "look:depth-look", "depth:approach", "depth:touch", "press:push", "depth:unload", "depth:rise",
        "look:depth-after",
    ]
    # TURN: straight down, about the fingertip — it stays where it was over the button
    (goal, quat), = arm.turns
    R0 = rp._rot(rp.reach_quat(ctx.model, BUTTON[:2], 30))
    tip = np.asarray(start) + R0 @ np.asarray(rp.CONTACT_TOOL)
    Rd = rp._rot(quat)
    assert Rd[2, 2] == pytest.approx(-1.0)
    assert np.asarray(goal) + Rd @ np.asarray(dp.DOWN_CONTACT_TOOL) == pytest.approx(tip)
    # TOUCH: straight down, fast to TOUCH_SLOW_M short of where the depth and
    # TF's mount put the lid (2026-10-23: all of it at press_speed met the lid
    # at 12.4 Nm), then slowly, no farther than the depth plus travel_m
    cfg = load_press_demo(PINK)
    fast, stroke = arm.lines[0], arm.lines[1]
    ahead_tf = dp.DOWN_CONTACT_TOOL[2] - LENS[2]
    assert fast[:2] == pytest.approx(list(goal[:2])) and goal[2] - fast[2] == pytest.approx(0.27 - ahead_tf - dp.TOUCH_SLOW_M)
    assert stroke[:2] == pytest.approx(list(goal[:2])) and goal[2] - stroke[2] == pytest.approx(0.27 + cfg.travel_m)
    g = legs["depth:touch"].guard
    assert g.trip == "touch" and g.gravity is not None and g.reference is not None
    assert legs["depth:approach"].guard.trip == "touch" and legs["depth:approach"].guard.reference is g.reference
    # PUSH: button_travel_m from where the torques felt the lid, backstopped
    assert arm.lines[2][2] == pytest.approx(goal[2] - TOUCH_AT - cfg.button_travel_m) and legs["press:push"].guard.trip == "press"
    # ... its backstop the TOTAL off the free-air reading over the box (touch_nm), not touch_nm - contact_nm on
    # top of a first sample the touch's overshoot already loaded; the recoil carries on back up the touch
    gp = legs["press:push"].guard
    assert gp.touch_nm == pytest.approx(ctx.model.touch_nm) and gp.gravity is not None
    assert gp.reference is g.reference and gp.reference is not None
    assert legs["press:push"].recoil_via is legs["depth:touch"].traj
    # RISE: back to where the touch began, judged for a pull at the fingertip
    assert arm.lines[-1][2] == pytest.approx(goal[2]) and legs["depth:rise"].guard.hang is not None


def test_only_the_lines_up_or_in_place_fly_faster_and_every_guard_is_as_before(ctx, monkeypatch, tmp_path):
    """Run 20:04 flew the turn (3.3 mm of tool travel) in 7.43 s and the rise
    in 5.56 s at CONTACT_SPEED. The turn pivots about the fingertip and the
    rise starts after the at-rest check: FREE_LINE_SPEED. Off the button and
    the push, as before; the touch flies at press_speed to TOUCH_SLOW_M short
    of the lid and at TOUCH_SPEED onto it (2026-10-23: 12.4 Nm at 0.35)."""
    from rammp_box_opening.constants import CONTACT_SPEED, FREE_LINE_SPEED

    cfg = load_press_demo(PINK)
    looks = [(_sighting(x=0.02), 0.27, "2/2"), (_sighting(), 0.27, "2/2"), (_sighting(up=12.0), None, "2/2")]
    result, events, arm, legs, _s = _press(ctx, monkeypatch, tmp_path, looks)
    assert dp.exit_code(result) == 0 and "depth:centre" in events
    speeds = {name: lg.speed for name, lg in legs.items()}
    assert speeds == {
        "depth:turn": FREE_LINE_SPEED, "depth:centre": FREE_LINE_SPEED, "depth:approach": cfg.press_speed,
        "depth:touch": dp.TOUCH_SPEED, "press:push": cfg.press_speed, "depth:unload": CONTACT_SPEED,
        "depth:rise": FREE_LINE_SPEED,
    }
    assert dp.CENTRE_SPEED == FREE_LINE_SPEED and CONTACT_SPEED < FREE_LINE_SPEED < cfg.press_speed
    over_box, down = arm.reads[0], arm.reads[1]  # 'over the box', then 'pointing down' after the turn
    turn, rise, unload = legs["depth:turn"].guard, legs["depth:rise"].guard, legs["depth:unload"].guard
    assert (turn.touch_nm, turn.trip, turn.reference, turn.hang) == (rp.DRAG_NM, "obstruction", over_box, None)
    assert turn.gravity is not None
    for g in (rise, unload):
        assert (g.touch_nm, g.trip, g.reference) == (ctx.model.touch_nm, "obstruction", down)
        assert g.hang[1] == rp.HANG_N and g.gravity is not None
    assert legs["depth:touch"].guard.touch_nm == cfg.contact_nm and legs["depth:touch"].guard.reference == down


def test_one_at_rest_read_where_the_arm_stands_and_a_fresh_one_after_a_turn(ctx, monkeypatch, tmp_path):
    """Each at-rest read holds the arm 0.45 s. The turn moved: its own read
    pointing down is the touch's reference."""
    looks = [(_sighting(), 0.27, "2/2"), (_sighting(up=12.0), None, "2/2")]
    _r, _e, arm, legs, _s = _press(ctx, monkeypatch, tmp_path, looks)
    assert arm.trace[:4] == ["rest:over the box", "depth:turn", "rest:over the button, pointing down",
                             "look:depth-look (settled)"]
    assert legs["depth:touch"].guard.reference == arm.reads[1]


def test_at_staging_one_at_rest_read_before_the_look(ctx, monkeypatch, tmp_path):
    looks = [(_sighting(), 0.27, "2/2"), (_sighting(up=12.0), None, "2/2")]
    result, _e, arm, legs, _s = _press(ctx, monkeypatch, tmp_path, looks, tilt=0)
    assert dp.exit_code(result) == 0
    assert arm.trace[:2] == ["rest:over the box", "look:depth-look (settled)"]  # no second read: the turn did not move
    assert legs["depth:touch"].guard.reference == arm.reads[0]


def test_the_callers_read_is_the_reference_when_nothing_moved_since(ctx, monkeypatch, tmp_path):
    """The reach-out read the stroke's start a moment before: the same
    reading at the same joints, not read again."""
    here = ([0.5, 0.0, 0.0, 0.0], None)
    looks = [(_sighting(), 0.27, "2/2"), (_sighting(up=12.0), None, "2/2")]
    _r, _e, arm, legs, _s = _press(ctx, monkeypatch, tmp_path, looks, here=here)
    assert arm.trace[:2] == ["depth:turn", "rest:over the button, pointing down"]
    assert legs["depth:turn"].guard.reference == here


def test_a_hand_that_slid_in_turning_is_not_turned_again(ctx, monkeypatch, tmp_path):
    """The reach-out's walk in landed 2 deg short of straight down: turning
    to exactly down from there is the wrist's singular case. No turn, and
    the caller's read is the reference. CENTRE takes the lean out instead:
    the button on the tilted fingers' line is off the fingertip's vertical."""
    here = ([0.5, 0.0, 0.0, 0.0], None)
    short = dp.down_quat(ContainerModel.load(PINK), BUTTON[:2], 2.0)
    seen = _seen((0.0, 0.0))
    looks = [seen, seen, seen.after]
    result, events, arm, legs, _s = _press(ctx, monkeypatch, tmp_path, looks, tilt=2, here=here, turned=short)
    assert dp.exit_code(result) == 0 and arm.turns == [] and "depth:turn" not in events and "depth:centre" in events
    assert arm.trace[0] == "look:depth-look (settled)" and not any(t.startswith("rest:over") for t in arm.trace)
    assert legs["depth:touch"].guard.reference == here and legs["depth:touch"].target[2] == pytest.approx(short)


def test_the_push_backstop_is_a_ceiling_off_free_air_not_on_top_of_the_touch_overshoot(ctx, monkeypatch, tmp_path):
    """Run 2026-10-22 20:04 replayed (joint_4, the trip reports): over the box
    it read 8.211 Nm in free air; the 0.35 touch overshot to 6.1 off that, so
    the push's first sample (2.738) was already 5.5 Nm in. The raw backstop
    (touch_nm - contact_nm = 7 Nm over THAT) rode to -4.972: 13.2 Nm off free
    air against touch_nm 10, on the lid's edge. Judged off the free-air
    reading the push trips at the first sample past touch_nm."""
    from dataclasses import replace

    from rammp_box_opening.runtime.guards import CompensatedGuard, TorqueGuard

    looks = [(_sighting(), 0.27, "2/2"), (_sighting(up=12.0), 0.255, "2/2")]
    _r, _e, _a, legs, _s = _press(ctx, monkeypatch, tmp_path, looks)
    cfg = load_press_demo(PINK)
    free_air = [8.211, -0.419, 0.926, -0.254]
    push = [[2.738, -0.857, 0.754, -0.41], [-2.857, -1.583, 2.064, -0.421], [-3.72, -1.562, 2.129, -0.439],
            [-4.204, -1.577, 2.297, -0.415], [-4.972, -1.727, 2.288, -0.428]]
    q = [0.0] * 7

    def first_trip(guard):
        guard.on_progress(0.01)
        for i, eff in enumerate(push):
            if guard.on_efforts(eff, q) if isinstance(guard, CompensatedGuard) else guard.on_efforts(eff):
                return i, free_air[0] - eff[0]
        return None, None

    spec = replace(legs["press:push"].guard, reference=(free_air, q))
    now = CompensatedGuard(spec.touch_nm, spec.gravity, reference=spec.reference)
    was = TorqueGuard(ctx.model.touch_nm - cfg.contact_nm)
    i_now, total_now = first_trip(now)
    i_was, total_was = first_trip(was)
    assert (i_was, round(total_was, 1)) == (4, 13.2)  # what the run did
    assert i_now == 1 and total_now < total_was  # the first sample the stream had past 10 Nm (11.1)
    assert all(free_air[0] - e[0] <= ctx.model.touch_nm for e in push[:i_now])


def test_the_touch_measures_how_far_in_front_of_the_lens_the_fingertips_are(ctx, monkeypatch, tmp_path):
    import json

    looks = [(_sighting(lid=0.27), 0.27, "2/2"), (_sighting(up=12.0), None, "2/2")]
    result, _e, _a, _l, _s = _press(ctx, monkeypatch, tmp_path, looks, touch_at=0.155)
    note = [json.loads(line) for line in open(next(tmp_path.glob("run-*.jsonl"))) if '"depth_touch"' in line][0]
    assert note["down_mm"] == pytest.approx(155.0) and note["fingertips_ahead_of_lens_mm"] == pytest.approx(115.0)
    assert result.popped


def test_at_staging_the_hand_already_points_down_and_does_not_turn(ctx, monkeypatch, tmp_path):
    looks = [(_sighting(), 0.27, "2/2"), (_sighting(up=12.0), None, "2/2")]
    result, events, arm, _l, _s = _press(ctx, monkeypatch, tmp_path, looks, tilt=0)
    assert "depth:turn" not in events and arm.turns == [] and dp.exit_code(result) == 0


def test_without_the_button_in_view_it_does_not_press_blind(ctx, monkeypatch, tmp_path):
    result, events, arm, _l, _s = _press(ctx, monkeypatch, tmp_path, [(None, 0.31, "the button in none")])
    assert result.free and "not pressing blind" in result.why and "depth:touch" not in events
    assert arm.lines == [] and dp.exit_code(result) == 1


def test_from_over_the_box_the_fingers_are_moved_over_the_button_the_camera_sees(ctx, monkeypatch, tmp_path):
    """Flatbot 2026-10-22 20:04: TF's tilted aim put the press 45 mm off the
    button's centre, on the lid's edge. Straight down, the camera sees the
    button from its own axis, LENS[:2] beside the fingers: the hand moves by
    the difference, and looks again."""
    looks = [(_sighting(x=0.02, y=-LENS[1] + 0.03), 0.27, "2/2"), (_sighting(x=0.001), 0.27, "2/2"),
             (_sighting(up=12.0), None, "2/2")]
    result, events, arm, legs, start = _press(ctx, monkeypatch, tmp_path, looks)
    assert dp.exit_code(result) == 0
    assert events[:6] == ["depth:turn", "look:depth-look", "depth:centre", "look:depth-look", "depth:approach", "depth:touch"]
    # the re-look follows a move with no at-rest read: it waits for still frames
    assert [t for t in arm.trace if t.startswith("look:")][:2] == ["look:depth-look (settled)", "look:depth-look"]
    assert "look:depth-after (settled)" in arm.trace
    goal, quat = arm.turns[0]
    R = rp._rot(quat)
    centre = np.asarray(goal) + R[:, 0] * 0.02 + R[:, 1] * 0.03  # along the hand's own x and y
    assert arm.lines[0] == pytest.approx(list(centre))
    assert legs["depth:centre"].guard.trip == "obstruction" and legs["depth:centre"].plan_free
    # the stroke then goes straight down from over the button
    assert arm.lines[1][:2] == pytest.approx(list(centre[:2]))


def test_each_look_after_a_move_or_the_press_is_told_where_the_button_must_be(ctx, monkeypatch, tmp_path):
    """The centring move goes level by what the camera asked, the hand's
    attitude held: the look after it is told where that button, unmoved,
    must be seen from there (2026-10-23 17:30 took a cabinet's circle). The
    after-look stands where the last look did: told where it saw it."""
    seen = _seen(hand_off=(0.02, 0.03))
    got = []

    def again(ctx_):
        out = seen(ctx_)
        got.append(out[0])
        return out

    looks = [seen, again, seen.after]
    result, events, arm, _l, _s = _press(ctx, monkeypatch, tmp_path, looks)
    assert dp.exit_code(result) == 0
    assert [e for e in events if e.startswith("look:")] == ["look:depth-look", "look:depth-look", "look:depth-after"]
    first, after_move, after_press = arm.expects
    assert first is None  # nothing is known before the first look
    assert after_move == pytest.approx((got[0].x, got[0].y, got[0].lid_depth), abs=1e-6)
    assert after_press == pytest.approx(dp.where_seen(got[0]), abs=1e-9)
    assert result.seen_at is not None  # ... and a look again after it, where the after-look saw it


def test_a_button_far_from_under_the_fingers_is_not_chased(ctx, monkeypatch, tmp_path):
    result, events, _a, _l, _s = _press(ctx, monkeypatch, tmp_path, [(_sighting(y=-LENS[1] + 0.12), 0.27, "2/2")])
    assert result.free and "farther than a re-centre" in result.why and "depth:centre" not in events


def test_a_move_that_does_not_bring_the_button_under_the_fingers_stops(ctx, monkeypatch, tmp_path):
    looks = [(_sighting(x=0.03), 0.27, "2/2"), (_sighting(x=0.029), 0.27, "2/2")]
    result, events, _a, _l, _s = _press(ctx, monkeypatch, tmp_path, looks)
    assert result.free and "did not bring it under" in result.why and "depth:touch" not in events


def test_the_touch_reports_where_the_mount_in_tf_puts_the_fingertips(ctx, monkeypatch, tmp_path):
    import json

    looks = [(_sighting(lid=0.27), 0.27, "2/2"), (_sighting(up=12.0), None, "2/2")]
    _press(ctx, monkeypatch, tmp_path, looks, touch_at=0.15)
    note = [json.loads(line) for line in open(next(tmp_path.glob("run-*.jsonl"))) if '"depth_touch"' in line][0]
    assert note["fingertips_ahead_of_lens_mm"] == pytest.approx(120.0)
    assert note["tf_says_mm"] == pytest.approx(1000 * (dp.DOWN_CONTACT_TOOL[2] - LENS[2]))


def test_a_hand_that_cannot_point_down_there_stops_with_nothing_else_flown(ctx, monkeypatch, tmp_path):
    result, events, _a, _l, _s = _press(ctx, monkeypatch, tmp_path, [], refuse_turn=True)
    assert result.free and "no straight-down hand" in result.why and events == [] and dp.exit_code(result) == 1


def test_no_wrist_frame_at_all_presses_nothing(ctx, monkeypatch, tmp_path):
    result, events, _a, _l, _s = _press(ctx, monkeypatch, tmp_path, [(None, None, "no frame")])
    assert result.free and "not pressing blind" in result.why and "depth:touch" not in events


def test_an_open_box_is_never_pressed_the_pads_feel_its_lid_beside_the_knob(ctx, monkeypatch, tmp_path):
    """17:27: the knob already 10 mm up, and the run stopped — pressing it
    would shut the box. Now nothing presses it: the closed pads go halfway
    across the lid's ring along the hand's x, feel the lid there, and come
    back up; the box is put at the button's centre, at the felt lid's height,
    the knob up — what the grip goes down to."""
    from rammp_box_opening.models.container import from_container

    result, events, arm, legs, start = _press(ctx, monkeypatch, tmp_path, [(_sighting(up=14.0), 0.27, "2/2")])
    assert events == ["depth:turn", "look:depth-look", "depth:beside", "depth:approach", "depth:touch", "depth:unload",
                      "depth:rise"]
    assert "press:push" not in events
    assert result.free and result.contact and result.popped and not result.felt and result.why is None
    assert dp.lid_ready(result) and dp.exit_code(result) == 0
    goal, quat = arm.turns[0]
    R = rp._rot(quat)
    d = dp.lid_touch_offset_m(ctx.model)
    beside = np.asarray(goal) + R[:, 0] * d  # along the hand's x: the pads' narrow side across the ring
    assert arm.lines[0] == pytest.approx(list(beside))
    assert legs["depth:beside"].guard.trip == "obstruction" and legs["depth:beside"].plan_free
    assert arm.lines[1][:2] == pytest.approx(list(beside[:2]))  # the touch straight down from there
    assert legs["depth:touch"].guard.trip == "touch"
    # the box: the button's centre — where the fingers stood before the move — at the lid's felt height
    lid_z = goal[2] - TOUCH_AT + dp.DOWN_CONTACT_TOOL[2] * R[2, 2]
    button = from_container(ctx.cpose, ctx.model.button_offset)
    assert button[:2] == pytest.approx(list(goal[:2]), abs=1e-6) and button[2] == pytest.approx(lid_z, abs=1e-6)
    assert result.touch_xyz == pytest.approx(tuple(button), abs=1e-4)
    # back up where the touch began, judged for a pull at the fingertip, as after a press
    assert arm.lines[-1][2] == pytest.approx(goal[2]) and legs["depth:rise"].guard.hang is not None


def test_the_aim_before_the_depth_press_saw_the_knob_up_and_that_is_enough(ctx, monkeypatch, tmp_path):
    """Either reading is enough: pressing an open box shuts it, while feeling
    the lid of a shut one costs a grip that closes on air and stops."""
    result, events, _a, _l, _s = _press(ctx, monkeypatch, tmp_path, [(_sighting(up=1.0), 0.27, "2/2")], knob_up_mm=10.0)
    assert "depth:beside" in events and "press:push" not in events and result.popped
    result, events, _a, _l, _s = _press(ctx, monkeypatch, tmp_path, [(_sighting(up=1.0), 0.27, "2/2"),
                                                                     (_sighting(up=0.5), None, "2/2")], knob_up_mm=2.0)
    assert "depth:beside" not in events and "press:push" in events


def test_the_pads_feel_an_open_lid_halfway_across_its_ring_clear_of_the_knob_and_the_edge(pink):
    d = dp.lid_touch_offset_m(pink)
    assert d == pytest.approx(0.5 * (0.022 + 0.0525))  # Ankou: halfway between the button's edge and the lid's
    assert d - dp.PAD_HALF_M[0] - 0.022 >= dp.LID_TOUCH_CLEAR_M
    assert 0.0525 - np.hypot(d + dp.PAD_HALF_M[0], dp.PAD_HALF_M[1]) >= dp.LID_TOUCH_CLEAR_M
    # a lid with no ring round its button to speak of: not felt, not pressed
    narrow = SimpleNamespace(button_diameter_m=0.044, dims=(0.07, 0.07, 0.12), button_offset=(0.0, 0.0, 0.12))
    assert dp.lid_touch_offset_m(narrow) is None


def test_an_open_box_with_no_lid_ring_to_feel_is_not_pressed(ctx, monkeypatch, tmp_path):
    monkeypatch.setattr(dp, "lid_touch_offset_m", lambda model: None)
    result, events, _a, _l, _s = _press(ctx, monkeypatch, tmp_path, [(_sighting(up=14.0), 0.27, "2/2")])
    assert result.free and "OPEN" in result.why and "depth:touch" not in events and "depth:beside" not in events


def test_no_contact_all_the_way_down_rises_and_says_so(ctx, monkeypatch, tmp_path):
    result, events, _a, _l, _s = _press(ctx, monkeypatch, tmp_path, [(_sighting(), 0.27, "2/2")], touch_at=None)
    assert result.free and "no contact" in result.why and "press:push" not in events
    assert events[-2:] == ["depth:unload", "depth:rise"]


def test_a_contact_where_these_fingertips_cannot_be_is_not_the_lid(ctx, monkeypatch, tmp_path):
    # the lid 30 cm below the lens, met 3 cm down: the fingertips would be 27 cm in front of the lens
    result, events, _a, _l, _s = _press(ctx, monkeypatch, tmp_path, [(_sighting(lid=0.30), None, "2/2")], touch_at=0.03)
    assert result.free and "not where" in result.why and "press:push" not in events


def test_a_push_the_torques_never_felt_bottom_out_is_not_a_press(ctx, monkeypatch, tmp_path):
    looks = [(_sighting(), 0.27, "2/2"), (_sighting(up=0.5), None, "2/2")]
    result, _e, _a, _l, _s = _press(ctx, monkeypatch, tmp_path, looks, push="arrived")
    assert not result.felt and not result.popped and dp.exit_code(result) == 1


def test_a_box_the_bench_camera_sees_moved_is_moved_the_way_it_went(ctx, monkeypatch, tmp_path):
    """The bench D405 is turned 180 deg about its lens axis: a button seen
    12 mm further along the camera's x moved 12 mm along the hand's -x."""
    from rammp_box_opening.models.container import from_container

    bench = (0.0, 0.075, -0.125)
    looks = [(_sighting(y=bench[1]), 0.27, "2/2"), (_sighting(x=0.012, y=bench[1], up=12.0), None, "2/2")]
    result, _e, arm, _l, _s = _press(ctx, monkeypatch, tmp_path, looks, lens=bench, r_hc=RZ180, measured=bench,
                                     touch_at=0.27 - (dp.DOWN_CONTACT_TOOL[2] - bench[2]))
    assert result.moved
    R = rp._rot(arm.turns[0][1])
    want = np.r_[BUTTON[0], BUTTON[1]] - R[:2, 0] * 0.012  # along the hand's -x
    assert from_container(ctx.cpose, ctx.model.button_offset)[:2] == pytest.approx(list(want))


def test_the_home_after_a_depth_press_that_stopped_says_why_first():
    """17:30: a depth press stopped over the box went home announcing "DEPTH
    PRESS done" and told why only after the home."""
    stopped = dp.DepthPress(free=True, why="moving over the button did not bring it under the fingers (21 -> 288 mm)")
    assert dp.home_why(stopped) == "STOP: " + stopped.why
    assert "MOVED" in dp.home_why(dp.DepthPress(free=True, contact=True, moved=True))
    assert dp.home_why(dp.DepthPress(free=True, contact=True, felt=True, popped=True)) == ""


def test_a_box_the_press_moved_stops_the_run(ctx, monkeypatch, tmp_path):
    looks = [(_sighting(), 0.27, "2/2"), (_sighting(x=0.012, up=12.0), None, "2/2")]
    result, _e, _a, _l, _s = _press(ctx, monkeypatch, tmp_path, looks)
    assert result.moved and dp.exit_code(result) == 1


ARRIVED = ("arrived", {"message": "ok", "progress": 1.0})


def _depth_flow(ctx, monkeypatch, tmp_path, press, **kw):
    """test_reach_press._flow with --depth-press: `press` stands in for
    press_down_from_here (called with the reach's ctx). The fake chain's fk
    stands at z 0.45 over the aimed button at 0.40."""
    from test_reach_press import _flow

    got = {}

    def fake(node, ctx_, cfg, runner, watcher, contact_tool, here=None, turned=None):
        got["contact"] = contact_tool
        return press(ctx_)

    monkeypatch.setattr(dp, "press_down_from_here", fake)
    real = rp.run_reach_press

    def with_flag(node, ctx_, cfg, runner, watcher, args, scene_fix, locator):
        return real(node, ctx_, cfg, runner, watcher, SimpleNamespace(execute=True, press_only=True, depth_press=True),
                    scene_fix, locator)

    monkeypatch.setattr(rp, "run_reach_press", with_flag)
    kw.setdefault("touch", ARRIVED)  # no TF stroke flies: the 4th motion is the way up or home
    code, rec, chain, home = _flow(ctx, monkeypatch, tmp_path, **kw)
    return code, rec, chain, home, got


def _popped(ctx_):
    return dp.DepthPress(free=True, contact=True, felt=True, popped=True)


def _lost(ctx_):
    """Touched and pushed, then the wrist camera did not find the box again."""
    return dp.DepthPress(free=False, contact=True, felt=True, lost=True, why="the wrist camera does not find it")


def _measured(ctx_, z=0.30):
    """The depth press as it leaves ctx: the box put where the fingertips met it."""
    from rammp_box_opening.models.container import from_container

    b = from_container(ctx_.cpose, ctx_.model.button_offset)
    ctx_.cpose = rp.cpose_for_button((b[0], b[1], z), ctx_.cpose.yaw, ctx_.model.button_offset)
    return dp.DepthPress(free=True, contact=True, felt=True, popped=True, touch_xyz=(b[0], b[1], z))


def test_the_reach_out_hands_over_to_the_depth_press_once_over_the_box(ctx, monkeypatch, tmp_path):
    """After the slide in, the TF stroke is not flown: the depth press is.
    Standing within HOME_CLEAR_M of the box's padded top, the hand goes
    straight up clear of it before home is planned — once."""
    from rammp_box_opening.worlds import planned_top_z

    code, rec, chain, _h, got = _depth_flow(ctx, monkeypatch, tmp_path, _popped)
    assert code == 0 and got["contact"] == rp.CONTACT_TOOL
    assert rec.events == ["reach:look", "aim:reach-aim", "reach:re-aim", "reach:slide-in", "depth:clear", "home"]
    assert "press:down" not in rec.events
    up = [xyz for xyz, _q in chain.lines][-1]
    assert up[:2] == pytest.approx([0.5, -0.4]) and up[2] == pytest.approx(planned_top_z(ctx.model, ctx.cpose) + rp.HOME_CLEAR_M)
    g = rec.legs["depth:clear"].guard
    assert g.trip == "obstruction" and g.touch_nm == rp.DRAG_NM and g.hang is not None and rec.legs["depth:clear"].plan_free


# ---- CENTRE on a mount that is known, in world xy ----------------------------------

def test_a_mount_in_tf_that_disagrees_with_the_measured_lens_is_not_centred_on(ctx, monkeypatch, tmp_path):
    """Flatbot 2026-10-22 20:04: TF's hand-measured mount put the lens 100 mm
    beside the fingers, the truth ~65. CENTRE on it 'converges' with the
    press 35 mm off, on the lid's edge — it stops with nothing pressed."""
    tf_dev = (0.0, 0.100, -0.22)
    result, events, arm, _l, _s = _press(ctx, monkeypatch, tmp_path, [_seen((0.0, 0.0))], lens=tf_dev)
    assert result.free and "mount" in result.why and "+100" in result.why and "+65" in result.why
    assert events == ["depth:turn", "look:depth-look"] and arm.lines == [] and dp.exit_code(result) == 1


def test_no_measured_lens_in_the_camera_config_is_not_centred_on(ctx, monkeypatch, tmp_path):
    result, events, arm, _l, _s = _press(ctx, monkeypatch, tmp_path, [_seen((0.02, 0.0))], measured=None)
    assert result.free and "wrist_lens_in_hand_m" in result.why
    assert events == ["depth:turn", "look:depth-look"] and arm.lines == []


def test_a_mount_in_tf_within_5_mm_of_the_measured_lens_centres(ctx, monkeypatch, tmp_path):
    tf = (0.002, LENS[1] + 0.003, LENS[2] + 0.01)
    seen = _seen((0.02, 0.03), lens=tf)
    looks = [seen, seen, seen.after]
    result, events, _a, _l, _s = _press(ctx, monkeypatch, tmp_path, looks, lens=tf)
    assert dp.exit_code(result) == 0
    assert events[:6] == ["depth:turn", "look:depth-look", "depth:centre", "look:depth-look", "depth:approach", "depth:touch"]


def test_a_turn_short_of_straight_down_centres_the_vertical_stroke_not_the_tool_axis(ctx, monkeypatch, tmp_path):
    """Exactly down sits on the wrist's singularity, the turn stops 5 deg
    short. The button on the tool axis 277 mm below the lens is 14 mm off the
    fingertip's vertical: CENTRE moves it under the vertical stroke, level."""
    seen = _seen((0.0, 0.0), depth=0.277)
    looks = [seen, seen, seen.after]
    result, events, arm, legs, _s = _press(ctx, monkeypatch, tmp_path, looks, refuse_first=2,
                                           touch_at=0.277 - (dp.DOWN_CONTACT_TOOL[2] - LENS[2]))
    goal, quat = arm.turns[0]
    R = rp._rot(quat)
    assert np.degrees(np.arccos(-R[2, 2])) == pytest.approx(5.0, abs=0.01)
    assert dp.exit_code(result) == 0 and "depth:centre" in events
    b = seen.world["b"]
    tip = np.asarray(arm.lines[0]) + R @ np.asarray(dp.DOWN_CONTACT_TOOL)
    assert np.hypot(*(tip[:2] - b[:2])) <= dp.CENTRE_TOL_M
    assert np.hypot(*(np.asarray(goal[:2]) - arm.lines[0][:2])) == pytest.approx(0.014, abs=0.001)
    # level: every waypoint of the centring move at the height it began
    for p in legs["depth:centre"].traj.points:
        assert p.positions[2] == pytest.approx(goal[2], abs=0.001)
    # and the stroke goes straight down from there
    assert arm.lines[1][:2] == pytest.approx(arm.lines[0][:2])


def test_the_bench_cameras_turn_in_the_hand_is_kept_when_centring(ctx, monkeypatch, tmp_path):
    """The bench D405, 180 deg about its lens axis: with the button under the
    fingers there is no move; 30 mm off in the hand's +y, the move goes 30 mm
    toward it — not 100 mm away."""
    bench = (0.0, 0.075, -0.125)
    seen = _seen((0.0, 0.0), lens=bench, r_hc=RZ180)
    looks = [seen, seen.after]
    result, events, arm, _l, start = _press(ctx, monkeypatch, tmp_path, looks, tilt=0, lens=bench, r_hc=RZ180,
                                            measured=bench, touch_at=0.27 - (dp.DOWN_CONTACT_TOOL[2] - bench[2]))
    assert dp.exit_code(result) == 0 and "depth:centre" not in events
    seen = _seen((0.0, 0.03), lens=bench, r_hc=RZ180)
    looks = [seen, seen, seen.after]
    result, events, arm, _l, start = _press(ctx, monkeypatch, tmp_path, looks, tilt=0, lens=bench, r_hc=RZ180,
                                            measured=bench, touch_at=0.27 - (dp.DOWN_CONTACT_TOOL[2] - bench[2]))
    assert dp.exit_code(result) == 0 and "depth:centre" in events
    R = rp._rot(rp.reach_quat(ctx.model, BUTTON[:2], 0))
    want = np.asarray(start[:3]) + R @ np.r_[0.0, 0.03, 0.0]
    assert arm.lines[0][:2] == pytest.approx(list(want[:2]), abs=0.001)


def test_a_touch_far_from_where_the_mount_puts_the_fingertips_pushes_nothing(ctx, monkeypatch, tmp_path):
    ahead_tf = dp.DOWN_CONTACT_TOOL[2] - LENS[2]
    looks = [(_sighting(lid=0.27), 0.27, "2/2")]
    result, events, _a, _l, _s = _press(ctx, monkeypatch, tmp_path, looks, touch_at=0.27 - (ahead_tf - 0.05))
    assert result.free and result.contact and "not pushing" in result.why and "press:push" not in events
    assert events[-2:] == ["depth:unload", "depth:rise"]
    # 12 mm (#128's mount against the 20:04 touch): pushed
    looks = [(_sighting(lid=0.27), 0.27, "2/2"), (_sighting(up=12.0), None, "2/2")]
    result, events, _a, _l, _s = _press(ctx, monkeypatch, tmp_path, looks, touch_at=0.27 - (ahead_tf + 0.012))
    assert "press:push" in events and dp.exit_code(result) == 0


def test_a_box_the_touch_put_lower_needs_no_line_up_before_home(ctx, monkeypatch, tmp_path):
    """20:04: the touch found the lid 76 mm under TF's. With the box put
    there, the hand at the stroke's start is clear of it and home plans."""
    code, rec, _c, home, _g = _depth_flow(ctx, monkeypatch, tmp_path, _measured)
    assert code == 0 and rec.events[-2:] == ["reach:slide-in", "home"]
    from rammp_box_opening.models.container import from_container

    assert from_container(home["cpose"], ctx.model.button_offset)[2] == pytest.approx(0.30)  # home planned around it


def test_a_line_up_that_does_not_fly_falls_back_to_the_back_off_and_logs_each_refusal(ctx, monkeypatch, tmp_path):
    import json

    from test_reach_press import _Chain

    real = _Chain.straight_line

    def no_way_up(self, q0, xyz, quat):
        if xyz[2] > 0.49:  # the line up (to 0.50; the look and the slide stand at ~0.47): a joint limit
            self.lines.append((list(xyz), list(quat)))
            return None, "joint_6 limit"
        return real(self, q0, xyz, quat)

    monkeypatch.setattr(_Chain, "straight_line", no_way_up)
    code, rec, _c, _h, _g = _depth_flow(ctx, monkeypatch, tmp_path, _popped, home_refusals=2)
    assert code == 1 and rec.events[-1] == "reach:back-off" and "depth:clear" not in rec.events
    rows = [json.loads(line) for line in open(next(tmp_path.glob("run-*.jsonl")))]
    refused = [r for r in rows if r["leg"] == "home_refused"]
    assert len(refused) == 2 and all("INVALID_START" in r["why"] for r in refused)


def test_a_depth_press_whose_home_failed_never_exits_0(ctx, monkeypatch, tmp_path, capsys):
    """The knob popped, but the arm did not get home: it holds over the box."""
    monkeypatch.setattr(rp, "_go_home", lambda *a, **k: False)
    code, _r, _c, _h, _g = _depth_flow(ctx, monkeypatch, tmp_path, _measured)
    assert code == 1 and "did not get home" in capsys.readouterr().out


def test_a_box_no_camera_finds_after_the_press_holds_the_arm(ctx, monkeypatch, tmp_path, capsys):
    code, rec, _c, _h, _g = _depth_flow(ctx, monkeypatch, tmp_path, _lost, scene_after="none")
    assert code == 1 and "home" not in rec.events and rec.events[-1] == "scene"
    assert "neither camera" in capsys.readouterr().out


def test_a_box_only_the_scene_camera_finds_after_the_press_goes_home_around_it(ctx, monkeypatch, tmp_path):
    from rammp_box_opening.models.container import from_container

    code, rec, _c, home, _g = _depth_flow(ctx, monkeypatch, tmp_path, _lost)
    assert code == 1  # the knob was not judged
    assert rec.events.count("scene") == 1 and rec.events[-1] == "home"
    assert from_container(home["cpose"], ctx.model.button_offset) == pytest.approx([0.66, -0.57, 0.40])
    # the scene camera saw it 50 mm away: home is planned around the box shifted by that
    code, rec, _c, home, _g = _depth_flow(ctx, monkeypatch, tmp_path / "moved", _lost, scene_after="moved")
    assert code == 1 and rec.events[-1] == "home"
    assert from_container(home["cpose"], ctx.model.button_offset) == pytest.approx([0.71, -0.57, 0.40])


def test_a_box_the_depth_press_moved_is_a_stop_before_the_arm_goes_home(ctx, monkeypatch, tmp_path, capsys):
    def moved(ctx_):
        return dp.DepthPress(free=True, contact=True, felt=True, popped=True, moved=True)

    code, rec, _c, _h, _g = _depth_flow(ctx, monkeypatch, tmp_path, moved)
    out = capsys.readouterr().out
    assert code == 1 and rec.events[-1] == "home"
    assert out.index("STOP: the press MOVED the box") < out.index("\nhome ")


def test_the_mission_record_keeps_what_the_reach_and_the_depth_press_found(ctx, monkeypatch, tmp_path):
    import json

    from rammp_box_opening.detection_set import MissionFrames

    ctx.mission_frames = MissionFrames(tmp_path / "missions", "ankou_pink.yaml", 0.2933, now=0)
    code, _r, _c, _h, _g = _depth_flow(ctx, monkeypatch, tmp_path, _measured)
    assert code == 0
    ctx.mission_frames.frames.append(("scene_0", {"x": np.zeros(1)}))
    doc = json.loads((ctx.mission_frames.close(exit_code=code) / "truth.json").read_text())
    # the aim through TF is a fix, never trusted truth; the arm's own touch is kept beside it
    assert doc["truth_source"] == "fix" and doc["truth_xyz"] == pytest.approx([0.66, -0.57, 0.40])
    assert doc["stage"] == "pressed" and doc["popped"] is True and doc["touch_xyz"] == pytest.approx([0.66, -0.57, 0.30])
    assert doc["table_z"] == pytest.approx(0.2933)


# ---- what the depth press leaves behind ---------------------------------------

def test_the_touch_puts_the_box_where_the_fingertips_met_it(ctx, monkeypatch, tmp_path):
    """TF's lid at 0.40 (BUTTON 0.24 here); the fingertips met it 240 mm
    down from 0.46, where the wrist camera saw the lid (so the touch passes
    the mount check): every world after the touch holds the box there, and
    the table is left as the scene camera measured it."""
    from rammp_box_opening.models.container import from_container

    ctx.model = ContainerModel.load(PINK)
    ctx.cpose = rp.cpose_for_button(BUTTON, 0.3, ctx.model.button_offset)
    ctx.worlds.table_top_z = 0.10
    lid = 0.24 + (dp.DOWN_CONTACT_TOOL[2] - LENS[2])
    looks = [(_sighting(lid=lid), lid, "2/2"), (_sighting(up=12.0), None, "2/2")]
    result, _e, _a, _l, _s = _press(ctx, monkeypatch, tmp_path, looks, touch_at=0.24)
    assert result.popped and result.free
    assert from_container(ctx.cpose, ctx.model.button_offset) == pytest.approx([BUTTON[0], BUTTON[1], 0.22])
    assert ctx.cpose.yaw == 0.3 and result.touch_xyz == pytest.approx((BUTTON[0], BUTTON[1], 0.22))
    assert ctx.worlds.table_top_z == 0.10
    rise = [kw for kind, kw in ctx.worlds.pushes if kind == "full"][-1]  # the rise's world
    assert from_container(rise["cpose"], ctx.model.button_offset)[2] == pytest.approx(0.22)
    assert result.ref is not None  # the free-air reading the way home is guarded against


def test_a_box_the_wrist_camera_loses_after_a_contact_holds_the_arm(ctx, monkeypatch, tmp_path):
    looks = [(_sighting(), 0.27, "2/2"), (None, None, "the button in none")]
    result, events, _a, _l, _s = _press(ctx, monkeypatch, tmp_path, looks)
    assert result.contact and result.felt and result.lost and not result.free
    assert dp.exit_code(result) == 1 and events[-1] == "look:depth-after"


def test_a_box_the_press_moved_is_moved_in_the_world_by_what_the_camera_saw(ctx, monkeypatch, tmp_path):
    from rammp_box_opening.models.container import from_container

    looks = [(_sighting(), 0.27, "2/2"), (_sighting(x=0.012, up=12.0), None, "2/2")]
    result, _e, arm, _l, _s = _press(ctx, monkeypatch, tmp_path, looks)
    assert result.moved
    R = rp._rot(arm.turns[0][1])
    want = np.r_[BUTTON[0], BUTTON[1]] + R[:2, 0] * 0.012  # along the hand's own x
    assert from_container(ctx.cpose, ctx.model.button_offset)[:2] == pytest.approx(list(want))


def test_the_scene_camera_is_asked_only_when_the_wrist_lost_the_box(ctx, tmp_path):
    from conftest import runner as make_runner

    from rammp_box_opening.models.container import ContainerPose

    r = make_runner(ctx.client, tmp_path)
    fix = SimpleNamespace(top_xyz=(0.6, -0.2, 0.4))
    asked = []
    loc = SimpleNamespace(locate=lambda s: asked.append(s) or SimpleNamespace(top_xyz=(0.6, -0.2, 0.4)))
    seen = dp.DepthPress(free=True, contact=True, popped=True)
    assert dp.after_scene_check(ctx, r, seen, fix, loc) is seen and asked == []
    lost = dp.DepthPress(free=False, contact=True, lost=True, why="x")
    assert dp.after_scene_check(ctx, r, lost, fix, None) is lost  # no scene camera: the arm holds
    ctx.cpose = ContainerPose(xyz=(0.6, -0.2, 0.28), yaw=0.0)
    out = dp.after_scene_check(ctx, r, lost, fix, loc)
    assert out.free and not out.moved and len(asked) == 1 and ctx.cpose.xyz == (0.6, -0.2, 0.28)
    tipped = SimpleNamespace(locate=lambda s: SimpleNamespace(top_xyz=(0.6, -0.2, 0.37)))
    out = dp.after_scene_check(ctx, r, lost, fix, tipped)
    assert out.free and out.moved


def test_an_arm_that_did_not_get_home_is_never_a_success(ctx, capsys):
    popped = dp.DepthPress(free=True, contact=True, felt=True, popped=True)
    assert dp.finish(ctx, popped, True) == 0
    assert dp.finish(ctx, popped, False) == 1 and "did not get home" in capsys.readouterr().out
    held = dp.DepthPress(free=False, contact=True, why="the way up stopped")
    assert dp.finish(ctx, held, False) == 1 and "did not get home" not in capsys.readouterr().out




def test_the_reach_out_slides_in_turning_down_in_one_leg(ctx, monkeypatch, tmp_path):
    """Run 20:04: the slide in (4.56 s), two at-rest reads and the turn
    (7.43 s) — one walk does both, as slow and as guarded as the slide."""
    import test_reach_press as trp

    from rammp_box_opening.constants import CONTACT_SPEED

    got = {}

    def press(node, ctx_, cfg, runner, watcher, contact_tool, here=None, turned=None):
        got.update(here=here, turned=turned)
        # as the real press does: the free-air reading it was handed is the one home is judged against
        return dp.DepthPress(free=True, contact=True, felt=True, popped=True, ref=here)

    monkeypatch.setattr(dp, "press_down_from_here", press)
    walks = []

    def walk(self, q0, xyz, quat, step_m=None, step_rad=None, max_step_rad=None):
        walks.append((list(xyz), list(quat), step_m, step_rad, max_step_rad))
        return [list(q0), [v + 0.01 for v in q0]], None

    monkeypatch.setattr(trp._Chain, "walk", walk, raising=False)
    real = rp.run_reach_press
    monkeypatch.setattr(rp, "run_reach_press", lambda n, c, cfg, r, w, a, f, l: real(
        n, c, cfg, r, w, SimpleNamespace(execute=True, press_only=True, depth_press=True), f, l))
    code, rec, chain, _h = trp._flow(ctx, monkeypatch, tmp_path, touch=ARRIVED)
    assert code == 0 and rec.events == ["reach:look", "aim:reach-aim", "reach:re-aim", "reach:slide-in", "depth:clear",
                                        "home"]
    leg = rec.legs["reach:slide-in"]
    (goal, down, step_m, step_rad, jump), = walks
    assert leg.speed == CONTACT_SPEED and leg.guard.reference == trp.AT_LOOK2 and leg.guard.touch_nm == rp.DRAG_NM
    assert leg.target[2] == pytest.approx(down) and got["turned"] == pytest.approx(down)
    assert (step_m, step_rad, jump) == (dp.TURN_STEP_M, dp.TURN_STEP_RAD, dp.TURN_MAX_JUMP_RAD)
    assert rp._rot(down)[2, 2] == pytest.approx(-1.0)  # exactly down, the first of TURN_SHORT_DEG
    # the pads end where the tilted fingertip would have ended the slide
    quat = rp.reach_quat(ctx.model, trp.BUTTON[:2], rp.REACH_TILTS_DEG[0])
    start = rp.tool_for_contact((trp.BUTTON[0], trp.BUTTON[1], trp.BUTTON[2] + rp.REACH_START_M), quat)
    tip = np.asarray(start) + rp._rot(quat) @ np.asarray(rp.CONTACT_TOOL)
    assert np.asarray(goal) + rp._rot(down) @ np.asarray(dp.DOWN_CONTACT_TOOL) == pytest.approx(tip)
    # one at-rest read after it, handed over: the depth press reads nothing again
    assert got["here"] == trp.AT_START and rec.reads == 4
    # the slide is solved by plan_reach and reach_from_here, to prove it flies, and not again from the
    # live joints; the stroke, which the depth press never flies, only by plan_reach
    assert sum(xyz == pytest.approx(start) for xyz, _q in chain.lines) == 2
    bottom = rp.tool_for_contact((trp.BUTTON[0], trp.BUTTON[1], trp.BUTTON[2] - load_press_demo(PINK).travel_m), quat)
    assert sum(xyz == pytest.approx(bottom) for xyz, _q in chain.lines) == 1


def test_where_no_walk_turns_down_on_the_way_it_slides_then_turns(ctx, monkeypatch, tmp_path):
    import test_reach_press as trp

    got = {}

    def press(node, ctx_, cfg, runner, watcher, contact_tool, here=None, turned=None):
        got.update(here=here, turned=turned)
        return dp.DepthPress(free=True, contact=True, felt=True, popped=True)

    monkeypatch.setattr(dp, "press_down_from_here", press)
    tried = []
    monkeypatch.setattr(trp._Chain, "walk", lambda self, q0, xyz, quat, **k: (tried.append(quat), (None, "a joint limit"))[1],
                        raising=False)
    real = rp.run_reach_press
    monkeypatch.setattr(rp, "run_reach_press", lambda n, c, cfg, r, w, a, f, l: real(
        n, c, cfg, r, w, SimpleNamespace(execute=True, press_only=True, depth_press=True), f, l))
    code, rec, _c, _h = trp._flow(ctx, monkeypatch, tmp_path, touch=ARRIVED)
    assert code == 0 and len(tried) == len(dp.TURN_SHORT_DEG)
    quat = rp.reach_quat(ctx.model, trp.BUTTON[:2], rp.REACH_TILTS_DEG[0])
    assert rec.legs["reach:slide-in"].target[2] == pytest.approx(quat)  # the tilted slide: the depth press turns
    assert got["turned"] is None and got["here"] == trp.AT_START


def test_the_walk_in_on_flatbot_turns_down_over_the_button_no_lower_than_the_slide(ctx, monkeypatch):
    """Flatbot's URDF, run 20:04's aimed button at 35 deg (the walk
    modelled before it was flown: 31 points, the fingertip within 0.6 mm of
    the slide's height)."""
    from rammp_box_opening.constants import HOME
    from rammp_box_opening.runtime import approach
    from rammp_box_opening.runtime.kinematics import ArmChain

    chain = ArmChain(path="src/rammp_box_opening/test/data/arm_flatbot_20261021.urdf", root="arm_mount")
    model = ContainerModel.load(PINK)
    ctx.model = model
    button, tilt = (0.4833, -0.1575, 0.4044), 35
    quat, start, _bottom, look = rp.reach_geometry(model, button, tilt, 0.015)
    q_look = chain.nearest_solution(HOME, look, quat)
    assert q_look is not None
    ctx.client.live = [float(v) for v in q_look]
    monkeypatch.setattr(approach, "_chain", lambda: chain)
    pts, down, why = dp.walk_in_down(ctx, start, quat)
    assert why is None, why
    tip = np.asarray(start) + rp._rot(quat) @ np.asarray(rp.CONTACT_TOOL)  # the slid-in fingertip
    R, t = chain.fk(pts[-1])
    R = np.asarray(R, float)
    lean = np.degrees(np.arccos(np.clip(-R[2, 2], -1.0, 1.0)))
    assert lean < 1.0 + max(dp.TURN_SHORT_DEG)
    assert np.linalg.norm(np.asarray(t) + R @ np.asarray(dp.DOWN_CONTACT_TOOL) - tip) < 0.001
    lowest = []
    for q in pts:
        Rq, tq = chain.fk(q)
        Rq = np.asarray(Rq, float)
        lowest.append(min(float((np.asarray(tq) + Rq @ np.asarray(p))[2]) for p in (rp.CONTACT_TOOL, dp.DOWN_CONTACT_TOOL)))
    assert min(lowest) >= tip[2] - 0.001  # never nearer the lid than the slide
    assert max(float(np.abs(np.asarray(b) - np.asarray(a)).max()) for a, b in zip(pts, pts[1:])) <= dp.TURN_MAX_JUMP_RAD


def test_a_look_straight_after_an_at_rest_read_takes_frames_at_once(monkeypatch):
    """look() waits LOOK_SETTLE_S for still frames; after an at-rest read
    the arm has been still 0.45 s already. Either way a frame shot before
    `after` never counts."""
    import time as _time

    from rammp_box_opening.tasks import press_demo

    monkeypatch.setattr(dp, "LOOK_SETTLE_S", 0.5)
    monkeypatch.setattr(press_demo, "save_aim_frame", lambda *a, **k: None)
    clock = {"ns": 1000}
    grab = SimpleNamespace(color=np.zeros((4, 4, 3)), depth=np.zeros((4, 4)), k=np.eye(3),
                           color_stamp=SimpleNamespace(sec=0, nanosec=1000))  # shot before the look began
    node = SimpleNamespace(get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=clock["ns"])))

    def spin(_node):
        clock["ns"] += 10
        if clock["ns"] % 1000 == 0:  # a new frame now and then, stamped now
            grab.color_stamp = SimpleNamespace(sec=0, nanosec=clock["ns"])

    monkeypatch.setattr(press_demo, "_spin_detect", spin)
    # the stale frame sees the button elsewhere: counted, it would not agree
    monkeypatch.setattr(dp, "find_in_camera", lambda c, d, k, m, dist=None, expect=None: (
        _sighting(x=0.05 if grab.color_stamp.nanosec == 1000 else 0.0), None))
    for settled in (True, False):
        clock["ns"], grab.color_stamp = 1000, SimpleNamespace(sec=0, nanosec=1000)
        t0 = _time.monotonic()
        s, _z, _lens, status = dp.look(node, SimpleNamespace(grab=grab), None, "t", settled=settled)
        took = _time.monotonic() - t0
        assert s is not None and status == "2/2 frames", (settled, status)
        assert (took < dp.LOOK_SETTLE_S) if settled else (took >= dp.LOOK_SETTLE_S), (settled, took)


def test_whatever_the_fast_stretch_meets_is_not_pushed(ctx, monkeypatch, tmp_path):
    """The fast stretch is a guarded touch too, and ends TOUCH_SLOW_M short
    of the lid — more than AHEAD_TOL_M, so every contact the press accepts
    is met slowly. Something met on the fast stretch stands too far above
    the lid: the slow stretch is not flown, nothing is pushed."""
    assert dp.TOUCH_SLOW_M > dp.AHEAD_TOL_M
    looks = [(_sighting(), 0.27, "2/2"), (_sighting(up=12.0), 0.255, "2/2")]
    result, events, arm, legs, start = _press(ctx, monkeypatch, tmp_path, looks, touch_at=0.08)
    assert "depth:approach" in events and "depth:touch" not in events and "press:push" not in events
    assert result.free and "not pushing" in result.why


def test_a_knob_still_rising_is_looked_at_again(monkeypatch):
    """16:02-16:03: the after-look read the knob 4.5 mm up, under
    KNOB_UP_MM; the next run's aim saw it 8 mm up. A press felt to bottom
    out is looked at again before it is judged not to have popped."""
    import dataclasses

    seen = iter([(_sighting(up=4.5), None, "2/2"), (_sighting(up=8.0), None, "2/2")])
    expects = []
    monkeypatch.setattr(dp, "look", lambda *a, **k: (expects.append(k.get("expect")), (*next(seen)[:2], None, "2/2"))[1])
    felt = dp.DepthPress(free=True, contact=True, felt=True, popped=False, knob_mm=4.5, seen_at=(0.0, -0.065, 0.27))
    again = dp.knob_up_after(None, SimpleNamespace(model=None), None, None, felt)
    assert again.popped and again.knob_mm == pytest.approx(8.0)
    assert expects and set(expects) == {(0.0, -0.065, 0.27)}  # looked for where the after-look saw it
    # nothing to look at again: not felt, already popped, or the arm not free
    for r in (dataclasses.replace(felt, felt=False), dataclasses.replace(felt, popped=True),
              dataclasses.replace(felt, free=False)):
        assert dp.knob_up_after(None, None, None, None, r) is r


def test_the_lid_is_gripped_only_off_a_popped_box_standing_where_it_was_seen():
    ok = dp.DepthPress(free=True, contact=True, felt=True, popped=True)
    assert dp.lid_ready(ok)
    # slid a little, still standing: gripped where it slid to (16:02: 6 mm)
    assert dp.lid_ready(dp.DepthPress(free=True, contact=True, popped=True, moved=True, shift_m=0.006, lid_dz_m=-0.001))
    for bad in (
        dp.DepthPress(free=True, contact=True, popped=False),  # the knob is not up
        dp.DepthPress(free=False, contact=True, popped=True),  # the arm holds
        dp.DepthPress(free=True, contact=True, popped=True, lost=True),  # the wrist lost the box
        dp.DepthPress(free=True, contact=True, popped=True, why="stopped short"),
        dp.DepthPress(free=True, contact=True, popped=True, moved=True, shift_m=0.03),  # moved too far
        dp.DepthPress(free=True, contact=True, popped=True, moved=True, shift_m=0.004, lid_dz_m=0.02),  # tipped
    ):
        assert not dp.lid_ready(bad), bad


def test_at_rest_the_lens_is_placed_even_when_tf_lags_the_newest_frame(monkeypatch):
    """Flatbot 2026-10-23 16:51: right over the box, both of the depth look's
    frames were newer than the newest arm TF, the lookup at their stamps
    failed, and the press stopped "no camera pose for the wrist's frames".
    The arm had not moved since the look began, so the camera's pose a
    little earlier — never before the arm stopped — places the lens."""
    from rclpy.time import Time

    from rammp_box_opening.runtime import approach

    stamp_ns = 1_000_000_000_000
    latest_ns = stamp_ns - 40_000_000  # TF is 40 ms behind the frame

    class Buffer:
        def lookup_transform(self, root, parent, when):
            if when.nanoseconds > latest_ns:
                raise RuntimeError("extrapolation into the future")
            return SimpleNamespace(transform=SimpleNamespace(
                translation=SimpleNamespace(x=0.0, y=0.065, z=-0.105), rotation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0)))

    grab = SimpleNamespace(tf_buffer=Buffer(), parent="wrist_camera_color_optical_frame", _base_frame="arm_mount",
                           mount_xyz=[0.0, 0.0, 0.0], mount_quat=[0.0, 0.0, 0.0, 1.0], stamp_offset_s=0.0,
                           color_stamp=Time(nanoseconds=stamp_ns).to_msg())
    monkeypatch.setattr(approach, "_chain", lambda: SimpleNamespace(fk=lambda q: (np.eye(3), np.zeros(3))))
    ctx = SimpleNamespace(client=SimpleNamespace(joints=lambda: [0.0] * 7))
    assert dp.lens_in_hand(ctx, grab) is None  # at the frame's own stamp: TF has nothing yet
    lens, r_hc = dp.lens_in_hand(ctx, grab, still_since_ns=stamp_ns - 1_000_000_000)
    assert lens == pytest.approx([0.0, 0.065, -0.105])
    # ... but never from before the arm stopped
    assert dp.lens_in_hand(ctx, grab, still_since_ns=stamp_ns - 20_000_000) is None
