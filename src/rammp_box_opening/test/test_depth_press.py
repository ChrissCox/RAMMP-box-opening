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

    def __init__(self, refuse_turn=False):
        self.lines, self.turns, self.refuse_turn = [], [], refuse_turn
        self.trace = []  # legs, looks and at-rest reads, in order (_press)

    def fk(self, q):
        return np.asarray(quat_to_mat(*q[3:7]), float), np.asarray(q[:3], float)

    def straight_line(self, q0, xyz, quat):
        self.lines.append(list(xyz))
        return [list(q0), [float(v) for v in xyz] + [float(v) for v in quat]], None

    def walk(self, q0, xyz, quat, step_m=None, step_rad=None, max_step_rad=None):
        if self.refuse_turn:
            return None, "a joint limit 40% of the way"
        self.turns.append((list(xyz), list(quat)))
        return [list(q0), [float(v) for v in xyz] + [float(v) for v in quat]], None


class _Gravity:
    def wrist_holding(self, q):
        return [0.0] * 4

    def tip_load(self, q, p):
        return [0.48, 0.0, 0.2, 0.0]


LENS = (0.0, 0.065, -0.105)  # the lens from the tool frame, in the hand: 65 mm beside the fingers, 105 mm behind


def _sighting(x=0.0, y=-LENS[1], lid=0.27, up=0.4):
    """The button seen at (x, y) from the lens axis: by default under the fingers."""
    return dp.LensSighting(x=x, y=y, lid_depth=lid, button_depth=lid - up / 1000.0, up_mm=up, raw=None)


def _press(ctx, monkeypatch, tmp_path, looks, tilt=30, touch_at=0.24, push="touch", refuse_turn=False, here=None,
           turned=None):
    """press_down_from_here with the hand `tilt` deg over BUTTON (at the
    reach-out's stroke start, 6 cm above TF's lid at 0.40), the wrist camera
    scripted by `looks` and the fingertips meeting the lid `touch_at` down
    the stroke. Every at-rest read is a distinct reading, ("rest", n, where)
    in arm.trace. Returns (DepthPress, events, arm, legs, start tool pose)."""
    from rammp_box_opening.runtime import approach
    from rammp_box_opening.runtime.runner import Runner

    cfg = load_press_demo(PINK)
    ctx.model = ContainerModel.load(PINK)
    ctx.press_cfg = cfg
    arm = _Arm(refuse_turn)
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

    def look(node, watcher, model, tag, runner=None, ctx=None, settled=False):
        events.append("look:" + tag)
        arm.trace.append("look:%s%s" % (tag, " (settled)" if settled else ""))
        s, z, status = seq.pop(0)[:3]
        return s, z, LENS, status

    monkeypatch.setattr(dp, "look", look)

    class R(Runner):
        def run(self, legs_, execute, lookahead=None):
            for lg in legs_:
                events.append(lg.name)
                arm.trace.append(lg.name)
                legs[lg.name] = lg
            name = legs_[-1].name
            if name == "depth:touch" and touch_at is not None:
                ctx.client.exec_script = [("touch", {"message": "torque guard trip", "progress": 0.7, "torque_peak": 3.2})]
            if name == "press:push":
                ctx.client.exec_script = [(push, {"message": "x", "progress": 0.6 if push == "touch" else 1.0, "torque_peak": 7.4})]
            top = list(ctx.client.live)
            out = super().run(legs_, execute, lookahead)
            if name == "depth:touch" and touch_at is not None:
                ctx.client.live = [top[0], top[1], top[2] - touch_at] + top[3:]  # stopped on the lid
            return out

    runner = R(ctx.client, log_dir=tmp_path)
    contact = rp.CONTACT_TOOL if tilt else dp.DOWN_CONTACT_TOOL
    result = dp.press_down_from_here(None, ctx, cfg, runner, SimpleNamespace(), contact, here=here, turned=turned)
    return result, events, arm, legs, start


def test_over_the_box_it_turns_down_measures_the_depth_and_presses_by_feel(ctx, monkeypatch, tmp_path):
    looks = [(_sighting(), 0.27, "2/2"), (_sighting(up=12.0), 0.255, "2/2")]
    result, events, arm, legs, start = _press(ctx, monkeypatch, tmp_path, looks)
    assert dp.exit_code(result) == 0 and result.free and result.felt and result.popped
    assert events == [
        "depth:turn", "look:depth-look", "depth:touch", "press:push", "depth:unload", "depth:rise", "look:depth-after",
    ]
    # TURN: straight down, about the fingertip — it stays where it was over the button
    (goal, quat), = arm.turns
    R0 = rp._rot(rp.reach_quat(ctx.model, BUTTON[:2], 30))
    tip = np.asarray(start) + R0 @ np.asarray(rp.CONTACT_TOOL)
    Rd = rp._rot(quat)
    assert Rd[2, 2] == pytest.approx(-1.0)
    assert np.asarray(goal) + Rd @ np.asarray(dp.DOWN_CONTACT_TOOL) == pytest.approx(tip)
    # TOUCH: straight down, no farther than the depth the camera measured plus travel_m
    cfg = load_press_demo(PINK)
    stroke = arm.lines[0]
    assert stroke[:2] == pytest.approx(list(goal[:2])) and goal[2] - stroke[2] == pytest.approx(0.27 + cfg.travel_m)
    g = legs["depth:touch"].guard
    assert g.trip == "touch" and g.gravity is not None and g.reference is not None
    # PUSH: button_travel_m from where the torques felt the lid, backstopped
    assert arm.lines[1][2] == pytest.approx(goal[2] - 0.24 - cfg.button_travel_m) and legs["press:push"].guard.trip == "press"
    # RISE: back to where the touch began, judged for a pull at the fingertip
    assert arm.lines[-1][2] == pytest.approx(goal[2]) and legs["depth:rise"].guard.hang is not None


def test_only_the_lines_up_or_in_place_fly_faster_and_every_guard_is_as_before(ctx, monkeypatch, tmp_path):
    """Run 20:04 flew the turn (3.3 mm of tool travel) in 7.43 s and the rise
    in 5.56 s at CONTACT_SPEED. The turn pivots about the fingertip and the
    rise starts after the at-rest check: FREE_LINE_SPEED. Off the button,
    and the touch and the push, as before."""
    from rammp_box_opening.constants import CONTACT_SPEED, FREE_LINE_SPEED

    cfg = load_press_demo(PINK)
    looks = [(_sighting(x=0.02), 0.27, "2/2"), (_sighting(), 0.27, "2/2"), (_sighting(up=12.0), None, "2/2")]
    result, events, arm, legs, _s = _press(ctx, monkeypatch, tmp_path, looks)
    assert dp.exit_code(result) == 0 and "depth:centre" in events
    speeds = {name: lg.speed for name, lg in legs.items()}
    assert speeds == {
        "depth:turn": FREE_LINE_SPEED, "depth:centre": FREE_LINE_SPEED, "depth:touch": cfg.press_speed,
        "press:push": cfg.press_speed, "depth:unload": CONTACT_SPEED, "depth:rise": FREE_LINE_SPEED,
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
    the caller's read is the reference."""
    here = ([0.5, 0.0, 0.0, 0.0], None)
    short = dp.down_quat(ContainerModel.load(PINK), BUTTON[:2], 2.0)
    looks = [(_sighting(), 0.27, "2/2"), (_sighting(up=12.0), None, "2/2")]
    result, events, arm, legs, _s = _press(ctx, monkeypatch, tmp_path, looks, here=here, turned=short)
    assert dp.exit_code(result) == 0 and arm.turns == [] and "depth:turn" not in events
    assert arm.trace[0] == "look:depth-look (settled)" and not any(t.startswith("rest:over") for t in arm.trace)
    assert legs["depth:touch"].guard.reference == here and legs["depth:touch"].target[2] == pytest.approx(short)


def test_the_touch_measures_how_far_in_front_of_the_lens_the_fingertips_are(ctx, monkeypatch, tmp_path):
    import json

    looks = [(_sighting(lid=0.27), 0.27, "2/2"), (_sighting(up=12.0), None, "2/2")]
    result, _e, _a, _l, _s = _press(ctx, monkeypatch, tmp_path, looks, touch_at=0.24)
    note = [json.loads(line) for line in open(next(tmp_path.glob("run-*.jsonl"))) if '"depth_touch"' in line][0]
    assert note["down_mm"] == pytest.approx(240.0) and note["fingertips_ahead_of_lens_mm"] == pytest.approx(30.0)
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
    assert events[:5] == ["depth:turn", "look:depth-look", "depth:centre", "look:depth-look", "depth:touch"]
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


def test_an_open_box_is_not_pressed(ctx, monkeypatch, tmp_path):
    result, events, _a, _l, _s = _press(ctx, monkeypatch, tmp_path, [(_sighting(up=14.0), 0.26, "2/2")])
    assert result.free and "OPEN" in result.why and "depth:touch" not in events


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


def test_a_box_the_press_moved_stops_the_run(ctx, monkeypatch, tmp_path):
    looks = [(_sighting(), 0.27, "2/2"), (_sighting(x=0.012, up=12.0), None, "2/2")]
    result, _e, _a, _l, _s = _press(ctx, monkeypatch, tmp_path, looks)
    assert result.moved and dp.exit_code(result) == 1


def test_the_reach_out_hands_over_to_the_depth_press_once_over_the_box(ctx, monkeypatch, tmp_path):
    """After the slide in, the TF stroke is not flown: the depth press is."""
    from test_reach_press import _flow

    got = {}

    def press(node, ctx_, cfg, runner, watcher, contact_tool, here=None, turned=None):
        got["contact"] = contact_tool
        return dp.DepthPress(free=True, contact=True, felt=True, popped=True)

    monkeypatch.setattr(dp, "press_down_from_here", press)
    real = rp.run_reach_press

    def with_flag(node, ctx_, cfg, runner, watcher, args, scene_fix, locator):
        return real(node, ctx_, cfg, runner, watcher, SimpleNamespace(execute=True, press_only=True, depth_press=True),
                    scene_fix, locator)

    monkeypatch.setattr(rp, "run_reach_press", with_flag)
    code, rec, _c, _h = _flow(ctx, monkeypatch, tmp_path)
    assert code == 0 and got["contact"] == rp.CONTACT_TOOL
    assert rec.events == ["reach:look", "aim:reach-aim", "reach:re-aim", "reach:slide-in", "home"]
    assert "press:down" not in rec.events


def test_the_reach_out_slides_in_turning_down_in_one_leg(ctx, monkeypatch, tmp_path):
    """Run 20:04: the slide in (4.56 s), two at-rest reads and the turn
    (7.43 s) — one walk does both, as slow and as guarded as the slide."""
    import test_reach_press as trp

    from rammp_box_opening.constants import CONTACT_SPEED

    got = {}

    def press(node, ctx_, cfg, runner, watcher, contact_tool, here=None, turned=None):
        got.update(here=here, turned=turned)
        return dp.DepthPress(free=True, contact=True, felt=True, popped=True)

    monkeypatch.setattr(dp, "press_down_from_here", press)
    walks = []

    def walk(self, q0, xyz, quat, step_m=None, step_rad=None, max_step_rad=None):
        walks.append((list(xyz), list(quat), step_m, step_rad, max_step_rad))
        return [list(q0), [v + 0.01 for v in q0]], None

    monkeypatch.setattr(trp._Chain, "walk", walk, raising=False)
    real = rp.run_reach_press
    monkeypatch.setattr(rp, "run_reach_press", lambda n, c, cfg, r, w, a, f, l: real(
        n, c, cfg, r, w, SimpleNamespace(execute=True, press_only=True, depth_press=True), f, l))
    code, rec, chain, _h = trp._flow(ctx, monkeypatch, tmp_path)
    assert code == 0 and rec.events == ["reach:look", "aim:reach-aim", "reach:re-aim", "reach:slide-in", "home"]
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
    code, rec, _c, _h = trp._flow(ctx, monkeypatch, tmp_path)
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
    monkeypatch.setattr(dp, "find_in_camera", lambda c, d, k, m, dist=None: (
        _sighting(x=0.05 if grab.color_stamp.nanosec == 1000 else 0.0), None))
    for settled in (True, False):
        clock["ns"], grab.color_stamp = 1000, SimpleNamespace(sec=0, nanosec=1000)
        t0 = _time.monotonic()
        s, _z, _lens, status = dp.look(node, SimpleNamespace(grab=grab), None, "t", settled=settled)
        took = _time.monotonic() - t0
        assert s is not None and status == "2/2 frames", (settled, status)
        assert (took < dp.LOOK_SETTLE_S) if settled else (took >= dp.LOOK_SETTLE_S), (settled, took)
