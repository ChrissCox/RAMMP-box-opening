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


def _press(ctx, monkeypatch, tmp_path, looks, tilt=30, touch_at=0.24, push="touch", refuse_turn=False):
    """press_down_from_here with the hand `tilt` deg over BUTTON (at the
    reach-out's stroke start, 6 cm above TF's lid at 0.40), the wrist camera
    scripted by `looks` and the fingertips meeting the lid `touch_at` down
    the stroke. Returns (DepthPress, events, arm, legs, start tool pose)."""
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
    ctx.client.settled_efforts = lambda *a, **k: ([0.0] * 4, list(ctx.client.live))
    events, legs = [], {}
    seq = list(looks)

    def look(node, watcher, model, tag, runner=None, ctx=None):
        events.append("look:" + tag)
        s, z, status = seq.pop(0)[:3]
        return s, z, LENS, status

    monkeypatch.setattr(dp, "look", look)

    class R(Runner):
        def run(self, legs_, execute, lookahead=None):
            for lg in legs_:
                events.append(lg.name)
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
    result = dp.press_down_from_here(None, ctx, cfg, runner, SimpleNamespace(), contact)
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


ARRIVED = ("arrived", {"message": "ok", "progress": 1.0})


def _depth_flow(ctx, monkeypatch, tmp_path, press, **kw):
    """test_reach_press._flow with --depth-press: `press` stands in for
    press_down_from_here (called with the reach's ctx). The fake chain's fk
    stands at z 0.45 over the aimed button at 0.40."""
    from test_reach_press import _flow

    got = {}

    def fake(node, ctx_, cfg, runner, watcher, contact_tool):
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
    down from 0.46: every world after the touch holds the box there, and
    the table is left as the scene camera measured it."""
    from rammp_box_opening.models.container import from_container

    ctx.model = ContainerModel.load(PINK)
    ctx.cpose = rp.cpose_for_button(BUTTON, 0.3, ctx.model.button_offset)
    ctx.worlds.table_top_z = 0.10
    looks = [(_sighting(), 0.27, "2/2"), (_sighting(up=12.0), None, "2/2")]
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
