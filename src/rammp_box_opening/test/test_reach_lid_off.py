"""The reach-out's lid off (tasks/reach_lid_off): the compliant reach-out
press as the whole run — the knob looked at from the look pose, the press
once more where it is not up, the grip, the lift, the carry and the
set-down — with the arm, the cameras and the planner faked; and its
reachability on flatbot's URDF for the boxes of 2026-10-27/28."""

import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

import test_reach_press as trp
from rammp_box_opening.models.container import ContainerModel, ContainerPose, load_press_demo
from rammp_box_opening.tasks import depth_press as dp
from rammp_box_opening.tasks import reach_lid_off as rlo
from rammp_box_opening.tasks import reach_press as rp

PINK = trp.PINK
BUTTON = trp.BUTTON  # (0.66, -0.57, 0.40): the fake fk's box
UP = (BUTTON[0], BUTTON[1], BUTTON[2] + 0.009)  # the aim's press point on a popped knob: its top
KNOB_ON = 0.39  # the knuckle closed on the knob (bench 2026-08-26: 0.387)
AIR = 0.79  # ... closed on air


def test_the_grip_goes_by_the_lid_no_lower_than_the_held_pads_nor_higher_than_the_buttons_stop():
    """The lid is not below the button the hold held the pads on, nor more
    than the button's travel above it; between, the press's top. Without a
    hold, the press's top."""
    assert rlo.grip_lid_z(0.3097) == 0.3097
    assert rlo.grip_lid_z(0.3097, 0.3056) == 0.3097  # 13:43's: the scene's top within the button's travel
    assert rlo.grip_lid_z(0.300, 0.3056) == 0.3056  # a top below the held pads is no lid
    assert rlo.grip_lid_z(0.325, 0.3056) == pytest.approx(0.3056 + rlo.KNOB_STROKE_M)  # past the stop is no lid


class _Chain(trp._Chain):
    """test_reach_press's fake chain, with an IK for the lid's spot."""

    def __init__(self, refuse=(), unreachable=()):
        super().__init__(refuse)
        self.unreachable = unreachable
        self.solved = []

    def nearest_solution(self, q_start, xyz, quat, seeds=()):
        self.solved.append(list(xyz))
        if any(what(xyz) for what in self.unreachable):
            return None
        return np.zeros(7)


class _Rec(trp._Rec):
    def __init__(self):
        super().__init__()
        self.plans = []  # (leg name, how many events came before it was planned)


def _walk_or_turn(ctx_, q_look, slide, start, quat, top_z=None):
    """reach_press.walk_or_turn for the fake chain: the turn in place where
    the slide ends, as test_reach_press._way_down's."""
    turn = [[float(v) for v in slide[-1]], [float(v) + 0.01 for v in slide[-1]]]
    return dp.WayDown(turn, list(trp.DOWN_QUAT), 0.0, False), None


def _full_flow(ctx, monkeypatch, tmp_path, aims, knuckle=KNOB_ON, slipped=False, press_only=False, chain=None,
               extent=None, ends=None, home_refusals=0, wrist=None, scene=None, refuse_carry=False):
    """run_reach_press as a full run (compliant --depth-press, no
    --press-only), everything faked: `aims` the wrist's aims in order, each
    (press point, knob mm) or None — the look's, then each after a press
    (and its recheck); `knuckle` what closing on the knob reads, `slipped`
    whether the fingers read that still at the table; `extent` the table
    the scene camera saw (no room map: the lid goes beside the box, in its
    zone); `ends`: {n: (outcome, info)} for the n-th goal sent; `wrist`
    the wrist's points at the spot look, `scene` the scene camera's (None:
    no camera data). Returns
    (exit code, the record, the chain, the run log's rows)."""
    from conftest import leg as make_leg

    from rammp_box_opening.runtime import approach
    from rammp_box_opening.runtime.runner import Runner

    monkeypatch.setenv("RAMMP_BOX_OPENING_STATE", str(tmp_path))  # never the robot's own gripper.json
    rec = _Rec()
    cfg = load_press_demo(PINK)
    ctx.model = ContainerModel.load(PINK)
    ctx.press_cfg = cfg
    ctx.lid_drop = ctx.lid_drops = None  # the run resolves its own
    ctx.worlds.table_extent = extent
    chain = chain or _Chain()
    monkeypatch.setattr(approach, "_chain", lambda: chain)
    monkeypatch.setattr(approach, "_gravity", lambda: trp._Gravity())
    monkeypatch.setattr(rp, "oblique_aim_available", lambda: True)
    refusals = {"home": home_refusals}

    def plan(ctx_, state, name, target, world, speed):
        rec.plans.append((name, len(rec.events)))
        if name == "lid:carry" and refuse_carry:
            raise RuntimeError("planning failed for lid:carry: MotionGenStatus.IK_FAIL")
        if name == "home" and refusals["home"] > 0:
            refusals["home"] -= 1
            raise RuntimeError("planning failed for home: MotionGenStatus.INVALID_START_STATE_WORLD_COLLISION")
        start = list(state.joints)
        lg = make_leg(name, start=start, end=[0.0] * 7 if name == "home" else [v + 0.05 for v in start], speed=speed,
                      world=world[0])
        lg.target = target
        return lg, state

    monkeypatch.setattr(rp, "_plan_motion", plan)
    monkeypatch.setattr(rp, "way_down", trp._way_down([], lambda tilt, tip: True))
    monkeypatch.setattr(rp, "walk_or_turn", _walk_or_turn)
    # the arm held on the button 32 mm above the stroke's end, the way up back along it
    monkeypatch.setattr(dp, "rise_back_up", lambda chain_, stroke, q, join_above_m=None: [list(q), list(stroke.q0)])
    held_pad = (BUTTON[0], BUTTON[1], BUTTON[2] - 0.006)

    def hold(ctx_, cfg_, runner_, stroke, top_z, seen):
        rec.events.append("hold")
        seen.update(held_joints=[float(v) + 0.002 for v in stroke.legs[-1][1][-1]], touch_xyz=held_pad)
        return dp.DepthPress(free=True, contact=True, felt=True, judged=False, press_n=15.0)

    monkeypatch.setattr(dp, "compliant_hold", hold)
    monkeypatch.setattr(rlo, "spot_look", lambda node, watcher: (rec.events.append("spot-look"), wrist)[1])

    class R(Runner):
        def run(self, legs, execute, lookahead=None):
            for lg in legs:
                rec.events.append(lg.name)
                rec.legs[lg.name] = lg
            sent = len(getattr(self.client, "gains_sent", []))
            out = super().run(legs, execute, lookahead)
            rec.runs.append(([lg.name for lg in legs], getattr(self.client, "gains_sent", [])[sent:]))
            return out

    runner = R(ctx.client, log_dir=tmp_path)
    queue = list(aims)

    def aim(node, watcher, yaw, lid_z, prior_xy, runner_, tag):
        rec.events.append("aim:" + tag)
        got = queue.pop(0) if queue else None
        if got is None:
            return None, "no circle", None, None
        return (got[0], 0.0), "3/3 frames", got[1], got[0]

    monkeypatch.setattr(rp, "_aim", aim)
    held = {"on": True}

    def gripper_cmd(pos):
        if pos is None:  # query only
            return True, (AIR if slipped else knuckle) if held["on"] else 0.0, False
        held["on"] = pos > 0.5
        return True, knuckle if pos > 0.5 else 0.0, False

    ctx.client.gripper_cmd = gripper_cmd
    ctx.client.lets_go = []
    ctx.client.release = lambda: ctx.client.lets_go.append(len(ctx.client.executed)) or True
    if ends:
        trp._goal_ends(ctx, monkeypatch, ends)
    fix = SimpleNamespace(pose=ContainerPose(xyz=(BUTTON[0], BUTTON[1], BUTTON[2] - 0.121), yaw=0.0), top_xyz=BUTTON,
                          table_points=scene, table_z_scene=BUTTON[2] - 0.121)
    locator = SimpleNamespace(owl=SimpleNamespace(enable=lambda: None, disable=lambda: None), prime=lambda: None,
                              locate=lambda s: (rec.events.append("scene"), fix)[1])
    args = SimpleNamespace(execute=True, press_only=press_only, depth_press=True, compliant=True)
    watcher = SimpleNamespace(table_z=BUTTON[2] - 0.121)
    code = rp.run_reach_press(None, ctx, cfg, runner, watcher, args, fix, locator)
    rows = []
    for f in sorted(tmp_path.glob("run-*.jsonl")):
        rows += [json.loads(line) for line in open(f)]
    return code, rec, chain, rows


WAY_IN = ["reach:re-aim", "reach:slide-in", "depth:turn", "depth:press"]
BACK = ["depth:rise", "depth:turn-back", "reach:slide-out"]
GRIP = ["grip:open", "reach:re-aim", "reach:slide-in", "depth:turn", "grip:approach", "grip:down"]
CLOSE = ["grip:close", "lid:lift", "lid:clear"]
CARRY = ["lid:carry", "place:down", "place:touch"]
HOME = ["place:open", "place:retreat", "home"]


def _executions(rec):
    """Each runner.run's legs with the gains of every execution it sent."""
    return [(names, gains) for names, gains in rec.runs]


def test_a_knob_up_after_the_press_is_gripped_lifted_carried_set_down_and_the_arm_goes_home(
        ctx, monkeypatch, tmp_path, capsys):
    """The full run: the way in, the hold, the way back to the look pose
    (the rise and the way down backwards), the look, then the grip's way
    down with the fingers opening on it, the close, the lift, the clear, the
    carry and the set-down, the release, the retreat and home — six
    executions after the look's, every one RIGID and unguarded, the lid's
    way to the table planned while the hand came down to the knob."""
    code, rec, chain, rows = _full_flow(ctx, monkeypatch, tmp_path, [(BUTTON, 0.4), (UP, 9.0)])
    assert code == 0
    assert rec.events == ["reach:look", "aim:reach-aim", *WAY_IN, "hold", *BACK, "aim:reach-after", *GRIP, *CLOSE, "spot-look", *CARRY,
                          *HOME]
    runs = _executions(rec)
    assert [names for names, _g in runs] == [["reach:look"], WAY_IN, BACK, GRIP, CLOSE, CARRY, HOME]
    # one goal each, every one under the run's one gains spec
    for names, gains in runs[1:]:
        assert gains == [dp.COMPLIANT_GAINS], names
    assert all(lg.guard is None for lg in rec.legs.values() if lg.kind.value == "motion")
    # the carry planned at the spot look, with the lid lifted; home while the carry flew
    planned = {name: at for name, at in rec.plans}
    assert rec.events.index("spot-look") < planned["lid:carry"] <= rec.events.index("lid:carry")
    assert rec.events.index("place:touch") < planned["home"] <= rec.events.index("place:open")
    # nothing chose the lid's spot before the lift: no camera data, so beside the box
    assert [r["leg"] for r in rows].index("lid_spot") > [r["leg"] for r in rows].index("reach_grip_result")
    spot, = [r for r in rows if r["leg"] == "lid_spot"]
    assert spot["tier"] == "beside the box" and spot["mapped"] is False
    assert spot["xyz"] == pytest.approx(list(ctx.lid_drop.xyz), abs=1e-4)
    hover = rec.legs["lid:carry"].target[1]
    assert hover[:2] == pytest.approx(list(ctx.lid_drop.xyz[:2]))
    assert ctx.lid_at is ctx.lid_drop  # the placed lid is in every world after the release
    knob, = [r for r in rows if r["leg"] == "reach_knob"]
    assert (knob["press"], knob["popped"], knob["looks"], knob["knob_up_mm"]) == (1, True, 1, 9.0)
    # the grip goes by the lid where the hold held the pads (6 mm below the
    # top) plus the button's stop, under the top: the top
    grip, = [r for r in rows if r["leg"] == "reach_grip"]
    assert grip["lid_z"] == pytest.approx(BUTTON[2]) and grip["pads_z"] == pytest.approx(BUTTON[2] - rlo.GRIP_BELOW_LID_M)
    out = capsys.readouterr().out
    assert "LID GRIPPED" in out and out.rstrip().endswith("arm home")
    assert ctx.client.lets_go == []  # nothing stopped: nothing let go


def test_a_lid_that_rose_with_its_knob_up_is_the_box_open_and_the_grip_follows_it_up(ctx, monkeypatch, tmp_path,
                                                                                       capsys):
    """2026-10-28 16:15:19: the press let the lid go and it rose 9.6 mm, the
    button 5.8 mm from where it was aimed, its knob 7.8 mm up — read as "the
    press moved the box" and the arm went home with the box open. A lid
    LID_POP_M higher with the button in place is the box open, and its knob
    up is something to hold: no press once more, and the grip goes by the
    lid where it now stands."""
    risen = (BUTTON[0] + 0.0058, BUTTON[1], BUTTON[2] - 0.0004 + 0.0096 + 0.0078)  # on the first aim's lid
    code, rec, _chain, rows = _full_flow(ctx, monkeypatch, tmp_path, [(BUTTON, 0.4), (risen, 7.8)])
    assert code == 0
    runs = _executions(rec)
    assert [names for names, _g in runs] == [["reach:look"], WAY_IN, BACK, GRIP, CLOSE, CARRY, HOME]  # pressed once
    knob, = [r for r in rows if r["leg"] == "reach_knob"]
    assert (knob["popped"], knob["lid_rose"], knob["knob_up_mm"], knob["stop"]) == (True, True, 7.8, None)
    assert knob["lid_dz_mm"] == pytest.approx(9.6)
    grip, = [r for r in rows if r["leg"] == "reach_grip"]
    assert grip["lid_z"] == pytest.approx(BUTTON[2] + 0.0096)  # the held-pads lid (the top) raised by the rise
    out = capsys.readouterr().out
    assert "knob UP 7.8 mm, and its LID ROSE +10 mm — the box is open" in out and "LID GRIPPED" in out


@pytest.mark.parametrize("knob_mm, dz, shift", [(-0.6, 0.0082, 0.0046), (0.9, 0.0094, 0.0030)],
                         ids=["16:23:53", "16:15:52"])
def test_a_lid_that_rose_with_its_knob_flush_is_neither_gripped_nor_pressed_again_the_arm_goes_home(
        ctx, monkeypatch, tmp_path, capsys, knob_mm, dz, shift):
    """2026-10-28 16:23:53: the press let the lid go and it rose 8.2 mm, the
    button 4.6 mm from where it was aimed, its knob -0.6 mm above it.
    Gripped as the box open, the fingers closed on air (0.768 against the
    band [0.30, 0.55]), opened, backed out and went home: a flush knob gives
    them nothing to hold — nor would 16:15:52's (the lid 9.4 mm up, the
    button 3.0 mm off, its knob 0.9). The look stops the run: home from the
    look pose, the fingers never opened, no press once more (a released lid
    is not pressed again), exit 1 — the record keeps the rise and says why."""
    risen = (BUTTON[0] + shift, BUTTON[1], BUTTON[2] - 0.0004 + dz + knob_mm / 1000)  # on the first aim's lid
    aims = [(BUTTON, 0.4), (risen, knob_mm), (risen, knob_mm)]
    code, rec, _chain, rows = _full_flow(ctx, monkeypatch, tmp_path, aims, knuckle=AIR)
    assert code == 1
    assert rec.events == ["reach:look", "aim:reach-aim", *WAY_IN, "hold", *BACK, "aim:reach-after", "aim:reach-after",
                          "home"]
    assert ctx.client.gripper_sent == [] and ctx.client.lets_go == []
    knob, = [r for r in rows if r["leg"] == "reach_knob"]
    assert (knob["popped"], knob["lid_rose"], knob["knob_up_mm"], knob["looks"]) == (False, True, knob_mm, 2)
    assert knob["lid_dz_mm"] == pytest.approx(1000 * dz) and knob["shift_mm"] == pytest.approx(1000 * shift)
    assert knob["stop"].startswith("the lid rose %+.0f mm (released) but its knob is flush" % (1000 * dz))
    assert not [r for r in rows if r["leg"] in ("reach_grip", "reach_grip_result", "lid_spot")]
    out = capsys.readouterr().out
    assert "knob %.1f mm above the lid, and its LID ROSE %+.0f mm — the box is open" % (knob_mm, 1000 * dz) in out
    assert ("STOP: the lid rose %+.0f mm (released) but its knob is flush (%.1f mm above the lid) — the fingers have "
            "nothing to hold, so the lid is not taken off; the box is left open" % (1000 * dz, knob_mm)) in out
    assert "pressing once more" not in out and "GRIP" not in out


def test_a_lid_that_rose_at_the_press_once_more_with_its_knob_flush_stops_the_run_there(ctx, monkeypatch, tmp_path,
                                                                                      capsys):
    """Not up after the first press, its lid where it was: pressed once more
    (owner 2026-10-28). That press lets the lid go, its knob flush: the same
    stop — home from the look pose, nothing gripped."""
    risen = (BUTTON[0], BUTTON[1], BUTTON[2] + 0.0099)  # the lid +9.4 mm on the first aim's -0.4, its knob 0.9 above
    aims = [(BUTTON, 0.4), (BUTTON, 2.0), (BUTTON, 1.5), (risen, 0.9), (risen, 0.9)]
    code, rec, _chain, rows = _full_flow(ctx, monkeypatch, tmp_path, aims)
    assert code == 1
    press = [*WAY_IN, "hold", *BACK]
    assert rec.events == ["reach:look", "aim:reach-aim", *press, "aim:reach-after", "aim:reach-after", *press,
                          "aim:reach-after", "aim:reach-after", "home"]
    knobs = [(r["press"], r["popped"], r["lid_rose"], r["stop"] is None) for r in rows if r["leg"] == "reach_knob"]
    assert knobs == [(1, False, False, True), (2, False, True, False)]
    out = capsys.readouterr().out
    assert "pressing once more" in out and "STOP: the lid rose +9 mm (released) but its knob is flush" in out
    assert "pressed 2 times" not in out and ctx.client.gripper_sent == []


@pytest.mark.parametrize("dz", [-0.009, 0.040])
def test_a_lid_lower_or_far_higher_after_the_press_is_still_the_box_moved(ctx, monkeypatch, tmp_path, dz):
    """Lower than before the press, or higher than a released lid rises
    (LID_POP_M): the press moved the box — home, nothing gripped."""
    after = (BUTTON[0], BUTTON[1], BUTTON[2] - 0.0004 + dz + 0.0009)
    code, rec, _chain, rows = _full_flow(ctx, monkeypatch, tmp_path, [(BUTTON, 0.4), (after, 0.9), (after, 0.9)])
    assert code == 1
    knob, = [r for r in rows if r["leg"] == "reach_knob"]
    assert (knob["popped"], knob["lid_rose"]) == (False, False)
    assert knob["stop"].startswith("the press moved the box: its lid")
    assert not any(n.startswith("grip:") for n in rec.events)


def test_a_knob_not_up_is_looked_at_again_then_pressed_once_more_and_its_lid_taken_off(ctx, monkeypatch, tmp_path,
                                                                                       capsys):
    """Not up after the press, nor at the look again: the same press once
    more from the look pose, its way in solved again from where the arm
    stands, the hold, the way back and the look; up — the lid comes off."""
    aims = [(BUTTON, 0.4), (BUTTON, 2.0), (BUTTON, 1.5), (UP, 8.0)]
    code, rec, _chain, rows = _full_flow(ctx, monkeypatch, tmp_path, aims)
    assert code == 0
    press = [*WAY_IN, "hold", *BACK]
    assert rec.events == ["reach:look", "aim:reach-aim", *press, "aim:reach-after", "aim:reach-after", *press,
                          "aim:reach-after", *GRIP, *CLOSE, "spot-look", *CARRY, *HOME]
    knobs = [(r["press"], r["popped"], r["looks"], r["knob_up_mm"]) for r in rows if r["leg"] == "reach_knob"]
    assert knobs == [(1, False, 2, 2.0), (2, True, 1, 8.0)]
    assert "pressing once more — the same press, from the look pose" in capsys.readouterr().out


def test_a_knob_still_not_up_after_the_second_press_sends_the_arm_home_with_nothing_gripped(ctx, monkeypatch,
                                                                                         tmp_path, capsys):
    aims = [(BUTTON, 0.4), (BUTTON, 2.0), (BUTTON, 1.5), (BUTTON, 3.0), (BUTTON, 2.5)]
    code, rec, _chain, _rows = _full_flow(ctx, monkeypatch, tmp_path, aims)
    assert code == 1
    press = [*WAY_IN, "hold", *BACK]
    assert rec.events == ["reach:look", "aim:reach-aim", *press, "aim:reach-after", "aim:reach-after", *press,
                          "aim:reach-after", "aim:reach-after", "home"]
    out = capsys.readouterr().out
    assert "STOP: pressed 2 times and the knob is not up (3.0 mm above the lid) — the lid is not taken off" in out
    assert ctx.client.gripper_sent == []  # the fingers never opened


def test_fingers_that_close_on_air_open_again_and_back_out_the_way_they_came(ctx, monkeypatch, tmp_path, capsys):
    """The close reads air (outside the grip band): nothing is lifted. The
    fingers open, the hand backs out the grip's way down to the look pose,
    and goes home from there: exit 1."""
    code, rec, _chain, rows = _full_flow(ctx, monkeypatch, tmp_path, [(BUTTON, 0.4), (UP, 9.0)], knuckle=AIR)
    assert code == 1
    back_out = ["grip:up", "depth:turn-back", "reach:slide-out"]
    assert rec.events == ["reach:look", "aim:reach-aim", *WAY_IN, "hold", *BACK, "aim:reach-after", *GRIP, *CLOSE,
                          "grip:open", *back_out, "home"]
    assert (CLOSE, []) in _executions(rec)  # the lift's run returned at the close: handed, nothing sent
    assert [r["leg"] for r in rows if r["leg"].startswith(("lid:", "place:"))] == []
    result, = [r for r in rows if r["leg"] == "reach_grip_result"]
    assert result["ok"] is False and "0.790" in result["detail"]
    assert "GRIP FAILED" in capsys.readouterr().out


def test_a_lid_that_slipped_on_the_way_is_released_the_arm_goes_home_and_the_run_says_so(ctx, monkeypatch, tmp_path,
                                                                                         capsys):
    code, rec, _chain, rows = _full_flow(ctx, monkeypatch, tmp_path, [(BUTTON, 0.4), (UP, 9.0)], slipped=True)
    assert code == 1 and rec.events[-3:] == HOME
    setdown, = [r for r in rows if r["leg"] == "reach_setdown"]
    assert setdown["kept"] is False
    assert "the lid slipped out of the fingers" in capsys.readouterr().out


def test_a_box_the_press_moved_is_not_gripped(ctx, monkeypatch, tmp_path, capsys):
    moved = (BUTTON[0] + 0.02, BUTTON[1], BUTTON[2] + 0.009)
    code, rec, _chain, _rows = _full_flow(ctx, monkeypatch, tmp_path, [(BUTTON, 0.4), (moved, 9.0)])
    assert code == 1 and rec.events[-2:] == ["aim:reach-after", "home"]
    assert "STOP: the press MOVED the box 20 mm" in capsys.readouterr().out


def test_a_box_the_wrist_no_longer_finds_after_the_press_is_not_pressed_again(ctx, monkeypatch, tmp_path, capsys):
    code, rec, _chain, _rows = _full_flow(ctx, monkeypatch, tmp_path, [(BUTTON, 0.4), None, None])
    assert code == 1 and rec.events[-3:] == ["aim:reach-after", "aim:reach-after", "home"]
    assert rec.events.count("depth:press") == 1
    assert "the knob is not judged" in capsys.readouterr().out


def test_an_open_box_has_its_lid_taken_off_unpressed(ctx, monkeypatch, tmp_path, capsys):
    """The aim reads the knob up before anything is pressed: pressing it
    would shut the box. The full run takes the lid off from the look pose,
    no stroke flown."""
    code, rec, _chain, rows = _full_flow(ctx, monkeypatch, tmp_path, [(UP, 9.0)])
    assert code == 0
    assert rec.events == ["reach:look", "aim:reach-aim", *GRIP, *CLOSE, "spot-look", *CARRY, *HOME]
    assert [(r["button_above_lid_mm"], r["opened"]) for r in rows if r["leg"] == "open_box"] == [(9.0, True)]
    grip, = [r for r in rows if r["leg"] == "reach_grip"]
    assert grip["lid_z"] == pytest.approx(BUTTON[2])  # the scene's top, above the aim's knob less its height
    assert "ALREADY OPEN" in capsys.readouterr().out


def test_with_press_only_an_open_box_still_stops_the_run(ctx, monkeypatch, tmp_path):
    code, rec, _chain, _rows = _full_flow(ctx, monkeypatch, tmp_path, [(UP, 9.0)], press_only=True)
    assert code == 1 and rec.events == ["reach:look", "aim:reach-aim", "home"]


# ---- the lid's spot, chosen with the lid lifted: never a refusal ---------------

BOX_XY = BUTTON[:2]
SMALL = (BOX_XY[0] - 0.22, BOX_XY[0] + 0.22, BOX_XY[1] - 0.22, BOX_XY[1] + 0.22)  # no spot beside the box fits
ROOMY = (BOX_XY[0] - 0.6, BOX_XY[0] + 0.6, BOX_XY[1] - 0.6, BOX_XY[1] + 0.6)
TABLE_Z = BUTTON[2] - 0.121


def _table(xy=BOX_XY, r=0.6, step=0.01, z=TABLE_Z):
    """A flat table's points round `xy`, and the box standing on it."""
    g = np.arange(-r, r + 1e-9, step)
    X, Y = np.meshgrid(g + xy[0], g + xy[1], indexing="ij")
    flat = np.stack([X.ravel(), Y.ravel(), np.full(X.size, z)], -1)
    box = flat[np.hypot(flat[:, 0] - xy[0], flat[:, 1] - xy[1]) < 0.0525].copy()
    flat = flat[np.hypot(flat[:, 0] - xy[0], flat[:, 1] - xy[1]) >= 0.0525]
    box[:, 2] = z + 0.121
    return np.concatenate([flat, box])


def _wall(pts, inner=0.15, outer=0.45, tall=0.30):
    """`pts` with everything from `inner` to `outer` round the box standing
    `tall`: no room for the lid anywhere the hand reaches."""
    pts = pts.copy()
    d = np.hypot(pts[:, 0] - BOX_XY[0], pts[:, 1] - BOX_XY[1])
    pts[(d > inner) & (d < outer), 2] = TABLE_Z + tall
    return pts


def test_a_small_table_no_longer_stops_the_run_before_anything_moves(ctx, monkeypatch, tmp_path, capsys):
    """2026-10-28: five full runs stopped "nowhere to set the lid down"
    before any motion. Nothing is resolved at the start now: the lid comes
    off, and with no spot beside the box in the set-down zone and no camera
    data, it goes to the nearest table point the hand reaches, off the box —
    placed, exit 0, and said once, as information."""
    code, rec, _chain, rows = _full_flow(ctx, monkeypatch, tmp_path, [(BUTTON, 0.4), (UP, 9.0)], extent=SMALL)
    assert code == 0 and rec.events[:2] == ["reach:look", "aim:reach-aim"] and rec.events[-3:] == HOME
    spot, = [r for r in rows if r["leg"] == "lid_spot"]
    assert spot["tier"] == "nearest"
    out = capsys.readouterr().out
    assert "nowhere to set the lid down" not in out and "STOP" not in out
    assert "the nearest table point the hand reaches, off the box" in out
    x0, x1, y0, y1 = SMALL
    assert x0 < ctx.lid_drop.xyz[0] < x1 and y0 < ctx.lid_drop.xyz[1] < y1  # on the table
    assert math.hypot(ctx.lid_drop.xyz[0] - BOX_XY[0], ctx.lid_drop.xyz[1] - BOX_XY[1]) > 0.105  # off the box


def test_a_press_only_run_chooses_no_lid_spot(ctx, monkeypatch, tmp_path, capsys):
    code, rec, _chain, rows = _full_flow(ctx, monkeypatch, tmp_path, [(BUTTON, 0.4)], extent=SMALL, press_only=True)
    assert code == 0 and ctx.lid_drop is None and not [r for r in rows if r["leg"] == "lid_spot"]
    assert "spot-look" not in rec.events


def test_a_clear_spot_the_wrist_sees_after_the_lift_is_where_the_lid_goes(ctx, monkeypatch, tmp_path, capsys):
    code, rec, _chain, rows = _full_flow(ctx, monkeypatch, tmp_path, [(BUTTON, 0.4), (UP, 9.0)], extent=ROOMY,
                                         wrist=_table())
    assert code == 0
    spot, = [r for r in rows if r["leg"] == "lid_spot"]
    assert spot["tier"] == "clear" and spot["mapped"] is True and spot["wrist_points"] > 0
    assert "where the cameras see room for it and the hand" in capsys.readouterr().out
    d = math.hypot(ctx.lid_drop.xyz[0] - BOX_XY[0], ctx.lid_drop.xyz[1] - BOX_XY[1])
    assert d >= 0.19  # lid_place_min_clear from the box
    seen, = [r for r in rows if r["leg"] == "seen_obstacles"]  # the carry's world from the same map
    assert seen["when"] == "wrist and scene camera, the lid lifted"


def test_nothing_clear_still_sets_the_lid_down_at_the_most_room_seen(ctx, monkeypatch, tmp_path, capsys):
    code, rec, _chain, rows = _full_flow(ctx, monkeypatch, tmp_path, [(BUTTON, 0.4), (UP, 9.0)], extent=ROOMY,
                                         wrist=_wall(_table()))
    assert code == 0 and rec.events[-3:] == HOME
    spot, = [r for r in rows if r["leg"] == "lid_spot"]
    assert spot["tier"] == "most room"
    out = capsys.readouterr().out
    assert "nothing reads clear" in out and "STOP" not in out and out.rstrip().endswith("arm home")


def test_no_camera_data_still_sets_the_lid_down_beside_the_box(ctx, monkeypatch, tmp_path, capsys):
    code, _rec, _chain, rows = _full_flow(ctx, monkeypatch, tmp_path, [(BUTTON, 0.4), (UP, 9.0)], extent=ROOMY)
    assert code == 0
    spot, = [r for r in rows if r["leg"] == "lid_spot"]
    assert (spot["tier"], spot["mapped"], spot["wrist_points"]) == ("beside the box", False, 0)
    assert "no camera shows the table: beside the box" in capsys.readouterr().out


def test_the_wrist_points_of_the_held_lid_are_not_an_obstacle(ctx, monkeypatch):
    """At the spot look the wrist sees the lid in the fingers, ~0.1 m over
    the table under the hand: taken out (wrist_table), it stands nowhere on
    the map; left in, it would stand there and block the spots beside it."""
    from rammp_box_opening.perception import clear_table
    from rammp_box_opening.runtime import approach

    chain = _Chain()  # fk: the tool at [0.5, -0.4, 0.45], the hand straight up (R eye): pads 11 mm above it
    monkeypatch.setattr(approach, "_chain", lambda: chain)
    ctx.model = ContainerModel.load(PINK)
    ctx.cpose = ContainerPose(xyz=(BOX_XY[0], BOX_XY[1], TABLE_Z), yaw=0.0)
    table = _table()
    pads = np.array([0.5, -0.4])
    g = np.arange(-0.05, 0.0501, 0.004)
    X, Y = np.meshgrid(g + pads[0], g + pads[1], indexing="ij")
    disc = np.hypot(X - pads[0], Y - pads[1]) < 0.0525
    lid = np.stack([X[disc], Y[disc], np.full(int(disc.sum()), TABLE_Z + 0.10)], -1)
    held = np.concatenate([table, lid])
    scene = SimpleNamespace(pose=ctx.cpose, table_points=table, table_z_scene=TABLE_Z)

    def standing_at_pads(tmap):
        X_, Y_ = tmap.centres()
        near = np.hypot(X_ - pads[0], Y_ - pads[1]) < 0.04
        return int(np.isfinite(tmap.tall[near]).sum())

    raw = clear_table.table_map([(table, TABLE_Z), (held, TABLE_Z)], BOX_XY)
    assert standing_at_pads(raw) > 0  # counted, the held lid stands on the map
    tmap = rlo.spot_map(ctx, held, [0.0] * 7, scene, None, TABLE_Z, lens_in_hand=(0.0, 0.0, 0.5))
    assert standing_at_pads(tmap) == 0
    assert len(rlo.wrist_table(ctx, held, [0.0] * 7, (0.0, 0.0, 0.5))) < len(table)  # the lid's column and the box out


def test_no_carry_that_plans_holds_the_lid_and_says_so(ctx, monkeypatch, tmp_path, capsys):
    """Every spot the hand reaches refused by the planner: the run stops
    holding the lid (exit 1) after CARRY_TRIES, not before the run."""
    code, rec, _chain, rows = _full_flow(ctx, monkeypatch, tmp_path, [(BUTTON, 0.4), (UP, 9.0)], extent=ROOMY,
                                         refuse_carry=True)
    assert code == 1 and "lid:carry" not in rec.events
    spot, = [r for r in rows if r["leg"] == "lid_spot"]
    assert spot["xyz"] is None and spot["refused"].count("IK_FAIL") == rlo.CARRY_TRIES
    assert "STOP: no carry to a spot the lid could go to plans" in capsys.readouterr().out


# ---- flatbot's URDF: the five boxes of the owner's brief ------------------------

FLATBOT_BOXES = [(0.51, -0.27), (0.57, -0.19), (0.716, -0.13), (0.668, -0.33), (0.60, -0.49)]
LID = 0.3125  # 16:23's lid; the scene's table ~0.10 below it


def _held_at_bottom(chain, stroke, top_z, cfg):
    """The press leg's sample whose pads stand nearest the button's bottom:
    where the spring holds the arm on a button that stops it there."""
    pts = stroke.legs[-1][1]
    bottom = top_z - cfg.button_travel_m
    return min(pts, key=lambda q: abs(dp._pads_z(chain, q) - bottom))


@pytest.mark.parametrize("box", FLATBOT_BOXES, ids=["%.3f,%.2f" % b for b in FLATBOT_BOXES])
def test_on_flatbot_the_whole_lid_off_is_reachable_for_every_box_of_the_brief(ctx, monkeypatch, tmp_path, box):
    """Flatbot's own kinematics, the pink canister's lid at 0.3125: the
    press is proven at 30 deg; from the hold the way back to the look pose
    — the rise and the way down backwards — is one the runner flies; the
    look pose sees the button at the frame's centre ~0.30 m off the lens,
    where the straight-down depth look could not stand (no turned-down hand
    over the far boxes rises the 56-69 mm it needs above the stroke's
    start); the grip comes down to the lid; the lift, the clear, the carry
    (IK standing in for the planner), the set-down and the retreat are
    lines the runner flies; and a spot beside the box is one the hand
    reaches straight down."""
    from rammp_box_opening.runtime.runner import Runner
    from rammp_box_opening.tasks import press_demo

    chain, cfg, home = trp._on_flatbot(ctx, monkeypatch)
    button = (box[0], box[1], LID)
    ctx.cpose = rp.cpose_for_button(button, 0.0, ctx.model.button_offset)
    plan, why = rp.plan_reach(ctx, cfg, button, home, depth=True)
    assert why is None and plan.tilt_deg == 30
    way, why, _kind = rp.solve_way_in(ctx, cfg, button, plan.tilt_deg, plan.look_leg.goal_joints)
    assert way is not None, why
    runner = Runner(ctx.client, log_dir=tmp_path)
    q_held = _held_at_bottom(chain, way.stroke, LID, cfg)
    back, kind, why = rlo.way_back_legs(ctx, way, q_held)
    assert kind is None, why
    assert [(lg.name, runner._refusal(lg)) for lg in back] == [(lg.name, None) for lg in back]
    assert back[-1].goal_joints == pytest.approx(rlo._look_joints(way))
    assert len(rlo.merge_groups(back)) == 1  # one execution
    # the look: the button at the frame's centre, ~0.30 m off the lens (the wrist's OAK: K of 10-27's frames)
    R, t = chain.fk(rlo._look_joints(way))
    lens = np.asarray(t) + np.asarray(R) @ np.array([0.0, 0.065, -0.105])
    c = np.asarray(R).T @ (np.asarray(button) - lens)
    assert 0.27 < float(np.linalg.norm(c)) < 0.33 and abs(c[0] / c[2]) < 0.1 and abs(c[1] / c[2]) < 0.1
    if math.hypot(*box) > 0.70:
        # no straight-down depth look at 245 mm from the lid: the hand at the stroke's start rises too little
        R0, t0 = chain.fk(way.stroke.q0)
        lens0 = np.asarray(t0) + np.asarray(R0) @ np.array([0.0, 0.065, -0.105])
        need = 0.245 - (float(lens0[2]) - LID)
        up, _why = chain.straight_line(way.stroke.q0, [t0[0], t0[1], t0[2] + need], way.stroke.quat)
        assert up is None
    # the grip, from the look pose the way back ends at
    lid_z = rlo.grip_lid_z(LID, dp._pads_z(chain, q_held))
    grip, why = rlo.solve_grip_way(ctx, cfg, button, plan.tilt_deg, back[-1].goal_joints, lid_z)
    assert grip is not None, why
    legs = rp.way_in_legs(ctx, cfg, grip)
    assert [(lg.name, runner._refusal(lg)) for lg in legs] == [(lg.name, None) for lg in legs]
    assert dp._pads_z(chain, legs[-1].goal_joints) == pytest.approx(lid_z - rlo.GRIP_BELOW_LID_M, abs=5e-4)
    # the spot: the first beside the box the hand reaches straight down
    spots = press_demo.lid_drop_candidates(ctx.model, ctx.cpose, LID - 0.10)
    spots = [ContainerPose(xyz=tuple(xyz), yaw=0.0) for xyz, _d in spots]
    reached = [d for d in spots if rlo.drop_reachable(ctx, d, home) is None]
    assert reached, [rlo.drop_reachable(ctx, d, home) for d in spots]

    def plan_ik(ctx_, state, name, target, world, speed):
        # the planner stood in for by the arm's IK: a goal it reaches, flown as two points
        if target[0] == "joints":
            q = list(target[1])
        else:
            q = chain.nearest_solution(list(state.joints), target[1], target[2], seeds=(home,))
            assert q is not None, name
        return rp.line_leg(ctx_, name, [list(state.joints), [float(v) for v in q]], target[2] if len(target) > 2 else
                           [0, 0, 0, 1], speed, None, world), state

    monkeypatch.setattr(rp, "_plan_motion", plan_ik)
    lift, why = rlo.plan_lift(ctx, cfg, legs[-1].goal_joints, grip.stroke.quat)
    assert lift is not None, why
    assert [lg.name for lg in lift.legs] == ["lid:lift", "lid:clear"]
    q_look = lift.legs[-1].goal_joints
    # where the lid goes, from the spot look: the first the hand reaches from there
    drop, tier, _says = next(rlo.spot_candidates(ctx, None, LID - 0.10, q_look, seeds=(home,)))
    assert tier == "beside the box"
    carry, why = rlo.plan_carry(ctx, cfg, q_look, drop)
    assert carry is not None, why
    home_legs, why = rlo.plan_home_after(ctx, cfg, carry)
    assert home_legs is not None, why
    names = [lg.name for lg in carry.legs + home_legs]
    assert names[:4] == ["lid:carry", "place:down", "place:touch", "place:retreat"] and names[-1] == "home"
    plan_free = [lg for lg in lift.legs + carry.legs + home_legs if lg.name not in ("lid:carry", "home")]
    assert [(lg.name, runner._refusal(lg)) for lg in plan_free] == [(lg.name, None) for lg in plan_free]
    # the lid lifted straight up off the rim, then up and back toward the arm, clear of the box's padded volume
    lift_leg, clear = lift.legs
    _R, t_grip = chain.fk(legs[-1].goal_joints)
    _R, t_lift = chain.fk(lift_leg.goal_joints)
    assert t_lift[2] - t_grip[2] == pytest.approx(rlo.lift_off_m(ctx.model), abs=1e-4)
    assert t_lift[:2] == pytest.approx(t_grip[:2], abs=1e-4)
    _R, t_clear = chain.fk(clear.goal_joints)
    from rammp_box_opening.worlds import planned_top_z

    assert t_clear[2] >= planned_top_z(ctx.model, ctx.cpose) + rp.HOME_CLEAR_M
    assert math.hypot(*t_clear[:2]) < math.hypot(*t_lift[:2])  # back toward the arm
