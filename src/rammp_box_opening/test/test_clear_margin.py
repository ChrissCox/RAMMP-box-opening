"""home_arm from a start too near a joint limit for the planner.

rammp-curobo refuses a start within joint_limit_margin_rad (0.10) of a URDF
limit, and cannot retime some starts outside it while the wrist is folded,
so home_arm was refused in both worlds from where a run left the arm
(2026-10-23: arm_joint_4 0.047 rad from -2.57, the tool 268 mm above
arm_mount). It now steps plan-free along the joint line toward HOME first,
the shortest step HOME plans from, and plans HOME from where it ends.

rclpy is never initialised: the node and the client are stubs."""

from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from conftest import FakeClient

from rammp_box_opening.constants import HOME
from rammp_box_opening.primitives import core
from rammp_box_opening.primitives.core import ClearMarginFree, PlanState
from rammp_box_opening.runtime import approach
from rammp_box_opening.runtime.kinematics import ArmChain
from rammp_box_opening.tasks import cli_common, home_arm

PKG = Path(__file__).resolve().parents[1]
FLATBOT_URDF = str(PKG / "test" / "data" / "arm_flatbot_20261021.urdf")
FLATBOT_WORLD = str(PKG / "config" / "world_flatbot.yaml")
CONTAINER = str(PKG / "config" / "containers" / "ankou_pink.yaml")

# the live joints home_arm was refused from, 2026-10-23 11:19
STUCK = [0.0617, -2.0574, -3.1385, -2.5233, -0.2102, -1.2174, 1.7257]
LINE = np.array(HOME) - np.array(STUCK)


def _along(q):
    """How far along the STUCK -> HOME joint line q is (0..1)."""
    return float(np.dot(np.array(q) - STUCK, LINE) / np.dot(LINE, LINE))


class _Bad:
    success = False
    message = "MotionGenStatus.FINETUNE_TRAJOPT_FAIL"


class LivePlanner(FakeClient):
    """Plans HOME only from 45% or more of the way along the line, as the
    live planner did (2026-10-23: refused from 0.29-0.42, planned from 0.45)."""

    def plan_to_joints(self, q7, start_joints):
        ok = _along(start_joints) >= 0.45
        res = super().plan_to_joints(q7, start_joints)
        return res if ok else _Bad()


class NeverPlans(FakeClient):
    def plan_to_joints(self, q7, start_joints):
        super().plan_to_joints(q7, start_joints)
        return _Bad()


@pytest.fixture(autouse=True)
def chain_state(monkeypatch, tmp_path):
    monkeypatch.setenv("RAMMP_BOX_OPENING_STATE", str(tmp_path))
    saved = (approach._ROOT, approach._URDF_PATH, approach._CHAIN, approach._GRAVITY)
    yield
    approach._ROOT, approach._URDF_PATH, approach._CHAIN, approach._GRAVITY = saved


def _flatbot_ctx(monkeypatch, live, urdf=FLATBOT_URDF, client=None):
    client = client or FakeClient()
    client.live = list(live)
    monkeypatch.setattr(cli_common, "init_runtime", lambda execute: (object(), client))
    monkeypatch.setattr(cli_common, "_arm_urdf_path", lambda node, frame: urdf)
    ctx, _runner = cli_common.build_ctx(Namespace(container=CONTAINER, bench_world=FLATBOT_WORLD, execute=False))
    return ctx


def test_home_arm_steps_toward_home_until_the_planner_can_plan_it(monkeypatch, capsys):
    ctx = _flatbot_ctx(monkeypatch, STUCK, client=LivePlanner())
    legs = home_arm.build_legs(ctx)
    assert [x.name for x in legs] == ["clear-margin", "home"]
    step = legs[0]
    assert step.guard is not None and step.guard.trip == "obstruction"
    assert step.speed == core.CONTACT_SPEED
    pts = np.array([p.positions for p in step.traj.points])
    assert np.allclose(pts[0], STUCK)
    for q in pts:  # on the straight joint line to HOME
        assert np.allclose(q, np.array(STUCK) + _along(q) * LINE, atol=1e-9)
    # the shortest of the steps it tried that HOME plans from: 0.288 clears
    # the margin, 0.382 is 0.05 rad more, 0.475 is the first at 0.45 or more
    tried = sorted({round(_along(q), 3) for q in ctx.client.joint_starts})
    assert tried == pytest.approx([0.288, 0.382, 0.475], abs=1e-3)
    assert _along(pts[-1]) == pytest.approx(0.475, abs=1e-3)
    assert np.max(np.abs(pts[-1] - STUCK)) == pytest.approx(0.254, abs=1e-3)
    # the tool only rises: +119 mm
    fb = ArmChain(root="arm_mount", path=FLATBOT_URDF)
    z = [fb.fk(q)[1][2] for q in pts]
    assert min(z) == pytest.approx(z[0]) and z[-1] - z[0] == pytest.approx(0.119, abs=0.002)
    # HOME is planned from the step's END, once per step in the bare world:
    # world_flatbot.yaml has no table, so no band to guard
    assert ctx.client.joint_starts[-1] == pytest.approx(list(step.goal_joints))
    assert legs[1].world == "bench_bare" and len(ctx.client.joint_starts) == 3
    out = capsys.readouterr().out
    assert "arm_joint_4" not in out and "joint_4 is 0.047 rad from its limit" in out
    assert "a plan-free step 48% of the way to HOME first (the largest joint 0.254 rad, the tool +119 mm up" in out
    # the no-table caution once, up front; the refused tries say nothing
    assert out.count("CAUTION") == 1 and "CAUTION: no table is known" in out


def test_a_joint_just_outside_the_margin_plans_from_the_start_first(monkeypatch):
    start = list(STUCK)
    start[3] = -2.42  # 0.15 rad of room: outside the margin, inside the trajopt's band
    ctx = _flatbot_ctx(monkeypatch, start)
    assert [x.name for x in home_arm.build_legs(ctx)] == ["home"]
    assert ctx.client.joint_starts == [pytest.approx(start)]


def test_a_start_clear_of_the_limits_is_planned_as_before(monkeypatch):
    ctx = _flatbot_ctx(monkeypatch, HOME)
    assert [x.name for x in home_arm.build_legs(ctx)] == ["home"]
    assert ctx.client.joint_starts == [pytest.approx(HOME)]


def test_without_the_arm_description_the_planner_has_the_last_word(monkeypatch):
    """No URDF, no limits to read: no step, and HOME is asked of the planner
    from the live joints as before (its own refusal names the joint)."""
    ctx = _flatbot_ctx(monkeypatch, STUCK, urdf=None)
    assert [x.name for x in home_arm.build_legs(ctx)] == ["home"]
    assert ctx.client.joint_starts == [pytest.approx(STUCK)]


def test_no_step_up_to_the_cap_that_plans_is_refused(monkeypatch):
    ctx = _flatbot_ctx(monkeypatch, STUCK, client=NeverPlans())
    with pytest.raises(RuntimeError, match="no plan-free step toward HOME of up to 0.35 rad per joint"):
        home_arm.build_legs(ctx)
    largest = max(np.max(np.abs(np.array(q) - STUCK)) for q in ctx.client.joint_starts)
    assert largest <= core.CLEAR_MARGIN_MAX_STEP_RAD + 1e-9


def test_a_line_that_would_lower_the_tool_is_refused(monkeypatch):
    start = list(HOME)
    start[3] = -2.50  # 0.07 rad from its limit
    goal = [0.0617, -1.9, -3.1385, -2.30, -0.2102, -1.4, 1.7257]
    fb = ArmChain(root="arm_mount", path=FLATBOT_URDF)
    q0, q1 = np.array(start), np.array(start) + 0.25 * (np.array(goal) - start)
    z = [fb.fk(q0 + u * (q1 - q0))[1][2] for u in np.linspace(0, 1, 26)]
    assert z[0] - min(z) > 0.02  # the line toward this goal goes down
    ctx = _flatbot_ctx(monkeypatch, start)
    with pytest.raises(RuntimeError, match="drops the tool 21 mm"):
        next(ClearMarginFree(goal).steps(ctx, PlanState(joints=start, chain=0)))


def test_a_goal_too_near_a_limit_is_refused(monkeypatch):
    goal = list(HOME)
    goal[5] = 2.05
    ctx = _flatbot_ctx(monkeypatch, STUCK)
    with pytest.raises(RuntimeError, match="the goal's joint_6 is 0.040 rad from its limit"):
        next(ClearMarginFree(goal).steps(ctx, PlanState(joints=STUCK, chain=0)))


def test_a_chain_rooted_off_the_planning_frame_is_refused(monkeypatch):
    """The drop is the chain root's z: on flatbot base_link leans 30 deg."""
    ctx = _flatbot_ctx(monkeypatch, STUCK)
    approach.set_chain_root("base_link", None)
    with pytest.raises(RuntimeError, match="chain root base_link != planning frame arm_mount"):
        next(ClearMarginFree(HOME).steps(ctx, PlanState(joints=STUCK, chain=0)))
