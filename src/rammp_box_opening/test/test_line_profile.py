"""A free-air plan-free line at contact speed flies a re-timed profile: no
faster anywhere than the dilated one it replaces, and shorter."""
import pytest

from conftest import FakeClient, runner as make_runner
from rammp_box_opening.constants import CONTACT_SPEED, JOINT_VMAX
from rammp_box_opening.runtime import approach, driver
from rammp_box_opening.runtime.guards import GuardSpec
from rammp_box_opening.runtime.legs import Kind, Leg
from rammp_box_opening.runtime.retime import check_like_executor
from rammp_box_opening.runtime.runner import _path_speed_peak
from rammp_box_opening.runtime.stamps import secs

NAMES = ["joint_%d" % i for i in range(1, 8)]
A = [0.0, 0.3, 0.0, 1.5, 0.0, 0.8, 0.0]
B = [0.1, 0.45, 0.05, 1.3, 0.05, 0.95, 0.1]


def _pts(a, b, n=76):
    return [[x + (y - x) * k / (n - 1) for x, y in zip(a, b)] for k in range(n)]


def _line(guard, speed=CONTACT_SPEED, name="reach:slide-in"):
    pts = _pts(A, B)
    return Leg(name=name, kind=Kind.MOTION, traj=approach.line_trajectory(NAMES, pts), speed=speed, guard=guard,
               world="full", chain=0, target=("pose", [0, 0, 0], [0, 0, 0, 1]), goal_joints=list(pts[-1]),
               plan_s=0.0, plan_free=True)


def _drag():
    return GuardSpec(touch_nm=1.5, trip="obstruction", gravity=lambda q: [0.0] * 4, reference=([0.0] * 4, A))


class _Rec(FakeClient):
    def __init__(self):
        super().__init__()
        self.live = list(A)
        self.flown = []

    def execute(self, traj_, speed, **kw):
        self.flown.append(traj_ if float(speed) == 1.0 else driver.dilate(traj_, speed))
        return super().execute(traj_, speed, **kw)


@pytest.mark.parametrize("scale", [1.0, 0.5])
def test_a_free_air_line_is_retimed_never_faster(tmp_path, scale):
    c = _Rec()
    r = make_runner(c, tmp_path)
    r.time_scale = scale
    lg = _line(_drag())
    before = driver.dilate(lg.traj, lg.speed * scale)
    res = r.run([lg], execute=True)
    assert res[-1].ok
    assert c.executed[-1][1] == 1.0  # the profile is baked in
    flown = c.flown[-1]
    assert not check_like_executor(flown, JOINT_VMAX)
    assert _path_speed_peak(flown) <= _path_speed_peak(before) * (1 + 1e-6)
    assert [p.positions for p in flown.points] == [p.positions for p in lg.traj.points]  # positions untouched
    t_old, t_new = secs(before.points[-1].time_from_start), secs(flown.points[-1].time_from_start)
    assert t_new < 0.7 * t_old, (t_old, t_new)


def test_contact_strokes_keep_their_dilated_time_base(tmp_path):
    c = _Rec()
    r = make_runner(c, tmp_path)
    touch = GuardSpec(touch_nm=3.0, trip="touch", gravity=lambda q: [0.0] * 4, reference=([0.0] * 4, A))
    r.run([_line(touch, speed=0.35, name="press:down")], execute=True)
    assert c.executed[-1][1] == pytest.approx(0.35)
    c2 = _Rec()
    r2 = make_runner(c2, tmp_path)
    raw = GuardSpec(touch_nm=1.5, trip="obstruction")  # no free-air reference: not judged at rest
    r2.run([_line(raw)], execute=True)
    assert c2.executed[-1][1] == pytest.approx(CONTACT_SPEED)


@pytest.mark.parametrize("what", ["push", "warped", "guard_from", "armed_late"])
def test_pushes_warped_and_late_armed_lines_keep_their_dilated_time_base(tmp_path, what):
    """Only a line whose guard judges an obstruction against a rest reading,
    armed from the first sample, may be re-timed: a press's recoil, a warp's
    slow zone and an arming fraction all count in the leg's own time base."""
    from dataclasses import replace

    ref = ([0.0] * 4, A)
    g = _drag()
    if what == "push":
        g = GuardSpec(touch_nm=10.0, trip="press", gravity=lambda q: [0.0] * 4, reference=ref)
    if what == "armed_late":
        g = replace(g, arm_after=0.3)
    lg = _line(g)
    if what == "warped":
        lg.warp = (1.0, CONTACT_SPEED, 0.5)
    if what == "guard_from":
        lg.guard_from = 10
    c = _Rec()
    c.exec_script = [("arrived", {"message": "ok", "progress": 1.0})]
    make_runner(c, tmp_path).run([lg], execute=True)
    assert c.executed[0][1] == pytest.approx(CONTACT_SPEED)


def test_the_preview_shows_the_time_the_line_will_fly(tmp_path):
    c = _Rec()
    r = make_runner(c, tmp_path)
    lg = _line(_drag())
    table = r.preview([lg])
    r.run([lg], execute=True)
    flown_s = secs(c.flown[-1].points[-1].time_from_start)
    row = [line for line in table.splitlines() if line.startswith("reach:slide-in")][0]
    assert float(row.split()[5]) == pytest.approx(flown_s, abs=0.006)
