"""press_demo.main() driven end to end with every ROS, camera and planner
edge faked: no rclpy.init, nothing talks to DDS or the arm. run_main is the
seam; the scene camera's approach (approach_from_scene) is scripted per
test, and everything the run writes lands in tmp_path."""

import json
import sys

import pytest
from conftest import FakeClient

PINK = "src/rammp_box_opening/config/containers/ankou_pink.yaml"
BENCH = "src/rammp_box_opening/config/world_bench.yaml"
BOX_XY = (0.45, -0.10)
TOUCH = ("touch", {"message": "contact", "progress": 0.97, "torque_peak": 3.2})


class IkFail:
    success = False
    message = "MotionGenStatus.IK_FAIL"
    trajectory = None


class _Node:
    def get_clock(self):
        class C:
            def now(self):
                class T:
                    nanoseconds = 0

                return T()

        return C()

    def get_logger(self):
        class L:
            def error(self, *a):
                pass

        return L()


class _Exec:
    def add_node(self, n):
        pass

    def spin(self):
        pass

    def shutdown(self):
        pass

    def remove_node(self, n):
        pass


class _Grab:
    color = None
    depth = None
    k = None
    color_stamp = None

    def missing(self):
        return []


class _Watcher:
    def __init__(self, node, cfg, model, table_z, base_frame=None, camera_cfg_name=None):
        self.model = model
        self.table_z = table_z
        self.grab = _Grab()
        self.roi = None
        self.active = False
        self.last_aim = None
        self.last_reject = None

    def to_container_pose(self, got):
        from rammp_box_opening.perception.depth_source import container_pose_from_top

        return container_pose_from_top(got[0], float(got[1]), self.model, self.table_z)

    def status(self):
        return "3/3 still frames found the button circle"

    def last_coarse(self):
        return None

    def coarse_fix(self, now=None):
        return None


LOCATES = []  # every timeout the scene locator was asked for, in order
HOME_FIRSTS = []  # the keywords of every home_first call, in order


def run_main(monkeypatch, tmp_path, argv, approach, client=None, centre=None):
    """press_demo.main() with `approach(ctx)` standing in for
    approach_from_scene (returns a SceneApproach), and `centre` for
    centre_over_button when given. Returns (exit code, client, run-log rows)."""
    import faulthandler

    import rclpy.executors

    from rammp_box_opening import worlds as worlds_mod
    from rammp_box_opening.models.container import ContainerModel
    from rammp_box_opening.perception import owl_source, scene, scene_calib, scene_source
    from rammp_box_opening.perception.depth_source import container_pose_from_top
    from rammp_box_opening.tasks import cli_common, press_demo

    client = client or FakeClient()
    model = ContainerModel.load(PINK)
    table_z = -0.057 + 0.03  # world_bench.yaml's table top
    top = (BOX_XY[0], BOX_XY[1], table_z + model.dims[2])

    class _Fix:
        pose = container_pose_from_top(top, 0.0, model, table_z)
        top_xyz = top
        n_points = 300
        score = 0.6
        table_z_scene = table_z
        table_extent = None

    class _Locator:
        last_why = None
        moved_note = None
        last_timing = {}
        grab = None

        def __init__(self, *a, **k):
            self.owl = type("O", (), {"enable": lambda s: None})()

        def locate(self, timeout):
            LOCATES.append(timeout)
            return _Fix

        def base_to_link(self):
            return None

    class _Rclpy:
        @staticmethod
        def create_node(name):
            return _Node()

        @staticmethod
        def spin_once(node, timeout_sec=0.0):
            return None

    monkeypatch.setattr(sys, "argv", ["press_demo", *argv, "--container", PINK, "--bench-world", BENCH])
    monkeypatch.setattr(cli_common, "init_runtime", lambda execute: (_Node(), client))
    monkeypatch.setattr(press_demo, "rclpy", _Rclpy)
    monkeypatch.setattr(faulthandler, "enable", lambda *a, **k: None)
    monkeypatch.setattr(rclpy.executors, "SingleThreadedExecutor", _Exec)
    monkeypatch.setattr(owl_source, "announce_container", lambda node, path: None)
    monkeypatch.setattr(
        owl_source, "OwlRung", lambda *a, **k: type("Owl", (), {"enable": lambda s: None, "disable": lambda s: None})()
    )
    monkeypatch.setattr(press_demo, "BoxTopWatcher", _Watcher)
    identity = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    monkeypatch.setattr(scene, "load_scene_calibration", lambda p=None: identity)
    monkeypatch.setattr(scene_source, "SceneLocator", _Locator)
    monkeypatch.setattr(scene_calib, "scene_tag_px", lambda p: None)
    monkeypatch.setattr(press_demo, "readiness_refusal", lambda *a, **k: None)
    monkeypatch.setattr(press_demo, "home_first", lambda *a, **k: HOME_FIRSTS.append(k) or True)
    monkeypatch.setattr(press_demo, "MissionFrames", lambda *a, **k: None)
    monkeypatch.setattr(press_demo, "CAPTURES_DIR", tmp_path / "captures")
    monkeypatch.setattr(press_demo, "residuals_path_for", lambda p: tmp_path / "residuals.jsonl")
    monkeypatch.setattr(press_demo, "dump_wrist_frame", lambda w, t: None)

    def fake_approach(node, ctx, cfg, runner, watcher, execute, scene_fix, lookahead=True):
        ctx.cpose = scene_fix.pose
        return approach(ctx)

    monkeypatch.setattr(press_demo, "approach_from_scene", fake_approach)
    if centre is not None:
        monkeypatch.setattr(press_demo, "centre_over_button", centre)
    real_init = worlds_mod.WorldStore.__init__
    monkeypatch.setattr(
        worlds_mod.WorldStore, "__init__", lambda self, path, out_dir=None: real_init(self, path, out_dir=tmp_path / "worlds")
    )
    monkeypatch.setenv("RAMMP_BOX_OPENING_STATE", str(tmp_path))  # the Runner's log, the gripper file
    with pytest.raises(SystemExit) as e:
        press_demo.main()
    rows = []
    for f in sorted((tmp_path / "runs").glob("run-*.jsonl")):
        rows += [json.loads(line) for line in open(f)]
    return e.value.code, client, rows


def _at_staging(button_up_mm=0.0):
    """The scene approach as it ends at staging over the scene's box, with
    the first aim landing on the button (no pre-planned descent)."""

    def approach(ctx):
        from rammp_box_opening.models.container import from_container
        from rammp_box_opening.tasks.press_demo import SceneApproach

        b = from_container(ctx.cpose, ctx.model.button_offset)
        return SceneApproach(got=((b[0], b[1], b[2]), 0.0), preplanned=None, status="3/3", button_up_mm=button_up_mm)

    return approach


def _no_centre(node, ctx, cfg, runner, watcher, execute, got):
    ctx.cpose = watcher.to_container_pose(got)
    return got, 0, 0.0


def _legs(rows):
    return [r["leg"] for r in rows]


def _off_by(monkeypatch, *offs):
    """The tool stands offs[i] (m) from over the button before aim i+1,
    the last one from then on."""
    from rammp_box_opening.tasks import press_demo

    offs = list(offs)
    monkeypatch.setattr(press_demo, "_off_button", lambda ctx, cfg: offs.pop(0) if len(offs) > 1 else offs[0])


def test_a_refused_recentre_plan_presses_from_the_last_aim(monkeypatch, tmp_path, capsys):
    """The tool stands 40 mm off the aimed button and the planner refuses
    the move over it (IK_FAIL near the reach edge): nothing was sent, so the
    arm still stands at staging and the press goes from the last aim — not
    a RuntimeError traceback with the arm hanging over the lid."""
    from rammp_box_opening.tasks import press_demo

    _off_by(monkeypatch, 0.04)
    monkeypatch.setattr(press_demo, "knob_height_mm", lambda *a, **k: 11.0)
    client = FakeClient()
    client.plans = [IkFail()]  # approach:recentre
    client.exec_script = [TOUCH]
    code, _c, rows = run_main(monkeypatch, tmp_path, ["--execute", "--press-only"], _at_staging(), client=client)
    out = capsys.readouterr().out
    assert "CENTRE: the move over the button was refused" in out and "pressing from the last aim" in out
    assert "approach:recentre" not in _legs(rows)  # nothing was sent
    assert "press:down" in _legs(rows)


def test_the_first_look_for_the_box_keeps_at_it_longer_than_a_re_check(monkeypatch, tmp_path):
    """Flatbot 2026-10-27 15:34:01: the scene camera refused the OWL's three
    candidates in 5 s and the run went home; the next run had the box at
    once. The first look, before anything moves, gets SCENE_FIRST_LOCATE_S."""
    from rammp_box_opening.tasks import press_demo

    LOCATES.clear()
    client = FakeClient()
    client.exec_script = [TOUCH]
    run_main(monkeypatch, tmp_path, ["--execute", "--press-only"], _at_staging(), client=client)
    assert LOCATES[0] == press_demo.SCENE_FIRST_LOCATE_S > press_demo.SCENE_LOCATE_S


def _refused(ctx):
    """The scene approach refused: the run goes home and stops (exit 1),
    right after the start these tests are about."""
    from rammp_box_opening.tasks.press_demo import SceneApproach

    return SceneApproach(failed=True, status="the approach to staging was refused (IK_FAIL)")


def test_a_compliant_depth_press_run_starts_from_the_spring_hold_it_finds(monkeypatch, tmp_path):
    """Flatbot 2026-10-27 16:24: the run after a compliant run found the arm
    0.03 rad off HOME, the joint spring's hold, and flew a POSITION home
    before anything else. A default --depth-press run (compliant, the scene
    camera's reach-out) takes a start within REST_TOL_RAD as there."""
    from rammp_box_opening.constants import REST_TOL_RAD

    HOME_FIRSTS.clear()
    code, _c, _rows = run_main(monkeypatch, tmp_path, ["--execute", "--depth-press"], _refused)
    assert code == 1
    assert [k.get("tol") for k in HOME_FIRSTS] == [REST_TOL_RAD]


def test_every_other_run_still_starts_at_its_exact_rest_pose(monkeypatch, tmp_path):
    """--guarded reads its at-rest reference at the rest pose and checks
    gravity against it; without --depth-press, or with --no-scene, there is
    no compliant reach-out. All of them keep AT_REST_RAD."""
    from rammp_box_opening.tasks import press_demo

    # --no-scene's run stops where the others do: its search's first step fails (exit 1)
    monkeypatch.setattr(press_demo, "search_for_box", lambda *a, **k: (None, "look"))
    for i, argv in enumerate((["--execute", "--depth-press", "--guarded"], ["--execute"],
                              ["--execute", "--depth-press", "--no-scene"])):
        HOME_FIRSTS.clear()
        code, _c, _rows = run_main(monkeypatch, tmp_path / str(i), argv, _refused)
        assert code == 1, argv
        assert [k.get("tol") for k in HOME_FIRSTS] == [press_demo.AT_REST_RAD], argv


def test_an_open_box_seen_by_the_re_aim_over_the_button_is_not_pressed(monkeypatch, tmp_path, capsys):
    """The first aim could not read the button's height (its lid ring off
    the frame's edge); the tool stood 40 mm off, moved over the button and
    aimed again — and that aim, from straight above, reads the knob 10 mm
    UP. The run stops and goes home before any press. A re-aim that reads
    the knob flush (2 mm) goes on to press."""
    from rammp_box_opening.tasks import press_demo

    for up_mm, pressed in ((10.0, False), (2.0, True)):
        _off_by(monkeypatch, 0.04, 0.0)
        first, asked = [], []

        def approach(ctx, first=first):
            a = _at_staging(button_up_mm=None)(ctx)
            first.append(a.got)
            return a

        def aim(node, watcher, yaw, runner=None, out=None, up_mm=up_mm, first=first, asked=asked, **kw):
            asked.append(out is not None)
            if out is not None:
                out["button_up_mm"] = up_mm
            return first[0], "3/3"

        monkeypatch.setattr(press_demo, "aim_button_at_staging", aim)
        monkeypatch.setattr(press_demo, "knob_height_mm", lambda *a, **k: 11.0)
        client = FakeClient()
        client.exec_script = [("arrived", {"message": "ok", "progress": 1.0}), TOUCH]
        code, _c, rows = run_main(
            monkeypatch, tmp_path / ("%g" % up_mm), ["--execute", "--press-only"], approach, client=client
        )
        out = capsys.readouterr().out
        assert asked == [True], up_mm  # the re-aim hands back the height it measured
        assert ("press:down" in _legs(rows)) is pressed, up_mm
        assert ("already OPEN" in out) is not pressed, up_mm
        if not pressed:
            assert code == 1 and "returning home" in out
            assert [r["button_above_lid_mm"] for r in rows if r["leg"] == "open_box"] == [10.0]


def test_a_refused_press_plan_at_staging_goes_home_with_a_stop_line(monkeypatch, tmp_path, capsys):
    """No pre-planned descent and the fresh press plan is refused: a STOP
    line and the recovery home, never a traceback with the arm over the box."""
    client = FakeClient()
    client.plans = [IkFail()]  # press:down
    code, _c, rows = run_main(
        monkeypatch, tmp_path, ["--execute", "--press-only"], _at_staging(), client=client, centre=_no_centre
    )
    out = capsys.readouterr().out
    assert code == 1
    assert "STOP before the press" in out and "IK_FAIL" in out and "returning home" in out
    assert "press:down" not in _legs(rows) and "home" in _legs(rows)


def test_a_refused_chained_press_plan_stops_with_a_stop_line(monkeypatch, tmp_path, capsys):
    """The chained approach + descent (a search fix, not at staging: the
    dry run's path) refused: the same honest stop."""
    from rammp_box_opening.tasks import press_demo

    monkeypatch.setattr(press_demo, "search_for_box", lambda *a, **k: (((BOX_XY[0], BOX_XY[1], 0.094), 0.0), None))
    client = FakeClient()
    client.plans = [IkFail()]  # approach:staging
    code, _c, _rows = run_main(monkeypatch, tmp_path, ["--no-scene", "--press-only"], _at_staging(), client=client)
    out = capsys.readouterr().out
    assert code == 1
    assert "STOP before the press" in out and "IK_FAIL" in out and "returning home" in out


"""main()'s shape around the perception thread, read from its source (no
rclpy): once the thread spins, every exit — a failed startup included — must
pass through the finally that stops it, or the interpreter tears rclpy down
under a live executor and aborts ("terminate called without an active
exception", exit 134 instead of the run's own message or 130)."""

import ast
from pathlib import Path

from rammp_box_opening.tasks import press_demo


def _main_body():
    tree = ast.parse(Path(press_demo.__file__).read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    return main.body


def _is_call(node, owner, attr):
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == attr
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == owner
    )


def _start_index(body):
    for i, stmt in enumerate(body):
        if any(_is_call(n, "pthread", "start") for n in ast.walk(stmt)):
            return i
    raise AssertionError("main() never starts the perception thread")


def _the_try(body):
    i = _start_index(body)
    assert i + 1 < len(body)
    return i, body[i + 1]


def test_the_try_opens_right_after_the_perception_thread_starts():
    i, nxt = _the_try(_main_body())
    assert isinstance(nxt, ast.Try), "line %d follows pthread.start(), not the try" % nxt.lineno
    final = ast.Module(body=nxt.finalbody, type_ignores=[])
    assert any(_is_call(n, "pexec", "shutdown") for n in ast.walk(final))
    # a startup Ctrl+C still exits 130
    assert any(
        isinstance(h.type, ast.Name) and h.type.id == "KeyboardInterrupt" for h in nxt.handlers
    )


def test_ctx_is_none_before_the_try():
    body = _main_body()
    i, _ = _the_try(body)
    assert any(
        isinstance(s, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "ctx" for t in s.targets)
        and isinstance(s.value, ast.Constant)
        and s.value.value is None
        for s in body[:i]
    )


def test_the_finally_reads_mission_frames_through_getattr():
    _, nxt = _the_try(_main_body())
    assert isinstance(nxt, ast.Try)
    final = ast.Module(body=nxt.finalbody, type_ignores=[])
    # no bare ctx.mission_frames: ctx may be None, startup having failed first
    assert not any(
        isinstance(n, ast.Attribute)
        and n.attr == "mission_frames"
        and isinstance(n.value, ast.Name)
        and n.value.id == "ctx"
        for n in ast.walk(final)
    )
    calls = [
        n for n in ast.walk(final)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "keep_mission_frames"
    ]
    assert len(calls) == 1
    arg = calls[0].args[0]
    assert isinstance(arg, ast.Call) and isinstance(arg.func, ast.Name) and arg.func.id == "getattr"
    assert isinstance(arg.args[0], ast.Name) and arg.args[0].id == "ctx"
    assert arg.args[1].value == "mission_frames"
    assert len(arg.args) == 3 and arg.args[2].value is None


def test_no_frames_to_keep_is_a_no_op():
    assert press_demo.keep_mission_frames(None, SystemExit(2)) is None
    assert press_demo.keep_mission_frames(getattr(None, "mission_frames", None), None) is None
