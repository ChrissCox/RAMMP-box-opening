"""press_demo.main() end to end, every ROS / camera / planner edge faked (no
rclpy.init, nothing talks to DDS or the arm): the lid drop spot is resolved
only for a run that will grip and place the lid — never for the depth press,
which ends at the press."""
import json
import sys
from pathlib import Path

import pytest
from conftest import FakeClient

PKG = Path(__file__).resolve().parents[1]
PINK = str(PKG / "config/containers/ankou_pink.yaml")
BENCH = str(PKG / "config/world_bench.yaml")
BOX_XY = (0.45, -0.10)
SMALL = (BOX_XY[0] - 0.25, BOX_XY[0] + 0.25, BOX_XY[1] - 0.25, BOX_XY[1] + 0.25)  # no lid spot fits
ROOMY = (BOX_XY[0] - 0.6, BOX_XY[0] + 0.6, BOX_XY[1] - 0.6, BOX_XY[1] + 0.6)


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


class _Touch:
    leg_name = "press:down"
    ok = True
    outcome = "touch"
    detail = "contact"


def run_main(monkeypatch, tmp_path, argv, approach, table_extent, depth=None, push=None):
    """press_demo.main() with `approach` standing in for approach_from_scene,
    `depth` for press_down_from_here and `push` for run_push. Returns
    (exit code, run-log rows)."""
    import faulthandler

    import rclpy.executors

    from rammp_box_opening import worlds as worlds_mod
    from rammp_box_opening.models.container import ContainerModel
    from rammp_box_opening.perception import owl_source, scene, scene_calib, scene_source
    from rammp_box_opening.perception.depth_source import container_pose_from_top
    from rammp_box_opening.tasks import cli_common, depth_press, press_demo

    client = FakeClient()
    model = ContainerModel.load(PINK)
    table_z = -0.057 + 0.03  # world_bench.yaml's table top
    top = (BOX_XY[0], BOX_XY[1], table_z + model.dims[2])

    class _Fix:
        pose = container_pose_from_top(top, 0.0, model, table_z)
        top_xyz = top
        n_points = 300
        score = 0.6
        table_z_scene = table_z

    _Fix.table_extent = table_extent

    class _Locator:
        last_why = None
        moved_note = None
        last_timing = {}
        grab = None

        def __init__(self, *a, **k):
            class O:
                def enable(self):
                    pass

            self.owl = O()

        def locate(self, timeout):
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

    def fake_approach(node, ctx, cfg, runner, watcher, execute, scene_fix):
        ctx.cpose = scene_fix.pose
        return approach(ctx)

    def no_centre(node, ctx, cfg, runner, watcher, execute, got):
        ctx.cpose = watcher.to_container_pose(got)
        return got, 0, 0.0

    real_init = worlds_mod.WorldStore.__init__

    def init(self, path, out_dir=None):
        real_init(self, path, out_dir=tmp_path / "worlds")
        self.table_extent = table_extent

    monkeypatch.setenv("RAMMP_BOX_OPENING_STATE", str(tmp_path))  # the Runner's log, the gripper file
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
    monkeypatch.setattr(
        scene, "load_scene_calibration", lambda p=None: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    )
    monkeypatch.setattr(scene_source, "SceneLocator", _Locator)
    monkeypatch.setattr(scene_calib, "scene_tag_px", lambda p: None)
    monkeypatch.setattr(press_demo, "readiness_refusal", lambda *a, **k: None)
    monkeypatch.setattr(press_demo, "home_first", lambda *a, **k: True)
    monkeypatch.setattr(press_demo, "MissionFrames", lambda *a, **k: None)
    monkeypatch.setattr(press_demo, "CAPTURES_DIR", tmp_path / "captures")
    monkeypatch.setattr(press_demo, "residuals_path_for", lambda p: tmp_path / "residuals.jsonl")
    monkeypatch.setattr(press_demo, "dump_wrist_frame", lambda w, t: None)
    monkeypatch.setattr(press_demo, "approach_from_scene", fake_approach)
    monkeypatch.setattr(press_demo, "centre_over_button", no_centre)
    monkeypatch.setattr(worlds_mod.WorldStore, "__init__", init)
    if depth is not None:
        monkeypatch.setattr(depth_press, "press_down_from_here", depth)
    if push is not None:
        monkeypatch.setattr(press_demo, "run_push", push)
    with pytest.raises(SystemExit) as e:
        press_demo.main()
    rows = []
    for f in sorted((tmp_path / "runs").glob("run-*.jsonl")):
        rows += [json.loads(line) for line in open(f)]
    return e.value.code, rows


def _approach(pressed=None):
    def approach(ctx):
        from rammp_box_opening.models.container import from_container
        from rammp_box_opening.tasks.press_demo import SceneApproach

        b = from_container(ctx.cpose, ctx.model.button_offset)
        return SceneApproach(
            got=((b[0], b[1], b[2]), 0.0), preplanned=None, pressed=pressed,
            press_leg=None, status="3/3", button_up_mm=0.0,
        )

    return approach


def test_the_depth_press_is_not_refused_over_a_lid_spot_it_never_uses(monkeypatch, tmp_path, capsys):
    from rammp_box_opening.tasks.depth_press import DepthPress, exit_code

    result = DepthPress(free=True, contact=True, felt=True, popped=True)
    called = []

    def depth(node, ctx, cfg, runner, watcher, tool):
        called.append(ctx.lid_drop)
        return result

    code, _rows = run_main(monkeypatch, tmp_path, ["--execute", "--depth-press"], _approach(), SMALL, depth=depth)
    out = capsys.readouterr().out
    assert "no lid drop spot" not in out
    assert called == [None]  # pressed, no lid spot resolved for it
    assert code == exit_code(result) and code != 4


def test_a_depth_press_run_that_falls_through_to_grip_and_place_still_needs_a_lid_spot(
    monkeypatch, tmp_path, capsys
):
    """The reach already touched in flight: the depth branch is skipped and
    the run goes on to grip and place — its drop spot is still checked."""

    def depth(*a, **k):
        raise AssertionError("the depth press must not run after an in-flight touch")

    code, _rows = run_main(
        monkeypatch, tmp_path, ["--execute", "--depth-press"], _approach(pressed=[_Touch()]), SMALL, depth=depth
    )
    assert code == 4 and "no lid drop spot" in capsys.readouterr().out


def test_a_fall_through_run_on_a_roomy_table_carries_a_checked_lid_spot(monkeypatch, tmp_path, capsys):
    """Never the configured lid_place unchecked: ctx.lid_drop is resolved
    before the press that leads to grip and place."""
    seen = []

    def push(ctx, *a, **k):
        seen.append(ctx.lid_drop)
        sys.exit(99)

    code, _rows = run_main(
        monkeypatch, tmp_path, ["--execute", "--depth-press"], _approach(pressed=[_Touch()]), ROOMY, push=push
    )
    assert code == 99 and len(seen) == 1 and seen[0] is not None
    assert "lid goes to" in capsys.readouterr().out


def test_the_normal_press_on_a_small_table_is_still_refused(monkeypatch, tmp_path, capsys):
    code, _rows = run_main(monkeypatch, tmp_path, ["--execute"], _approach(), SMALL)
    assert code == 4 and "no lid drop spot" in capsys.readouterr().out
