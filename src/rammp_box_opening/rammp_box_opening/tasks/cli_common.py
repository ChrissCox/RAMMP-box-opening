"""Shared CLI plumbing: args, node/client/runner wiring, safety refusals.

Dry-run is the default for every CLI. --execute arms the client — the
only software gate on motion, since the kinova-gen3-ros2 driver executes
whatever it is sent — and requires a measured container config
(measure_me: false). There is no typed confirmation (owner, 2026-09-16):
the human on the physical e-stop is the gate. Every CLI refuses to start
off Cyclone DDS, the middleware sheppy's driver and planner containers speak.
"""

import argparse
import os
import sys
from pathlib import Path

import rclpy
from rclpy.signals import SignalHandlerOptions

from rammp_box_opening.models.container import ContainerModel
from rammp_box_opening.primitives.core import Ctx
from rammp_box_opening.runtime.abort import AbortFlag, install_sigint
from rammp_box_opening.runtime.client import PlannerClient
from rammp_box_opening.runtime.driver import rmw_refusal
from rammp_box_opening.runtime.runner import Runner
from rammp_box_opening.worlds import WorldStore


def _share_path(*parts):
    from ament_index_python.packages import get_package_share_directory

    return Path(get_package_share_directory("rammp_box_opening"), *parts)


def default_container_yaml():
    """The container every CLI and both OWL detector nodes fall back to:
    the round clear canister with the pink push-button lid (owner,
    2026-09-22: "this box is the new default"). The OXOs stay selectable
    with --container."""
    return str(_share_path("config", "containers", "ankou_pink.yaml"))


def default_bench_yaml():
    return str(_share_path("config", "world_bench.yaml"))


def make_parser(desc):
    ap = argparse.ArgumentParser(
        description=desc, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--execute",
        action="store_true",
        help="after previewing, execute: arms arm AND gripper motion on the "
        "kinova-gen3-ros2 driver (without it nothing is ever sent to it)",
    )
    ap.add_argument(
        "--container",
        default=None,
        help="container config YAML (default: installed ankou_pink.yaml, the pink canister; "
        "oxo_pop.yaml / oxo_pop_small.yaml are the OXOs)",
    )
    ap.add_argument(
        "--bench-world",
        default=None,
        help="bench world YAML (default: installed world_bench.yaml)",
    )
    ap.add_argument(
        "--speed-scale",
        type=float,
        default=1.0,
        help="whole-run slow mode: dilate every motion by 1/x (0 < x <= 1), "
        "guarded strokes included — a first attempt at a new placement",
    )
    return ap


def refuse_unmeasured(model, execute, measuring=False):
    """An unmeasured container (measure_me: true — geometry estimated, not
    taped) may not be flown at. `measuring` is the observation that
    measures it (press_demo --detect-only: the look pose and home, in the
    bench world above the keep-out band, never near the container), which
    the flag must not lock out (2026-09-22: it did)."""
    if execute and model.measure_me and not measuring:
        sys.exit(
            "container config still carries measure_me: true — run the "
            "measurement worksheet (docs/HARDWARE_BRINGUP.md) and flip it "
            "before any hardware execution (dry-run is fine)."
        )


def init_runtime(execute):
    """rclpy + SIGINT ownership + node + client, shared by every CLI.

    `execute` (the --execute flag) arms the client; nothing else does.
    A shell off Cyclone is refused before rclpy is touched: Fast DDS
    discovers the driver and planner containers and then loses their data,
    which reads as a stalled arm instead of a wiring error.

    Owning SIGINT means an in-flight stroke gets its cancel delivered on
    a live context before we exit (runtime/abort.py; proven by
    scripts/abort_e2e.py — rclpy's default handler makes it a race)."""
    why = rmw_refusal(os.environ)
    if why:
        sys.exit(why)
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    abort = AbortFlag()
    install_sigint(abort)
    node = rclpy.create_node("rammp_box_opening")
    return node, PlannerClient(node, abort=abort, motion_enabled=bool(execute))


def build_ctx(args):
    """Ctx + Runner for a CLI that has not detected a container (cpose
    None): only pre-detection (bench-world) legs can be planned from it."""
    cfg = args.container or default_container_yaml()
    bench = args.bench_world or default_bench_yaml()
    model = ContainerModel.load(cfg)
    refuse_unmeasured(model, args.execute)
    node, client = init_runtime(args.execute)
    ctx = Ctx(
        model=model,
        cpose=None,
        client=client,
        worlds=WorldStore(bench),
        config_path=cfg,
    )
    return ctx, Runner(client)


def apply_speed_scale(runner, args):
    scale = float(getattr(args, "speed_scale", 1.0))
    if not 0.0 < scale <= 1.0:
        sys.exit("--speed-scale must be in (0, 1]")
    runner.time_scale = scale
    if scale < 1.0:
        print("[runner] SLOW MODE: every motion dilated by 1/%.2f" % scale)


def run_task(args, build_legs):
    ctx, runner = build_ctx(args)
    apply_speed_scale(runner, args)
    try:
        legs = build_legs(ctx)
    except RuntimeError as e:
        # a refused plan is an honest line, never a traceback — the arm
        # holds where it is
        sys.exit("plan refused — arm holds: %s" % e)
    try:
        results = runner.run(legs, execute=args.execute)
    except KeyboardInterrupt:
        sys.exit(130)  # the abort path already reported what was confirmed
    bad = [r for r in results if not r.ok]
    if bad:
        sys.exit(1)
