"""Shared plumbing for the stub-isolated e2e harnesses.

Every harness process runs under one sourced shell chain on an isolated
ROS domain. A harness refuses to start beside a real arm driver or
planner — the stubs serve the real driver and /rammp_curobo names, and discovery
binding on a shared graph is a coin flip — and puts the ros2 CLI daemon
back down on exit: a daemon left bound to the isolated domain makes
`ros2 node list` in normal shells come up empty (field lesson 8).
"""

import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DOMAIN = os.environ.get("ABORT_E2E_DOMAIN", "77")
CONTAINER_YAML = REPO / "src/rammp_box_opening/config/containers/oxo_pop.yaml"


class Shell:
    """The sourced zsh every harness process runs under: the isolated
    domain on Cyclone DDS (the driver's middleware), the harness's own stub
    knobs, then the humble -> ~/rammp_deps_ws -> this-repo overlays."""

    def __init__(self, stub_env=""):
        # loopback only: ROS_LOCALHOST_ONLY, unless a Cyclone config is
        # already loaded — the bench's (~/.config/rammp-bench/
        # cyclonedds-local.xml, from ~/.profile since 2026-09-28) pins DDS to
        # loopback itself, and Cyclone refuses a participant when both
        # configure the interfaces ("rcl node's rmw handle is invalid")
        self.chain = (
            "export ROS_DOMAIN_ID=%s; "
            "if [ -n \"$CYCLONEDDS_URI\" ]; then export ROS_LOCALHOST_ONLY=0; else export ROS_LOCALHOST_ONLY=1; fi; "
            "export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp; %s"
            "source /opt/ros/humble/setup.zsh; "
            "source ~/rammp_deps_ws/install/setup.zsh; "
            "source %s/install/setup.zsh; " % (DOMAIN, stub_env, REPO)
        )

    def run(self, cmd, **kw):
        return subprocess.run(
            ["zsh", "-c", self.chain + cmd], capture_output=True, text=True, **kw
        )

    def spawn(self, cmd, log, extra_env="", **popen):
        """Start `cmd` in its own session with stdout+stderr in `log`, so
        kill() can take the whole process group down."""
        return subprocess.Popen(
            ["zsh", "-c", self.chain + extra_env + cmd],
            stdout=open(log, "w"),
            stderr=subprocess.STDOUT,
            start_new_session=True,
            **popen,
        )

    def refuse_real_stack(self):
        self.run("ros2 daemon stop", timeout=30)  # a daemon bound elsewhere lies
        probe = self.run("timeout 20 ros2 node list", timeout=30)
        nodes = probe.stdout
        if any(n in nodes for n in ("/kinova_gen3_node", "/rammp_curobo", "/controller_manager")):
            sys.exit(
                "REAL arm stack or planner visible on ROS_DOMAIN_ID=%s:\n%s\n"
                "This harness serves fake driver and planner names — refusing "
                "(discovery binding on a shared graph is a coin flip)."
                % (DOMAIN, nodes)
            )

    def daemon_reset(self):
        """Call on the way out: the probe rebinds the ros2 CLI daemon to the
        isolated domain, and normal shells would inherit that."""
        self.run("ros2 daemon stop", timeout=30)


def workdir(prefix):
    """The harness's scratch folder. Every process it spawns also keeps its
    STATE there (run logs, captures, generated worlds, calibration
    residuals — constants.state_dir): synthetic runs do not belong beside
    the bench's, and the every-run captures are pruned to the newest few."""
    tmp = Path(tempfile.mkdtemp(prefix=prefix))
    os.environ["RAMMP_BOX_OPENING_STATE"] = str(tmp / "state")
    print("workdir %s (domain %s)" % (tmp, DOMAIN))
    return tmp


def measured_config(tmp, edit=None, name="oxo_measured.yaml"):
    """A copy of the shipped container yaml with measure_me flipped
    (--execute refuses an unmeasured config), `edit`ed further if asked.
    `name` lets one harness keep several variants side by side."""
    text = CONTAINER_YAML.read_text().replace("measure_me: true", "measure_me: false")
    # no re-centring over the button: the stub planner has no real IK (the
    # arm's forward kinematics say nothing about where it "stands") and the
    # stub camera does not ride the arm, so the loop could never converge.
    # Unit-tested instead (test_tasks: recentre*).
    text = text.replace("recentre_max_moves: 2", "recentre_max_moves: 0", 1)
    if edit is not None:
        text = edit(text)
    cfg = tmp / name
    cfg.write_text(text)
    return cfg


def kill(proc):
    if proc is None:
        return
    for sig in (signal.SIGINT, signal.SIGKILL):
        try:
            os.killpg(os.getpgid(proc.pid), sig)
            proc.wait(timeout=5)
            return
        except (ProcessLookupError, subprocess.TimeoutExpired):
            continue


def wait_for(path, needle, timeout, proc=None, what="", also_dump=()):
    """True once `needle` appears in the log at `path`; False on timeout.
    Exits with the log tails if `proc` dies first."""
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if needle in Path(path).read_text():
            return True
        if proc is not None and proc.poll() is not None:
            dumps = "".join(
                "\n--- %s ---\n%s" % (Path(p).name, Path(p).read_text()[-2000:])
                for p in (path, *also_dump)
            )
            sys.exit("%s died before %r:%s" % (what, needle, dumps))
        time.sleep(0.2)
    return False
