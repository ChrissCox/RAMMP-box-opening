"""scripts/sheppy_task.sh, the wrapper every sheppy task starts through, run
for real against a fake ~/RAMMP-box-opening under tmp_path: its drift gate
(--execute refuses while install/ is not src/) and how it starts the task.

Nothing real is built or started. The fake install/setup.bash has the
colcon chain line the wrapper looks for and only exports ROS_DISTRO, and the
task is a stub that prints the env it got and its argv. Since 2026-10-27 the
wrapper skips humble's second source, finds agreeing trees with two diffs
before drift()'s per-file loop, and execs the console script itself instead
of `ros2 run`: about 0.58 s off every launch."""

import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "sheppy_task.sh"

# the drift report as the wrapper has printed it since 2026-10-22
HEAD = "[sheppy_task] the installed rammp_box_opening is not the source:"
REFUSED = "[sheppy_task] REFUSED --execute — rebuild first: colcon build --packages-select rammp_box_opening"
GOES_ON = "[sheppy_task] not executing: running the installed copy anyway"
STALE = (
    "stale in install/ (a rebuild keeps it: rm -rf build/rammp_box_opening "
    "install/rammp_box_opening, then rebuild): "
)

PKG = {
    "__init__.py": "",
    "constants.py": "PUSH_MM = 6.7\n",
    "tasks/__init__.py": "",
    "tasks/press_demo.py": "def main():\n    return 0\n",
}
CONFIG = {
    "world_flatbot.yaml": "base_frame: base_link\n",
    "containers/ankou_pink.yaml": "name: ankou_pink\n",
}
# the console script's stand-in
STUB = """#!/bin/sh
echo "rmw=$RMW_IMPLEMENTATION"
echo "ld=$LD_LIBRARY_PATH"
echo "ament=$AMENT_PREFIX_PATH"
echo "install_saw=$INSTALL_SAW"
for a in "$0" "$@"; do echo "arg=$a"; done
"""


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


class Tree:
    """A fake ~/RAMMP-box-opening: the package and its config in src/, the
    same copied into install/ as a build leaves them, and one task. cmp on
    the wrapper's PATH notes each call: only drift()'s per-file loop uses it."""

    def __init__(self, tmp):
        self.home = tmp / "snor lax"  # a space: every path the wrapper builds stays one word
        self.bin = tmp / "bin"
        self.cmp_log = tmp / "cmp.log"
        spy = '#!/bin/sh\necho "$*" >> "%s"\nexec "%s" "$@"\n' % (self.cmp_log, shutil.which("cmp"))
        _write(self.bin / "cmp", spy)
        (self.bin / "cmp").chmod(0o755)
        ws = self.home / "RAMMP-box-opening"
        self.src = ws / "src" / "rammp_box_opening"
        self.inst = ws / "install" / "rammp_box_opening"
        self.inst_py = self.inst / "lib" / "python3.10" / "site-packages" / "rammp_box_opening"
        self.setup = ws / "install" / "setup.bash"
        for rel, text in PKG.items():
            _write(self.src / "rammp_box_opening" / rel, text)
            _write(self.inst_py / rel, text)
        for rel, text in CONFIG.items():
            _write(self.src / "config" / rel, text)
            _write(self.inst / "share" / "rammp_box_opening" / "config" / rel, text)
        # the bytecode each side compiled for itself is never drift
        _write(self.src / "rammp_box_opening" / "__pycache__" / "constants.cpython-310.pyc", "src")
        _write(self.inst_py / "__pycache__" / "constants.cpython-310.pyc", "install")
        _write(self.setup, 'COLCON_CURRENT_PREFIX="/opt/ros/humble"\nexport ROS_DISTRO=humble\n')
        self.exe = self.inst / "lib" / "rammp_box_opening" / "press_demo"
        _write(self.exe, STUB)
        self.exe.chmod(0o755)
        self.world = self.inst / "share" / "rammp_box_opening" / "config" / "world_flatbot.yaml"

    def env(self):
        return {"HOME": str(self.home), "PATH": "%s:%s" % (self.bin, os.environ.get("PATH", "/usr/bin:/bin"))}

    def run(self, *argv):
        return subprocess.run(
            ["bash", str(SCRIPT), *argv], env=self.env(), capture_output=True, encoding="utf-8", timeout=60
        )


def _got(out, key):
    return [line[len(key) + 1 :] for line in out.splitlines() if line.startswith(key + "=")]


@pytest.fixture
def tree(tmp_path):
    return Tree(tmp_path)


def test_the_wrapper_parses():
    assert subprocess.run(["bash", "-n", str(SCRIPT)]).returncode == 0


@pytest.mark.parametrize("execute", [[], ["--execute"]])
def test_agreeing_trees_start_the_installed_task_on_the_flatbot_world(tree, execute):
    r = tree.run("press_demo", *execute, "--depth-press")
    assert r.returncode == 0, r.stderr
    assert r.stderr == ""
    assert _got(r.stdout, "arg") == [str(tree.exe), "--bench-world", str(tree.world), *execute, "--depth-press"]
    assert _got(r.stdout, "rmw") == ["rmw_cyclonedds_cpp"]
    assert _got(r.stdout, "ld")[0].startswith(str(tree.inst / "lib" / "rammp_curobo_interfaces") + ":")
    # the install's chain stood in for humble's setup.bash: nothing real was sourced
    assert _got(r.stdout, "ament") == [""]
    # the two diffs settled it, __pycache__ apart: drift()'s 57-cmp loop was
    # 81-97 ms of every launch on the robot, the diffs 7 ms
    assert not tree.cmp_log.exists()


def test_the_task_is_the_process_sheppy_started_and_its_exit_code_reaches_sheppy(tree):
    """sheppy starts a node in a new session and stops it with SIGINT to
    the whole group (os.killpg), so the task got it under `ros2 run` as
    well. The wrapper execs the task now: the task is that very process,
    and Ctrl+C's 130 is what sheppy reads."""
    pidfile = tree.home.parent / "hold.pid"
    hold = tree.inst / "lib" / "rammp_box_opening" / "hold"
    _write(hold, "#!/bin/sh\ntrap 'exit 130' INT\necho $$ > \"%s\"\nwhile :; do sleep 0.05; done\n" % pidfile)
    hold.chmod(0o755)
    p = subprocess.Popen(
        ["bash", str(SCRIPT), "hold"],
        env=tree.env(),
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        encoding="utf-8",
    )
    try:
        deadline = time.monotonic() + 30
        while not (pidfile.exists() and pidfile.read_text().endswith("\n")):
            assert p.poll() is None, p.stderr.read()
            assert time.monotonic() < deadline, "the task never started"
            time.sleep(0.02)
        assert int(pidfile.read_text()) == p.pid
        os.killpg(p.pid, signal.SIGINT)
        assert p.wait(timeout=30) == 130
    finally:
        if p.poll() is None:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()


def _edited_py(t):
    _write(t.src / "rammp_box_opening" / "constants.py", "PUSH_MM = 12.0\n")
    return ["differs: constants.py"]


def _stale_py(t):
    _write(t.inst_py / "tasks" / "old_press.py", "")
    return [STALE + "tasks/old_press.py"]


def _new_subpackage(t):  # one "Only in" line to diff
    _write(t.src / "rammp_box_opening" / "lift" / "__init__.py", "")
    _write(t.src / "rammp_box_opening" / "lift" / "lid.py", "DROP_MM = 30\n")
    return ["differs: lift/__init__.py", "differs: lift/lid.py"]


def _edited_container(t):
    _write(t.src / "config" / "containers" / "ankou_pink.yaml", "name: ankou_pink\npush_mm: 12\n")
    return ["differs: config/containers/ankou_pink.yaml"]


DRIFTS = [_edited_py, _stale_py, _new_subpackage, _edited_container]


def _report(stderr):
    """(first line, the drift lines sorted, last line): find walks in
    directory order, so two lines of one change may come either way."""
    lines = stderr.splitlines()
    return lines[0], sorted(lines[1:-1]), lines[-1]


@pytest.mark.parametrize("change", DRIFTS, ids=lambda f: f.__name__.lstrip("_"))
def test_drift_refuses_execute_with_todays_report(tree, change):
    lines = change(tree)
    r = tree.run("press_demo", "--execute")
    assert r.returncode == 3
    assert r.stdout == ""  # the task never started
    assert _report(r.stderr) == (HEAD, sorted(lines), REFUSED)


@pytest.mark.parametrize("change", DRIFTS, ids=lambda f: f.__name__.lstrip("_"))
def test_drift_on_a_dry_run_says_so_and_runs_the_installed_copy(tree, change):
    lines = change(tree)
    r = tree.run("press_demo")
    assert r.returncode == 0, r.stderr
    assert _report(r.stderr) == (HEAD, sorted(lines), GOES_ON)
    assert _got(r.stdout, "arg") == [str(tree.exe), "--bench-world", str(tree.world)]


def _readme_in_config(t):  # setup.py installs only config's YAMLs
    _write(t.src / "config" / "README.md", "the scene camera's calibration\n")


def _notes_in_the_package(t):
    _write(t.src / "rammp_box_opening" / "perception" / "notes.txt", "")


@pytest.mark.parametrize("stray", [_readme_in_config, _notes_in_the_package], ids=lambda f: f.__name__.lstrip("_"))
def test_a_stray_file_diff_sees_is_not_drift(tree, stray):
    """diff sees the trees differ, so drift()'s loop runs, and it finds
    nothing installed that differs: the verdict it always gave."""
    stray(tree)
    r = tree.run("press_demo", "--execute")
    assert r.returncode == 0, r.stderr
    assert r.stderr == ""
    assert _got(r.stdout, "arg") == [str(tree.exe), "--bench-world", str(tree.world), "--execute"]
    assert tree.cmp_log.exists()  # it cost the full loop, nothing more


@pytest.mark.parametrize(
    "argv, says",
    [
        (["no_such_task"], "[sheppy_task] no such task: no_such_task"),
        ([".."], "[sheppy_task] no such task: .."),  # a directory: -x alone passes it
        (["../rammp_box_opening/press_demo"], "[sheppy_task] usage:"),  # a path to the stub itself
        ([""], "[sheppy_task] usage:"),
        ([], "[sheppy_task] usage:"),
    ],
    ids=["unknown", "dotdot", "slash", "empty", "none"],
)
def test_a_name_that_is_no_task_starts_nothing(tree, argv, says):
    r = tree.run(*argv, *(["--execute"] if argv else []))
    assert r.returncode != 0
    assert r.stdout == ""
    assert says in r.stderr


@pytest.mark.skipif(not Path("/opt/ros/humble/setup.bash").is_file(), reason="needs ROS 2 humble")
def test_an_install_without_humbles_chain_gets_humble_first(tree):
    """If a rebuild ever drops humble's chain line from install/setup.bash,
    humble is sourced as before, ahead of the install's own setup."""
    tree.setup.write_text('export INSTALL_SAW="${AMENT_PREFIX_PATH:-}"\n')
    r = tree.run("press_demo")
    assert r.returncode == 0, r.stderr
    assert "/opt/ros/humble" in _got(r.stdout, "install_saw")[0].split(":")
