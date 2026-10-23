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
