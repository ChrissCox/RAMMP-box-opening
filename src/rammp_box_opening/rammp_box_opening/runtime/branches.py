"""Which way round a continuous joint is written.

The Gen3's joints 1, 3, 5 and 7 turn without end (both URDF limits
infinite). The arm's transport reports them wrapped into (-pi, pi]
(kinova-gen3-driver kortex_transport.cpp), and the driver compares every
target with the measurement the short way round — to the ARM, q and
q + 2*pi are one angle. Not to the planner: cuRobo bounds them as plain
revolute joints at +/-6.0 rad and plans the straight line between the
numbers it is handed. HOME holds joint_3 at 3.142, a hair past +pi, so at
HOME it reads -3.1411 some days and +3.1413 on others; handed -3.1411, the
planner turned joint_3 six radians — a full circle, 4.7 s — to reach a box
on the left of the table (bench 2026-09-25). Whatever hands a live reading
to the planner, or lines one up against a planned path, writes it on the
right branch first.
"""

from rammp_curobo.geometry import ang_diff

from rammp_box_opening.constants import HOME

CONTINUOUS_JOINTS = (0, 2, 4, 6)  # joint_1, joint_3, joint_5, joint_7
PLANNER_CONTINUOUS_LIMIT_RAD = 6.0  # where the planner's URDF bounds them


def on_branch(q, ref):
    """`q` with each continuous joint written as the angle nearest `ref`'s
    — the same physical angle, on `ref`'s side of the wrap."""
    out = [float(v) for v in q]
    for i in CONTINUOUS_JOINTS:
        out[i] = float(ref[i]) + ang_diff(out[i], float(ref[i]))
    return out


def planner_branch(q):
    """`q` as the planner should be handed it: each continuous joint on the
    branch of HOME — the planner's retract config, which its IK solutions
    sit near — unless that falls outside the planner's bounds, where it is
    left as reported."""
    out = [float(v) for v in q]
    for i in CONTINUOUS_JOINTS:
        c = float(HOME[i]) + ang_diff(out[i], float(HOME[i]))
        if abs(c) <= PLANNER_CONTINUOUS_LIMIT_RAD:
            out[i] = c
    return out
