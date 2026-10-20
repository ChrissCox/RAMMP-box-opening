"""home_arm: return to HOME — the recovery move and the abort-drill mover.

    ros2 run rammp_box_opening home_arm --execute
    ros2 run rammp_box_opening home_arm --execute --lift-first 0.08   # start refused: lift straight up first

Plans in the pre-detection BENCH world first: no container pose is
known to an isolated home (a failed run may have left the box anywhere),
so the whole placement band is blocked to container height and the move
stays above anything that could be standing there. A recovery home
usually STARTS inside that band, though — the arm is holding at a press,
a grip or a set-down — and then the band makes the start itself invalid.
So a refused start falls back to the bare table world with a printed
caution: the move then ignores where the box may be, and the operator
clears the bench before running it (review 2026-09-02). The attended
abort drill Ctrl+Cs this mid-motion. When even the bare world refuses
the START (the fingers are at the table), --lift-first M flies a plan-free
straight vertical line of M metres first and plans HOME from its end.
"""

from rammp_box_opening.constants import HOME, TRANSIT_SPEED
from rammp_box_opening.primitives.core import LiftFree, PlanState, _plan_motion
from rammp_box_opening.tasks import cli_common


def build_legs(ctx, lift_first_m=0.0):
    """HOME from live; with lift_first_m > 0, a plan-free straight lift of
    that much FIRST (primitives.LiftFree), and HOME planned from where the
    lift ends. For the pose the planner refuses as a start — fingers at
    the table after an abort or a jog (2026-09-17: refused in both worlds
    40 mm above the table, 23 deg off vertical)."""
    state = PlanState(joints=list(ctx.client.joints()), chain=0)
    legs = []
    if lift_first_m > 0:
        lift, state = LiftFree(lift_first_m).plan(ctx, state)
        legs += lift
        print(
            "[home_arm] plan-free lift of %.0f mm first (straight up, attitude held, "
            "obstruction-guarded) — HOME is planned from where it ends" % (1000 * lift_first_m)
        )
    guarded = ctx.worlds.push_name("bench", model=ctx.model)
    try:
        leg, _ = _plan_motion(
            ctx, state, "home", ("joints", list(HOME)), guarded, TRANSIT_SPEED
        )
        return legs + [leg]
    except RuntimeError as e:
        print("[home_arm] home refused in the guarded bench world (%s)" % e)
    # there is no typed confirmation any more (--execute alone arms a run),
    # so this cannot ask for the bench to be cleared first: it says what the
    # move about to fly does NOT know, and the dry-run preview is the place
    # to read it before adding --execute
    print(
        "[home_arm] CAUTION: planning in the bare table world — this move "
        "ignores where the box may be. With --execute it flies NOW; run "
        "without it first if the bench is not clear."
    )
    bare = ctx.worlds.push_name("bench", model=None, tag="bare")
    leg, _ = _plan_motion(
        ctx, state, "home", ("joints", list(HOME)), bare, TRANSIT_SPEED
    )
    return legs + [leg]


def main():
    ap = cli_common.make_parser(__doc__)
    ap.add_argument(
        "--lift-first",
        type=float,
        default=0.0,
        metavar="M",
        help="plan-free straight lift of M metres before planning home (the pose the "
        "planner refuses as a start: fingers at the table after an abort or a jog); "
        "0.08 is usually enough",
    )
    args = ap.parse_args()
    cli_common.run_task(args, lambda ctx: build_legs(ctx, args.lift_first))
