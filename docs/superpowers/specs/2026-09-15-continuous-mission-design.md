# One continuous mission — design

Owner decisions, 2026-09-15. Replaces the fly-to-a-bench-pose opening and the
rests either side of it, and removes the bench constants a wheelchair
invalidates. Measured against run-20260915-125518 (18.1 s wall, scan to home).

## Where the time goes today

| motion | s | work? |
| --- | --- | --- |
| scan flight to the look pose | 3.39 (+0.21 plan) | no |
| press descent / push / recoil | 3.34 | yes |
| retreat, grip descent, close, lift | 5.60 | yes |
| carry, set-down, release | 3.95 | yes |
| retreat + home | 3.85 | no |

Planning is already hidden under motion (gaps between legs ~0) and detection
itself costs 0.54 s. The waste is the opening flight and the parking move.

## Decisions

1. **The look is a wrist tilt, not a flight.** Measured on the real planner from
   HOME: flying to the scan pose 3.89 s, re-orienting in place 4.01 s (the
   wrist-flat -> tool-down IK family change swings ~3 rad either way), a
   joint-space `joint_6 -= pi/2` 1.72 s with no planner call. The look is
   therefore a joint-space target derived from the arm's own rest pose:
   `LOOK = HOME with joint_6 - pi/2`. No bench Cartesian pose, so it holds
   wherever the chair stands.
2. **A sweep does the searching.** The look sees 0.58 x 0.33 m at 0.30 m range
   (camera at [0.36, 0.00, 0.27], nadir). If nothing commits, the base joint
   sweeps a bounded arc at low speed, stopping the instant a coarse fix
   commits. This is also how the box is found on a chair.
3. **Two detection paths.** Coarse: during motion, stillness gate off, button
   circle not required, looser agreement — answers "is there a box, roughly
   where", stops the sweep, never aims the press. Precise: unchanged (still
   frames, circle required, three agreeing within 15 mm, residual gate) and
   taken at rest; it is what the press and grip are aimed with.
4. **Continuity.** Contacts keep their rests — press touch, push, grip close,
   set-down are physical stops. The rests between approach, staging and descent
   go away. cuRobo v1.0.0 plans start and end at rest, so this is done by
   composing those segments into ONE guarded execution client-side (the
   re-timer already flows chained unguarded legs); the via-point action
   proposed in RAMMP-CuRobo (#33) remains the deeper fix and drops into the
   same seam.
5. **The table comes from the camera.** The depth frames the detector already
   uses carry the table plane; fit it and drop the measured `table_z`. The lid
   drop spot becomes box-relative rather than a fixed base-frame point.
6. **Runs still end at HOME.** Unchanged, so the start-pose guard keeps working.

## Flow

```
press:close dispatched (rides everything below)
look: plan_to_joints(LOOK)          coarse detection runs during the motion
  |- coarse fix commits  -> stop the motion
  |- nothing by the end  -> sweep joint_1 through +/-arc, slow, same stop rule
  |- nothing at all      -> NO BOX, home, exit 2
precise fix at rest (~0.5 s)        table plane fitted from the same frames
press: approach + staging + descent as ONE guarded execution
  ... push, recoil, retreat, grip, carry, set-down, release unchanged ...
home
```

## Risks

- The nadir view is narrower than the old scan pose, so the sweep is
  load-bearing rather than a fallback.
- The first tilt is planned before any table fit exists: it runs in the
  pre-detection world, and it is the one motion that precedes perception.
- A coarse fix biased by 10-15 mm that a precise fix happens to agree with
  would fly a slightly wrong pre-plan; the residual gate against the fitted
  table catches gross errors, and disagreement beyond 5 mm forces a replan.
- Merging the approach into the guarded descent means the guard is armed
  through free air; a trip there is a genuine collision, but it is a new place
  for the guard to fire.

## What implementation changed (2026-09-15)

Measured against the live planner and camera while building it.

**The look needed a lift, and then it needed a bigger one.** Turning the
wrist alone leaves the fingertips at z 0.154 — above the pre-detection
keep-out band (top z 0.105) but inside cuRobo's collision activation
distance, and the goal is refused outright (`IK_FAIL`). Bisected against the
live planner: refused at lift 0.00 and 0.10, solved from 0.15. The look
opens the elbow by `LIFT_RAD` and takes the same angle back on the wrist, so
the camera still points straight down.

`LIFT_RAD` 0.25 cleared the band and FAILED IN THE FIELD (2026-09-15, first
`--execute` run). What has to be in frame is the whole box TOP, which sits
`dims.z` nearer the camera than the table and is therefore seen through a
smaller window, and a top touching the image border is refused. The box
stood at base [0.48, 0.12]; the lift-0.25 look reached x 0.55 at lid height
and the top's far edge sat at 0.54, leaving 10 mm. The semantic model
reported it jammed against the frame edge while the plateau geometry found
nothing in 234 frames. Sizing the look by what it sees of the TABLE was the
mistake.

`LIFT_RAD` 0.8 costs 3.42 s against 2.74 s, still under the 3.89 s scan
flight it replaces, and sees 2.8x the lid-height area:

| lift | look | lid-height frame |
| --- | --- | --- |
| 0.25 | 2.74 s | 0.47 × 0.26 m |
| 0.80 | 3.42 s | 0.77 × 0.44 m, x 0.28-0.71, y ±0.38 |

The narrow axis is x, because the image's long axis lies along base y. The
sweeps cost 1.50 s and 2.86 s and swing that narrow axis somewhere new.

**One budget for the search.** Each step gets a bounded beat (0.5 s after an
empty pass, 2.0 s after a coarse sighting) and the whole search shares
`detect.timeout_s`; whatever is left is spent standing where the search
ended. Without this a coarse glimpse that never confirmed would strand the
arm staring at it while the directions it had not looked in went unlooked.

**The semantic ladder needed its own clock.** It used to be bounded by the
detect time left, which the search now spends MOVING — so the cloud rung had
nothing to call with, every time. It gets its own budget, and the wait after
it is a beat rather than another full detect window.

**Continuity landed in the merge rules, not in the task.** `can_merge` now
lets a guarded leg be the TAIL of a group: the approach and the descent it
leads into are re-timed into one profile that flows through the junction,
and the torque guard arms at the junction rather than watching the
approach's own dynamics. Two rules keep it honest — nothing merges AFTER a
contact, and a leg whose timing is already baked in (a warped descent)
cannot join a group profile. The guarded tail's expected-contact fraction is
rescaled onto the merged path, or an honest contact would read as an early
strike. The staged press is built as one chain and costs the same ONE
execution the single-solve press does (proven by the `staged` goal audit in
`scripts/press_demo_e2e.py`).

**Still open.** A coarse sighting can stop the look part-way through its
tilt, leaving the camera at an angle the precise fix may not commit from;
the confirm beat bounds the cost and the sweeps re-tilt, but it wants
watching on the first live run. The cuRobo via-point action (RAMMP-CuRobo
#33) remains the deeper fix for continuity and is not part of this work.


## Second pass: the choppiness (2026-09-15, after two live runs)

Two full runs succeeded. They still read as slow and choppy, so the
executions were counted and the stops that were doing no work removed.
Measured on the live planner against the box the second run found.

| | before | after |
| --- | --- | --- |
| reach and press | 5.69 s, 1 execution | 2.89 s, 1 execution |
| lift, carry, set-down | 4.84 s, 3 executions | 2.88 s, 1 execution |
| grip:down | 4.66 s | 4.10 s |
| executions per run | 10 | 8 |

**`merge_press` is off.** The one-solve press existed to remove the staging
stop, and there is no staging stop any more: the approach is chained into
the guarded descent and re-timed into a single execution. So the merge
bought nothing and cost 2.8 s, because it cruises free air at
`warp_fast_speed` where the chained approach cruises at `TRANSIT_SPEED`,
and it plans its lateral run in the reduced world where the container is
invisible. The path still works and is still tested; it is just slower.

**Grip and place became one chain.** Planned together under the post-press
retreat, so the lift, the carry and the set-down merge into one execution.
The set-down gave up its warp to join that group, the group profile giving
it the same fast-then-slow.

**The slip check moved from the lift to the set-down.** A leg carrying a
verify closes its merge group, so the check was what forced the stop after
the lift. It now runs at the bottom of the set-down, the last moment before
the fingers open and the moment it matters. A slipped lid was already
caught there as "never felt the surface"; now it is named.

**`warp_fast_speed` 0.5 -> 0.75** (owner). The free-air part of a guarded
descent now matches the cruise the chained press approach already uses, and
the loader's bound became `TRANSIT_SPEED` rather than a separate number:
such a segment may go as fast as the arm ever moves in the open, never
faster. The only leg it still governs is grip:down. What remains of that
leg's time is its slow zone, which is `grip_speed` 0.15 over
`warp_slow_frac` 0.30 of the path, about two thirds of the descent's
duration.

**Stops that remain, each earning its keep:** the settle after the search
that the precise fix needs; the press contact; the push and the recoil off
it; the retreat to the hop, because the fingers open there and must be open
before descending onto the knob; the grip close; the set-down contact; the
release.


## The chained press tripped on nothing, and why (2026-09-15)

First `--execute` run after the second pass: `press:down -> touch (guard
tripped EARLY at 75% of the stroke, contact expected ~94%)`, 91 mm above the
button with nothing there. Peak 5.91 Nm, the same magnitude as a real press,
which is what made it look like a strike.

Replayed against the live planner with that run's own box pose:

| | |
| --- | --- |
| descent begins at | time 0.700 |
| guard could trip from | time 0.750 |
| it tripped at | time 0.750 |

Tripping at exactly the first armed instant is the signature of a deviation
that was already there, not one that arrived. The re-timed profile reaches
the descent cruise BEFORE the junction and holds it steady, so the motion
was not the problem. The BASELINE was: it was captured at the junction and
first compared 0.14 s later, and the corner between a 0.75 cruise and a 0.35
one leaves a torque offset a 3 Nm touch threshold cannot tell from a touch.
The peak of 5.91 Nm was accumulated across the whole approach against the
baseline taken at its start, not measured at the trip.

Two changes, both in `runner._run_motion`:

- **The baseline is taken AT the arming point**, not at the junction. Any
  steady offset the corner leaves is then absorbed into the baseline rather
  than measured against a stale one.
- **The settle after a junction is a TIME** (`GROUP_SETTLE_S` 0.25 s), not a
  fraction. `WARP_SETTLE_FRAC` was tuned for a warp's gentle 8-sample ramp;
  a fraction of a short trajectory is a shorter wait for the same physical
  event.

On that run's geometry the guard now arms at time 0.790 instead of 0.750,
which is 38% along the descent rather than 21%, or 69 mm above the button
rather than 91 mm. The cost is 22 mm of descent no longer guarded. The
verify's own early-strike floor sits at 0.795, so the guard now arms
essentially where a trip can first be read as a successful press.

## Decision 5 reversed: the table does NOT come from the camera (2026-09-16)

Field: two runs with the box right under the look, 438 frames, no fix, the
last reject "no points at container height". The detection code had not
changed since the runs that found the box the day before.

Root cause, shown on the 1 September capture (camera at ~0.51 m, the same
height as today's look) with only the table source varied:

| table used for the lid band | frames that find the top |
| --- | --- |
| surveyed, -0.027 m | 39 / 94 |
| fitted from the camera, as shipped | 7 / 94 |

Passive stereo reads a blank bench as a broad smear, not a peak: on those
frames it spans roughly -0.07 to +0.02 m around the true -0.027, and the
MODE of that smear sits up to 35 mm above the real surface. The fit took the
mode. `BAND_TOL_M` is 35 mm, so a table read that high puts the lid on the
edge of its own search band, and the detector reports that as "no points at
container height". The first two frames of a run hit, then three agreeing
fits lock in and every later frame is rejected — the regression test
`test_detection_does_not_follow_a_table_the_camera_reads_high` reproduces
exactly that, "2/6 frames found a container top", before the fix.

The fit is removed from the live path: the band, the residual gate, the
pinned origin and the lid drop all use the surveyed table again. Through the
watcher's own tick the capture then finds the top in 39 frames, 34 with the
button circle. Taking the table from the scene remains a reasonable goal for
a wheelchair, but not from this camera's depth of a blank surface; the arm's
own first touch is a better-conditioned measurement of it.

## Arm driver crashes seen the same morning

Not caused by the mission, recorded here because they read as "the arm
stopped working". From `~/.sheppy/logs/arm/` and container events:

- 09:35, exit 133: `std::runtime_error: not connected !!!` before the node
  came up. The driver started before the arm was reachable, and connects
  once with no retry.
- 09:55, exit 133, 81 s after start:
  `KDetailedException WRONG_SERVOING_MODE, must be low level servoing mode`
  from BaseCyclic Refresh. The arm left low-level servoing while the
  driver's 1 kHz loop was running, and that loop has no handler around
  Refresh, so the process terminates. Anything that takes the arm to
  single-level servoing does this: a protective stop, the e-stop, the Kinova
  web app, the joystick, or the wrist buttons. The driver's own
  `clear_faults()` already knows a protective stop drops low-level
  servoing; the RT loop dies before anything can call it. The planner log
  shows a plan refused 13 s earlier because the arm's live state was in
  self-collision, so the arm was already somewhere unusual.

The mission's gripper path cannot cause the second: it goes through the
cyclic interconnect command, which the driver chose precisely because the
high-level gripper API needs single-level servoing. Both are driver-side,
in kinova-gen3-ros2.

What the mission did get wrong: `press:close` was the first actuation and
nothing checked the planner first. After a stack restart cuRobo takes
17-21 s to load and set_world waits 5 s, so a run started in that window
closed the fingers and then died. `readiness_refusal` now waits up to 30 s
for the planner, and for the driver when executing, before anything moves.


## The scene camera (2026-09-16)

Built on the owner's go: calibrate the scene camera to the arm, locate the
box from it before the arm moves, remove the last perception stop, set the
two speed knobs.

**Calibration without motion or a marker to attach.** The cabinet's ArUco
tag is 57-58 mm (four depth readings on white paper; the "50 mm" label is
wrong) and only 32 pixels wide in the scene camera, where its pose flips
between mirror solutions (34 deg scatter). So the tag gives position only;
the door and table PLANES, thousands of depth points each, give
orientation. The wrist camera sees the tag and the door at 30 cm from HOME
and is already calibrated to the arm. The solve refuses when the
door-to-table angle disagrees between the frames by more than 2 deg — the
first attempt was refused at 3.7 deg, which found two things: the table is
tilted 1.5 deg from base +z (now `table_normal_base` in world_bench.yaml,
from the wrist camera's own view of the table), and the wrist's door
normal must be fitted on a broad patch with the paper masked (3.1 deg of
scatter became 0.3). Result: mismatch 0.47 deg, the arm's fingertips
project onto the fingers in the scene image to ~10 px, the table to 17 mm.
`scripts/calibrate_scene_camera.py`; launch publishes base_link ->
scene_camera_link; `scripts/stub_scene.py` renders the same synthetic box
from that pose for the harness.

**Locating the box.** OWL on the scene image says which thing is the box;
the depth inside its pixel box at lid height gives the position. Two field
corrections: the box's NEAR face crosses the lid band from an oblique view
and dragged the centre 5 cm toward the camera (only the highest slab
counts now), and a single stray pixel stretched the min-area rectangle
(footprint by 10-90 % spreads, yaw by the tightest 3-97 % box). Live:
1.8 s per fix, ~1550 lid points, yaw agreeing with the wrist's. Its
position was 5 cm from the wrist's last fix of the day before — the box
had probably been moved; the mission now records the scene-vs-wrist pair
every run, and those pairs are the calibration's refinement data.

**The mission.** Scene fix -> approach to staging in ONE goal from HOME
(2.6 s at transit 1.0) with the descent pre-planned on the way -> wrist
confirms at staging -> the pre-planned descent flies when the wrist
agrees within 5 mm. Falls back to the wrist search from wherever it is.
Perception moved onto its own thread and node: under one executor the
detector's timer starved (the stub bench delivered 1.6 frames/s to it and
the wrist could not confirm inside a one-second reach; the fix is also
what makes in-flight confirmation possible on the bench). A stop the leg
asked for (stop_when) is never judged by its contact verify.

**The last stop, gated.** `detect.confirm_in_flight` chains the approach
into the guarded descent as ONE goal; the wrist confirms in flight; the
goal is cut at the junction only when nothing has confirmed by then or the
wrist disagrees with the scene. Gated on the wrist camera's timestamp lag
being measured (`stamp_offset_s`; `record_scan_frames.py --lag-sweep` and
`--analyze-lag`, `perception/lag.py`), because a frame shot in motion is
otherwise placed where the arm was, not where the shutter fired. Proven in
the harness (7 goals, reached and touched in one motion); the lag itself
needs one recorded bench run.

**Knobs set.** `TRANSIT_SPEED` 1.0, `grip_speed` 0.35. On the live
planner the scene-first mission's timed motion is 9.1 s (approach 2.6,
descent 1.1, grip 2.4, lift+carry+set-down 2.8), the search gone.
Resting tool-down instead of at HOME would save a further 1.1 s at each
end (measured); left to the owner.

**Still open.** The residual pairs will say how good the calibration is
at the box; four well-spread runs are enough to refine it.

## First attended run on the scene path (2026-09-16)

"It missed the centre of the box and didn't press down hard enough."
Four separate things, from the run's own JSONL rows and the ROS graph:

1. **The scene OWL saw no box in 3.0 s**, so the run fell back to the
   wrist search — which found the box and opened it. The launch-started
   scene instance published NOTHING even when enabled (0 messages in
   45 s, GPU idle, 19 % CPU) while the launch's wrist instance
   heartbeated; a fresh instance started by hand on the same camera
   found the box within 1.4 s of enabling, at scores 0.18-0.23. Two
   differences between them: the launch's instance had never run an
   inference before the mission enabled it, and the mission enabled
   both instances in the same instant. Mitigations, since the wedge
   itself was not reproduced: the node now runs one warm-up inference
   on a blank frame before it reports ready (the first CUDA pass is the
   slow, allocating one), the mission no longer enables the wrist
   instance at start, and the scene instance gets its own score floor
   (`min_score` 0.12 from the launch) because 0.18-0.23 against a 0.18
   floor is a coin flip per frame and the lid-slab geometry behind the
   bbox rejects anything that is not a box top anyway.
2. **The press missed the centre by ~6 mm — and it was the planner, not
   the pads or the camera.** First read as a constant +6 mm x and
   trimmed (`press_offset_xy [-0.006, 0]`); that trim is REVERTED, see
   the next section: the miss is radial (along the arm's bearing), began
   with the 2026-09-14 migration, and is the descent plan bowing.
3. **It didn't press hard enough.** The stroke met the 7.0 Nm backstop
   at 5.6 Nm of rise without popping the button — off-centre by 6 mm,
   the pads sat on the button's edge and the force went into tipping
   the lid. `touch_nm` 7.5 stays as a backstop; a centred press pops
   inside `button_travel_m` without reaching it. The `torque_peak` of
   7.1 on that row is the chained approach's dynamics over the whole
   merged group, not the touch force; logging the touch alone is a
   separate small change still to make.
4. **"terminate called without an active exception / Aborted" at exit**
   was the perception thread still spinning while rclpy shut down; the
   mission now stops its executor and joins the thread before the nodes
   are destroyed. The "LID PULLED" line printed on a failed grip was a
   print bug (the grip closed on air at 0.786, outside the band) and
   now says GRIP FAILED.

## The 6 mm miss was the planner's descent bowing (2026-09-16)

"It keeps hitting just to the left of the button." The circle detector
was the suspect; the run history cleared it in one table. Tool-at-touch
minus detected-centre, attended presses only: 2026-09-03/04 (11 presses)
x -0.3..-0.8 mm, y ~0; from 2026-09-14 (5 presses) +6.6, +2.1, +5.6,
+5.8, +6.1 mm — and not along world x but RADIALLY, along the arm's
bearing to the box (bearing -15 deg: (+6.1, -1.7); +6 deg: (+5.8, +0.6)).
The one run that tripped nearer its planned end (14.8 mm above the fix
instead of 23-25) missed by 2.1 instead of 6. What changed on 09-14 was
the planner: v1.0.0's PlanToPose has no `approach_offset_m`, so the
vertical final approach became two ordinary plans, and a plan between
two vertically aligned poses is vertical only at its ends.

Measured on the live planner (plan-only, FK of every waypoint through
the driver's URDF, which agrees with cuRobo to 0.01 mm): the final
60 mm segment bows radially outward 7.4 / 4.9 / 6.8 mm for the three
boxes above, widest 26-29 mm above the target — which is exactly the
trip height (travel_m 15 + TCP_OFFSET 11). Predicted vs measured misses
agree to a millimetre.

**Fix** (`runtime/kinematics.py`, `runtime/approach.py`): the waypoint
stays a planner result, exact and collision-checked; the stretch below
it is built from the arm's own kinematics — forward kinematics from the
driver's URDF (tool_frame = end_effector_link + 0.120 m, cuRobo's
definition) and a damped-least-squares IK walked down the line in 2 mm
steps holding the goal attitude. Live through the mission's client:
0.07 mm from vertical over the whole descent, 31 points in ~100 ms,
joint steps <= 0.006 rad, and cuRobo accepts the line's end joints as
a valid target (its own plan to them ends at exactly our joints). One
planner round trip instead of two: staging-to-target now plans in
0.31-0.35 s wall. If the line is refused (joint limit, singularity) the
planner's plan flies and the mission prints "NOTE press:down: PLANNER
descent (may bow ...)". Every vertical approach benefits: press, the
grip's descent onto the knob (0.05), the pre-planned descent from the
scene fix (0.04).

`press_offset_xy` is back to zero: it was the bow, and a trim would
have double-corrected. The circle detector was never wrong.

Two things found on the way, both outside the code: the D405 dropped
off USB at 13:56 (`No such device` in the wrist_camera container; the
sensor closed) — re-seat and restart that node; and the launch-started
scene OWL instance holds its subscriptions and answers parameters but
receives no colour frames while the same code started by hand does. Its
`output="screen"` prints are the place to look.

## Two more from the bench (2026-09-16, late afternoon)

**"The D405 doesn't see the box, the scene camera does."** Both true, and
neither camera is wrong. Since 13:33 the box on the table is a different
container: the scene puts its lid top at z 0.064 (the model's is 0.085,
and at 13:33 the wrist measured 0.0855 on the box it pressed), the wrist
measures its top 0.11-0.12 m square (model 0.075). The scene locator's
gates are loose by design (lid band +/-35 mm, footprint >= 0.4 of the
model) so it finds it; the wrist's are the press's own (footprint within
tolerance of the model, a button circle of the model's diameter) so it
refuses — correctly, since the mission has no numbers for that box's
button or knob. The mission now saves the wrist's frame (colour, depth,
K, pose, the reject reason) whenever the wrist fails to confirm at
staging or the search ends empty: ~/.ros/rammp_box_opening/captures/
wrist-<staging|nobox>-<stamp>/, replayable with
`record_scan_frames.py --analyze`.

**"The arm keeps flipping."** The press attitude was yaw-steered to the
box's bearing (the 2026-08-25 family); the LOOK pose is `[180, 0, 0]`
yawed +90 deg, and the steered attitude at a box on the arm's right was
105 deg away from it — the wrist turned that much between search and
approach. The attitude is now FIXED in the world at the LOOK orientation
(`press_attitude_rpy_deg [180, 0, 90]`, `press_yaw_steer false`): tool
down, wrist facing straight ahead, from look through approach, press,
grip and place. Reach checked on the live planner at 20 positions over
the table (x 0.32-0.62, y -0.30..0.30) from HOME and from LOOK: all
reachable, straight descents everywhere. `press_yaw_steer: true` brings
the bearing family back per container.

## First full run with the fixed wrist: the lid's drop spot (2026-09-17)

The press landed -0.9 mm radial / -0.1 mm tangential with the full 4 mm
push: the straight descent holds. Then `place:lid:transit` was IK_FAIL and
the mission ended in a traceback with the arm above the box.

Not the attitude. The carry pose above the drop spot [0.493, -0.164, 0.166]
failed in every attitude tried (fixed +90, -90 and 0 deg, and the old
bearing-steered one) in the run's world, and planned in a table-only
world. The run's world holds the pressed container padded to a 135 mm
cuboid aligned to the BASE axes (CONTACT_SHIFT_PAD_M, whatever the box's
yaw). The drop spot sat lid_place_min_clear (156 mm) from the box along
the direction to the configured lid_place — 27 deg off -y this time —
which puts it about 20 mm nearer that cuboid's corner than a spot on an
axis. Around the same box at the same distance, the four axis spots
planned and the eight diagonal ones did not.

Fix: drop spots lie on the base axes through the box, the side nearest the
configured lid_place first (`lid_drop_candidates`); `build_place_legs`
tries the next spot when the planner refuses the carry or the set-down
(and says so); a refused grip/place plan after the press sends the arm
home instead of a traceback. Checked plan-only with the real builders —
approach, press, push, grip, carry, set-down — at 32 box poses (x
0.40-0.62, y -0.22..0.14, yaw 10 and 35 deg): all plan with the fixed
wrist, none needed the fallback.

Found on the way: the Gemini dropped off USB (the D405 now enumerates on
the port the Gemini used), so the scene path fell back to the wrist
search; and pytest 9.1.1 landed in ~/.local on 2026-09-16 16:34, which
ROS Humble's launch_testing pytest plugin cannot load under — the unit
suite runs with `-p no:launch_testing -p no:launch_ros`.

## The staging aim is the button's circle, nothing else (2026-09-17)

"It missed the knob; it didn't see the box right over it; the start is too
slow." One run, three findings, and for once a picture: the mission now
saves the wrist frame whenever the wrist fails at staging, and the frame
from this run shows the 4.1-inch OXO dead centre in the wrist's view with
the button plainly visible — refused 103/104 frames as "footprint 0.11x0.10
vs model 0.07x0.07". The box IS 0.10 m square (its lid plateau covers
0.098-0.104 m in that frame; the button is 0.047 m across): the model was
the 2.9-inch box's, and the plateau detector's footprint gate, right for a
scan across a table, was refusing the box the arm was parked over.

**The flow is now the owner's (2026-09-17):** the scene camera's OWL finds
the box and the arm flies over it; from staging the press is aimed by
OpenCV's circle alone (`perception/button_aim.py`) — the largest circle of
the button's diameter, at lid height, on a flat lid (the ring around it is
lid too, which is what tells the button from the lid's rounded corners) —
with no OWL and no footprint gate; the OWL-gated wrist scan runs only when
the scene camera finds nothing. `test/data/wrist_staging_big_box.npz` is
that refused frame, and the aim finds the button on it 4 mm from where the
wrist's own search placed it a minute later. `oxo_pop.yaml` is the
4.1-inch box now (dims 0.104, button 0.047); the 2.9-inch one is
`oxo_pop_small.yaml`.

**The missed knob.** The touch met the surface 17.7 mm ABOVE the predicted
button top: the knob was already up from the previous run, the "press"
pushed it back down (sealing the lid), and the grip closed on air. A touch
10-35 mm above the button top now means "already popped": no push, retreat,
grip (`already_popped`, `build_push_legs(push=False)`).

**The scene camera's 5 cm.** The scene put the box at [0.520, -0.005]; the
wrist found it at [0.573, -0.022] — and the day before, [0.5165, -0.003]
against [0.562, -0.025]. The calibration's solve had the camera ~5 cm
short along +x; an interim +0.052/-0.017 m xy correction is in
`camera_scene.yaml` (z left alone: the lid reads 15 mm low and the near
table 19 mm high, which is a rotation). With the circle aim, staging only
has to put the button inside the wrist's view, so this no longer decides
a run; every run at staging now records a real scene-vs-wrist pair for
the refinement.

**The slow start.** The scene OWL is enabled the moment the locator
exists, so its first bbox (~1.2 s of tick + inference) lands during the
readiness checks instead of after them; the SCENE line now prints the
locate time and the time since start.

Plan-only, real builders, 32 box poses x both attitudes with the new
dims: 64/64 plan every phase.

## Why the OWL instances went deaf (2026-09-17, found)

The launch-started scene OWL instance went silent three times in a day —
ticks running, parameters answering, the watchdog re-subscribing every
5 s to no effect, image callbacks simply never delivered again — and each
time right after it had run inferences. The wrist instance, which had not
inferred since its start, kept heartbeating. Both grabbers created their
`TransformListener` with its default spin thread, so every node owning a
grabber was spun by TWO executors: its owner's (the OWL node's
`rclpy.spin`, the mission's perception executor) and the listener's own.
rclpy does not guard a node's callbacks against that; a 0.65 s inference
dispatched on one thread while the other keeps taking from the same
subscriptions is how a node ends up alive and deaf. `spin_thread=False`
in both grabbers (whoever spins the node feeds TF). Stress-tested on the
bench: three 20 s inference windows, 20/19/19 bboxes, alive with 2 Hz
heartbeats after each, zero re-subscribes — where before it died inside
the first. The watchdog stays as a backstop.

Also from the profile of the same afternoon: the scene locate's 3.9 s was
3.7 s of point-cloud lift, of which 1.6-2.3 s was the RANSAC table fit
behind a number that is only printed (now 3000 subsampled points) and the
rest a full-density 320k-point cloud (now every second pixel: ~300 lid
points at 0.7 m). The OWL's own share was 0.13 s once pre-enabled.

## The button's rim, and pressing where a finger would (2026-09-17, late)

"It went to the box but didn't press it, and it wasn't the centre of the
button, it was the top" — with the box CLOSED. The run's numbers: the arm
went exactly where it aimed (-0.1 mm), and the touch met a surface 9.6 mm
above the lid plane; the morning's press on the same closed box (aimed
from 49 cm by the plateau path) met it at +2.3 mm and popped it. The
depth profile of the button on the saved staging frame explains the
difference: the 4.1-inch OXO's button is not flat — a ring ~7 mm proud of
the lid, strongest on one side, around a ~5 mm dimple where a finger
presses. Closed pads sent onto the ring press nothing; a 4 mm push there
is theatre. (My first reading, "the box was already popped", was wrong:
+9.6 mm is the rim, +17.7 mm is a popped knob.)

Three things, built:

1. **The press point is the dimple** (`button_aim.press_point`): inside
   the seam circle Hough finds, the centroid of the lowest depth band
   when it sits >= 2.5 mm below the disc's median, near the middle, on
   >= 30 pixels; the seam centre otherwise (the small box's flat button).
   Its z is the disc's median — what the pads meet around the dimple.
2. **The aimed frame is saved every run** (`save_aim_frame`: circle,
   seam centre, press point drawn; the aim logged with its pixels,
   dimple depth and capture path; the newest 20 kept). The rim miss had
   no frame and was read blind.
3. **The touch's height is judged** (`touch_verdict`): +0..5 mm press;
   5..12 mm the RIM — no push, back off to staging height, aim again on
   the dimple and press again once, then stop honestly if the re-aim
   lands on the same point; 12..35 mm a popped knob — no push, grip;
   above that not this button — stop. The popped threshold moved from
   10 to 12 mm so a rim contact can never read as "popped".

> **CORRECTION, 2026-09-21 — there is no rim and no dimple; item 1 is
> removed.** The first reading above was the right one: the +9.6 mm
> surface WAS the button, already popped. On 2026-09-21 the 13:35 aim frame
> shows the whole disc +10 mm above its own lid ring before the press (a
> difference inside one frame, so no calibration enters it); that morning's
> pop check read 9.2 mm; the touch met it at +10.8 mm; the 11:26 frame of
> the same box, closed, has the disc flush (+1 mm). This knob stands about
> 10 mm proud when up with the lid on the container (+17.7 mm was measured
> once, 09-17 11:22). The "dimple" was stereo noise at the sensor's 1 mm
> steps: it fired on one real aim in four, moved that press 13 mm off the
> circle's centre, and the owner saw it land "too far to the right". The
> press point is the centre of the seam circle and nothing overrides it
> (`button_aim.find_button`; regression on that day's frame,
> `test_the_press_point_is_the_centre_of_the_circle`).
>
> **The thresholds, settled by the bench the same afternoon (14:16).** The
> full mission pressed, the box OPENED (owner's eyes; the pop check read
> [11.1, 10.1, 11.1] mm), "popped" still began at 12 mm, so it pressed
> again — and that second stroke rode the soft raised knob all the way
> down under its 3 Nm guard (touch at -3.7 mm) and shut the box. Two runs
> at 14:14 had pressed an already-open box shut the same way. Now:
> the CAMERA decides — `KNOB_UP_MIN_MM` 6 (down reads -0.8..+1, up
> 9.2..11.1): the pop check from the hop, and at the AIM the button against
> its own lid ring in the same frame (`ButtonSighting.above_lid_m`):
> already up means the box is open and is NOT pressed
> (`refuse_open_box`: stop, say so, home). The lid-height gate lets the
> centre stand up to `KNOB_MAX_UP_M` 25 mm above the lid so a high knob
> (+17.7 was met once) is still seen as the button instead of sending the
> mission to the unchecked wrist search. The TOUCH is the fallback only:
> `POPPED_MIN_M` 12 -> 8 mm (closed touches -3.7..+6.2, raised +9.6,
> +10.8, +17.7). Harness: `open` scenario (stub_d405 --knob-up).

**A live aim view was built and removed the same day (2026-09-21).** A
page on 127.0.0.1 showing the wrist camera, every Hough circle with its
verdict, and the arm's target. It found the dimple bug's symptom at a
glance; once the press worked the owner had it taken out ("it makes it
longer" — it also had the aimer gate all 8 circles per frame to feed the
page). What remains for a miss: the aim frame saved every run
(captures/wrist-aim-*), the pop-check frame (wrist-pop-*), and the run log.

Not built (4 and 5 of the five): confirming the pop optically before the
grip, and the aim's 4.4 s (frames rejected as moving / pose not yet at
the stamp).

## Correction: no "rim" band; the push presses until the stop (2026-09-17, later)

The 5-12 mm "rim" band refused the very next run: the touch read +6.2 mm
with the pads on the button (owner watching), the mission would not push,
re-aimed to the same point and went home. Two samples were not a
threshold. The band is gone: below 12 mm is the button, 12-35 mm a popped
knob, above that not this button. The re-aim path went with it.

What the owner asked for is what the push now does: "press down until it
feels the force". The push's bound `button_travel_m` is 10 mm (was 4): the
touch stops at a light 3 Nm contact with the button barely loaded, and 4 mm
from there popped the 4.1-inch button only from a +2 mm contact (10:52), not
from +6 mm. The push ends when the button's stop is felt — the backstop,
touch_nm 7.5 minus the 3 Nm contact, above the in-contact baseline — with
10 mm as the room it has; the 2026-09-03 video of an unbounded 7 Nm stroke
compressing the container is why it stays bounded. The aim frames stay.

## Four of the ten (2026-09-17, evening): refine, refit, confirm the pop, and one that measured out

**Refine the scene calibration from the runs** (`perception/scene_refine.py`,
`scripts/refine_scene_calibration.py`). Every staging run records a
scene-vs-wrist pair; the script fits the rigid correction that maps the
scene's points onto the wrist's — Kabsch when >= 4 pairs span >= 0.15 m,
a translation otherwise — reports the rms before/after per pair, and
`--apply` writes camera_scene.yaml with a backup. The harness had written
~90 synthetic pairs into the real file; those are purged, and a run with
its own `--scene-calib` now records next to that file. Two real pairs so
far, at one spot: a translation of +5.7/-3.9/+24.2 mm (applied; rms 25 ->
3 mm); the mission prints the pair count, mean offset and spread after
every run and says when a rotation fit is available. Place the box at
different spots to get there.

**Re-fit the pre-planned descent instead of re-planning it**
(`runtime/approach.refit_descent`). A leg planned with a vertical approach
now carries its waypoint (index, joints); when the aim moves the target
by up to 30 mm, the leg is rebuilt from the arm's own kinematics with no
planner round trip. Live: 115 ms against a 350 ms re-plan, 0.012 mm from
vertical, and cuRobo accepts the end joints. The 5 mm reuse tolerance
stays for the untouched plan; beyond 30 mm it re-plans as before.

*Where the correction is made (2026-09-21).* The first version kept the
planner's part down to the waypoint and jogged sideways THERE, 56 mm above
the button: the arm descended over the scene camera's spot and swerved
onto the button at the last moment — the owner saw it "shift to the right
as it was pressing, almost missing it" (the touch itself landed 0.4 mm
from the aim). The re-fit now corrects at the height the leg STARTS from
— staging, where the arm stands still after aiming: a level line to above
the new target, one straight line down to the approach offset above it,
a rest, and the same final line every planned descent ends with. The
guarded part is unchanged (same rest, same arming height: replayed
offline, pads 41 mm above the button either way); the whole descent is
over the aimed axis from staging height down.

**Confirm the pop before the grip** (`knob_height_mm`, `_press_again`).
After the push and the retreat the wrist reads the knob's height at the
button's pixel from the hop (median depth in a 41 px window over up to 3
frames): >= 12 mm up is a confirmed pop; below that the fingers close, the
arm rises to staging height, presses again once, and reads again; a
second miss stops the run honestly. The stub bench pops its button after
the first trip (stub_arm publishes /stub_bench/knob_up, stub_d405 renders
the knob 16 mm proud) so the harness exercises the check.

**OWL at half resolution — measured, not done.** OWLv2's processor
resizes every frame to its fixed 960x960 before the model, so 1280x720,
640x360 and 422x238 inputs all cost 0.62 s on the Orin, and the smaller
ones only score lower (0.259 -> 0.225). The time is the model's; the knob
was not added.

## Two containers in the scene (2026-09-17, evening)

A black 19x9 cm lidded tray sat next to the OXO box. The scene OWL scored
them 0.255 and 0.255, its single best box flipped between them every tick,
and the lid-slab gates let the tray through (it is about as tall) — the
locator put "the box" 30 cm from where it was, on whichever box arrived
first. Now the OWL node publishes its top three boxes (six fields each,
best first; consumers that read the first six see what they always did),
and the locator tries each against the container's geometry: at lid
height, a face not a strip, and — new — no longer than 1.4x the model's
longer side, which the tray fails ("the top is 190x90 mm, the container
104x104 — another container"). The first candidate that fits is the box;
the failure message lists what was refused. Verified in unit tests and
the harness; the live check waits for the stack (it was down at the end
of the day) — the next run's SCENE line, or "none of the OWL's N
candidates is this container", is the proof either way.

