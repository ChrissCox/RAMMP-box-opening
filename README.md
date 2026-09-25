# RAMMP-box-opening

Open an OXO POP container with the Kinova Gen3 7-DoF + Robotiq 2F-85:
find the box with the wrist camera, press its lid button, pull the lid
and set it aside — one autonomous mission (`press_demo`) composed of
guarded legs. This repo is a **client** of sheppy's two arm containers
(rammp-deployments, `december_2026` manifest): the kinova-gen3-ros2 driver
executes and grips, the RAMMP-CuRobo v1.0.0 planner plans. Interfaces and
the robot description come from the `~/rammp_deps_ws` overlay — no copied
code, no submodule, no modifications to either repo.

Design spec (authoritative): `docs/superpowers/specs/2026-08-14-box-opening-design.md`
Implementation plan: `docs/superpowers/plans/2026-08-17-box-opening-phase-0-1.md`
Hardware sessions: `docs/HARDWARE_BRINGUP.md`

## Build (zsh)

The dependency overlay, once (and again when a pinned source moves): the
`rammp-interfaces-ros2` v1.0.0 clone in its `src/`, plus two packages built
in place from their own checkouts.

```zsh
source /opt/ros/humble/setup.zsh
cd ~/rammp_deps_ws
colcon build --base-paths src ~/RAMMP-CuRobo/rammp_curobo_interfaces \
    ~/kinova-gen3-ros2/kinova_gen3_description --cmake-args -DBUILD_TESTING=OFF
```

This workspace:

```zsh
source /opt/ros/humble/setup.zsh
source ~/rammp_deps_ws/install/setup.zsh
cd ~/RAMMP-box-opening
colcon build --symlink-install
source install/setup.zsh
python3 -m pytest src/rammp_box_opening/test -q -p no:launch_testing -p no:launch_ros
```

Python files are COPIED into `build/` at build time (the egg-link points
there, not at `src/`), so rebuild before any `ros2 run` or e2e after
editing code. YAMLs are symlinked and update in place.

## Runtime sourcing chain (execution shells)

humble → `~/rammp_deps_ws` → this ws, with
`export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp` in EVERY shell, explicitly —
the driver and planner containers speak Cyclone DDS, and a Fast DDS shell
discovers them and then loses their data (every CLI refuses to start off
Cyclone). Leave `ROS_LOCALHOST_ONLY` unset: the containers do not set it.

## Safety stance

Dry-run is the default everywhere: every CLI plans and previews without
motion unless given `--execute`, and motion CLIs are run by a human
with a hand on the physical e-stop. `--execute` alone arms every CLI —
no typed confirmation (owner decisions 2026-08-24 for `press_demo`,
2026-09-16 for the rest): autonomous once started, Ctrl+C stops
everything. No autonomous motion from agent sessions, ever. The
driver executes whatever it is sent — it has no dry-run parameter — so
`--execute` is what arms the client, arm and gripper alike; without it
nothing is ever sent. The client also applies the gates the driver lacks
before every goal (`runtime/driver.py`): the trajectory must start within
0.05 rad of the live arm, stay under the joint velocity limits, and run
forward in time.

Ctrl+C during motion is OWNED, not inherited: motion CLIs install their
own SIGINT handler (`runtime/abort.py`) so an in-flight stroke gets its
cancel delivered on a live context and confirmed by the server before
the process exits — rclpy's default handler makes that a race that ends
in a traceback instead of an answer. Proven by `scripts/abort_e2e.py`
(stub driver + stub planner, isolated `ROS_DOMAIN_ID=77`, goal-count
audit; refuses to run beside a real stack). Re-run it after any change to the client,
runner, or CLI wiring — it is the software half of the attended abort
drill in `docs/HARDWARE_BRINGUP.md`.

## Where the container may sit — measured tool-down reach band

`scripts/reach_probe.py` (offline, in-process planner, nothing can move)
swept tool-down poses — the press-attitude family every contact leg
uses — across the bench at two heights for a container on the MEASURED
table (top −0.027 m, reconciled 2026-08-24 from RAMMP-CuRobo's Orbbec
measurement): contact (button top, z=0.133 at the time of the sweep) and
staging (+0.08, z=0.213). Result (2026-08-24, 5 cm pitch, 459 plans;
raw data `docs/reach_map.json`):

```
     x 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75
y -0.45  #   #   #   #   #   #   #   #   .   .   .   .
y -0.35  #   #   #   #   #   #   #   #   #   .   .   .
y -0.25  #   #   #   #   #   #   #   #   #   #   .   .
y -0.15  #   #   #   #   #   #   #   #   #   #   o   .
y -0.05  #   #   #   #   #   #   #   #   #   #   #   .
y +0.00  #   #   #   #   #   #   #   #   #   #   #   .
y +0.05  #   o   #   #   #   #   #   #   #   #   #   .
y +0.15  #   #   #   #   #   #   #   #   #   #   o   .
y +0.25  #   #   #   #   #   #   #   #   #   #   .   .
y +0.35  #   #   #   o   #   #   #   #   #   .   .   .
y +0.45  #   #   #   #   #   #   #   #   .   .   .   .
(# = staging AND contact plan, o = one height only, . = neither;
 every-other row shown — full 19-row map in docs/reach_map.json)
```

Read it as: **place the container with its button between x 0.20 and a
radial edge of ~0.70 m, anywhere in y ±0.45** — 180/228 grid points are
usable at both heights, with a few isolated single-point holes (planner
stochasticity; nudge an inch if one bites). This is the complement of
the wrist-flat lore: the wrist-flat TRANSIT family fails inside ~0.5 m
radius, while TOOL-DOWN work covers the whole band — which is why every
mission pose uses the tool-down family.

For the mission, the practical zone is tighter than the reach band: the
box must be in VIEW before it can be reached, and what has to be in view is
the whole box TOP: it sits `dims.z` nearer the camera than the table, so it
is seen through a smaller window, and a top touching the image border is
refused. From the LOOK (HOME's hand lifted and wrist turned down) the camera
stands at (0.49, 0, 0.48) looking straight down, and at lid height the frame
covers **x 0.28–0.71, y ±0.38** — the narrow axis is x. The base then sweeps
`scan.search_arc_rad` (0.6 rad) each way, carrying that view centre to
(0.41, ∓0.28) and swinging the narrow axis somewhere new. A
container outside what the search covers exits honestly via the no-box
path. (`open_box.park_tool_down: true` makes the look pose the resting
pose too — off by default; the run ends at HOME.)

Caveats (also in the JSON meta): bare-bench world — the container's own
collision model shrinks this only locally; every PROBE plan starts at
HOME (mission legs chain from the previous leg's predicted end, and a
mission may start from the look pose); probe yaw is the point bearing (joint_7
absorbs tool-down yaw). Re-run whenever bench geometry changes:

```zsh
python3 scripts/reach_probe.py            # ~2 min on the Orin; --quick to smoke
```

## The mission (owner design 2026-08-24; press proven live 2026-08-25)

One launch, one command, autonomous once started:

```zsh
# rammp-deployments/december_2026: arm, planner, both cameras AND this
# package's launch (robot TF, scene TF, two OWL nodes — the `box_opening` node)
sheppy up box-opening
ros2 run rammp_box_opening press_demo --execute      # in its own shell
```

Without sheppy, the launch runs by hand instead:
`ros2 launch rammp_box_opening press_demo.launch.py` (`camera:=true` adds
a host D405 driver). Its output under sheppy is `sheppy logs box_opening`.

Flow (`tasks/press_demo.py`, one log line per state):

0. **SCENE** — before the arm moves at all, the fixed scene camera finds
   the box (`perception/scene_source.py`): its own OWL instance says which
   thing in the frame is the box, and the depth points inside that box
   at lid height, lifted through the calibration in
   `config/camera_scene.yaml` (`scripts/calibrate_scene_camera.py`, no
   motion, no marker to attach), give the box's position to a few
   centimetres. The arm flies straight to staging above it in one goal
   from HOME; from there the press is aimed by the button's CIRCLE alone
   (`perception/button_aim.py`: OpenCV's largest circle of the button's
   diameter, at lid height, on a flat lid — no OWL, no box-footprint gate
   on this path; owner's design 2026-09-17). The press point is the
   circle's CENTRE, nothing else. The descent pre-planned during the
   approach flies as it is when the aim agrees within 2 mm, and is
   re-fitted to the aim otherwise (a level move at staging height, then
   straight down; no planner call). When the aim finds the button more
   than 5 mm from under the tool, the arm first moves over it and aims
   again (up to twice): every press is aimed from directly above the
   button, wherever the box stands. A
   box that is ALREADY OPEN is not pressed (an open OXO pressed is an OXO
   shut): the aim measures the button against its own lid ring in the same
   frame, and at 6 mm or more the run stops and says so — this knob stands
   about 10 mm proud when up. After the push the wrist confirms the pop the
   same way (>= 6 mm) before the grip; only a knob that is really down is
   pressed once more. A touch 8-35 mm above the button top also means the
   knob is up (no push), but the raised knob is soft and a stroke can ride
   it down unnoticed, so the camera is the judge. Every run records
   the scene-versus-wrist residual
   (`~/.ros/rammp_box_opening/scene_calib/residuals.jsonl`), which is the
   data that refines the calibration. With `detect.confirm_in_flight`
   (only once the wrist camera's timestamp lag has been measured, see
   `config/camera_d405_wrist.yaml`) the approach and the guarded descent
   are ONE goal and the wrist confirms on the way: the arm never stops
   above the box. No scene fix, or no button found at staging, and the
   OWL-gated search below runs from wherever the arm is. `--no-scene`
   forces it. `config/containers/ankou_pink.yaml` is the pink canister
   (the default); `oxo_pop.yaml` the 4.1-inch OXO, `oxo_pop_small.yaml` the
   2.9-inch one (`--container`).
1. **LOOK** — the arm's own rest pose with the hand lifted and the wrist
   pitched down (`primitives/look.py`): 3.42 s, where flying to a fixed
   bench pose measured 3.89 s. Seeing nothing, the
   base SWEEPS `scan.search_arc_rad` each way at `SEARCH_SPEED`, and the
   motion is cancelled the instant the camera sees a box. Planned in the
   bench world, whose unseen-container band blocks the whole placement
   zone to container height. A step the arm already stands at is skipped.
   Nothing in any of this is expressed in the bench frame, so it holds
   wherever the chair is standing.
2. **DETECT** — the lid-top plateau above the table
   (`perception/depth_source.py`: band, smoothness, footprint and border
   gates, button-circle refine), on TWO paths over the same frames.
   COARSE runs while the arm moves — no stillness gate, no button circle,
   looser agreement — and only ever stops a search. PRECISE is taken at
   rest and is what the press is aimed with. Both look for the lid at the
   SURVEYED table height plus `dims.z`. The table is not fitted from the
   camera: passive stereo reads a blank bench as a smear whose mode sits
   ~35 mm above the real surface, and a fit taken from it moved the lid
   band off the lid (field 2026-09-16). Gated by the persistent OWL
   node's bbox (`owl_detector`, `perception/owl_source.py`) — depth
   answers first, and the semantic ladder (`vlm.backends`: local OWL,
   then the cloud) is walked only when the whole search could not answer.
   No stable fix → home, exit 2. The fingers close (`press:close`) before
   the look and ride it.
3. **PRESS** — two stages. The TOUCH: the approach CHAINED into ONE
   guarded stroke toward `press_demo.travel_m` below the lid plane,
   re-timed into a single execution — the approach cruises at transit
   speed, the descent at `press_demo.speed`, and the guard
   (`press.contact_nm`) arms at the junction between them. A trip near the
   expected contact is the surface, measured by the arm's own fingertip
   TF; an EARLY trip reports as a strike, and full travel with no trip as
   "no surface" — both failures. The PUSH: `press.button_travel_m` past
   that contact, until the button's stop is felt (`press.touch_nm` is its
   backstop). Then a reflex recoil, a LAZY retreat to the hop with
   `grip:open` riding it, and a look at the knob from the hop: not popped
   → one more press.
4. **GRIP** — guarded descent to `grip_clear_m` above the lid around the
   popped knob (a trip = struck it), band-verified close
   (`open_box.grip_band`; 0.8 = closed on air).
5. **PLACE** — the lift, the carry and the set-down as ONE execution, since
   nothing happens between them the arm has to be still for. The drop spot
   is derived from the DETECTED box every run (`lid_place` says only which
   side to prefer, and the height is the surveyed table). The
   set-down is guarded at `setdown_touch_nm` (4.0 Nm — the trip IS the
   success) and re-checks the grip band before releasing, so a lid that
   slipped on the way is caught before the fingers open. Then lazy retreat
   to the carry height, home (or the look pose).

Each next phase is planned as a Runner lookahead while the previous
phase's last unguarded motion flies. `--press-only` stops after the
press (retreat to staging, home); `--detect-only` reports fixes for
15 s and homes (the wrist-mount calibration observation). All knobs live
in the container's YAML (`config/containers/ankou_pink.yaml` by default).

Containers are YAML models in `config/containers/`, chosen with `--container`:
the round clear canister with the pink push-button lid (`ankou_pink.yaml`, THE
DEFAULT since 2026-09-22 — the same press-and-pop mechanism as the OXO;
geometry estimated from photos and flagged `measure_me` until taped), the
4.1-inch OXO (`oxo_pop.yaml`) and the 2.9-inch (`oxo_pop_small.yaml`). Every
CLI and both OWL detector nodes fall back to the default; the detectors take
their prompts from whichever container the mission is run with.

Every contact descent (press, grip, set-down) arrives from a waypoint
straight above its target: the v1.0.0 planner has no approach constraint,
so the waypoint is a planner result and the stretch below it is the arm's
own straight line (`runtime/kinematics.py`: FK/IK from the driver's URDF;
`runtime/approach.py`), flown as one trajectory with a brief rest at the
waypoint. A planner plan for that stretch bowed 5-7 mm sideways — the
constant off-centre press of 2026-09-14..16.

Proven off-bench by `scripts/press_demo_e2e.py` (stub driver + stub planner +
synthetic D405/OWL on an isolated domain, shipped config minus the cloud
rung): the actual depth pipeline recovers the box origin to ~1 mm and
its yaw to a fraction of a degree, the full leg chain runs with the
guard-trip/cancel/replan path exercised, a button that does not pop the
first time is pressed again (`repress`), and an empty table homes and
exits 2. Rebuild, then run it and `scripts/abort_e2e.py` after any change
(`press_demo_e2e.py repress` runs one scenario). The harnesses keep their
run logs and captures in their own work folder (`RAMMP_BOX_OPENING_STATE`),
never beside the bench's in `~/.ros/rammp_box_opening`.
