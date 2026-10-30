# RAMMP-box-opening

Open a push-button container with the Kinova Gen3 7-DoF + Robotiq 2F-85:
find the box with the scene camera, press its lid button with the hand
tilted toward it, pull the lid and set it aside — one autonomous mission
(`press_demo`). This repo is a **client** of sheppy's two arm containers
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

With `--symlink-install` Python runs live from `src/` (`build/` links to
it). The robot's install is built WITHOUT it — `colcon build
--packages-select rammp_box_opening` copies Python into `install/` — so a
code edit is not what `ros2 run` flies until rebuilt, and
`scripts/sheppy_task.sh` refuses `--execute` while `install/` differs from
`src/`. YAMLs are symlinked under `--symlink-install` and update in place.

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

This band is straight-down reach, and the mission no longer presses
straight down (2026-10-28): every box is pressed by the reach-out, the
hand tilted back toward the arm (`tasks/reach_press.py`), which reaches
farther (flatbot: boxes 0.50-0.82 m out). For the mission the box must
stand where the scene camera sees it — it alone finds the box — and where
the reach-out's look and press plan: both are proven by the arm's own
kinematics before the arm moves toward the box, and a box they do not
reach is refused with nothing moved. (`open_box.park_tool_down: true`
makes HOME's wrist turned down the resting pose — off by default; the run
ends at HOME.)

Caveats (also in the JSON meta): bare-bench world — the container's own
collision model shrinks this only locally; every PROBE plan starts at
HOME (mission legs chain from the previous leg's predicted end, and a
mission may start from the look pose); probe yaw is the point bearing (joint_7
absorbs tool-down yaw). Re-run whenever bench geometry changes:

```zsh
python3 scripts/reach_probe.py            # ~2 min on the Orin; --quick to smoke
```

## The mission (owner design 2026-08-24; one run, the reach-out, since 2026-10-28)

One launch, one command, autonomous once started:

```zsh
# rammp-deployments/december_2026: arm, planner, both cameras AND this
# package's launch (robot TF, scene TF, the OWL nodes — the `box_opening` node)
sheppy up box-opening
ros2 run rammp_box_opening press_demo --execute      # in its own shell
```

Without sheppy, the launch runs by hand instead:
`ros2 launch rammp_box_opening press_demo.launch.py` (`camera:=true` adds
a host D405 driver). Its output under sheppy is `sheppy logs box_opening`.

Every box is pressed by the REACH-OUT, however near it stands (owner,
2026-10-28: "make the arm always go to that 45 degree looking pose even if
it's close enough to go straight up and down. completely scrap the home ->
straight vertical if close enough mode"). Flow (`tasks/press_demo.py`, then
`tasks/reach_press.py` and `tasks/reach_lid_off.py`; one log line per
state):

0. **SCENE** — before the arm moves toward it, the fixed scene camera finds
   the box (`perception/scene_source.py`): sheppy's scene segmenter, or the
   scene OWL instance where the segmenter cannot answer, says which thing
   in the frame is the box, and the depth points inside it at lid height,
   lifted through the calibration in `config/camera_scene.yaml`
   (`scripts/calibrate_scene_camera.py`, no motion, no marker to attach) —
   or through TF on a robot whose TF carries the scene camera's mount
   (world `external_static_tf`: flatbot) — give the box's position to a
   few centimetres. What the scene camera sees standing on the table goes
   into every world the planner plans in. The scene camera alone finds the
   box (owner, 2026-10-22): no scene fix, and the run homes and exits 2.
   The fingers close as the run starts. `config/containers/ankou_pink.yaml`
   is the pink canister (the default); `oxo_pop.yaml` the 4.1-inch OXO,
   `oxo_pop_small.yaml` the 2.9-inch one (`--container`).
1. **SPOT** — where the lid will go: among what the scene camera sees
   standing on the table (`perception/clear_table`; no lid pose is
   configured), the first spot the hand reaches. Resolved before the arm
   moves toward the box; nowhere → exit 4, nothing moved.
2. **LOOK** — the hand tilted 30-45 deg back toward the arm (the smallest
   tilt that plans), pulled back and raised clear of the box, the press
   proven by the arm's own kinematics before it flies there. From the look
   pose the wrist aims at the button's CIRCLE (`perception/button_aim.py`,
   the oblique aim: OpenCV's circle of the button's diameter at lid height
   — no OWL, no box-footprint gate; owner's design 2026-09-17); the press
   point is its centre. No button → home, exit 2. A box ALREADY OPEN — its
   button 6 mm or more above its own lid ring in the same frame; this knob
   stands about 10 mm proud when up — is not pressed (an open box pressed
   is a box shut): its lid comes straight off.
3. **PRESS** — the way in as ONE execution from the aimed look pose: the
   re-aim, the slide in turning the hand straight down over the button, and
   the stroke (`tasks/depth_press.py`), all in the driver's joint
   impedance: its springs push the button through and hold it there
   `press.push_hold_s`. Nothing the run reads stops the arm — no torque
   guard, no at-rest check; the e-stop and Ctrl+C do.
4. **LID** — back up the stroke and out to the look pose, where the wrist
   judges the knob; one more press where it is not up (still not up:
   STOP, home). A lid that rose (released) with its knob flush is neither
   gripped nor pressed again: STOP, home, the box left open — lift the lid
   off by hand, or shut it, before the next run (its aim takes a flush
   knob for a shut box and would press it). The fingers open as the hand
   comes down onto the knob and close on it (on air: they open, the hand
   backs out, home); the lift off the canister, the carry and the set-down
   on the spot are ONE execution; the fingers open, and home.

`--press-only` stops after the press (rise, home; the lid stays on).
`--guarded` flies the same reach-out position-controlled, with its torque
guards and at-rest checks (as before 2026-10-27), and ends at the press.
`--detect-only` reports the scene camera's fixes for 15 s with nothing
moving; `--wrist-detect` is the wrist-mount calibration observation (HOME's
wrist turned down, 15 s of wrist fixes, then home). Without `--execute`
the run is previewed up to the look pose. `--depth-press` is still
accepted from older command lines and changes nothing. Exit codes: 0 done
(the lid set down and the arm home; `--press-only`, `--guarded`: pressed),
1 a stop, 2 no box or no button, 3 the move home first refused, 4 nowhere
for the lid, 130 Ctrl+C. All knobs live in the container's YAML
(`config/containers/ankou_pink.yaml` by default).

Containers are YAML models in `config/containers/`, chosen with `--container`:
the round clear canister with the pink push-button lid (`ankou_pink.yaml`, THE
DEFAULT since 2026-09-22 — the same press-and-pop mechanism as the OXO;
geometry estimated from photos and flagged `measure_me` until taped), the
4.1-inch OXO (`oxo_pop.yaml`) and the 2.9-inch (`oxo_pop_small.yaml`). Every
CLI and both OWL detector nodes fall back to the default; the detectors take
their prompts from whichever container the mission is run with.

Every contact line of the run — the slide in, the stroke, the grip's
descent, the set-down — is the arm's own straight line
(`runtime/kinematics.py`: FK/IK from the driver's URDF), never a planner
path: a planner plan for such a stretch bowed 5-7 mm sideways (the
constant off-centre press of 2026-09-14..16).

The stub harness that flew the old straight-down mission end to end
(`scripts/press_demo_e2e.py`, with synthetic D405 and scene cameras) went
with that mission on 2026-10-28: the reach-out's compliant legs, its tilted
look and a wrist camera that rides the arm are beyond its stubs. The
reach-out and its lid off are tested offline (`test/test_reach_press.py`,
`test_reach_lid_off.py`, `test_depth_press.py`, `test_press_demo_main.py`);
`scripts/abort_e2e.py` still proves the Ctrl+C path against a stub driver
and planner — rebuild and run it after any client, runner or CLI change. It
keeps its run logs in its own work folder (`RAMMP_BOX_OPENING_STATE`), never
beside the bench's in `~/.ros/rammp_box_opening`.
