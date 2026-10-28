# Hardware bringup and attended sessions

Every motion in this document is run BY A HUMAN with a hand on the
physical e-stop. Agent sessions prepare, analyze logs, and never execute.
Spec: `docs/superpowers/specs/2026-08-14-box-opening-design.md` (§6, §8).

## 1. Session preconditions

- Exactly ONE arm stack at a time: sheppy's `arm` node owns the Gen3. Never
  start the old `kortex_bringup` beside it (`pgrep -fa kortex` must be empty).
- Bringup order (human, separate shells, each with
  `export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp` typed explicitly —
  non-interactive shells skip `~/.zshrc`; leave `ROS_LOCALHOST_ONLY` unset):
  1. Driver and planner (T1): in `~/rammp-deployments/december_2026`, run
     `sheppy`, select `kinova_gen3_driver` for `arm` and `curobo_planner`
     for `planner`, start both. The planner's first plan after a cold start
     compiles CUDA kernels for minutes; the kernel-cache volume makes that a
     one-time cost. The planner node MUST mount
     `/home/abra/.ros/rammp_box_opening/worlds` at the same path (read-only):
     this repo hands it world files by path, and preflight fails without it.
  2. Robot TF, scene-camera TF and the OWL detectors: sheppy's `box_opening`
     node (in the `bench` profile) runs
     `ros2 launch rammp_box_opening press_demo.launch.py` as a host process;
     `sheppy logs box_opening` shows its output (the two "owl_detector ...
     ready" lines, re-subscribe notes). By hand instead, from a task shell
     (below), when sheppy is not in use. Either way it stops stray HOST
     camera drivers and OWL nodes first, never sheppy's containers. The
     driver publishes `/joint_states` but no TF; this launch's
     `robot_state_publisher` is what makes the finger-tip and camera-mount
     frames exist.
  3. Task shells: humble → `~/rammp_deps_ws/install/setup.zsh` →
     `~/RAMMP-box-opening/install/setup.zsh`.
- The driver executes whatever it receives: there is no server-side dry-run.
  `--execute` arms the client for arm AND gripper motion; without it nothing
  is sent. Treat any `--execute` run as capable of moving both.

## 2. Measurement worksheet (once, before the first `--execute`)

All measurements are from base_link: +x forward, +z up, tape measure,
metres. Record into the two config files (`colcon build
--symlink-install` runs the YAMLs and the python from `src/`, so they
update in place; the robot's copy install needs a rebuild after any edit,
and `scripts/sheppy_task.sh` refuses `--execute` until it has one).

### 2a. Bench world — `src/rammp_box_opening/config/world_bench.yaml`

1. Table top height relative to base_link and extents. This number is
   load-bearing twice: the planner's table cuboid, AND the depth
   detector's search band (container candidates are looked for at
   table + `dims.z`) and the pinned container origin z. A wrong table
   height fails detection honestly ("top N mm from nominal … table_z
   needs recalibrating") rather than pressing on a guess. It is measured,
   not fitted from the camera: a camera fit was tried and read the blank
   bench ~35 mm high, which put the lid outside its own search band
   (field 2026-09-16).
2. Any wall/shelf within reach: add as cuboids, same schema.
3. Reconcile ONCE with `~/RAMMP-CuRobo`'s `world_real_bench.yaml` (theirs
   also carries placeholder geometry — measuring supersedes both; after
   this, OUR generated worlds are what the planner uses at runtime).

### 2b. Container model — `src/rammp_box_opening/config/containers/<container>.yaml` (`ankou_pink.yaml` is the default)

With the container at its bench spot, lid on:

1. `dims`: outer x, y, z of the body (lid ON). `dims.z` sets the
   detector's expected lid height above the table and the container
   cuboid's height.
2. `lid_dims`: the lid alone (it is carried and placed as a cuboid).
3. `button_offset`: bottom-center origin → button TOP center (the flush
   button top IS the container top on this box).
4. `button_diameter_m`: caliper the round button — it sizes the circle
   the detector aims the press at.
5. `press.touch_nm`: a `--guarded` press's trip threshold (the compliant
   run has no guard: its joint springs push 15 N,
   `depth_press.COMPLIANT_PRESS_N`, through `press.button_travel_m` and hold
   `press.push_hold_s`). Bracketed at the bench (6.1 stopped short of the
   seal, 8.1 scooted the box); tune only with `torque_peak` from the run log
   in front of you.
6. `press_demo.travel_m`: press the button by hand with a caliper — the
   travel at which the lid releases, plus ~2 mm margin. A `--guarded`
   touch is position-controlled with the torque guard as a stop; too-large
   travel means the guard is your only brake.
7. Where the lid goes: nothing to set. It goes where the cameras see room
   for it and the hand, every run (`perception/clear_table`), at the
   measured table height; `open_container.lid_place` is gone (2026-10-27:
   nothing in the environment is configured). The run refuses (exit 4)
   when nowhere has room.
8. `open_box.grip_band`: close the fingers on the POPPED knob by hand and
   record the position feedback the runner prints (0.387 at the bench);
   the band must hold it and exclude 0.8 (closed on air).
9. Flip `measure_me: false` — the CLIs refuse `--execute` until then;
   that refusal is the worksheet's completion gate.
10. Wrist-mount check (needs the camera AT the look pose, so this comes
    after the flip): run §4 steps 2–3, then `press_demo --execute
    --wrist-detect` with the container placed at a tape-measured spot. The
    arm turns its wrist down — a joint move into free air, no flight —
    reports every fix for 15 s and homes; the printed top-face centre vs
    the tape solves the wrist-mount error. If it reports no fix, check
    `ros2 topic hz /wrist_camera/aligned_depth_to_color/image_raw`
    (sheppy's wrist_camera node up, aligned depth on?), then the status
    line's last reject reason, then where the box is standing — from the
    look the camera sees 0.58 × 0.33 m of table around (0.36, 0). A failed
    detect always returns the arm home.
11. Scene camera calibration (once, and again whenever the camera or the
    cabinet tag is moved). With the launch up and the arm at HOME:
    `python3 scripts/calibrate_scene_camera.py`. It moves nothing: the wrist
    camera sees the cabinet's ArUco tag from HOME and places the tag and the
    door in base_link, the scene camera sees the same tag, door and table,
    and the door and table planes give the orientation (a 57 mm tag seen
    from a metre cannot). It refuses to write when the two cameras disagree
    on the door-to-table angle by more than 2 deg. Then rebuild, restart the
    launch, and check `~/.ros/rammp_box_opening/scene_calib/<stamp>.jpg`:
    the green circles are the arm's own fingertips drawn through the new
    calibration and must sit on the fingers. `--check` repeats that check
    against the saved file. The table's tilt in base_link that the solve
    uses is `table_normal_base` in world_bench.yaml, measured from the wrist
    camera's own view of the table.
12. Wrist timestamp lag (optional: the run aims only with the arm at
    rest): run `python3 scripts/record_scan_frames.py --lag-sweep`
    during any ordinary `--execute` run, then
    `python3 scripts/record_scan_frames.py --analyze-lag <capture dir>` and
    put the offset it reports in `camera_d405_wrist.yaml` `stamp_offset_s`.
    Until then frames shot while the arm moves are placed where the arm
    was, not where the shutter fired.

## 3. Every-session preflight

1. `ros2 run rammp_box_opening preflight` — must PASS, and moves nothing:
   the shell is on Cyclone; `/joint_states` fresh WITH effort fields
   (guarded legs refuse without them); the driver's
   `/execute_joint_trajectory` and the planner's actions reachable; the
   planner loads a world file this repo wrote (fails = the worlds mount is
   missing from the planner node); TF for the finger tips (fails = T2 not
   running); `/gripper_state` reports a gripper present; the wrist
   camera's colour, camera_info and ALIGNED depth all streaming (fails =
   the D405 is off USB, or the camera was launched without
   `align_depth.enable:=true`). WARN lines do not fail it but say what is
   degraded: a silent scene camera (the mission then finds no box and
   stops before the arm moves toward one: the wrist no longer searches),
   an owl_detector not running, or up but deaf (`sheppy restart
   box_opening`).
2. **Abort drill** (every session, no exceptions): start
   `ros2 run rammp_box_opening home_arm --execute` (it flies as soon as you
   run it — `--execute` alone arms it), then
   Ctrl+C mid-motion. PASS = the arm stops and holds immediately AND the
   CLI prints `cancel delivered; the driver stops and holds` (that line
   is the server's confirmation, not a hope). A session does not proceed
   past a failed drill. The software half is stub-proven off-bench by
   `python3 scripts/abort_e2e.py` (isolated domain, no arm) — run it
   after any client/runner/CLI change, BEFORE burning bench time on the
   live drill.

## 4. The mission (owner design 2026-08-24; one run, the reach-out, since 2026-10-28)

Attended, e-stop in hand. `--execute` alone arms it — NO typed
confirmation (owner decision: autonomous once started, Ctrl+C stops
everything; the abort drill above is the proof it does). There is one
run, the full box opening: every box is pressed by the REACH-OUT — the
hand tilted back toward the arm, one fingertip straight down onto the
button — however near it stands (owner, 2026-10-28: "make the arm always
go to that 45 degree looking pose even if it's close enough to go straight
up and down. completely scrap the home -> straight vertical if close
enough mode"), and then the lid comes off and is set down.

1. Off-bench, after any code change: rebuild, then
   `python3 scripts/abort_e2e.py` — it must PASS before bench time. (The
   stub harness that flew the old straight-down mission end to end,
   `scripts/press_demo_e2e.py`, went with that mission; the reach-out and
   its lid off are covered offline by `test/test_reach_press.py`,
   `test_reach_lid_off.py` and `test_depth_press.py`.)
2. Bringup (§1): `sheppy up bench` — arm, planner, both cameras and the
   `box_opening` node, which runs this package's launch (robot TF, the
   scene camera's TF from `config/camera_scene.yaml` when it exists, and
   the OWL instances; stops host strays). The mission shell needs
   `~/rammp_deps_ws` and this repo's install sourced on top of
   `~/ros2_ws`. The OWL models load once there — `sheppy restart
   box_opening` (or restart a hand-run launch) at the start of
   each session: an OWL instance left running for an hour or two stopped
   receiving camera frames and answered nothing (2026-09-16; the cameras
   themselves were still at 30 Hz).
3. Preflight + abort drill (§3). The container where the scene camera
   sees it and the reach-out reaches it (README, "Where the container may
   sit"), lid on, button flush.
4. Dry-run: `ros2 run rammp_box_opening press_demo` — read the SCENE line
   (the box where it really stands, to a few cm; if not, stop and check
   the scene camera's calibration or mount) and the lid's spot; the
   preview ends at the look pose.
5. First run on this stack: `ros2 run rammp_box_opening press_demo
   --execute --press-only --speed-scale 0.3` — watch one slow press before
   the full run (`--speed-scale` is the SAME run watched slowly). Then
   `ros2 run rammp_box_opening press_demo --execute`. Expected, one log
   line per state: SCENE: box at … — reaching out → lid goes to … (where
   the cameras see room for it and the hand) → COMPLIANT … → REACH-OUT:
   hand tilted N deg … → AIM: button circle … and AIMED from the look pose
   … → ONE MOTION from the look pose to below the button (the re-aim, the
   slide in turning the hand down, the stroke) and the hold → back to the
   look pose → AFTER PRESS 1: knob UP … — the box is open (not up:
   "pressing once more" and AFTER PRESS 2; still not up: STOP, home) →
   GRIP: the fingers open on the way down … → CLOSE, then ONE MOTION … →
   LID GRIPPED → RELEASE, then ONE MOTION … → DONE — box open, lid set
   down at …, arm home, exit 0. No box → the arm goes home, exit 2; no
   button from the look pose → home, exit 2; nowhere for the lid → exit 4
   with nothing moved. Every move flies in the driver's joint impedance
   and nothing the run reads stops it: the e-stop and Ctrl+C do.
   `--press-only` stops after the press (rise, home; the lid stays on).
   `--guarded` flies the same reach-out position-controlled, with its
   torque guards and at-rest checks (as before 2026-10-27), and ends at
   the press. `--depth-press` is still accepted and changes nothing: the
   depth press is the run.
6. Repeat from different container positions. Exit criterion: repeatable
   opened-and-lid-off runs, verified in
   `~/.ros/rammp_box_opening/runs/run-*.jsonl`.
7. Optional: `open_box.park_tool_down: true` rests the arm at HOME's wrist
   turned down between runs instead of the factory HOME (saves the 2.4-2.9
   rad wrist flip twice per run). It then holds the tool out over the
   table. Off by default — the run ends at HOME (owner, 2026-09-04).

## 5. Retired: the Phase-1 primitive ladder

The per-primitive CLIs (approach / press / grasp / lift / place /
retreat / open_container / pickup_container) and the hand-measured
`bench_pose` they planned from were retired with the Phase-0/1 tier once
the camera-driven mission ran live. `home_arm` is the only isolated
mover left (§3). Exercise the phases through `press_demo` itself:
`--detect-only` for the scene camera, `--wrist-detect` for the wrist's
mount, `--press-only` for the press, the full run for the lid.

## 6. When something trips

- The runner stops at the first unexpected outcome with the arm holding
  and prints leg name, outcome, torque peak, progress. The same row is
  appended to `~/.ros/rammp_box_opening/runs/run-*.jsonl` (the metrics
  source — do not delete).
- "GRIP FAILED — …: the fingers closed on air or on something else" →
  the knob was not up where the look put it, or the box moved: the fingers
  open, the hand backs out the way it came down, home. "the fingers read N
  at the table, outside the grip band" → the lid slipped on the way.
- "refused before sending: trajectory starts N rad from the live arm" →
  the plan went stale (the arm moved after planning); the runner replans
  from live on drift, so repeated refusals mean something else is moving
  the arm. Only one controller at a time.
- A leg fails with a driver code: `HALTED` → e-stop or ownership revoked
  (clear it, `home_arm`); `NOT_AUTHORIZED` → the `arm` node is running with
  arbitration enforced (the manifest sets `arbitration_mode: disabled`);
  "goal rejected by the driver" → a streaming session is open on it.
- A gripper leg fails "gripper at X" without moving → `/gripper_state`
  `present` false, or the setpoint never reached the driver (check
  `ros2 topic info /setpoint/gripper` shows the driver subscribed). The
  gripper's speed/force are `GRIPPER_SPEED`/`GRIPPER_FORCE` in
  `constants.py` (force is a current ceiling); re-check `open_box.grip_band`
  against the knob after changing them.
- Guard refuses to run: `/joint_states` has no effort fields — the arm
  bringup is wrong, not the guard.
- A failed post-touch replan (retreat or home refused from the contact
  pose) stops the mission with the arm holding. Clear the area and run
  `home_arm --execute`; when that is refused as an INVALID START in both
  worlds (the fingers are at the table after an abort or a jog — 40 mm
  up, 23 deg off vertical was refused on 2026-09-17), run
  `scripts/sheppy_task.sh home_arm --execute --lift-first 0.08` (on
  flatbot it must be that wrapper: it passes world_flatbot.yaml, and the
  lift refuses when the arm chain's root is not the planning frame) — a
  plan-free straight vertical lift of 80 mm at contact speed (the arm's own kinematics, attitude
  held, obstruction-guarded — anything met on the way UP stops it), then
  HOME planned from where the lift ends. Straight up is the one direction
  that cannot meet the table; keep the space above the fingers clear.
- press_demo refuses to start "N rad from any rest pose" → the previous
  run ended badly. Recover with `home_arm --execute` before anything
  else; never plan a mission from wreckage.
- "SCENE: none of the OWL's N candidate(s) is this container: ..." → the
  scene's boxes were all the wrong size or height for `dims` (another
  container, a lid off, the wrong `--container`); the run goes home and
  exits 2 (the wrist does not search). With two containers on the table
  the right one is chosen by its footprint — the OWL offers its top three.
- "SCENE: the scene OWL is SILENT" / "saw no box in 3.0 s" with the box
  plainly in the scene image → no box: the run goes home and exits 2.
  `ros2 topic hz /rammp_box_opening/owl_bbox_scene`
  should show ~2 Hz heartbeats even disabled. Silent with "re-subscribed"
  lines in `sheppy logs box_opening` → the node stopped receiving frames
  (2026-09-16/17: two executors spinning one node, fixed in the grabbers;
  a recurrence means something else) → `sheppy restart box_opening`.
  Alive but "no box" → the box scores under the scene floor (the container
  config's `vlm.owl_min_score_scene`: ankou_pink 0.05, the OXO 0.12);
  `ros2 topic echo` the bbox topic, 5th field.
- For a container whose config names `scene_detect.segmenter_classes`
  (ankou_pink: `[food container]`), the box comes from sheppy's always-on
  scene_segmenter first: "SCENE: ... (SEG 0.88, 2 sightings
  (segmenter+segmenter); ...)" in about 0.4 s, with the scene OWL off. The
  OWL is the fallback (segmenter down, a frame showing two places, or no fix
  in 1 s): "OWL 0.31, 2 sightings (owl+owl)" in about 2.6 s. A failure names
  both: "scene segmenter: ...; scene OWL: ...". "scene segmenter: ABSENT:
  nothing publishes /perception/object_detections" → the sheppy profile does
  not select `scene_segmenter: rfdetr` (the OWL is asked at once); "SILENT: no
  /perception/object_detections in 1.0 s" → it is selected but sends nothing
  (`sheppy logs scene_segmenter`); `ros2 topic hz /perception/object_detections`
  should show ~5 Hz. "two places pass as this container, [x, y] and [x, y],
  and the OWL did not settle which" → the segmenter boxed a second bowl whose
  band also passes; once that is seen, only a pair with an OWL sighting
  commits, and an OWL pair of a cut lid only at one of the two places (with
  "scene OWL: its boxes of the cut lid wait on the segmenter's", it was at
  neither). "the OWL's box of the cut lid is not confirmed by the
  segmenter's" → the segmenter saw the container in this look, but never
  where the OWL's boxes of the cut lid put it; a segmenter that has seen
  nothing confirms nothing, and those boxes commit on their own
  ("owl+owl"). "it watches '', not the scene camera (...); the mock?" →
  the segmenter runs as sheppy's mock. "does not import" → this install does
  not chain `~/ros2_ws/install` (rammp_perception_interfaces).
- A round box the scene frame cuts at its side is finished from its lid's
  arc in view ("cut off at the frame's left edge, ..." in the reasons).
  "... fits two places 35+ mm apart, one past its box: the N points of the
  other are too few to settle it" → too little of the lid is in view to
  tell it from its mirror (half of it in view is plenty), so that frame
  gives no sighting. "... stand 20-50 mm over its top where the lid's
  circle is: the arc took in something taller than the lid" → something
  taller stands right beside or behind the cut box, so that box is refused
  and the frame's other boxes are tried. With no fix at all the run goes
  home and exits 2, as for any scene refusal.
- "lid spot(s) refused by the planner (…) — setting the lid down at [x, y]
  instead" → normal: the first spot could not be planned and the next was
  used. "STOP: nowhere to set the lid down: …" or "STOP: the hand reaches
  none of the N spot(s) the lid could go to (…) — move the box nearer the
  arm; nothing moved" → resolved before anything moves (exit 4): clear
  the table around the box, or move the box.
- "SCENE: scene camera streams missing: colour, ..." → the Gemini driver
  has no device: `docker logs sheppy-scene_camera` stops at "Loaded node"
  with no device lines. Check it is on USB (`lsusb | grep -i orbbec`;
  2026-09-17 it had vanished and the D405 sat on its old port), re-plug
  it, `sheppy restart scene_camera`. Until then the run finds no box
  (exit 2).
- "STOP: no button from the look pose (…) — nothing was pressed" → the
  saved frame (`captures/wrist-reach-look-<stamp>/frame_000.jpg`) shows
  what the wrist saw; the status names the last reject. The button's
  `button_diameter_m` wrong for this box, or the lid not where the scene
  camera put it, are the usual causes. `oxo_pop.yaml` is the 4.1-inch OXO,
  `oxo_pop_small.yaml` the 2.9-inch, `ankou_pink.yaml` the round clear
  canister with the pink push-button lid (THE DEFAULT since 2026-09-22:
  same press-and-pop mechanism as the OXO) — run another with
  `--container <yaml>`. The two OWL detector nodes follow the mission's `--container`
  automatically (their prompts and score floor live in it; the mission
  announces its file on a latched topic and the nodes reload) — the log
  line "container -> ... queries [...]" in `sheppy logs box_opening`
  confirms it. Before that (2026-09-22) they kept the OXO's prompts
  whatever the mission was run with.
- A NEW container is a new YAML in `config/containers/` (copy the nearest
  one). Three numbers to tape before it may fly: the lid's outer
  diameter (or the box's sides) as `dims` x/y, the closed container's top
  height above the table as `dims` z and `button_offset` z, and the
  button's diameter; then flip `measure_me`. Prompts: sweep phrasings
  against a few phone photos of it before the bench (the pink canister's
  sweep found "clear", "jar", "canister" and "container" alone at 0.00 and
  every "... pink lid" phrase at 0.36-0.67 — the lid is the anchor; the
  OXO's own prompts score the OXO 0.08-0.13 on real wrist frames).
  `press_demo --wrist-detect --execute --container <yaml>` measures the
  top's footprint and height from the wrist as a cross-check — the one
  `--execute` an unmeasured container allows (it flies the look pose and
  home only, never near the box).
- "ALREADY OPEN: the knob stands N mm above its lid — not pressing it,
  which would shut the box; taking the lid off" (N >= 6, read by the aim
  against the button's own lid ring) → the box was left open by the run
  before; the full run takes its lid off unpressed (a press would shut it
  — three runs did, 2026-09-21). With `--press-only` or `--guarded` it is
  "STOP: the button already stands N mm above its lid — this box is
  already OPEN" and the arm goes home. This knob stands ~10 mm proud when
  up.
- "AFTER PRESS 1: knob UP N mm — the box is open" (N >= 6; this knob
  reads 9-11 up and within a millimetre of 0 down) → normal. "the knob is
  not up (…): pressing once more — the same press, from the look pose" →
  the first press did not pop it; "STOP: pressed 2 times and the knob is
  not up" → the button did not pop under the spring's push: check the box.
- "AFTER PRESS N: knob …, and its LID ROSE +M mm — the box is open" → the
  press let the lid go, the button where it was aimed. With the knob UP
  the grip goes by the lid where it now stands. With the knob flush it is
  "STOP: the lid rose +M mm (released) but its knob is flush (…)" and
  home, the box left open: the fingers have nothing to hold, and a
  released lid is not pressed again (pushing it down may latch it). Lift
  the lid off by hand, or shut it and run again.
- "scene camera: … — reaching out to its box anyway" → on the last run
  that recorded a scene-versus-wrist pair the scene camera's box stood
  more than 5 cm from the button the wrist found. Those pairs were
  recorded by the straight-down staging aim, one a run; the reach-out
  records none, so `scripts/refine_scene_calibration.py` refines only from
  the pairs already on file
  (`~/.ros/rammp_box_opening/scene_calib/residuals.jsonl`).
- After `sheppy restart box_opening` the detectors take ~9 s to load; a run
  started sooner reports "the scene OWL is SILENT". Wait for preflight's
  "owl_detector (scene camera) up and receiving frames".
- "the SCENE CAMERA HAS MOVED since its calibration: the tag is at ..." →
  the scene camera was re-aimed or knocked. Recalibrate with the arm at HOME:
  `python3 scripts/calibrate_scene_camera.py` (it only listens), then
  `sheppy restart box_opening`. The calibration records where it saw the
  cabinet's tag (`tag_px` in camera_scene.yaml) and every run compares.
- `--press-only` ends after the press: the rise and home, the lid on. A
  compliant press is judged by its spring's hold ("PRESSED: ~N N held on
  the button (a compliant run does not look at the knob)"), not by a look
  at the knob.
- A `--guarded` press "met a stop" at a rise below `touch_nm` with the
  button not popped → the backstop caught it: the stroke was off-centre
  (above) or the button needs more than `touch_nm` — raise it in 0.5 Nm
  steps, never past the runner's obstruction ceiling.
- "camera streams missing" for the WRIST camera with its container up →
  `docker logs sheppy-wrist_camera`: `xioctl ... No such device` means
  the D405 dropped off USB (seen 2026-09-16 13:56). Re-seat the cable
  (or power-cycle the hub), restart the wrist_camera node, confirm with
  `ros2 topic hz /wrist_camera/color/image_raw` (~30 Hz).
