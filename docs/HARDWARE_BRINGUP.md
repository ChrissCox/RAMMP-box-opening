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
--symlink-install` symlinks the YAMLs, so they update in place; python
is copied at build, so rebuild after code edits).

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
5. `press.touch_nm`: the press guard's trip threshold. Bracketed at the
   bench (6.1 stopped short of the seal, 8.1 scooted the box); tune only
   with `torque_peak` from the run log in front of you.
6. `press_demo.travel_m`: press the button by hand with a caliper — the
   travel at which the lid releases, plus ~2 mm margin. The stroke is
   position-controlled with the torque guard as a stop; too-large travel
   means the guard is your only brake.
7. `open_container.lid_place`: which SIDE of the box the lid should go,
   inside the reach band (README "Where the container may sit"). The drop
   spot itself is derived from the DETECTED box every run — the minimum
   clearance away, at the surveyed table height — so this is a preference,
   not a position. The run refuses (exit 4) when no direction from the box
   lands inside the set-down zone.
8. `open_box.grip_band`: close the fingers on the POPPED knob by hand and
   record the position feedback the runner prints (0.387 at the bench);
   the band must hold it and exclude 0.8 (closed on air). Also
   `grip_clear_m` (fingertip stop above the lid plane) and
   `grip_offset_xy` (a mm-scale trim) — read their comments.
9. Flip `measure_me: false` — the CLIs refuse `--execute` until then;
   that refusal is the worksheet's completion gate.
10. Wrist-mount check (needs the camera AT the look pose, so this comes
    after the flip): run §4 steps 2–3, then `press_demo --execute
    --detect-only` with the container placed at a tape-measured spot. The
    arm turns its wrist down — a joint move into free air, no flight —
    reports every fix for 15 s and homes; the printed top-face centre vs
    the tape solves the wrist-mount error. If it reports no fix, check
    `ros2 topic hz /wrist_camera/aligned_depth_to_color/image_raw`
    (sheppy's wrist_camera node up, aligned depth on?), then the status
    line's last reject reason, then where the box is standing — from the
    look the camera sees 0.58 × 0.33 m of table around (0.36, 0). The full
    mission sweeps the base to cover more; `--detect-only` does not. A
    failed detect always returns the arm home.
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
12. Wrist timestamp lag (once, before `detect.confirm_in_flight` may be
    turned on): run `python3 scripts/record_scan_frames.py --lag-sweep`
    during any ordinary `--execute` run, then
    `python3 scripts/record_scan_frames.py --analyze-lag <capture dir>` and
    put the offset it reports in `camera_d405_wrist.yaml` `stamp_offset_s`.
    Until then frames shot while the arm moves are placed where the arm
    was, not where the shutter fired, and the wrist may only confirm the
    box at rest.

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
   `align_depth.enable:=true`). WARN lines do not fail it — the mission has
   a fallback — but say what is degraded: a silent scene camera (the
   mission searches with the wrist instead), an owl_detector not running,
   or up but deaf (`sheppy restart box_opening`).
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

## 4. The mission (owner design 2026-08-24; press proven live 2026-08-25)

Attended, e-stop in hand. `--execute` alone arms it — NO typed
confirmation (owner decision: autonomous once started, Ctrl+C stops
everything; the abort drill above is the proof it does).

1. Off-bench, after any code change: rebuild, then
   `python3 scripts/abort_e2e.py && python3 scripts/press_demo_e2e.py`
   — both must PASS before bench time.
2. Bringup (§1): `sheppy up bench` — arm, planner, both cameras and the
   `box_opening` node, which runs this package's launch (robot TF, the
   scene camera's TF from `config/camera_scene.yaml` when it exists, and
   two OWL instances, wrist and scene; stops host strays). The mission
   shell needs `~/rammp_deps_ws` and this repo's install sourced on top of
   `~/ros2_ws`. The OWL models load once there — `sheppy restart
   box_opening` (or restart a hand-run launch) at the start of
   each session: an OWL instance left running for an hour or two stopped
   receiving camera frames and answered nothing (2026-09-16; the cameras
   themselves were still at 30 Hz).
3. Preflight + abort drill (§3). Container in the reach band AND the
   camera's view zone (README), lid on, button flush, nothing else
   box-sized on the bench (two container-sized tops in view is refused
   as ambiguous unless the OWL bbox picks one).
4. Dry-run: `ros2 run rammp_box_opening press_demo` — read the leg
   preview and the BOX line (container origin must match reality to
   ~1 cm; if not, stop and check the mount / `table_z`).
5. First run on this stack: `ros2 run rammp_box_opening press_demo
   --execute --press-only --speed-scale 0.3` — the driver, the gripper
   setpoint path and the two-plan approach are new, so watch one slow
   press before the full mission. (`--speed-scale` is the SAME run watched
   slowly: every guard takes its baseline and arms at the same point along
   the path as at full speed — its settle time dilates with the motion. It
   did not before 2026-09-21, and slow mode false-tripped a press that full
   speed flew.) Then
   `ros2 run rammp_box_opening press_demo --execute`. Expected, one log
   line per state: SCENE (or LOOK) → BOX at … (fix committed;
   `press:close` went out before the look) → PRESS: the touch ("surface
   found"), then the push → PRESSED — "met a stop" or "full … push" (the
   lid should visibly release) → recoil, retreat to the hop with
   `grip:open` riding it, POP CONFIRMED → GRIP: guarded descent,
   band-verified close, LID PULLED → PLACE: carry, guarded set-down
   ("surface felt … set down" at 4.0 Nm), release, retreat to the carry
   height, home → DONE, exit 0. No box → the arm parks home and it exits 2.
   Pre-detection legs (scan, no-box home) plan above an unseen-container
   keep-out band covering the whole placement zone to container height —
   a failed detection never sweeps low through the container it could
   not see. Keep OTHER tall objects out of the band.
   `--press-only` stops after the press (retreat to staging, home).
6. Repeat from different container positions in the band. Exit
   criterion: repeatable pressed-and-opened runs, verified in
   `~/.ros/rammp_box_opening/runs/run-*.jsonl`.
7. Optional: `open_box.park_tool_down: true` rests the arm at the LOOK
   pose between runs instead of the factory HOME (saves the 2.4-2.9 rad
   wrist flip twice per run; a run that starts there skips the look
   itself, since the search skips any step the arm already stands at). It
   then holds the tool out over where the box goes. Off by default — the
   run ends at HOME (owner, 2026-09-04).

## 5. Retired: the Phase-1 primitive ladder

The per-primitive CLIs (approach / press / grasp / lift / place /
retreat / open_container / pickup_container) and the hand-measured
`bench_pose` they planned from were retired with the Phase-0/1 tier once
the camera-driven mission ran live. `home_arm` is the only isolated
mover left (§3). Exercise the phases through `press_demo` itself:
`--detect-only` for perception, `--press-only` for the press, the full
run for grip and place.

## 6. When something trips

- The runner stops at the first unexpected outcome with the arm holding
  and prints leg name, outcome, torque peak, progress. The same row is
  appended to `~/.ros/rammp_box_opening/runs/run-*.jsonl` (the metrics
  source — do not delete).
- `press:down` "guard tripped EARLY at N% of the stroke" → the stroke
  struck something above the button: the box is not where the fix said
  (scooted by a previous press, or a wrong mount), or something
  is in the corridor. Stop, look, re-place; re-check the BOX line.
- `grip:down` trips (obstruction) → the open fingers struck the knob or
  rim instead of straddling it — the press scooted the box, or
  `grip_offset_xy` is off. `grip:close` "grip X vs band" fails → closed on
  air (0.8: the knob was not popped, or the fix was off) or on something
  too wide.
- `place:lid:down` "full stroke with no trip — never felt the surface" →
  the drop spot is higher than modelled or `setdown_touch_nm` is too
  high; "tripped at N% … set-down NOT confirmed" → struck something on
  the way down. The lid is still held; clear the area, `home_arm`.
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
- press_demo NO BOX with the container plainly in view: read the status
  line — "camera streams missing" means the driver (aligned depth
  included) isn't up; "0/N frames … (last reject: …)" names the gate that
  refused: "no points at container height" → `dims.z`, or a table the fit
  could not find (a view with no flat dominant surface: clutter, a slope,
  or the arm looking off the edge);
  "footprint" → `dims` xy; "touches the image border" → move the box
  toward the scan axis; "ambiguous" → a second box-sized top is in view
  and the OWL node did not pick one; "camera moving" only → the arm never
  parked still.
- A failed post-touch replan (retreat or home refused from the contact
  pose) stops the mission with the arm holding. Clear the area and run
  `home_arm --execute`; when that is refused as an INVALID START in both
  worlds (the fingers are at the table after an abort or a jog — 40 mm
  up, 23 deg off vertical was refused on 2026-09-17), run
  `home_arm --execute --lift-first 0.08`: a plan-free straight vertical
  lift of 80 mm at contact speed (the arm's own kinematics, attitude
  held, obstruction-guarded — anything met on the way UP stops it), then
  HOME planned from where the lift ends. Straight up is the one direction
  that cannot meet the table; keep the space above the fingers clear.
- press_demo refuses to start "N rad from any rest pose" → the previous
  run ended badly. Recover with `home_arm --execute` before anything
  else; never plan a mission from wreckage.
- "SCENE: none of the OWL's N candidate(s) is this container: ..." → the
  scene's boxes were all the wrong size or height for `dims` (another
  container, a lid off, the wrong `--container`); the run falls back to
  the wrist search. With two containers on the table the right one is
  chosen by its footprint — the OWL offers its top three.
- "SCENE: the scene OWL is SILENT" / "saw no box in 3.0 s" with the box
  plainly in the scene image → the run falls back to the wrist search
  (slower, still fine). `ros2 topic hz /rammp_box_opening/owl_bbox_scene`
  should show ~2 Hz heartbeats even disabled. Silent with "re-subscribed"
  lines in `sheppy logs box_opening` → the node stopped receiving frames
  (2026-09-16/17: two executors spinning one node, fixed in the grabbers;
  a recurrence means something else) → `sheppy restart box_opening`.
  Alive but "no box" → the box scores under the scene floor (`min_score`
  0.12 in the launch); `ros2 topic echo` the bbox topic, 5th field.
- "lid spot [x, y] refused by the planner — setting the lid down at [x, y]
  instead" → normal: the first drop spot beside the box could not be
  planned and the next side was used. "STOP before the grip — ... no drop
  spot beside the box plans" → every side of the box is refused (a box
  near the edge of the set-down zone or the reach): the arm goes home;
  move the box toward the middle of the table.
- "SCENE: scene camera streams missing: colour, ..." → the Gemini driver
  has no device: `docker logs sheppy-scene_camera` stops at "Loaded node"
  with no device lines. Check it is on USB (`lsusb | grep -i orbbec`;
  2026-09-17 it had vanished and the D405 sat on its old port), re-plug
  it, `sheppy restart scene_camera`. The mission meanwhile falls back to
  the wrist search.
- "the wrist did not find the button at staging (N/M still frames found the
  button circle (last reject: ...))" → the saved frame
  (`captures/wrist-staging-<stamp>/frame_000.jpg`) shows what it saw.
  "no circle of A-B px radius" → the button's `button_diameter_m` is
  wrong for this box, or the lid is not at `dims.z`; "ring around it is
  not lid" on every circle → the box is smaller than the model (the
  annulus around its button reaches the table). `oxo_pop.yaml` is the
  4.1-inch OXO, `oxo_pop_small.yaml` the 2.9-inch, `ankou_pink.yaml` the
  round clear canister with the pink push-button lid (THE DEFAULT since
  2026-09-22: same press-and-pop mechanism as the OXO) — run another with
  `--container <yaml>`. The search that follows still uses the
  plateau detector, whose footprint gate wants the model's `dims`.
  The two OWL detector nodes follow the mission's `--container`
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
  `press_demo --detect-only --execute --container <yaml>` measures the
  top's footprint and height from the wrist as a cross-check — the one
  `--execute` an unmeasured container allows (it flies the look pose and
  home only, never near the box).
- "STOP: the button already stands N mm above its lid — this box is
  already OPEN" (N >= 6, read by the aim against the button's own lid
  ring) → the box was left open by the run before. Nothing is pressed (a
  press would shut it — three runs did, 2026-09-21); the arm goes home.
  Push the button down flush and run again. This knob stands ~10 mm proud
  when up.
- "the button is ALREADY popped — the pads met the knob N mm above the
  lid" (N >= 8) → the same thing found by touch, on a path that has no aim
  (the wrist search); the mission skips the push and grips. The raised
  knob is soft: a stroke can also ride it down without noticing, which is
  why the aim's camera check above comes first.
- "STOP: the pads met a surface N mm above the button top — not this
  box's button" (N > 35) → something is on the lid, or this is not the
  box the model describes.
- "POP CONFIRMED — the knob stands N mm above the lid" (N >= 6; this
  knob reads 9-11 up and within a millimetre of 0 down) → normal. "NOT
  POPPED — the knob reads N mm ... pressing once more" → the push did not
  pop it; the arm closes, rises to staging and presses again. "STOP:
  pressed to the stop twice and the knob still reads N mm" → the button
  did not pop under the configured force (`touch_nm`): check the box,
  then the force. A stop over the box still gets home: when the blind
  recovery home is refused from there ("home refused from here ... rising
  over the box first"), the arm rises back to staging height above the
  button and homes from it.
- "scene calibration: N pair(s) on file, mean offset [...] mm, spread S m"
  printed after every staging run → the calibration's own residuals.
  When it says a rigid refinement is available (>= 4 pairs over 0.15 m —
  put the box at different spots), run
  `python3 scripts/refine_scene_calibration.py --apply` (backup kept);
  with pairs at one spot it applies a shift only. Run it without
  `--apply` first: it prints the fit and what each pair would be left
  with. Only pairs recorded AFTER the calibration file was last corrected
  count (the file's `refined_at`, else its modification time) — older ones
  measured a calibration that no longer exists, and both the mission and
  the script say how many they left out.
- "CENTRE: the button is N mm from under the tool — moving over it" →
  normal whenever the scene camera's fix is more than 5 mm off: the arm
  moves over the button at staging height and aims again, so every press
  is aimed from directly above the button (the only place presses have
  landed dead centre; aimed 11 cm off, one hit the button's edge and did
  not open the box, 2026-09-23). "still N mm off after 2 moves" → the
  wrist camera's mount calibration, not the scene. The wrist search's fix
  gets the same treatment: staging, a close-up aim, centring, then the
  press. `detect.recentre_max_moves` (default 2; 0 = aim once).
- "the search saw the box but not well enough to aim from there — going
  over it to look close up" → normal when the box stands near the robot
  (the bottom of the search's view, behind the gripper's fingers) or the
  button does not resolve from half a metre up: the search stops at the
  first sighting — a whole top, or a PARTIAL one (cut by the image edge or
  the fingers, at least 45 % of the lid) — and the arm goes over it; the
  close-up aim decides the press. "STOP: the search saw a box here but the
  button is not in view close up" → a sighting is never pressed on; look
  at the saved captures/wrist-staging-* frame.
- After `sheppy restart box_opening` the detectors take ~9 s to load; a run
  started sooner reports "the scene OWL is SILENT". Wait for preflight's
  "owl_detector (scene camera) up and receiving frames".
- "the SCENE CAMERA HAS MOVED since its calibration: the tag is at ..." →
  the scene camera was re-aimed or knocked; its fixes are refused and the
  wrist searches instead. Recalibrate with the arm at HOME:
  `python3 scripts/calibrate_scene_camera.py` (it only listens), then
  `sheppy restart box_opening`. The calibration records where it saw the
  cabinet's tag (`tag_px` in camera_scene.yaml) and every run compares.
- `--press-only` ends at the hop, reads the knob, presses once more when it
  did not pop, then rises out of the corridor and goes home — "PRESSED"
  alone no longer means the box opened; "POP CONFIRMED" does.
- "PRESS from staging — the pre-planned descent re-fitted N mm to the aim
  (no plan)" → normal: the aim moved the target a few mm and the descent
  was rebuilt without the planner. You will see the arm shift sideways by
  that N mm AT STAGING HEIGHT, then come straight down. A large N every
  run (10 mm and more) is the scene camera's calibration, not the press:
  see the "scene calibration" line above.
- The press lands off the button centre by the SAME amount every run
  (compare `press:down`'s `contact_xyz` against the BOX line across
  run-*.jsonl, in the arm's radial/tangential directions) → first check
  the run output for "NOTE press:down: PLANNER descent": the straight
  final stretch was refused and the planner's own plan flew, which bows
  5-7 mm radially (2026-09-16). Only a miss on a straight descent is a
  pad-vs-tool offset for `press_offset_xy`. Random misses are the fix.
- PRESSED "met a stop" at a rise below `touch_nm` with the button not
  popped → the backstop caught it: the stroke was off-centre (above) or
  the button needs more than `touch_nm` — raise it in 0.5 Nm steps,
  never past the runner's obstruction ceiling.
- "camera streams missing" for the WRIST camera with its container up →
  `docker logs sheppy-wrist_camera`: `xioctl ... No such device` means
  the D405 dropped off USB (seen 2026-09-16 13:56). Re-seat the cable
  (or power-cycle the hub), restart the wrist_camera node, confirm with
  `ros2 topic hz /wrist_camera/color/image_raw` (~30 Hz).
