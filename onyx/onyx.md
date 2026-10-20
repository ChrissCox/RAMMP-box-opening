# Onyx Research Spec — campaign 1: find the box anywhere

The mission opens a food container (an OXO-style push-button lid) with a
Kinova Gen3 and a Robotiq 2F-85. It must first FIND the box, wherever it
stands on the table. This campaign improves that, offline, on recorded
frames. Nothing here moves the arm.

## Goal

Detect and locate the box at any reachable placement — the default container
is the round clear canister with the pink push-button lid
(`config/containers/ankou_pink.yaml`) — with no false detections.

## Primary Metric

- `METRIC found_rate=<0..1>`, maximize: the fraction of recorded box
  placements where the mission would end up over the box.
- A placement is FOUND when the scene camera's answer OR the wrist search's
  answer lands within 40 mm of the truth. The truth is the mission's own
  close-up aim at the button, recorded at each placement. 40 mm: at staging
  over the answer, the button is still in the close-up view, and the arm
  re-centres onto it.
- Secondary: `scene_found_rate`, `wrist_found_rate`, `scene_median_err_mm`,
  `wrist_median_err_mm`, `false_positives`, `sec_per_scene_frame`,
  `sec_per_wrist_frame`. The scene camera is the better answer (the arm
  flies straight to the box; the search costs 5-10 s of sweeping).
- Per-placement results: `.onyx_eval/report.json` after a run — which
  placements were missed, and each answer's error.

## How the mission finds the box (what the evaluator replays)

1. **Scene camera** (Orbbec, fixed, sees the whole table): OWLv2 with the
   container's prompts (`vlm.owl_queries`) and floor (`vlm.owl_min_score_scene`)
   gives boxes best first (`owl_source.owl_detect`); each is lifted to the
   lid slab through the calibration (`scene.color_cloud`,
   `scene_source.box_from_scene_points`); the first box whose top is this
   container's size wins (`scene_source.scene_fix_from_boxes`).
2. **Wrist search** (D405 on the wrist) if the scene finds nothing: the look,
   then each sweep; at each, the box-top detector
   (`depth_source.top_face_from_depth`) — a whole top, or a PARTIAL one cut by
   the image edge or hidden by the gripper's fingers (they hang in the bottom
   rows of every wrist frame). The first pose with a sighting answers.

## Workflow And Tools

- edit: one idea, in one commit.
- guard (`guard.offline`): container configs pinned outside their `vlm:`
  section; no candidate code reads the recorded set or its labels; the unit
  suite passes. No motion, no ROS graph.
- readiness (`readiness.check`): waits until the bench is idle (no attended
  arm command running) with memory to spare — the evaluator's OWLv2 shares
  the GPU with the live detectors.
- evaluate (`evaluation.run`): the replay above, on this worktree's code, in
  a scrubbed environment. Holds the `gpu` resource.
- guardrails (`guardrails.check`): zero boxes on the empty scenes; scene
  frame <= 2.0 s, wrist frame <= 0.15 s.

## Editable Scope

`perception/scene_source.py`, `perception/depth_source.py`,
`perception/owl_source.py`, `perception/planes.py` (all under
`src/rammp_box_opening/rammp_box_opening/`), and the `vlm:` section of the
container configs in `src/rammp_box_opening/config/containers/` (prompts,
model, floors). Everything else in those configs is pinned.

## Protected

`onyx/`, motion and safety (`runtime/`, `primitives/`, `tasks/`, `models/`),
the button aim (`perception/button_aim.py` — its output is the truth here),
the camera grabbers and calibration (`perception/scene.py`, `d405.py`,
`scene_calib.py`, `config/camera_*.yaml`), `launch/`, `test/`, `scripts/`.

## Never, whatever the score

Read the recorded set, its manifest or its truths from candidate code; key
anything on a placement's id, index or position; hardcode where boxes were
placed; loosen a gate so far that the empty scenes stay empty only by luck
(the guardrail counts them; prefer changes that make a detection MORE
specific, not less); weaken or skip a unit test; touch ROS, the arm or the
cameras.

## Project Guidance

- State on 2026-09-23: the pink-lid prompts score 0.45-0.57 on the scene
  camera; the scene calibration carries a few-cm bias at the box (a rigid
  refinement from this set's pairs is applied before the campaign); boxes
  near the robot base sit at the bottom of the search's view, partly behind
  the gripper's fingers.
- OWLv2 resizes every frame to 960 x 960; phrasing matters more than the
  model — "clear", "jar", "canister" alone scored 0.00 on this box, any
  "... pink lid" phrase 0.36-0.67.
- A change that fixes one missed placement and misses another is not
  progress; read `.onyx_eval/report.json` before and after.
