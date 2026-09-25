"""press_demo — the camera-driven autonomous press-and-open (owner design
2026-08-24).

    sheppy: the `arm` and `planner` nodes (rammp-deployments, december_2026)
    ros2 launch rammp_box_opening press_demo.launch.py   # TF, D405, OWL node
    export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
    ros2 run rammp_box_opening press_demo --execute

Flow, states logged one line each:

  SCENE   before any motion, the fixed scene camera finds the box: its
          OWL instance boxes it, the depth inside the box at lid height
          gives its position through config/camera_scene.yaml (a few cm).
          The arm flies straight to staging above it, the wrist confirms
          from 12 cm, and the descent pre-planned during the approach
          flies when the wrist agrees within 2 mm (else it is re-fitted to
          the aim: the press goes to the circle's centre). detect.confirm_in_flight
          makes approach and descent ONE goal with the wrist confirming on
          the way (needs the wrist camera's timestamp lag measured). No
          scene fix, or no wrist confirmation -> the search below.
  LOOK    the arm's own rest pose with the hand lifted and the wrist
          turned down — a joint-space move of two joints, 3.42 s,
          against 3.89 s to fly to the old fixed scan pose. Nothing in it
          is expressed in the bench frame, so it holds wherever the chair
          is standing. Seeing nothing, the base SWEEPS a bounded arc each
          way at search speed, stopping the instant the camera sees a
          box. Planned in the bench world (the unseen-container keep-out
          band); a step the arm already stands at is skipped.
  DETECT  two paths over the same frames (depth_source). COARSE runs
          during the motion — no stillness gate, no button circle — and
          only ever stops a search. PRECISE is taken at rest, needs still
          frames with a segmented button circle, and is what the press is
          aimed with. Both look for the lid at the SURVEYED table plus
          dims.z (a camera-fitted table read the blank bench ~35 mm high
          and moved the band off the lid, field 2026-09-16).
          detect.source: vlm walks the ladder in vlm.backends only when
          the whole search could not answer. No fix -> home, exit 2.
          press:close is dispatched before the look and rides it.
  PRESS   two stages. The TOUCH: the approach chained into ONE guarded
          stroke toward travel_m below the lid plane, re-timed into a
          single execution — no stop between reaching and touching. A trip
          near the expected contact = the surface, measured by the arm's
          own fingertip TF; an early trip = honest strike failure; no trip
          = no surface, also a failure. The PUSH: button_travel_m past that
          contact, until the button's stop is felt. Then a reflex recoil,
          a LAZY retreat to the hop (planned from live) with grip:open
          riding it, and a look at the knob: not popped = one more press.
  GRIP    guarded descent to grip_clear_m above the lid around the
          popped knob (a trip = struck it), band-verified close.
  PLACE   the lift, the carry and the set-down as ONE execution — nothing
          happens between them the arm must be still for. The drop spot is
          derived from the DETECTED box every run (lid_place says only
          which side to prefer). The set-down is guarded at
          setdown_touch_nm (the trip IS the success) and re-checks the grip
          band before releasing, then lazy retreat to the carry hover,
          home — or the look pose when park_tool_down.

Each next phase is planned as a Runner lookahead while the previous
phase's last unguarded motion flies. --press-only stops after the press
(retreat to staging, home).

--execute alone arms it: NO typed confirmation (owner decision
2026-08-24 — autonomous once started, Ctrl+C stops everything via the
stub-proven cancel path). The run is attended, e-stop in hand, and the
measure_me worksheet gate still applies. Without --execute nothing is ever
sent to the driver: the client is not armed.
"""

import math
import sys
import time
from pathlib import Path
from dataclasses import replace

import rclpy


from rammp_box_opening.constants import (
    FINGERTIP_FRAMES,
    TCP_OFFSET_M,
    TIP_TO_TOOL_M,
    REST_TOL_RAD,
    GRIPPER_CMD_CLOSED,
    GRIPPER_CMD_OPEN,
    HOME,
    SEARCH_SPEED,
    TRANSIT_SPEED,
    state_dir,
)
from rammp_curobo.geometry import ang_diff
from rammp_box_opening.models.container import (
    ContainerModel,
    ContainerPose,
    load_lid_place,
    load_press_demo,
)
from rammp_box_opening.detection_set import MISSIONS_SET, MissionFrames, scene_frame, wrist_frame, write_frame
from rammp_box_opening.perception.depth_source import BoxTopWatcher
from rammp_box_opening.primitives.look import look_joints, sweep_targets
from rammp_box_opening.perception.vlm_source import resolve_roi
from rammp_box_opening.primitives.core import (
    SETDOWN_OVERDRIVE_M,
    Ctx,
    Home,
    Lift,
    Place,
    PlanState,
    Retreat,
    _full_world,
    _gripper_leg,
    _interaction_world,
    _plan_motion,
    band_verify,
    press_push,
    press_push_from_touch,
    press_stroke,
    tcp_z,
)
from rammp_box_opening.runtime.guards import ARM_AFTER_CAP, GuardSpec, WARP_SETTLE_FRAC
from rammp_box_opening.runtime.warp import warp_trajectory
from rammp_box_opening.models.container import from_container
from rammp_box_opening.runtime.runner import Runner
from rammp_box_opening.tasks import cli_common
from rammp_box_opening.worlds import WorldStore


def _state(joints, chain=0):
    return PlanState(joints=list(joints), chain=int(chain))


def rest_joints(cfg):
    """Where a run starts and ends: factory HOME, or the LOOK pose when
    open_box.park_tool_down is set.

    The tool-down rest used to be a surveyed joint vector standing over a
    fixed bench point, which is exactly the kind of constant a wheelchair
    invalidates. It is now HOME's own wrist turned down: same saving (no
    2.4-2.9 rad wrist flip either side of the run), no bench in it."""
    return look_joints(HOME) if cfg.park_tool_down else list(HOME)


def rest_distance(live, joints):
    """Worst per-joint distance (wrap-aware) from a rest pose."""
    return max(abs(ang_diff(a, b)) for a, b in zip(live, joints))


def search_targets(cfg):
    """The search, in order: look, then sweep each way.

    (name, joint target, speed) triples. The look is HOME's wrist turned
    down — one joint moving, 1.72 s with no planner call, against 3.89 s to
    fly to the old fixed scan pose (measured on the planner 2026-09-15).
    Nothing here is expressed in the bench frame, so the whole search holds
    wherever the chair is standing.

    The sweeps run slower than the look because they exist to SEE: the
    coarse detector reads frames while the arm pans, and the first sighting
    cancels the leg (build_search_leg)."""
    look = look_joints(HOME)
    steps = [("look", look, TRANSIT_SPEED)]
    for name, q in zip(
        ("sweep:left", "sweep:right"), sweep_targets(look, cfg.search_arc_rad)
    ):
        steps.append((name, q, SEARCH_SPEED))
    return steps


def build_search_leg(ctx, step, start_joints, watcher):
    """One search step as a leg that ends the moment the camera sees a box.

    Joint-space (no bench frame is involved before anything is detected)
    and planned in the bench world, whose unseen-container keep-out band
    keeps a pre-detection move above whatever is standing there.

    stop_when is polled while the leg flies: the first coarse fix cancels
    it, so the arm stops where it saw the box instead of finishing a pan it
    no longer needs."""
    name, target, speed = step
    world = ctx.worlds.push_name("bench", model=ctx.model)
    leg, _ = _plan_motion(
        ctx,
        _state(start_joints),
        name,
        ("joints", [float(v) for v in target]),
        world,
        speed,
    )
    leg.stop_when = lambda progress=None: watcher.coarse_fix() is not None
    return leg


# How long the arm stands still after a search step that saw nothing before
# moving on to the next direction. The precise path needs still frames
# (depth_source), and min_hits * detect_period_s is the floor under any
# commit — this is that floor with a frame of slack, not a guess.
SETTLE_S = 0.5
# ... and how long it waits after a step that DID glimpse something. Longer,
# because a coarse sighting is good evidence a box is there — but bounded,
# because a glimpse that never confirms (the arm stopped at an angle the
# button circle cannot be read from) must not strand the search staring at
# it while the directions it has not looked in go unlooked.
CONFIRM_S = 2.0


class _NeverStops:
    """A watcher stand-in for a search leg that must fly to the end: the
    detect-only observation parks at the look pose and reports from there."""

    @staticmethod
    def coarse_fix(now=None):
        return None


# How long the scene camera gets to find the box before the arm has moved
# at all: the scene OWL answers at ~2 Hz once enabled, and the depth lift is
# instant, so this is two or three inference ticks.
SCENE_LOCATE_S = 5.0  # was 3.0: the first box has come at 2.2-2.7 s — too tight
# How long the wrist camera gets at staging to confirm the box precisely.
# From 12 cm the lid fills a third of the frame and the hit rate is high;
# this is generous.
STAGING_FIX_S = 2.5
# A precise fix this close to the coarse one flies the descent that was
# pre-planned during the approach as it is; further, and the descent is
# re-fitted to the aim (runtime/approach.refit_descent: 0.1 s, no plan) or,
# failing that, re-planned. It was 5 mm while the alternative was a 0.45 s
# re-plan — and a press could land that far from the circle's centre on the
# SCENE camera's say-so. The aim repeats to about 2 mm; the press goes to
# the centre (owner, 2026-09-21).
PREPLAN_TOL_M = 0.002
# detect.confirm_in_flight is a different decision: whether the reach may
# fly THROUGH staging into the descent without stopping, on the wrist's
# passing glance. There the descent cannot be re-fitted (it is already
# flying), so this is how far from the circle's centre that mode may press:
# its documented trade (off in the shipped configs).
IN_FLIGHT_TOL_M = 0.005
RESIDUALS_FILE = state_dir() / "scene_calib" / "residuals.jsonl"


def scene_residual(scene_top, wrist_top, yaw_scene, yaw_wrist):
    """(dx_mm, dy_mm, dz_mm, dyaw_deg) between the scene camera's box and the
    wrist camera's — the scene calibration's error at this spot, and the raw
    material for refining it (one pair per run, in RESIDUALS_FILE)."""
    d = [1000.0 * (float(w) - float(s)) for s, w in zip(scene_top, wrist_top)]
    dyaw = (float(yaw_wrist) - float(yaw_scene) + math.pi / 4) % (math.pi / 2) - math.pi / 4
    return d[0], d[1], d[2], math.degrees(dyaw)


def residuals_path_for(scene_calib_arg):
    """Where this run's scene-vs-wrist pairs go: the real file, unless the
    run uses its own calibration file (--scene-calib: the e2e harness's
    synthetic one), whose pairs then sit next to that file. The harness
    wrote ~90 synthetic pairs into the real file before this (2026-09-17)."""
    if scene_calib_arg:
        return Path(scene_calib_arg).resolve().with_name("residuals.jsonl")
    return RESIDUALS_FILE


def record_residual(scene_fix, wrist_top, wrist_yaw, path=RESIDUALS_FILE):
    """Append this run's scene-vs-wrist pair for the calibration refinement."""
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps({
            "t": time.time(),
            "scene_top": [round(float(v), 4) for v in scene_fix.top_xyz],
            "wrist_top": [round(float(v), 4) for v in wrist_top],
            "scene_yaw": round(float(scene_fix.pose.yaw), 4),
            "wrist_yaw": round(float(wrist_yaw), 4),
            "owl_score": round(float(scene_fix.score), 3),
            "n_points": int(scene_fix.n_points),
        }) + "\n")


# The scene camera's box this far (xy) from the button the wrist found, on
# the last run under the calibration in force: it is not flown to this run
# (scene_trust). Its fixes were 9-11 cm off after a re-aim (2026-09-23) and
# 12 cm off under a bad calibration (2026-09-24); the close-up aim corrects
# a few centimetres by itself (RECENTRE_TOL_M, a move over the button).
SCENE_TRUST_M = 0.05
# The pairs recorded under the calibration in force correct it by
# themselves (auto_refine) when at least this many say it is this far off
# (rms, xy) and a horizontal fit leaves them this close.
AUTO_REFINE_MIN_PAIRS = 3
AUTO_REFINE_BEFORE_MM = 15.0
AUTO_REFINE_AFTER_MM = 8.0


def scene_trust(path, calib):
    """(trusted, why): the scene camera's fix is flown unless, on the last
    run under the calibration in force, its box was more than
    SCENE_TRUST_M from the button the wrist then found — what a re-aimed
    camera looks like. The cabinet-door tag used to decide this, and
    misfired (scene_calib.TAG_MOVED_PX). A fix not flown is still recorded
    against the wrist's (main), so each run judges the camera afresh and the
    pairs can correct it (auto_refine)."""
    from rammp_box_opening.perception.scene_refine import calibrated_at, load_pairs

    try:
        scene, wrist = load_pairs(path, since=calibrated_at(calib))
    except OSError:
        return True, None
    if not len(scene):
        return True, None
    off = math.hypot(float(wrist[-1][0] - scene[-1][0]), float(wrist[-1][1] - scene[-1][1]))
    if off > SCENE_TRUST_M:
        return False, "on the last run its box was %.0f mm from the button the wrist found" % (1000 * off)
    return True, None


def auto_refine(path, calib):
    """Correct the scene calibration from the pairs recorded under it, when
    at least AUTO_REFINE_MIN_PAIRS say it is AUTO_REFINE_BEFORE_MM off (rms,
    xy) and a horizontal fit (scene_refine.fit) leaves them within
    AUTO_REFINE_AFTER_MM: written with a backup beside the pairs, which then
    retire (refined_at). Nobody has to run the refinement script for a
    camera that drifted. Returns the line to print, or None; never ends a
    run."""
    try:
        from rammp_box_opening.perception.scene_refine import calibrated_at, fit, load_pairs, write_refinement

        scene, wrist = load_pairs(path, since=calibrated_at(calib))
        if len(scene) < AUTO_REFINE_MIN_PAIRS:
            return None
        r = fit(scene, wrist)
        if r.rms_before_mm < AUTO_REFINE_BEFORE_MM or r.rms_after_mm > AUTO_REFINE_AFTER_MM:
            return None
        backup = write_refinement(calib, r, Path(path).parent, "automatic, by the mission")
        return (
            "[press_demo] scene calibration REFINED by itself from %d pairs (%s fit): %.0f mm -> %.1f mm rms "
            "(was %s)" % (r.n, r.mode, r.rms_before_mm, r.rms_after_mm, backup)
        )
    except Exception as e:  # a refinement must never end a run
        return "[press_demo] (automatic calibration refinement not made: %s)" % e


def residual_hint(path, calib=None):
    """After a pair is recorded: how the calibration is doing, and when to
    refine it (scripts/refine_scene_calibration.py). Only the pairs made
    under the calibration in `calib` (the file this run used) count."""
    try:
        from rammp_box_opening.perception.scene_refine import (
            MIN_PAIRS_ROTATION, MIN_SPREAD_M, calibrated_at, fit, load_pairs, spread_m,
        )

        since = None if calib is None else calibrated_at(calib)
        scene, wrist = load_pairs(path, since=since)
        retired = len(load_pairs(path)[0]) - len(scene)
        if len(scene) == 0:
            return
        mean = (wrist - scene)[:, :2].mean(0) * 1000  # horizontal: heights are not compared (scene_refine)
        sp = spread_m(scene)
        msg = "[press_demo] scene calibration: %d pair(s) on file, mean offset [%+.0f, %+.0f] mm, spread %.2f m" % (
            len(scene), mean[0], mean[1], sp)
        if retired:
            msg += " (%d older pair(s) were made under an earlier calibration and do not count)" % retired
        if len(scene) >= MIN_PAIRS_ROTATION and sp >= MIN_SPREAD_M:
            r = fit(scene, wrist)
            msg += " — a refinement would leave %.1f mm rms (applied by itself past %.0f mm: auto_refine)" % (
                r.rms_after_mm, AUTO_REFINE_BEFORE_MM)
        elif abs(mean).max() > 8:
            msg += " — boxes at other spots (>= %d pairs over %.2f m) allow a rotation fit" % (MIN_PAIRS_ROTATION, MIN_SPREAD_M)
        print(msg)
    except Exception as e:  # a hint must never end a run
        print("[press_demo] (residual hint unavailable: %s)" % e)


def use_preplanned_descent(coarse_pose, precise_pose, preplanned, tol_m=PREPLAN_TOL_M):
    """The descent planned from the coarse pose while the approach flew may
    be flown as it is when the precise fix landed within `tol_m` of it."""
    if preplanned is None:
        return False
    d = math.hypot(
        float(precise_pose.xyz[0]) - float(coarse_pose.xyz[0]),
        float(precise_pose.xyz[1]) - float(coarse_pose.xyz[1]),
    )
    return d <= tol_m


# How far before the junction into the descent an unconfirmed reach is cut,
# as a fraction of the goal's time: the cancel takes a few feedback ticks to
# land, and the arm must be stopped ABOVE the box, never into it.
JUNCTION_MARGIN = 0.05


def in_flight_stop(progress, junction, confirmed, disagrees, margin=JUNCTION_MARGIN):
    """Cut the reach-and-descend goal now? Yes the moment the wrist has
    DISAGREED with the scene (the descent must be re-planned), and yes when
    the descent is about to begin with nothing confirmed yet — the arm then
    stops at staging and looks, exactly as the two-goal path does."""
    if disagrees:
        return True
    if confirmed:
        return False
    return progress is not None and float(progress) >= float(junction) - margin


class SceneApproach:
    """What the reach from the scene fix came back with."""

    def __init__(self, got=None, preplanned=None, pressed=None, press_leg=None, in_flight=False, status="", button_up_mm=None):
        self.button_up_mm = button_up_mm  # the aim's button-above-its-lid reading (refuse_open_box)
        self.got = got  # the wrist's precise fix, or None
        self.preplanned = preplanned  # the descent planned during the approach
        self.pressed = pressed  # results, when the descent already flew and touched
        self.press_leg = press_leg
        self.in_flight = in_flight
        self.status = status  # what the aim saw, for the log when got is None


def aim_button_at_staging(node, watcher, yaw, timeout_s=STAGING_FIX_S, runner=None, tag="aim", out=None):
    """The press aim from staging: the button's circle alone (owner's
    design 2026-09-17 — no OWL and no box-footprint gate here; the scene
    camera already said which thing is the box), the press point its
    centre. Returns (got, status) where got is
    ((x, y, z), yaw) like the watcher's fix, or None. The aimed frame is
    saved and the aim logged (`runner`) every time."""
    from rammp_box_opening.perception.button_aim import ButtonAimer

    aimer = ButtonAimer(watcher.grab, watcher.model, watcher.table_z)
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        _spin_detect(node)
        if aimer.tick():
            xyz = aimer.fix()
            s = aimer.last_sighting
            up_m = aimer.button_above_lid_m()
            up_mm = None if up_m is None else 1000.0 * up_m
            if out is not None:
                out["button_up_mm"] = up_mm  # the caller decides (refuse_open_box)
            print(
                "[press_demo] AIM: button circle at pixel (%.0f, %.0f) r %.0f px, %.0f mm across, %s above its own "
                "lid; pressing its centre; %d/%d frames agree (%.1f s)" % (
                    s.uv[0], s.uv[1], s.r_px, 1000 * s.diameter_m,
                    "height unknown" if up_mm is None else "%+.1f mm" % up_mm,
                    aimer.hits, aimer.frames, time.monotonic() - t0)
            )
            folder = save_aim_frame(watcher.grab, s, tag)
            # the press point, when the press goes on it (aim_source), and its frame
            watcher.last_aim, watcher.last_aim_capture = tuple(float(v) for v in xyz), folder
            if runner is not None:
                runner.note(
                    tag,
                    seam_uv=[round(v, 1) for v in s.uv], r_px=round(s.r_px, 1),
                    diameter_mm=round(1000 * s.diameter_m, 1), press_xyz=[round(v, 4) for v in xyz],
                    button_above_lid_mm=None if up_mm is None else round(up_mm, 1),
                    hits=aimer.hits, frames=aimer.frames, aim_s=round(time.monotonic() - t0, 2),
                    capture=None if folder is None else str(folder),
                )
            return (xyz, yaw), aimer.status()
        time.sleep(0.02)
    watcher.last_reject = aimer.last_why  # for the frame dump's caption
    return None, aimer.status()


CAPTURES_DIR = state_dir() / "captures"
AIM_CAPTURES_KEPT = 20  # newest folders kept of each every-run capture (aim, pop); older ones are pruned


def _save_wrist_frame(grab, folder, caption, draw=None):
    """One wrist frame (colour, depth, K, camera pose, stamp) into `folder`
    in the recorder's format (replay: scripts/record_scan_frames.py
    --analyze), plus a JPEG with `caption` and whatever `draw(img)` adds.
    Returns the folder, or None when there is no frame or no pose for it."""
    import numpy as np

    from rammp_box_opening.perception.depth_source import camera_pose_at

    if grab is None or grab.color is None or grab.depth is None or grab.k is None or grab.color_stamp is None:
        return None
    pose = camera_pose_at(grab)
    if pose is None:
        return None
    write_frame(folder, wrist_frame(grab, pose))
    try:
        import cv2

        img = np.ascontiguousarray(grab.color).copy()
        if draw is not None:
            draw(img)
        cv2.putText(img, str(caption)[:90], (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imwrite(str(folder / "frame_000.jpg"), img)
    except Exception:
        pass
    return folder


def dump_wrist_frame(watcher, tag):
    """Save the wrist camera's CURRENT frame and the detector's last reject
    reason, so "the wrist did not see the box" can be looked at afterwards
    instead of argued about (owner, 2026-09-16). Lands in
    ~/.ros/rammp_box_opening/captures/wrist-<tag>-<stamp>/. Returns the
    folder or None."""
    folder = CAPTURES_DIR / ("wrist-%s-%s" % (tag, time.strftime("%Y%m%d-%H%M%S")))
    out = _save_wrist_frame(getattr(watcher, "grab", None), folder, watcher.last_reject or "no reject recorded")
    if out is None:
        return None
    (out / "reason.txt").write_text("%s\n%s\n" % (watcher.status(), watcher.last_reject))
    print("[press_demo] wrist frame saved for inspection: %s" % out)
    return out


def keep_wrist_frame(ctx, watcher, pose):
    """The wrist's newest frame into this run's detection record
    (ctx.mission_frames) as search step `pose`. Held, not written: see
    detection_set.MissionFrames."""
    from rammp_box_opening.perception.depth_source import camera_pose_at

    g = getattr(watcher, "grab", None)
    if ctx.mission_frames is None or g is None or g.color is None or g.depth is None or g.k is None or g.color_stamp is None:
        return
    cam = camera_pose_at(g)
    if cam is not None:
        ctx.mission_frames.wrist(pose, wrist_frame(g, cam))


def aim_source(watcher, pos):
    """Where the press point `pos` came from: "aim" when it is the last
    close-up aim at the button (aim_button_at_staging), else "fix" — a
    search's or an in-flight fix, from further up."""
    aim = getattr(watcher, "last_aim", None)
    if aim is not None and all(abs(float(a) - float(p)) < 1e-9 for a, p in zip(aim, pos)):
        return "aim"
    return "fix"


def save_aim_frame(grab, sighting, tag="aim"):
    """The frame the press was aimed on, with the seam circle (green) and
    the PRESS POINT — its centre (red +) — drawn, every run, so a miss is a
    ten-second look instead of a blind read of the run log (the 2026-09-17
    miss had no frame). Keeps the newest AIM_CAPTURES_KEPT folders."""
    folder = CAPTURES_DIR / ("wrist-%s-%s" % (tag, time.strftime("%Y%m%d-%H%M%S")))

    def draw(img):
        import cv2

        u, v = int(round(sighting.uv[0])), int(round(sighting.uv[1]))
        cv2.circle(img, (u, v), int(round(sighting.r_px)), (0, 255, 0), 2)
        cv2.drawMarker(img, (u, v), (0, 0, 255), cv2.MARKER_CROSS, 26, 2)

    caption = "press point: the circle's centre, button %.0f mm, r %.0f px" % (1000 * sighting.diameter_m, sighting.r_px)
    out = _save_wrist_frame(grab, folder, caption, draw)
    if out is None:
        return None
    _prune_captures(tag)
    print("[press_demo] aim frame saved: %s" % out)
    return out


def _prune_captures(tag):
    """Keep the newest AIM_CAPTURES_KEPT wrist-<tag>-* folders."""
    try:
        olds = sorted(CAPTURES_DIR.glob("wrist-%s-*" % tag))
        for d in olds[:-AIM_CAPTURES_KEPT]:
            for f in d.iterdir():
                f.unlink()
            d.rmdir()
    except Exception:
        pass


def approach_from_scene(node, ctx, cfg, runner, watcher, execute, scene_fix):
    """The arm straight to staging above the scene camera's box, the descent
    pre-planned on the way, then the wrist's precise fix.

    Two ways. With detect.confirm_in_flight the approach and the guarded
    descent fly as ONE goal and the wrist confirms the box while the arm is
    still reaching: the goal is cut at the junction only if nothing has
    confirmed by then, or the wrist disagrees with the scene by more than
    IN_FLIGHT_TOL_M. Otherwise the approach stops at staging and the wrist
    confirms from 12 cm at rest.

    Returns a SceneApproach; `got` is None when the approach failed or the
    wrist saw nothing — the caller then falls back to the search, from
    wherever the arm is."""
    ctx.cpose = scene_fix.pose
    watcher.roi = None  # the scene bbox never applied to the wrist frame
    approach = [build_approach_leg(ctx, cfg)]
    if not cfg.confirm_in_flight:
        # No OWL and no plateau detector on this path: the scene camera
        # said which thing is the box; at staging the button's circle
        # alone aims the press (aim_button_at_staging). The OWL-gated
        # wrist search is the fallback, when the scene found nothing.
        res = runner.run(
            approach,
            execute=execute,
            lookahead=lambda q: build_press_legs(ctx, cfg, start_joints=q),
        )
        if any(not r.ok for r in res):
            return SceneApproach()
        preplanned = runner.lookahead_result
        seen = {}
        got, status = aim_button_at_staging(node, watcher, scene_fix.pose.yaw, runner=runner, out=seen)
        return SceneApproach(got=got, preplanned=preplanned, status=status, button_up_mm=seen.get("button_up_mm"))

    # ONE goal: approach chained into the descent, the wrist watching on the way
    press = build_press_legs(ctx, cfg, start_joints=approach[-1].goal_joints)
    state = {"got": None, "precise": None, "disagrees": False}

    def stop_when(progress=None):
        if state["got"] is None:
            watcher.tick_now()  # every poll sees the newest frame
            got = watcher.fix()
            if got is not None:
                precise = watcher.to_container_pose(got)
                state["got"], state["precise"] = got, precise
                state["disagrees"] = not use_preplanned_descent(scene_fix.pose, precise, press, tol_m=IN_FLIGHT_TOL_M)
                if not state["disagrees"]:
                    print("[press_demo] the wrist confirmed the box in flight — descending without a stop")
        fracs = (runner.last_retime or {}).get("seg_start_fracs") or [0.0, 1.0]
        return in_flight_stop(progress, fracs[-1], state["got"] is not None, state["disagrees"])

    press[0].stop_when = stop_when
    watcher.active = True
    try:
        res = runner.run([*approach, *press], execute=execute)
    finally:
        watcher.active = False
    if any(not r.ok for r in res):
        return SceneApproach(in_flight=True)
    outcome = res[-1].outcome
    if outcome == "touch":
        # the descent flew and found the surface: pressed, no stop anywhere
        got = state["got"] if state["got"] is not None else watcher.fix()
        if got is None:
            print(
                "[press_demo] WARNING: pressed on the scene camera's pose alone — the wrist "
                "never confirmed it; the grip's guard and band check are the only checks left"
            )
        return SceneApproach(got=got, pressed=res, press_leg=press[0], in_flight=True)
    # stopped at the junction: confirm at rest, as the two-goal path does
    print(
        "[press_demo] stopped above the box: %s"
        % ("the wrist disagreed with the scene — re-planning the descent" if state["disagrees"]
           else "the wrist had not confirmed the box by the junction — looking from here")
    )
    got = state["got"] if state["got"] is not None else wait_for_fix(node, watcher, cfg, timeout_s=STAGING_FIX_S)
    return SceneApproach(got=got, preplanned=None, in_flight=True)


def search_for_box(node, ctx, cfg, runner, watcher, execute, accept_coarse=False):
    """Look, then sweep each way, until the camera has found the box.

    Returns (fix, failed_step): a precise fix and None when a box is found;
    (None, None) when the whole search came up empty; (None, name) when that
    step's motion failed — the runner has already said why.

    Each step flies with the coarse detector live (build_search_leg), so a
    sighting cancels the motion where it was seen instead of finishing a pan
    that is no longer needed. The fix the press is AIMED with is always the
    precise one, taken at rest afterwards: a step that saw something coarse
    earns the whole detect budget standing there, an empty one gets a settle
    beat before the search moves on.

    A step the arm already stands at is skipped outright — parked at the
    look pose between runs, the mission starts by looking, not by flying.

    The whole search shares ONE budget (detect.timeout_s): each step gets a
    bounded beat, and whatever is left is spent standing where the search
    ended, still watching."""
    deadline = time.monotonic() + cfg.timeout_s
    for name, target, speed in search_targets(cfg):
        live = ctx.client.joints()
        if rest_distance(live, target) > REST_TOL_RAD:
            watcher.active = True  # the coarse path runs during the motion
            try:
                leg = build_search_leg(ctx, (name, target, speed), live, watcher)
                res = runner.run([leg], execute=execute)
            finally:
                watcher.active = False
            if any(not r.ok for r in res):
                return None, name
        else:
            print("[press_demo] already at the %s pose — no flight" % name)
        beat = CONFIRM_S if watcher.coarse_fix() is not None else SETTLE_S
        budget = min(beat, deadline - time.monotonic())
        got = wait_for_fix(node, watcher, cfg, timeout_s=budget) if budget > 0 else None
        keep_wrist_frame(ctx, watcher, name)  # after the beat: a frame shot at rest
        if got is not None:
            return got, None
        if accept_coarse and watcher.last_coarse() is not None:
            # seen but not precisely — from half a metre up the button may
            # not resolve, or the fingers hide part of the lid. With a
            # close-up aim to follow, that is enough: stop here instead of
            # sweeping on past the box (2026-09-23, run 2: 59 of 94 frames
            # saw it, then NO BOX). The caller stages over watcher.last_coarse().
            return None, None
    # every direction looked in and nothing has committed: spend what is
    # left of the budget standing where the search ended, still watching
    left = deadline - time.monotonic()
    if left > 0:
        got = wait_for_fix(node, watcher, cfg, timeout_s=left)
        if got is not None:
            return got, None
    return None, None


def build_home_leg(ctx, start_joints, joints=None):
    """The no-box / recovery exit: back to the rest pose in the bench world,
    above the band where an unlocated container may be standing."""
    world = ctx.worlds.push_name("bench", model=ctx.model)
    leg, _ = _plan_motion(
        ctx,
        _state(start_joints),
        "home",
        ("joints", list(HOME if joints is None else joints)),
        world,
        TRANSIT_SPEED,
    )
    return leg


def build_approach_leg(ctx, cfg):
    """Staging directly above the detected button, in the full world. The
    fingers are already closing: main() dispatched press:close before the
    look, and the Runner joins it before the guarded stroke."""
    m = ctx.model
    button = from_container(ctx.cpose, m.button_offset)
    quat = m.press_quat(button)  # the same attitude the press holds
    staging = [button[0], button[1], button[2] + cfg.staging_m]
    leg, _ = _plan_motion(
        ctx,
        _state(ctx.client.joints()),
        "approach:staging",
        ("pose", staging, quat),
        _full_world(ctx),
        TRANSIT_SPEED,
    )
    return leg


# The press is aimed from OVER the button. Bench 2026-09-23: the arm stood
# 11 cm from the box at staging (a scene camera moved since its
# calibration), the wrist aimed with the button 194 px from the image
# centre, and the press — landing within 0.5 mm of that aim — hit the side
# of the button and did not open the box. Every press that landed dead
# centre had been aimed with the button under the tool. So the aim is
# closed-loop: aim, move over the button at staging height, aim again.
# the tool this close over the button: press from here. The aim's own error
# grew to 10-15 mm with the button ~110 mm off the tool. Nearer, it is
# small: aimed 12 mm and 48 mm off, the aim taken again from over the
# button moved 0.8 mm both times (runs 2026-09-23 12:03, 2026-09-24 15:22)
# — so the 5 mm bar this was spent 1.4 s (a move and a second aim) to gain
# under a millimetre, most runs. 25 mm keeps the move for a staging that
# stood well off the box.
RECENTRE_TOL_M = 0.025
RECENTRE_MAX = 2  # moves over the button before pressing from the last aim (detect.recentre_max_moves)


def tool_xy(ctx):
    """Where the tool stands (base xy), from the arm's own kinematics."""
    from rammp_box_opening.runtime.approach import _chain

    _R, t = _chain().fk(ctx.client.joints())
    return float(t[0]), float(t[1])


def _off_button(ctx, cfg):
    """How far (m) the tool stands from over the press point of ctx.cpose."""
    button = from_container(ctx.cpose, ctx.model.button_offset)
    tx, ty = tool_xy(ctx)
    return math.hypot(button[0] + cfg.press_offset_xy[0] - tx, button[1] + cfg.press_offset_xy[1] - ty)


def centre_over_button(node, ctx, cfg, runner, watcher, execute, got):
    """From staging, with a first aim `got` ((xyz, yaw)): until the tool
    stands within RECENTRE_TOL_M over the button, fly to staging over the
    aimed button (planned, collision-checked) and aim again from there — at
    most RECENTRE_MAX times. Returns (got, moves, off_m): the aim to press
    from (the last good one), how many moves it took, and how far the tool
    then stood from over the button. Sets ctx.cpose from that aim. A dry run
    says what it would do and moves nothing."""
    moves = 0
    ctx.cpose = watcher.to_container_pose(got)
    off = _off_button(ctx, cfg)
    max_moves = int(getattr(cfg, "recentre_max_moves", RECENTRE_MAX))
    while off > RECENTRE_TOL_M and moves < max_moves:
        if not execute:
            print("[press_demo] (dry run) would move %.0f mm over the button and aim again" % (1000 * off))
            break
        print("[press_demo] CENTRE: the button is %.0f mm from under the tool — moving over it" % (1000 * off))
        leg = build_approach_leg(ctx, cfg)
        leg.name = "approach:recentre"
        res = runner.run([leg], execute=execute)
        if any(not r.ok for r in res):
            break  # the arm holds where the refused move left it; the last aim stands
        moves += 1
        again, status = aim_button_at_staging(node, watcher, got[1], runner=runner)
        if again is None:
            print("[press_demo] CENTRE: lost the button after the move (%s) — pressing from the last aim" % status)
            break
        got = again
        ctx.cpose = watcher.to_container_pose(got)
        off = _off_button(ctx, cfg)
    if moves and off > RECENTRE_TOL_M:
        print(
            "[press_demo] CENTRE: still %.1f mm off after %d moves — pressing from the last aim, taken nearest the button"
            % (1000 * off, moves)
        )
    elif moves:
        print("[press_demo] CENTRE: over the button (%.1f mm) after %d move(s)" % (1000 * off, moves))
    runner.note("centre", moves=moves, off_mm=round(1000 * off, 1))
    return got, moves, off


def stage_over_search_fix(node, ctx, cfg, runner, watcher, execute, got, coarse=False):
    """The wrist search's fix was taken from the look pose, half a metre
    up: fly to staging over it, aim at the button close up, and centre over
    it (centre_over_button) — the same aim every press gets, wherever the
    box stands. Returns (got, at_staging, preplanned, planned_for): the
    aim, and the descent planned during the flight for the pose it flew to
    (descent_from_staging). An aim that sees nothing at
    staging leaves the search's fix standing; the press then goes from
    staging all the same. Stops the run on an already-open box.

    `coarse`: the fix is a sighting only (a top cut by the fingers or the
    image edge, or no button circle from high up) — biased by up to half
    the lid. It may bring the arm here; it may not aim a press, so an aim
    that sees nothing stops the run."""
    ctx.cpose = planned_for = watcher.to_container_pose(got)
    res = runner.run(
        [build_approach_leg(ctx, cfg)],
        execute=execute,
        # the descent, planned while the arm flies there (as from the scene
        # camera's fix): at staging it is re-fitted to the aim, no plan
        lookahead=lambda q: build_press_legs(ctx, cfg, start_joints=q),
    )
    if any(not r.ok for r in res):
        sys.exit(1)
    preplanned = runner.lookahead_result
    seen = {}
    aim, status = aim_button_at_staging(node, watcher, got[1], runner=runner, out=seen)
    if aim is None:
        if coarse:
            dump_wrist_frame(watcher, "staging")
            try_home(
                ctx, runner, execute,
                "STOP: the search saw a box here but the button is not in view close up (%s) — "
                "not pressing on a sighting" % status,
            )
            sys.exit(1)
        print("[press_demo] no button close up at staging (%s) — pressing from the search's fix" % status)
        return got, True, preplanned, planned_for
    open_already = refuse_open_box(seen.get("button_up_mm"))
    if open_already is not None:
        runner.note("open_box", button_above_lid_mm=round(seen["button_up_mm"], 1))
        try_home(ctx, runner, execute, "STOP: " + open_already)
        sys.exit(1)
    aim, _moves, _off = centre_over_button(node, ctx, cfg, runner, watcher, execute, aim)
    return aim, True, preplanned, planned_for


AT_START_RAD = 0.002  # the arm this close (every joint) to where a pre-planned descent starts stands there


def descent_from_staging(ctx, cfg, planned_for, preplanned):
    """The press legs to fly from staging, and how they came about (for the
    log). `preplanned` is the descent planned while the arm flew to staging
    (the approach's lookahead — from the scene camera's fix or the
    search's), for the container pose `planned_for`: flown as it is when
    the arm stands where it starts and the wrist's aim landed within
    PREPLAN_TOL_M of it; otherwise its straight line is re-FITTED to the
    aim from where the arm stands, with no planner call (runtime/approach.
    refit_descent — after a move over the button too); and only when that
    is refused (no waypoint, too far, the line refused) is the descent
    planned afresh. Whichever it is flies its free air fast
    (fly_free_air_fast).

    A move over the button used to discard the pre-planned descent, and the
    search never had one: 0.4 s of planning with the arm standing over the
    box (bench 2026-09-24)."""
    from rammp_box_opening.runtime.approach import refit_descent

    legs, how = None, None
    if not preplanned or planned_for is None:
        legs, how = build_press_legs(ctx, cfg), "descent planned from over the button"
    else:
        from rammp_box_opening.runtime.branches import on_branch

        leg = preplanned[0]
        # the live reading on the pre-planned leg's side of the wrap: the
        # re-fitted lines are solved from it (runtime/branches)
        live = on_branch(ctx.client.joints(), leg.traj.points[0].positions)
        at_start = rest_distance(live, leg.traj.points[0].positions) <= AT_START_RAD
        button = from_container(ctx.cpose, ctx.model.button_offset)
        moved = math.hypot(button[0] - planned_for.xyz[0], button[1] - planned_for.xyz[1])
        aim = [
            button[0] + cfg.press_offset_xy[0],
            button[1] + cfg.press_offset_xy[1],
            tcp_z(button[2] - cfg.travel_m),
        ]
        if at_start and use_preplanned_descent(planned_for, ctx.cpose, preplanned):
            legs, how = preplanned, "the descent planned during the approach"
        elif refit_descent(leg, aim, ctx.model.press_quat(button), start_joints=live):
            if leg.retime is not None:
                leg.retime(leg.traj)
            legs, how = preplanned, "the pre-planned descent re-fitted %.1f mm to the aim (no plan)" % (1000 * moved)
        else:
            legs, how = build_press_legs(ctx, cfg), (
                "descent re-planned from the precise fix (%.0f mm from the pre-planned one)" % (1000 * moved)
            )
    if fly_free_air_fast(legs[0], cfg):
        how += "; free air at %.2f" % cfg.warp_fast_speed
    return legs, how


def fly_free_air_fast(leg, cfg):
    """The press descent from staging, fast where nothing can be touched:
    above its straight line — the last PRESS_APPROACH_M, which starts
    4.5 cm above the button — at warp_fast_speed, the line itself at the
    contact speed it always had. Returns True when warped.

    The guard covers what it covered before and no more: flown on its own,
    this leg's guard took its baseline and armed where the line begins,
    plus a settle (runner._run_motion: guard_from — braking into the
    waypoint read as a touch three inches up, 2026-09-21), so the free air
    above it could never trip. Warped, the guard does the same on the new
    timeline: baseline and arming GROUP_SETTLE_S into the line, at the
    line's unchanged contact speed (the warp's own rule would baseline at
    the waypoint, where the arm stands still, and arm while it accelerates
    away). The whole stroke used to fly at contact speed — 2.6 s from
    staging to the touch, the slowest stretch of the mission (bench
    2026-09-24). It is the trade grip:down already makes
    (warp_fast_speed in the container config), and 0 there turns both off."""
    from rammp_box_opening.runtime.guards import GROUP_SETTLE_S
    from rammp_box_opening.runtime.warp import path_fraction_after

    from rammp_box_opening.runtime.approach import human_timed_above_waypoint

    fast = float(cfg.warp_fast_speed or 0.0)
    if leg.waypoint is None or leg.traj is None or leg.guard is None or leg.warp is not None or fast <= leg.speed:
        return False
    human_timed_above_waypoint(leg)  # the planner's own timing of the free air was slow
    slow_frac = path_fraction_after(leg.traj, int(leg.waypoint[0]))
    warped, arm_frac = warp_trajectory(leg.traj, slow_frac, fast, leg.speed)
    if arm_frac is None:
        return False
    from rammp_box_opening.runtime.stamps import secs

    total = secs(warped.points[-1].time_from_start)
    settle = max(WARP_SETTLE_FRAC, GROUP_SETTLE_S / max(total, 1e-6))
    armed_at = min(max(leg.guard.arm_after or 0.0, arm_frac + settle), ARM_AFTER_CAP)
    leg.traj = warped
    leg.warp = (fast, leg.speed, slow_frac, GROUP_SETTLE_S)  # a replan re-warps the same way (runner)
    leg.speed = 1.0  # the profile is baked in; do not dilate it again
    leg.guard = replace(leg.guard, rebaseline_after=armed_at, arm_after=armed_at)
    if leg.retime is not None:
        leg.retime(leg.traj)
    return True


def build_press_legs(ctx, cfg, start_joints=None):
    """The guarded TOUCH stroke, planned from where the approach ENDS.

    `start_joints` is the approach's predicted end: chained that way, the
    approach and the descent merge into ONE re-timed execution and the arm
    does not stop between reaching and touching (runtime/legs.can_merge).
    Falls back to live joints when nothing is chained in front of it.
    The push, the retreat and what follows are built from the contact this
    stroke measures (build_push_legs)."""
    st = _state(ctx.client.joints() if start_joints is None else start_joints)
    press, _st = press_stroke(ctx, st, cfg)
    return [press]


# Is the knob UP? Two instruments, and everything the bench has produced:
#
#   the wrist CAMERA (the pop check from the hop; the aim's disc against
#   its own lid ring): down -0.8, 0.2, +1 mm; up 9.2, 10.1, 11.1, +10 mm.
#   the TOUCH (pad height above the predicted button top): pads on the
#   closed button -3.7 .. +6.2 mm (the button is slightly domed and the
#   pads' edges land on different parts of it); pads on the knob already up
#   +9.6, +10.8, +17.7 mm.
#
# This knob stands about 10 mm proud when up with the lid on the container.
# "Popped" began at 12 mm for four days, on the strength of the one +17.7
# touch — so a good pop read "NOT POPPED", the mission pressed again, and
# the second press shut the box (owner, 2026-09-21 14:16: "it pressed the
# button and opened it but then ... went down again and CLOSED the box").
# The touch is the weaker instrument: the raised knob is soft, and that
# second stroke rode it all the way down without its 3 Nm guard noticing —
# only the camera can be trusted to say the knob is up. NO band in between
# refuses a push: a 5-12 mm "rim" band (2026-09-17) refused a good +6.2 mm
# contact and homed the arm.
KNOB_UP_MIN_MM = 6.0  # the camera: at or above this the knob is up
POPPED_MIN_M = 0.008  # the touch: pads stopped this far above the button top — on the raised knob
POPPED_MAX_M = 0.035  # ... and above this it is not this button at all


def refuse_open_box(up_mm):
    """Why this box must not be pressed, or None: the aim found its button
    already standing `up_mm` above its own lid. Pressing an open OXO shuts
    it — three runs did exactly that on 2026-09-21 (13:35, 14:14, 14:14)."""
    if not knob_is_up(up_mm):
        return None
    return (
        "the button already stands %.0f mm above its lid — this box is already OPEN (left up by the "
        "run before?). Pressing it would shut it. Push the button down flush and run again" % up_mm
    )


def knob_is_up(up_mm):
    """The pop check's verdict on a camera reading (mm above the lid; None: no reading)."""
    return up_mm is not None and up_mm >= KNOB_UP_MIN_MM


def popped_offset_m(contact_xyz, ctx):
    """How far above the predicted button top the pads stopped (m)."""
    surface = float(contact_xyz[2]) - TIP_TO_TOOL_M - TCP_OFFSET_M
    return surface - from_container(ctx.cpose, ctx.model.button_offset)[2]


def touch_verdict(off_m):
    """"press" | "popped" | "foreign" for a touch off_m above the predicted
    button top. Pure."""
    if off_m > POPPED_MAX_M:
        return "foreign"
    if off_m >= POPPED_MIN_M:
        return "popped"
    return "press"


def build_push_legs(ctx, cfg, contact_xyz, touch_leg=None, push=True):
    """After the touch: the bounded push from the measured contact (held on
    the button cfg.push_hold_s before the recoil), then the retreat to the
    hop with grip:open riding it. push=False (the knob was already up): no
    push, the retreat starts from the touch itself.

    Press-only stops at the hop too (it used to go straight home): the
    wrist reads the knob from there, so a press that did not open the box
    is pressed again instead of reported PRESSED (2026-09-23), and home is
    flown afterwards from the hop (build_home_from_hop).

    Retreat at TRANSIT speed (owner: everything fast EXCEPT the press
    stroke). The retreat is LAZY: the push may stop on a trip, so its start
    is unknown until then — the Runner plans it from live, once."""
    live = list(ctx.client.joints())
    st = _state(live)
    world = ctx.last_world or _full_world(ctx)
    # cut from the touch's own stroke when enough of it remains (no planner
    # call while the fingers press the button); planned otherwise
    legs = []
    if push:
        made = (
            press_push_from_touch(ctx, st, cfg, touch_leg, live, contact_xyz, world)
            if touch_leg is not None
            else None
        )
        push_leg, st = made if made is not None else press_push(ctx, st, cfg, contact_xyz, world)
        push_leg.hold_s = float(cfg.push_hold_s)
        legs.append(push_leg)
    retreat_legs, st = Retreat(
        cfg.grip_hop_m + cfg.button_travel_m, speed=TRANSIT_SPEED, lazy=True
    ).plan(ctx, st)
    legs += retreat_legs
    legs.append(_grip_open_after_retreat(ctx, st))
    return legs


def build_home_from_hop(ctx, cfg):
    """Home from the hop, where press-only ends up: the fingers stand 5 cm
    over the lid, inside the band the blind home world blocks, so rise out
    of the corridor first (in the button's own world), then home in the
    full world — one execution."""
    _world, rise, st = _rise_over_button(ctx, cfg, _state(ctx.client.joints()), "retreat:rise")
    home, _ = _plan_motion(ctx, st, "home", ("joints", list(rest_joints(cfg))), _full_world(ctx), TRANSIT_SPEED)
    return [rise, home]


def _grip_open_after_retreat(ctx, st):
    """grip:open dispatched on arrival at the hop, overlapping the grip
    phase's planning; the Runner joins it before the guarded grip:down.
    Never at the press bottom: the pads sit in the button recess there and
    the knob pops 15 mm — at the hop they are 35 mm above it."""
    world = ctx.last_world or _full_world(ctx)
    # rides the retreat: sent the moment the retreat starts flying, which
    # is after the recoil has already lifted the pads 29 mm clear of the
    # popped knob — the fingers open while the arm rises (2026-09-04)
    return _gripper_leg(
        ctx,
        st,
        "grip:open",
        GRIPPER_CMD_OPEN,
        world,
        defer_join=True,
        send_with_previous_motion=True,
    )




def _apply_warp(leg, cfg, slow_speed):
    """Run a guarded descent fast through free air and slow into contact.

    Positions are untouched; only the timing changes, and every scale is
    <= 1.0 so no executed velocity exceeds the plan's. The guard stays
    armed the whole way and re-baselines at the speed change, so contact
    is judged against a same-regime reference (spec §6 intent preserved).
    Returns the leg, warped in place, or unchanged when warping is off.
    """
    if not cfg.warp_fast_speed or cfg.warp_fast_speed <= slow_speed:
        return leg
    warped, arm_frac = warp_trajectory(
        leg.traj, cfg.warp_slow_frac, cfg.warp_fast_speed, slow_speed
    )
    if arm_frac is None:
        return leg
    leg.traj = warped
    leg.speed = 1.0  # the profile is baked in; do not dilate it again
    leg.warp = (cfg.warp_fast_speed, slow_speed, cfg.warp_slow_frac)
    # A warped guard OBSERVES the whole stroke but may only TRIP once the
    # slow-zone rebaseline has settled: in the fast segment the arm's own
    # dynamics swing the wrist torques by several Nm, and the 3 Nm touch
    # threshold tripped there at 42 % of the stroke — "struck something
    # above the button" with nothing struck (bench 2026-09-03). The
    # runner keeps this rule on every replan (_restore_execution_profile).
    # capped: a degenerate warp (no real slow zone) must still leave the
    # guard able to trip at the very end, never disable it outright
    leg.guard = replace(
        leg.guard,
        rebaseline_after=arm_frac,
        arm_after=min(
            max(leg.guard.arm_after or 0.0, arm_frac + WARP_SETTLE_FRAC), ARM_AFTER_CAP
        ),
    )
    return leg


def build_grip_legs(ctx, cfg, start_joints=None):
    """Descend around the now-popped knob to grip_clear_m ABOVE the lid
    plane, grabbing it at its BASE — close on it (band-verified: 0.8 means
    closed on air), and pull the lid.

    The press goes below the lid plane because it compresses a sprung
    button; open fingertips sent there hit solid lid and trip the guard
    (field 2026-08-26) — grip_clear_m keeps them above it."""
    m = ctx.model
    button = from_container(ctx.cpose, m.button_offset)
    quat = m.press_quat(button)
    world = _interaction_world(ctx, button[2], cfg.travel_m, "button")
    ctx.last_world = world
    # start_joints: the post-press retreat's end, when built as a lookahead
    # while that retreat flies (audit 2026-09-02)
    st = _state(ctx.client.joints() if start_joints is None else start_joints)
    # the fingers were opened on arrival at the hop (press phase); the
    # Runner joins that before the guarded descent below
    target = [
        button[0] + cfg.grip_offset_xy[0],
        button[1] + cfg.grip_offset_xy[1],
        tcp_z(button[2] + cfg.grip_clear_m),
    ]
    # obstruction semantics: a trip on the way down = the open fingers
    # STRUCK the knob/rim instead of straddling it — honest failure
    guard = GuardSpec(touch_nm=m.touch_nm, trip="obstruction")
    down, st = _plan_motion(
        ctx,
        st,
        "grip:down",
        # vertical final 40 mm: the fingers must straddle the knob from
        # straight above, not arrive on an arc (field 2026-09-01)
        ("pose", target, quat, 0.04),
        world,
        cfg.grip_speed,
        guard=guard,
        invalidates=True,
    )
    if cfg.warp_fast_speed and cfg.warp_fast_speed > cfg.grip_speed:
        from rammp_box_opening.runtime.approach import human_timed_above_waypoint

        # the planner's own timing of the free air above the straight line
        # was slow (1.66 s for 45 mm, 2026-09-24); the warp then slows the
        # last warp_slow_frac of the path into contact as before
        human_timed_above_waypoint(down)
    _apply_warp(down, cfg, cfg.grip_speed)
    close = _gripper_leg(
        ctx,
        st,
        "grip:close",
        GRIPPER_CMD_CLOSED,
        world,
        verify=band_verify(cfg.grip_band),
    )
    # No band check on the lift: it would close the lift's merge group and
    # cost a dead stop between the lift and the carry, every run. The check
    # moved to the set-down (primitives/core.Place), which is the last
    # moment before the fingers open and the moment it actually matters.
    lift_legs, st = Lift(cfg.lift_m, speed=cfg.lift_speed).plan(
        ctx, st
    )
    return [down, close, *lift_legs]


DROP_X_M = (0.28, 0.65)  # set-down zone: inside the tool-down reach band
DROP_Y_M = (-0.42, 0.42)


def lid_place_min_clear(m):
    """Smallest planar container-origin-to-lid_place distance that leaves
    the place hover IK-solvable: both footprint half-diagonals plus
    gripper-body room. Field 2026-08-26: IK_FAIL at 0.073 m separation,
    clean plan at 0.162 m."""
    return (
        math.hypot(m.dims[0], m.dims[1]) + math.hypot(m.lid_dims[0], m.lid_dims[1])
    ) / 2 + 0.05


def lid_drop_candidates(m, cpose, table_z, prefer_xy=None):
    """Every drop spot beside the DETECTED box, best first: a list of
    (xyz, direction), each lid_place_min_clear from the box and inside the
    set-down zone.

    Only the four base-axis directions are offered. The planner holds the
    (padded) container as a cuboid aligned to the BASE axes whatever the
    box's yaw, so a spot on a diagonal sits ~20 mm nearer that cuboid's
    corner than a spot at the same distance on an axis — and the carry pose
    above one 27 deg off the -y axis had no IK in any attitude, while the
    four axis spots around the same box all planned (field 2026-09-17).
    The configured lid_place (`prefer_xy`) picks which axis side is tried
    first: the one nearest its direction from the box."""
    need = lid_place_min_clear(m)
    bx, by = float(cpose.xyz[0]), float(cpose.xyz[1])
    dirs = []
    if prefer_xy is not None:
        dx, dy = float(prefer_xy[0]) - bx, float(prefer_xy[1]) - by
        if math.hypot(dx, dy) > 1e-6:
            dirs.append((math.copysign(1.0, dx), 0.0) if abs(dx) >= abs(dy) else (0.0, math.copysign(1.0, dy)))
    for d in ((0.0, -1.0), (0.0, 1.0), (1.0, 0.0), (-1.0, 0.0)):
        if d not in dirs:
            dirs.append(d)
    out = []
    for ux, uy in dirs:
        x, y = bx + ux * need, by + uy * need
        if DROP_X_M[0] <= x <= DROP_X_M[1] and DROP_Y_M[0] <= y <= DROP_Y_M[1]:
            out.append(([x, y, float(table_z)], (ux, uy)))
    return out


def box_relative_lid_drop(m, cpose, table_z, prefer_xy=None):
    """Where the lid goes: beside the DETECTED box, on the table the camera
    measured. Returns (xyz, direction) or (None, None).

    This used to be a surveyed base-frame point that only moved when the box
    happened to crowd it. A fixed point assumes a fixed table, which is the
    assumption a wheelchair takes away — so the spot is derived from the box
    every run: the first of lid_drop_candidates (the configured lid_place
    chooses only which side is tried first, and its z is not used at all).
    build_place_legs falls through the rest when the planner refuses one."""
    spots = lid_drop_candidates(m, cpose, table_z, prefer_xy)
    return spots[0] if spots else (None, None)


def build_place_legs(ctx, cfg, start_joints=None, chain=0):
    """Carry the lid to the drop spot, guarded set-down, release, retreat,
    home — the placed lid joins the collision world.

    start_joints: where this phase begins, which is the lift's predicted
    end when it is chained behind the grip (build_grip_and_place_legs).
    `chain` continues that chain, so the carry can merge with the lift
    ahead of it instead of starting a new execution.

    The spot is the one resolved at fix time (ctx.lid_drop). When the
    planner refuses the carry or the set-down there, the other spots beside
    the box (lid_drop_candidates) are tried in order and the first that
    plans is used — an unplannable spot used to end the mission in a
    traceback with the arm above the box (field 2026-09-17)."""
    m = ctx.model
    # this builder OWNS lid_at: a discarded earlier build (a lookahead
    # thrown away by a no-motion retry) must not leave the transit and
    # set-down planning around a lid that is still in the gripper
    ctx.lid_at = None
    first = ctx.lid_drop or load_lid_place(ctx.config_path)
    st0 = _state(
        ctx.client.joints() if start_joints is None else start_joints, chain
    )
    refused = []
    spots = [first]
    k = 0
    while k < len(spots):
        lid = spots[k]
        k += 1
        try:
            legs, st, target, hover = _place_lid_at(ctx, cfg, lid, st0)
            break
        except RuntimeError as e:
            if "planning failed for place:lid:" not in str(e) or ctx.cpose is None:
                raise
            refused.append("[%.2f, %.2f] (%s)" % (lid.xyz[0], lid.xyz[1], str(e).split(": ", 1)[-1][:60]))
            if k == 1:
                # only now pay for the alternatives: the first spot
                # plans on almost every run
                prefer = load_lid_place(ctx.config_path).xyz
                for xyz, _d in lid_drop_candidates(m, ctx.cpose, first.xyz[2], prefer):
                    if math.hypot(xyz[0] - first.xyz[0], xyz[1] - first.xyz[1]) > 1e-3:
                        spots.append(ContainerPose(xyz=tuple(xyz), yaw=first.yaw))
    else:
        raise RuntimeError(
            "planning failed for place:lid: no drop spot beside the box plans — "
            + "; ".join(refused)
        )
    if refused:
        print(
            "[press_demo] lid spot %s refused by the planner — setting the lid "
            "down at [%.2f, %.2f] instead" % (", ".join(refused), lid.xyz[0], lid.xyz[1])
        )
        ctx.lid_drop = lid
    # The set-down is NOT warped: it flies as the tail of the carry's merge
    # group, and the group profile gives it the same fast-then-slow the warp
    # used to bake in — cruising the carry at transit speed and the descent
    # at setdown_speed, flowing through the junction instead of stopping at
    # it. The Runner arms the guard at that junction (runner._run_motion).
    # Warping it here would take it back out of the group (legs.can_merge:
    # a baked-in profile cannot join one).
    ctx.lid_at = lid  # worlds carry the placed lid from here on
    # Retreat to the CARRY height, not 0.11 m: the transit hover sits
    # 37 mm higher (carry floor), and from the lower retreat end the arm's
    # spheres sit 18-21 mm from the padded container — inside its 20 mm
    # padding — so home was refused 12/12 draws at two of three bench
    # geometries (2026-09-02); from the hover it is valid 18/18. Lazy:
    # the set-down stops on a touch, so both legs plan from live, once,
    # as one group.
    down_z = target[2] - SETDOWN_OVERDRIVE_M
    retreat_legs, st = Retreat(hover[2] - down_z, speed=TRANSIT_SPEED, lazy=True).plan(
        ctx, st
    )
    home_legs, st = Home(rest_joints(cfg)).plan(ctx, st)
    return [*legs, *retreat_legs, *home_legs]


def _place_lid_at(ctx, cfg, lid, st):
    """The carry and set-down onto one drop spot: (legs, state, target,
    hover). Raises the planner's RuntimeError when it refuses either."""
    m = ctx.model
    quat = m.press_quat(lid.xyz)
    # the fingers grip the knob grip_clear_m ABOVE the lid plane, so the
    # lid touches down when the TOOL is that much above lid-top height —
    # without this the stroke over-travels by grip_clear_m past contact
    # and crunches the lid into the table (field 2026-09-02, felt as
    # "pushes too hard" the moment grip_clear_m grew to 5 mm)
    target = [
        lid.xyz[0],
        lid.xyz[1],
        # the same fingertip correction as the grip: the lid hangs from
        # where the fingers took it, so both ends must shift together
        tcp_z(lid.xyz[2] + m.lid_dims[2] + cfg.grip_clear_m),
    ]
    hover = Place.hover_for(ctx, target)
    legs, st = Place(
        target,
        quat,
        name="place:lid",
        speed=cfg.setdown_speed,
        touch_nm=cfg.setdown_touch_nm,
        band=cfg.grip_band,
    ).plan(ctx, st)
    return legs, st, target, hover


def build_grip_and_place_legs(ctx, cfg, start_joints=None):
    """The whole tail after the press, as ONE chain: descend on the knob,
    close, lift, carry, set down, release, retreat, home.

    Built together so the unguarded stretch in the middle — the lift, the
    carry and the descent onto the drop spot — merges into a single
    execution. Those were three goals with two full stops between them, and
    nothing happens at either stop that the arm has to be still for. The
    contacts keep their own executions: grip:close and the set-down are
    physical events, and nothing merges across one."""
    grip = build_grip_legs(ctx, cfg, start_joints=start_joints)
    place = build_place_legs(
        ctx, cfg, start_joints=grip[-1].goal_joints, chain=grip[-1].chain
    )
    return [*grip, *place]


def run_push(ctx, cfg, runner, args, touch, touch_leg=None, node=None, watcher=None, retries=1):
    """The push stage from the touch's measured contact, then the retreat
    and what follows; the grip phase is planned while the retreat flies.
    Returns (results, touch).

    The touch's height decides (touch_verdict): on the button → push until
    the button's stop is felt; on a popped knob → no push, grip; on
    something else → stop and go home. After the push the wrist looks at
    the knob from the hop (knob_height_mm): not up → one more press from
    staging height, then an honest stop. Exits honestly when the arm could
    not measure its contact."""
    if not args.execute:
        # dry-run: the touch never ran, so there is no contact to push from
        return [], touch
    contact = touch[-1].contact_xyz if touch else None
    if contact is None:
        try_home(
            ctx,
            runner,
            args.execute,
            "the touch tripped but no fingertip TF was readable — cannot "
            "bound the push (are %s in the tree?)" % (FINGERTIP_FRAMES[0],),
        )
        sys.exit(1)
    off = popped_offset_m(contact, ctx)
    verdict = touch_verdict(off)
    runner.note("touch", offset_mm=round(1000 * off, 1), verdict=verdict)
    if verdict == "foreign":
        try_home(
            ctx, runner, args.execute,
            "STOP: the pads met a surface %.0f mm above the button top — not this box's "
            "button (something on the lid, or not the box)" % (1000 * off),
        )
        sys.exit(1)
    push = verdict != "popped"
    if not push:
        print(
            "[press_demo] the button is ALREADY popped — the pads met the knob %.1f mm above "
            "the lid; not pressing it back down, lifting off to grip it" % (1000 * off)
        )
    legs = build_push_legs(ctx, cfg, contact, touch_leg=touch_leg, push=push)
    res = runner.run(
        legs,
        execute=args.execute,
        # the WHOLE tail is planned here, under the retreat: grip, carry
        # and place as one chain, so the lift flows into the carry and the
        # set-down without a stop (build_grip_and_place_legs)
        lookahead=None
        if args.press_only
        else (lambda q: build_grip_and_place_legs(ctx, cfg, start_joints=q)),
    )
    if any(not r.ok for r in res):
        sys.exit(1)
    if push and node is not None and watcher is not None:
        seen = {}
        up_mm = knob_height_mm(node, watcher, ctx, evidence=seen)
        popped = knob_is_up(up_mm)
        runner.note(
            "pop_check", knob_up_mm=None if up_mm is None else round(up_mm, 1), popped=popped, retries_left=retries, **seen
        )
        if ctx.mission_frames is not None:
            ctx.mission_frames.note(popped=None if up_mm is None else bool(popped))
        if up_mm is None:
            print("[press_demo] POP CHECK: no wrist frame to judge the knob by — carrying on")
        elif popped:
            print("[press_demo] POP CONFIRMED — the knob stands %.1f mm above the lid" % up_mm)
        elif retries > 0:
            print(
                "[press_demo] NOT POPPED — the knob reads %.1f mm above the lid after the push; "
                "pressing once more" % up_mm
            )
            return _press_again(ctx, cfg, runner, args, node, watcher, retries - 1)
        else:
            try_home(
                ctx, runner, args.execute,
                "STOP: pressed to the stop twice and the knob still reads %.1f mm above the lid — "
                "the button did not pop (jammed, or not this mechanism)" % up_mm,
            )
            sys.exit(1)
    return res, touch


KNOB_WINDOW_PX = 20  # half-size of the depth window read at the button's pixel
KNOB_FRAMES = 3


def knob_height_mm(node, watcher, ctx, timeout_s=1.5, evidence=None):
    """From the hop: how far the button's top stands above the lid plane,
    by the wrist camera (median depth in a window around the button's
    pixel, over up to KNOB_FRAMES still frames), in mm — or None.

    The frame it judged is kept every run (captures/wrist-pop-*), with the
    window drawn on it; `evidence` (a dict) receives what each frame read,
    the window's pixel and the folder. The first reading this check made
    on the real box — 9.2 mm after a centred push that met the button's
    stop, 2026-09-21 — left nothing to say whether the knob was down or
    the reading was low."""
    import numpy as np

    from rammp_box_opening.perception.depth_source import DEPTH_MAX_M, DEPTH_MIN_M, camera_pose_at

    g = watcher.grab
    button = from_container(ctx.cpose, ctx.model.button_offset)
    lid_z = float(button[2])
    heights, last_stamp, uv = [], None, None
    # Only frames SHOT after the arm came to rest here are judged: one still
    # in the pipe from before the push shows the button as it was. The
    # harness read [-0.0, -0.0, 16.0] mm and called a popped knob "not
    # popped" (2026-09-21).
    at_rest_s = node.get_clock().now().nanoseconds * 1e-9
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s and len(heights) < KNOB_FRAMES:
        _spin_detect(node)
        if g.color_stamp is None or g.depth is None or g.k is None:
            time.sleep(0.02)
            continue
        stamp = (g.color_stamp.sec, g.color_stamp.nanosec)
        if stamp == last_stamp or stamp[0] + 1e-9 * stamp[1] < at_rest_s:
            time.sleep(0.01)
            continue
        last_stamp = stamp
        cam = camera_pose_at(g)
        if cam is None:
            continue
        rot, trans = cam
        k = np.asarray(g.k, float)
        pc = rot.T @ (np.asarray(button, float) - np.asarray(trans, float))
        if pc[2] <= 0.05:
            continue
        u, v = int(round(k[0, 0] * pc[0] / pc[2] + k[0, 2])), int(round(k[1, 1] * pc[1] / pc[2] + k[1, 2]))
        h, w = g.depth.shape
        if not (KNOB_WINDOW_PX <= u < w - KNOB_WINDOW_PX and KNOB_WINDOW_PX <= v < h - KNOB_WINDOW_PX):
            continue
        win = g.depth[v - KNOB_WINDOW_PX : v + KNOB_WINDOW_PX + 1, u - KNOB_WINDOW_PX : u + KNOB_WINDOW_PX + 1]
        valid = win[(win > DEPTH_MIN_M) & (win < DEPTH_MAX_M) & np.isfinite(win)]
        if valid.size < 40:
            continue
        zc = float(np.median(valid))
        p = rot @ np.array([(u - k[0, 2]) / k[0, 0] * zc, (v - k[1, 2]) / k[1, 1] * zc, zc]) + np.asarray(trans, float)
        heights.append(1000 * (float(p[2]) - lid_z))
        uv = (u, v)
    up_mm = None if not heights else float(np.median(heights))
    if uv is not None:

        def draw(img):
            import cv2

            a, b = (uv[0] - KNOB_WINDOW_PX, uv[1] - KNOB_WINDOW_PX), (uv[0] + KNOB_WINDOW_PX, uv[1] + KNOB_WINDOW_PX)
            cv2.rectangle(img, a, b, (0, 255, 0), 2)

        folder = CAPTURES_DIR / ("wrist-pop-%s" % time.strftime("%Y%m%d-%H%M%S"))
        out = _save_wrist_frame(g, folder, "knob %.1f mm above the lid (up from %.0f)" % (up_mm, KNOB_UP_MIN_MM), draw)
        if out is not None:
            _prune_captures("pop")
        if evidence is not None:
            evidence.update(frames_mm=[round(h, 1) for h in heights], uv=uv, capture=None if out is None else str(out))
    return up_mm


def _rise_over_button(ctx, cfg, st, name):
    """A rise back to staging height above the BUTTON, from wherever the arm
    stands in the corridor it came down. Built from the button and nothing
    else: ctx.last_world / last_pose / lid_at belong to whatever was planned
    last, and after a press that is the LOOKAHEAD — the lid's set-down world
    and a pose above the drop spot, 20 cm away (the refused "rise" of bench
    2026-09-21). The name starts with "retreat": a rise out of the corridor
    just descended may fly at transit speed in an interaction world
    (runner._refusal), like the retreat and the lift."""
    m = ctx.model
    button = from_container(ctx.cpose, m.button_offset)
    ctx.lid_at = None  # nothing has been placed: no phantom lid in these worlds
    world = _interaction_world(ctx, button[2], cfg.travel_m, "button")
    ctx.last_world = world
    staging = [button[0], button[1], button[2] + cfg.staging_m]
    rise, st = _plan_motion(ctx, st, name, ("pose", staging, m.press_quat(button)), world, TRANSIT_SPEED)
    return world, rise, st


def build_restage_legs(ctx, cfg):
    """From the hop, fingers open over a knob that did not pop: close them
    and rise back to staging height above the button (_rise_over_button)."""
    st = _state(ctx.client.joints())
    world, rise, _st = _rise_over_button(ctx, cfg, st, "retreat:restage")
    close = _gripper_leg(ctx, st, "press:close", GRIPPER_CMD_CLOSED, world)
    return [close, rise]


def _press_again(ctx, cfg, runner, args, node, watcher, retries):
    """One more press, from staging height, and the same judgement after it."""
    try:
        legs = build_restage_legs(ctx, cfg)
    except RuntimeError as e:
        try_home(ctx, runner, args.execute, "STOP before pressing again — %s" % e)
        sys.exit(1)
    res = runner.run(legs, execute=args.execute)
    if any(not r.ok for r in res):
        sys.exit(1)
    legs = build_press_legs(ctx, cfg)
    res = runner.run(legs, execute=args.execute)
    if any(not r.ok for r in res):
        sys.exit(1)
    touch = [r for r in res if r.leg_name.startswith("press")]
    return run_push(ctx, cfg, runner, args, touch, touch_leg=legs[-1], node=node, watcher=watcher, retries=retries)


def report_press_depth(runner, touch, press, ctx, model):
    """How deep the press actually went, measured by the arm.

    The touch's fingertip TF is the button's true top height in the frame
    the arm is commanded in — no camera, no dims.z, no TCP constant. It is
    printed against what the camera predicted, with the push's depth, and
    logged, so 'it pressed too low' is a number."""
    if not touch or touch[-1].contact_xyz is None:
        return
    tip = [float(v) for v in touch[-1].contact_xyz]
    tip_z = tip[2]
    surface = tip_z - TIP_TO_TOOL_M - TCP_OFFSET_M  # the pad face, base z
    button = from_container(ctx.cpose, model.button_offset)
    predicted = button[2]
    # where the pads LANDED against the aimed centre, in the arm's own
    # radial/tangential directions: the 2026-09-14..16 miss was +6 mm
    # radial from a bowing descent and only this number showed it
    bearing = math.atan2(button[1], button[0])
    dx, dy = tip[0] - button[0], tip[1] - button[1]
    radial = dx * math.cos(bearing) + dy * math.sin(bearing)
    tangential = -dx * math.sin(bearing) + dy * math.cos(bearing)
    pushed = None
    if press:
        end = press[-1].contact_xyz
        pushed = (tip_z - float(end[2])) * 1000 if end is not None else None
    runner.note(
        "press_depth",
        surface_z=round(surface, 4),
        predicted_button_z=round(predicted, 4),
        surface_vs_predicted_mm=round((surface - predicted) * 1000, 1),
        landed_radial_mm=round(radial * 1000, 1),
        landed_tangential_mm=round(tangential * 1000, 1),
        pushed_mm=None if pushed is None else round(pushed, 1),
        push_outcome=press[-1].outcome if press else None,
    )
    print(
        "[press_demo] pads landed %+.1f mm radial / %+.1f mm tangential of the aimed centre"
        % (radial * 1000, tangential * 1000)
    )
    print(
        "[press_demo] button surface at z %.4f by touch — the camera predicted "
        "%.4f (%+.1f mm); push %s"
        % (
            surface,
            predicted,
            (surface - predicted) * 1000,
            "met a stop after %.1f mm" % pushed
            if pushed is not None
            else ("ran its full bound" if press else "did not run"),
        )
    )


def _spin_detect(node):
    try:
        rclpy.spin_once(node, timeout_sec=0.1)
    except RuntimeError as e:
        # rclpy teardown artifact: our SIGINT handler raising inside a
        # subscription take surfaces as RuntimeError, not KeyboardInterrupt
        raise KeyboardInterrupt from e


def wait_for_fix(node, watcher, cfg, timeout_s=None):
    """Spin (the watcher ticks on its timer) until a fresh stable fix.

    The window is never purged here: the watcher keeps its in-flight
    samples — geometry is lifted with frame-stamp TF, the still-camera
    filter drops moving frames, the freshness window (1 s) means only
    the scan's parked tail can support a commit anyway, and the
    3-agreeing gate still stands — so a fix is often ready the moment
    the arm parks instead of half a second later (owner: detect during
    the flip, 2026-09-01)."""
    limit = cfg.timeout_s if timeout_s is None else timeout_s
    t0 = time.monotonic()
    watcher.active = True  # the detector only works inside a detect window
    try:
        while time.monotonic() - t0 < limit:
            _spin_detect(node)
            watcher.tick_now()  # not at the timer's mercy
            got = watcher.fix()
            if got is not None:
                return got
        return None
    finally:
        watcher.active = False


# cuRobo takes 17-21 s to load after sheppy starts the planner (measured
# 2026-09-16 from the planner's own log), and it advertises nothing until it
# has. A run started inside that window is waited for, not failed.
PLANNER_WAIT_S = 30.0


def readiness_refusal(client, execute, planner_wait_s=PLANNER_WAIT_S):
    """Why this run must not start, or None. Checked before ANYTHING is
    actuated — the gripper close included.

    Field 2026-09-16: the stack had just been restarted, the fingers closed,
    and only then did the run die on `set_world service unavailable`. The
    close was the first actuation and the planner was still loading."""
    if not client.planner_reachable(timeout_s=1.0):
        print(
            "[press_demo] waiting up to %.0f s for the planner to finish "
            "loading" % planner_wait_s
        )
        if not client.planner_reachable(timeout_s=planner_wait_s):
            return (
                "the planner is not up after %.0f s — is sheppy's `planner` node "
                "running? (docker logs sheppy-planner should end in "
                "'rammp_curobo ready'). Nothing moved." % planner_wait_s
            )
    if execute and not client.driver_reachable(timeout_s=5.0):
        return (
            "the arm driver is not answering — is sheppy's `arm` node up? "
            "(its log is in ~/.sheppy/logs/arm/). Nothing moved."
        )
    return None


def try_home(ctx, runner, execute, why):
    """Recovery home that cannot crash the exit path: a refused home plan
    leaves the arm holding with an honest line (2026-08-25 review). Goes
    to the mission's rest pose (ctx.press_cfg, when main set it).

    The home is planned in the blind bench world, whose whole placement
    band is blocked to container height — so a run that stops OVER the box
    (at the hop, at a touch) starts inside the band and is refused by
    construction: "home plan refused (... the start state collides with the
    world ...) — arm holds" (bench 2026-09-21). When the mission has located
    this box, it first rises out of the corridor it came down, back to
    staging height above the button (_rise_over_button: planned against the
    button's interaction world), and goes home from there."""
    print("[press_demo] %s — returning home" % why)
    cfg = getattr(ctx, "press_cfg", None)
    rest = None if cfg is None else rest_joints(cfg)
    try:
        legs = [build_home_leg(ctx, ctx.client.joints(), rest)]
    except RuntimeError as e:
        if cfg is None or getattr(ctx, "cpose", None) is None:
            print("[press_demo] home plan refused (%s) — arm holds" % e)
            return
        print("[press_demo] home refused from here (%s) — rising over the box first" % e)
        try:
            _world, rise, st = _rise_over_button(ctx, cfg, _state(ctx.client.joints()), "retreat:rise")
            home, _ = _plan_motion(ctx, st, "home", ("joints", list(HOME if rest is None else rest)), _full_world(ctx), TRANSIT_SPEED)
        except RuntimeError as e2:
            print("[press_demo] home plan refused (%s) — arm holds" % e2)
            return
        legs = [rise, home]
    runner.run(legs, execute=execute)


def detect_only_report(node, watcher, ctx, cfg, runner, execute):
    """Mount-calibration observation: hold at the look pose reporting every
    fresh fix (top-face centre, yaw, footprint), then park home.

    Place the container at a TAPE-MEASURED spot first; the printed base
    pose vs truth solves the wrist-mount error."""
    watcher.reset()
    watcher.active = True
    print("[press_demo] DETECT-ONLY: reporting fixes for 15 s")
    t0 = time.monotonic()
    last = None
    try:
        while time.monotonic() - t0 < 15.0:
            _spin_detect(node)
            got = watcher.fix()
            if got is None:
                continue
            pos, yaw = got
            key = tuple(round(float(v), 4) for v in pos)
            if key == last:
                continue
            last = key
            f = watcher.last_debug
            print(
                "fix: top [%.3f, %.3f, %.3f] yaw %.1f | footprint %.3fx%.3f m | %d px"
                % (
                    pos[0],
                    pos[1],
                    pos[2],
                    math.degrees(float(yaw)),
                    f.footprint[0],
                    f.footprint[1],
                    f.n_px,
                )
            )
    finally:
        watcher.active = False
    print("[press_demo] detect-only done (%s) — homing" % watcher.status())
    runner.run(
        [build_home_leg(ctx, ctx.client.joints())], execute=execute
    )


# This close (every joint, wrap-aware) to the rest pose is AT it; anywhere
# else, the run's first motion goes there (home_first).
AT_REST_RAD = 0.01


def home_first(ctx, runner, execute, rest):
    """The mission's first motion: the arm to its exact rest pose (`rest`,
    HOME) unless it already stands there, so every run starts from the same
    place (owner, 2026-09-25). Planned like home_arm's move: in the bench
    world, whose keep-out band keeps it above wherever a box may stand.
    Returns False — nothing moved — when that is refused, with the
    recovery named: an arm that far down (holding at the table after an
    abort) is home_arm's case, whose bare-world and lift-first fallbacks
    want the operator's eye on the bench, not an automatic start.

    It used to REFUSE any start more than 1.2 rad from a rest pose and fly
    from wherever it stood inside that; planning from a failure pose had
    swung the arm half upside down once (2026-09-01) — every joint goal
    now goes the short way round (runtime/branches), and the bench world
    and the trajectory sanity gate still stand between it and that."""
    live = ctx.client.joints()
    off = rest_distance(live, rest)
    if off <= AT_REST_RAD:
        return True
    print(
        "[press_demo] HOME FIRST: the arm is %.2f rad from its rest pose — going there before anything else" % off
    )
    world = ctx.worlds.push_name("bench", model=ctx.model)
    try:
        leg, _ = _plan_motion(ctx, _state(live), "home", ("joints", list(rest)), world, TRANSIT_SPEED)
    except RuntimeError as e:
        print(
            "[press_demo] not starting: the move home was refused in the bench world (%s) — the arm is "
            "probably down at the table. Clear the bench and recover by hand:\n"
            "    ros2 run rammp_box_opening home_arm --execute    (add --lift-first 0.08 if that is refused too)" % e
        )
        return False
    res = runner.run([leg], execute=execute)
    return all(r.ok for r in res)


def keep_mission_frames(frames, exc):
    """Write this run's detection record (ctx.mission_frames), with how the
    run ended (`exc`: the SystemExit in flight, or None), and say where it
    went. Never changes how the run ends: a failure here, or a Ctrl+C
    during the write, is a line in the log (a placement left without its
    truth.json is not part of the set: detection_set.write_manifest)."""
    if frames is None:
        return
    code = exc.code if isinstance(exc, SystemExit) else (0 if exc is None else type(exc).__name__)
    try:
        t0 = time.monotonic()
        out = frames.close(exit_code=code)
    except (Exception, KeyboardInterrupt) as e:  # a full disk, a second Ctrl+C
        print("[press_demo] (this run's frames were not kept: %s)" % (str(e) or type(e).__name__))
        return
    if out is not None:
        print(
            "[press_demo] frames kept for detection work: %s (%s; %s; %.1f s)"
            % (out, ", ".join(frames.doc["frames"]), frames.doc["truth_note"], time.monotonic() - t0)
        )


def main():
    t_main = time.monotonic()  # "it takes too long to detect the box": timed
    # If the process ever dies on a fatal signal, say which thread was doing
    # what: a run ended "terminate called without an active exception" after
    # its honest STOP line (bench 2026-09-21 14:16) and left nothing to go on.
    import faulthandler

    faulthandler.enable()
    ap = cli_common.make_parser(__doc__)
    ap.add_argument(
        "--press-only",
        action="store_true",
        help="stop after the press (the step-1 demo): press, retreat, home",
    )
    ap.add_argument(
        "--detect-only",
        action="store_true",
        help="scan, report fixes + camera-frame diagnostics for 15 s, home, "
        "exit — the wrist-mount calibration observation (no press)",
    )
    ap.add_argument(
        "--no-scene",
        action="store_true",
        help="ignore the scene camera even when it is calibrated: search with "
        "the wrist camera as before (the stub harness, and A/B runs)",
    )
    ap.add_argument(
        "--scene-calib",
        default=None,
        help="scene camera calibration yaml (default: the installed "
        "config/camera_scene.yaml; the stub harness passes its own)",
    )
    args = ap.parse_args()
    cfg_path = args.container or cli_common.default_container_yaml()
    bench = args.bench_world or cli_common.default_bench_yaml()
    model = ContainerModel.load(cfg_path)
    cfg = load_press_demo(cfg_path)
    cli_common.refuse_unmeasured(model, args.execute, measuring=args.detect_only)
    node, client = cli_common.init_runtime(args.execute)
    worlds = WorldStore(bench)
    runner = Runner(client)
    cli_common.apply_speed_scale(runner, args)
    # The persistent owl_detector node owns the OWL model; there is no
    # in-process copy (a second OWLv2 beside cuRobo on one GPU, field
    # 2026-09-01). The rung also owns the node's ENABLE gate: inference
    # runs only inside the mission's detect windows, because at 100 % GPU
    # duty it doubled every cuRobo solve (measured 2026-09-02).
    # PERCEPTION ON ITS OWN THREAD. The cameras, the detector's tick and the
    # OWL rungs live on a second node spun by a background executor, so the
    # control loop's spin cadence (a 1 kHz joint-state stream, action
    # feedback, planning calls) never starves them: on the stub bench the
    # single executor gave the detector 1.6 frames a second, and the wrist
    # could not confirm a box in a one-second reach (2026-09-16).
    pnode = rclpy.create_node("rammp_box_opening_perception")
    from rclpy.executors import SingleThreadedExecutor

    pexec = SingleThreadedExecutor()
    pexec.add_node(pnode)
    import threading

    pthread = threading.Thread(target=pexec.spin, name="perception", daemon=True)
    pthread.start()
    # the owl_detector nodes were started by the launch with no container
    # of their own: tell them which one this run is for, so their prompts
    # and floor are this container's
    from rammp_box_opening.perception.owl_source import announce_container

    announce_container(pnode, cfg_path)

    impls = {}
    owl = None
    watcher_holder = {}
    if cfg.detect_source == "vlm":
        from rammp_box_opening.perception.vlm_source import fetch_box_roi

        impls["claude"] = fetch_box_roi  # bounded at call time, below
        if "owl" in cfg.vlm_backends:
            from rammp_box_opening.perception.owl_source import OwlRung

            # listener starts NOW, not at detect time: the node sees the
            # box as the arm settles and its bbox gates the depth watcher,
            # so the fix is usually ready within a few still frames
            owl = OwlRung(pnode, cfg, watcher_holder)
            impls["owl"] = owl
    # the box found by geometry: the lid plateau above the surveyed table
    watcher = BoxTopWatcher(pnode, cfg, model, worlds.table_top_z)
    watcher_holder["watcher"] = watcher
    # The scene camera, when it has been calibrated to the arm
    # (scripts/calibrate_scene_camera.py): it finds the box before the arm
    # moves, and the search below becomes the fallback.
    locator = None
    if not args.no_scene and not args.detect_only:
        from rammp_box_opening.perception.scene import load_scene_calibration

        T_base_scene_link = load_scene_calibration(args.scene_calib)
        if T_base_scene_link is not None:
            from rammp_box_opening.perception.scene_source import SceneLocator

            from rammp_box_opening.perception.scene import scene_calib_path
            from rammp_box_opening.perception.scene_calib import scene_tag_px

            locator = SceneLocator(
                pnode, cfg, model, worlds.table_top_z, T_base_scene_link, spins=False,
                tag_px=scene_tag_px(args.scene_calib or scene_calib_path()),
            )
            # enable its OWL NOW: the first bbox takes ~1.2 s (tick + inference)
            # and the readiness checks below are that long — by the time
            # locate() asks, the box is already on the topic
            locator.owl.enable()
    # camera on from here to exit
    ctx = Ctx(
        model=model,
        cpose=None,
        client=client,
        worlds=worlds,
        config_path=cfg_path,
    )

    ctx.press_cfg = cfg  # recovery homes go to the mission's rest pose
    if args.execute and not args.detect_only:
        # every run keeps what its cameras saw and, when the press lands,
        # where the button really was: a detection set that grows with the
        # bench's own runs (detection_set.MissionFrames; written at the end)
        ctx.mission_frames = MissionFrames(state_dir() / "detection_sets" / MISSIONS_SET, Path(cfg_path).name, worlds.table_top_z)
    try:
        why = readiness_refusal(client, args.execute)
        if why:
            sys.exit("[press_demo] not starting: %s" % why)
        # THE FIRST MOTION: the arm to its exact rest pose, when it is not
        # already there (home_first) — every run starts from the same place
        if not home_first(ctx, runner, args.execute, rest_joints(cfg)):
            sys.exit(3)
        live = client.joints()
        # The wrist's OWL is enabled where the wrist starts looking (the
        # reach, or the search), NOT here: the scene camera's instance runs
        # first, and two instances inferring at once on this GPU is the
        # contention that left the scene one silent (field 2026-09-16).
        # The fingers shut NOW and ride the look (they sit in the
        # bottom rows of the wrist camera's frame, the box in its middle —
        # no occlusion, capture 20260901-130610). The join lands before the
        # guarded touch, which needs them closed. No closed fingers = no
        # press: an open aperture strikes the lid and can still read
        # "pressed", so a refused close ends the run here.
        if not args.detect_only and not runner.start_gripper(
            "press:close", GRIPPER_CMD_CLOSED, args.execute
        ):
            try_home(
                ctx,
                runner,
                args.execute,
                "press:close refused or failed — the fingers are not closed",
            )
            sys.exit(1)
        t_detect = time.monotonic()
        if args.detect_only:
            # the calibration observation parks at the look pose and stays
            # there: a sighting must not cut its motion short
            name, target, speed = search_targets(cfg)[0]
            if rest_distance(live, target) > REST_TOL_RAD:
                res = runner.run(
                    [build_search_leg(ctx, (name, target, speed), live, _NeverStops)],
                    execute=args.execute,
                )
                if any(not r.ok for r in res):
                    sys.exit(1)
            detect_only_report(node, watcher, ctx, cfg, runner, args.execute)
            sys.exit(0)

        # THE SCENE CAMERA FIRST. It sees the whole workspace from where it
        # stands, so the box is found before the arm has moved at all, and
        # the arm flies straight to staging above it. The wrist camera then
        # confirms from 12 cm, where it is good, and the descent that was
        # pre-planned during the approach flies at once when it agrees.
        got, preplanned, scene_fix, at_staging, reach = None, None, None, False, SceneApproach()
        scene_seen = None  # the scene camera's box, flown to or not (scene_trust)
        planned_for = None  # the pose `preplanned` was planned for
        if locator is not None:
            t_scene = time.monotonic()
            scene_fix = locator.locate(SCENE_LOCATE_S)
            if ctx.mission_frames is not None:
                ctx.mission_frames.scene(
                    scene_frame(locator.grab, locator.T_base_link),
                    fix_xyz=None if scene_fix is None else scene_fix.top_xyz,
                    score=None if scene_fix is None else scene_fix.score,
                    why=locator.last_why if scene_fix is None else None,
                )
            if locator.moved_note and scene_fix is not None:
                print("[press_demo] scene camera: %s" % locator.moved_note)  # e.g. the tag was not in view
            scene_seen = scene_fix  # recorded against the wrist's button, flown or not
            if scene_fix is not None:
                trusted, distrust = scene_trust(
                    residuals_path_for(args.scene_calib), args.scene_calib or scene_calib_path()
                )
                if not trusted:
                    print(
                        "[press_demo] scene camera: box at [%.3f, %.3f], NOT FLOWN — %s; searching with the wrist "
                        "(this run's pair judges it again)" % (scene_fix.pose.xyz[0], scene_fix.pose.xyz[1], distrust)
                    )
                    scene_fix = None
            elif locator.last_why:
                print("[press_demo] scene camera: %s — searching with the wrist" % locator.last_why)
            if scene_fix is not None:
                sp = scene_fix.pose
                print(
                    "[press_demo] SCENE: box at [%.3f, %.3f] yaw %.1f deg from %d lid points "
                    "(OWL %.2f; the scene reads the table at %.3f, surveyed %.3f; %.1f s to locate "
                    "[%s], %.1f s since start) — flying to staging"
                    % (sp.xyz[0], sp.xyz[1], math.degrees(sp.yaw), scene_fix.n_points,
                       scene_fix.score, scene_fix.table_z_scene, watcher.table_z,
                       time.monotonic() - t_scene,
                       " ".join("%s %.2f" % (k[:-2], v) for k, v in locator.last_timing.items()),
                       time.monotonic() - t_main)
                )
                runner.note(
                    "scene_fix",
                    origin_xyz=[round(float(v), 4) for v in sp.xyz],
                    yaw_deg=round(math.degrees(sp.yaw), 2),
                    top_xyz=[round(float(v), 4) for v in scene_fix.top_xyz],
                    n_points=scene_fix.n_points,
                    owl_score=round(scene_fix.score, 3),
                    table_z_scene=None if scene_fix.table_z_scene != scene_fix.table_z_scene
                    else round(scene_fix.table_z_scene, 4),
                )
                reach = approach_from_scene(
                    node, ctx, cfg, runner, watcher, args.execute, scene_fix
                )
                got, preplanned, planned_for = reach.got, reach.preplanned, scene_fix.pose
                at_staging = got is not None
                open_already = refuse_open_box(reach.button_up_mm)
                if open_already is not None:
                    runner.note("open_box", button_above_lid_mm=round(reach.button_up_mm, 1))
                    try_home(ctx, runner, args.execute, "STOP: " + open_already)
                    sys.exit(1)
                if at_staging and not reach.in_flight:
                    # a move over the button leaves the pre-planned descent
                    # behind the arm: it is re-fitted from where it stands
                    got, _moves, _off = centre_over_button(node, ctx, cfg, runner, watcher, args.execute, got)
                if got is None:
                    print(
                        "[press_demo] the wrist did not find the button at staging (%s) — "
                        "searching from here" % (reach.status or watcher.status())
                    )
                    dump_wrist_frame(watcher, "staging")
                    ctx.cpose = None
        if got is None:
            print("[press_demo] LOOK: wrist down over the table, sweeping if empty")
            if owl is not None:
                owl.enable()  # a fresh window: the first one may have expired
            # DEPTH FIRST, INSTANTLY, AND WHILE MOVING. One box-sized plateau
            # on the bench is unambiguous geometry: the coarse path reads it
            # during the look and the sweeps and stops the arm where it saw
            # it, and the precise fix the press is aimed with is taken
            # standing there. Blocking on a semantic model first cost 13 s in
            # the field (2026-09-01), so the ladder runs ONLY when the whole
            # search could not answer — semantics on demand.
            # a sighting is enough to go and look close up (every search fix
            # is aimed at staging): only with --execute, since a dry run
            # has no close-up aim to follow
            got, failed = search_for_box(
                node, ctx, cfg, runner, watcher, args.execute,
                accept_coarse=args.execute and not args.detect_only,
            )
            if failed is not None:
                sys.exit(1)
        seen = watcher.last_coarse() if (got is None and args.execute and not args.detect_only) else None
        if got is None and seen is None and cfg.detect_source == "vlm" and watcher.grab.color is not None:
            # (no colour frame at all = the wrist camera is down: there is
            # nothing to show a model, and status() below says which streams
            # are missing. This used to WAIT for a frame, forever, on a node
            # the camera is not even subscribed on — a D405 that dropped off
            # USB hung the mission here instead of ending it.)
            # The ladder gets its OWN budget, not what the search left: the
            # search spends the detect window MOVING, and measuring the
            # cloud rung against that clock left it with nothing to call
            # with every time (live dry run 2026-09-15). Still bounded —
            # on the no-internet target an unbounded call stalled far past
            # any budget (review 2026-09-02).
            t_ladder = time.monotonic()
            bound = dict(impls)
            if "claude" in bound:
                # budget read when the rung is CALLED (the owl rung may have
                # waited 2 s first), never frozen early
                bound["claude"] = lambda img, c: fetch_box_roi(
                    img,
                    c,
                    budget_s=cfg.timeout_s - (time.monotonic() - t_ladder),
                )
            roi, lines = resolve_roi(watcher.grab.color, cfg, impls=bound)
            for ln in lines:
                print("[press_demo] VLM %s" % ln)
            watcher.roi = roi  # None = ungated, honest refusals stand
            # a bbox only helps a plateau the geometry found ambiguous, and
            # a commit needs min_hits agreeing frames — a beat, not another
            # full detect window
            got = wait_for_fix(node, watcher, cfg, timeout_s=CONFIRM_S)
        if owl is not None:
            owl.disable()  # detect window closed: give the GPU back
        coarse_search = False
        if got is None and args.execute and not args.detect_only:
            seen = watcher.last_coarse()
            if seen is not None:
                print(
                    "[press_demo] the search saw the box but not well enough to aim from there (%s) — "
                    "going over it to look close up" % watcher.status()
                )
                got, coarse_search = seen, True
        if got is None:
            # benign detect timeout: park home, exit 2
            dump_wrist_frame(watcher, "nobox")
            try_home(ctx, runner, args.execute, "NO BOX — %s" % watcher.status())
            sys.exit(2)

        if not at_staging and args.execute and not args.detect_only:
            got, at_staging, preplanned, planned_for = stage_over_search_fix(
                node, ctx, cfg, runner, watcher, args.execute, got, coarse=coarse_search
            )
        pos, _yaw = got
        ctx.cpose = watcher.to_container_pose(got)
        if ctx.mission_frames is not None:
            source = aim_source(watcher, pos)
            ctx.mission_frames.located(pos, source, getattr(watcher, "last_aim_capture", None) if source == "aim" else None)
        print(
            "[press_demo] BOX at [%.3f, %.3f, %.3f] (%s) -> container origin "
            "[%.3f, %.3f, %.3f] yaw %.1f deg"
            % (
                pos[0],
                pos[1],
                pos[2],
                watcher.status(),
                ctx.cpose.xyz[0],
                ctx.cpose.xyz[1],
                ctx.cpose.xyz[2],
                math.degrees(ctx.cpose.yaw),
            )
        )
        if scene_seen is not None and at_staging and aim_source(watcher, pos) == "aim":
            # the pair: where the scene camera put the box, against the
            # button the close-up aim found — whether the scene fix was
            # flown or not (scene_trust), and it corrects the calibration
            # by itself when enough of them agree (auto_refine)
            dx, dy, dz, dyaw = scene_residual(scene_seen.top_xyz, pos, scene_seen.pose.yaw, ctx.cpose.yaw)
            print(
                "[press_demo] scene camera was off by [%+.0f, %+.0f] mm, %+.1f deg here "
                "(recorded for the calibration refinement)" % (dx, dy, dyaw)
            )
            runner.note("scene_residual", dx_mm=round(dx, 1), dy_mm=round(dy, 1), dz_mm=round(dz, 1), dyaw_deg=round(dyaw, 2))
            residuals = residuals_path_for(args.scene_calib)
            record_residual(scene_seen, pos, ctx.cpose.yaw, path=residuals)
            refined = auto_refine(residuals, args.scene_calib or scene_calib_path())
            if refined:
                print(refined)
            else:
                residual_hint(residuals, args.scene_calib or scene_calib_path())
        runner.note(
            "fix",
            top_xyz=[round(float(v), 4) for v in pos],
            origin_xyz=[round(float(v), 4) for v in ctx.cpose.xyz],
            yaw_deg=round(math.degrees(ctx.cpose.yaw), 2),
            top_residual_mm=round((pos[2] - (watcher.table_z + model.dims[2])) * 1000, 1),
            roi=list(watcher.roi) if watcher.roi is not None else None,
            status=reach.status if at_staging else watcher.status(),
            detect_s=round(time.monotonic() - t_detect, 2),
        )

        if not args.press_only:
            # the box lands wherever it lands: resolve the drop spot NOW,
            # before any container-directed motion — configured lid_place
            # when clear, slid away from the box when crowded, refusal
            # only when nothing in the set-down zone clears
            lid = load_lid_place(ctx.config_path)
            drop, _side = box_relative_lid_drop(
                model, ctx.cpose, watcher.table_z, lid.xyz
            )
            if drop is None:
                try_home(
                    ctx,
                    runner,
                    args.execute,
                    "no lid drop spot beside the box at [%.2f, %.2f] falls "
                    "inside the set-down zone (need %.0f mm clear, x %s y %s) "
                    "— move the box"
                    % (
                        ctx.cpose.xyz[0],
                        ctx.cpose.xyz[1],
                        lid_place_min_clear(model) * 1000,
                        list(DROP_X_M),
                        list(DROP_Y_M),
                    ),
                )
                sys.exit(4)
            print(
                "[press_demo] lid goes to [%.2f, %.2f, %.3f] — %.0f mm from "
                "the box, on the side the config prefers"
                % (
                    drop[0],
                    drop[1],
                    drop[2],
                    math.hypot(
                        drop[0] - ctx.cpose.xyz[0], drop[1] - ctx.cpose.xyz[1]
                    )
                    * 1000,
                )
            )
            ctx.lid_drop = ContainerPose(xyz=tuple(drop), yaw=lid.yaw)

        print(
            "[press_demo] PRESS target origin [%.3f, %.3f, %.3f] yaw %.1f deg"
            % (ctx.cpose.xyz[0], ctx.cpose.xyz[1], ctx.cpose.xyz[2], math.degrees(ctx.cpose.yaw))
        )
        if at_staging and reach.pressed is not None:
            # the reach carried its descent and touched: nothing to fly here
            runner.note("press_plan", in_flight=True)
            print("[press_demo] reached and touched in one motion")
            touched, touch_leg = reach.pressed, reach.press_leg
        else:
            if at_staging:
                # the arm already stands at staging above the box: the descent
                # alone — the one pre-planned during the approach when the aim
                # agrees with the scene's fix, re-FITTED to the aim when it
                # moved the target a little, re-planned only otherwise
                press_legs, how = descent_from_staging(ctx, cfg, planned_for, preplanned)
                runner.note("press_plan", from_staging=True, preplanned=press_legs is preplanned, how=how)
                print("[press_demo] PRESS from staging — %s" % how)
            else:
                # The approach and the descent are built as ONE chain and run
                # as ONE execution: the descent is planned from where the
                # approach ends, so the Runner re-times them into a single
                # profile that flows through the junction (legs.can_merge).
                # What this gives up is a close-range re-fix at a staging
                # stop (owner: no recalibrating mid flight). Each half is
                # planned in its OWN world, so the lateral run happens where
                # the container is visible.
                runner.note("press_plan", chained=True)
                approach = build_approach_leg(ctx, cfg)
                press_legs = [
                    approach,
                    *build_press_legs(ctx, cfg, start_joints=approach.goal_joints),
                ]
            touch_leg = press_legs[-1]
            # what follows the touch (the push's xy, the retreat) is relative
            # to the pose this stroke is ACTUALLY aimed at — a re-fitted or
            # pre-planned one was aimed before the last pose was commanded
            ctx.last_pose = (list(touch_leg.target[1]), list(touch_leg.target[2]))
            touched = runner.run(press_legs, execute=args.execute)
            if any(not r.ok for r in touched):
                sys.exit(1)
        touch = [r for r in touched if r.leg_name.startswith("press")]
        res, touch = run_push(
            ctx, cfg, runner, args, touch, touch_leg=touch_leg, node=node, watcher=watcher
        )
        press = [r for r in res if r.leg_name.startswith("press:push")]
        if ctx.mission_frames is not None:
            ctx.mission_frames.note(stage="pressed")
        print(
            "[press_demo] PRESSED — %s"
            % (press[-1].detail if press else "no push ran (dry-run)")
        )
        report_press_depth(runner, touch, press, ctx, model)
        if args.press_only:
            if args.execute:
                # the pop is confirmed (or pressed again) at the hop; home from there
                res = runner.run(build_home_from_hop(ctx, cfg), execute=args.execute)
                if any(not r.ok for r in res):
                    sys.exit(1)
            runner.finish()
            sys.exit(0)

        # No pre-grip re-look: at the hop the lid does not fit in the depth
        # frame (its far edge projects past the last row), so the border
        # gate refused every attempt — 1.2 s per run for nothing (audit
        # 2026-09-02). A press that scoots the box is caught by the guarded
        # grip:down (an obstruction trip) and the band verify (closed on
        # air), both honest failures.
        print(
            "[press_demo] GRIP AND PLACE: onto the knob, close, then lift, "
            "carry and set down as one motion"
        )
        try:
            legs = runner.lookahead_result or build_grip_and_place_legs(ctx, cfg)
        except RuntimeError as e:
            # nothing of the tail has moved yet: the button is popped and
            # the fingers are open above the box — go home, never a
            # traceback with the arm left hanging (field 2026-09-17)
            try_home(ctx, runner, args.execute, "STOP before the grip — %s" % e)
            sys.exit(1)
        res = runner.run(legs, execute=args.execute)
        grip = [r for r in res if r.leg_name == "grip:close"]
        if grip and grip[-1].ok:
            print("[press_demo] LID PULLED — %s" % grip[-1].detail)
        elif grip:
            print(
                "[press_demo] GRIP FAILED — %s: the fingers closed on air, so the "
                "button did not pop or the grip missed the knob" % grip[-1].detail
            )
        if any(not r.ok for r in res):
            sys.exit(1)  # grip or set-down failed — arm holds
        runner.finish()
        if ctx.mission_frames is not None:
            ctx.mission_frames.note(stage="done")
        print("[press_demo] DONE — box open, lid placed, arm home")
    except KeyboardInterrupt:
        sys.exit(130)  # the abort path already reported what was confirmed
    finally:
        ending = sys.exc_info()[1]  # how the run ended, for its record
        # Stop the perception thread BEFORE the interpreter tears rclpy
        # down: an executor still inside spin() on another thread at exit
        # ends in "terminate called without an active exception" and an
        # abort that eats the exit code (field 2026-09-16).
        pexec.shutdown()
        pthread.join(timeout=3.0)
        try:
            pexec.remove_node(pnode)
            pnode.destroy_node()
        except Exception:
            pass
        # the run is over and the arm stopped: now its frames are written
        keep_mission_frames(ctx.mission_frames, ending)


if __name__ == "__main__":
    main()
