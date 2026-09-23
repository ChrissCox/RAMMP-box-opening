"""Refining the scene camera's calibration from what the wrist measured.

Every staging run records one pair: where the scene camera put the button
top and where the wrist's aim found it, both in base_link (press_demo
record_residual). The pairs are the calibration's own residuals, and a
rigid correction fitted to them is the refinement — no marker, no motion.

With few pairs, or pairs all at one spot, only a translation is
identifiable; a rotation needs pairs spread over the table (>= 4, spanning
>= 0.15 m). The fit says which it did.

A pair measures the calibration it was recorded under, and no other: once
a correction is written, the pairs it was fitted from describe a camera
that no longer exists (calibrated_at, load_pairs' `since`).
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

MIN_PAIRS_ROTATION = 4
MIN_SPREAD_M = 0.15


@dataclass(frozen=True)
class Refinement:
    D: np.ndarray  # 4x4 correction, base -> base: p_wrist ~= D @ p_scene
    mode: str  # "translation" | "rigid"
    n: int
    rms_before_mm: float
    rms_after_mm: float
    spread_m: float


def calibrated_at(calib_path):
    """When the calibration in `calib_path` took effect (unix time): the
    `refined_at` the refinement script writes, else the file's own time —
    or None when there is no such file."""
    p = Path(calib_path)
    if not p.exists():
        return None
    try:
        with open(p) as f:
            at = (yaml.safe_load(f) or {}).get("refined_at")
    except (OSError, yaml.YAMLError):
        at = None
    return float(at) if at is not None else p.stat().st_mtime


def load_pairs(path, synthetic=None, since=None):
    """(scene_xyz, wrist_xyz) pairs from a residuals.jsonl; `synthetic`
    is a wrist point to drop (the e2e harness's stub box, before the
    harness got its own file). With `since` (calibrated_at), only the pairs
    recorded under the calibration in force: two pairs from before the
    2026-09-17 correction sat in the file for days, still reporting the
    24 mm it had fixed."""
    scene, wrist = [], []
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if since is not None and float(r.get("t", 0.0)) < since:
                continue
            w = [float(v) for v in r["wrist_top"]]
            if synthetic is not None and np.allclose(w, synthetic, atol=1e-6):
                continue
            scene.append([float(v) for v in r["scene_top"]])
            wrist.append(w)
    return np.asarray(scene, float).reshape(-1, 3), np.asarray(wrist, float).reshape(-1, 3)


def spread_m(pts):
    if len(pts) < 2:
        return 0.0
    d = pts[:, None, :2] - pts[None, :, :2]
    return float(np.sqrt((d ** 2).sum(-1)).max())


def fit(scene, wrist):
    """The rigid correction D with wrist ~= D @ scene (Kabsch when the
    pairs allow a rotation, a translation otherwise)."""
    scene = np.asarray(scene, float); wrist = np.asarray(wrist, float)
    n = len(scene)
    if n == 0:
        raise ValueError("no pairs")
    before = float(np.sqrt(np.mean(np.sum((scene - wrist) ** 2, axis=1))))
    sp = spread_m(scene)
    D = np.eye(4)
    if n >= MIN_PAIRS_ROTATION and sp >= MIN_SPREAD_M:
        cs, cw = scene.mean(0), wrist.mean(0)
        H = (scene - cs).T @ (wrist - cw)
        U, _S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1] *= -1
            R = Vt.T @ U.T
        D[:3, :3] = R
        D[:3, 3] = cw - R @ cs
        mode = "rigid"
    else:
        D[:3, 3] = (wrist - scene).mean(0)
        mode = "translation"
    fitted = scene @ D[:3, :3].T + D[:3, 3]
    after = float(np.sqrt(np.mean(np.sum((fitted - wrist) ** 2, axis=1))))
    return Refinement(D=D, mode=mode, n=n, rms_before_mm=1000 * before, rms_after_mm=1000 * after, spread_m=sp)


def apply(T_base_link, refinement):
    """The refined base_link -> scene_camera_link transform."""
    return refinement.D @ np.asarray(T_base_link, float)
