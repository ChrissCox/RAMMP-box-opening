"""Collision-world generation: bench + container-derived cuboid variants.

Worlds are a plan-time concern (spec §6). SetWorld is write-only, so the
client tracks the one it last pushed (PlannerClient.set_world); this module
only builds and writes the variants. Cuboids only — the planner drops
other shapes.
"""

import hashlib
from pathlib import Path

import yaml

from rammp_box_opening.constants import state_dir

ERR_TALL_M = 0.02  # container cuboid extra height (err tall, spec §3)
# Reduction plane below the deepest command. EMPIRICAL 2026-08-25 (margin
# probe vs the real planner): the finger collision spheres reach ~4 cm
# below tool_frame and padding adds 2 cm — at 0.03 the hover AND press
# goals were in collision (IK_FAIL); 0.08 plans.
PLANE_MARGIN_M = 0.08
# The planner pads every obstacle by this much per side, except the names on
# its no_pad list — which RAMMP-CuRobo v1.0.0 keeps to the pedestal alone
# (core/rammp_curobo/config.py PLANNER_DEFAULTS). The planner this repo first
# ran against exempted the table too. Padded, a table written at its TRUE
# height sits 2 cm too tall and swallows the arm's own base collision
# spheres: every start state reads as in collision. So a world FILE carries
# these true surfaces lowered by the padding the planner adds back; the
# builders below, and table_top_z, keep the true geometry.
PLANNER_PADDING_M = 0.02
PADDED_TRUE_SURFACES = ("table",)


def _as_planned(world):
    """`world` as the planner must be given it: every TRUE surface lowered by
    the padding the planner adds back. Applied last, after any capping, so a
    table capped at the reduction plane still ends at the plane once padded."""
    obstacles = []
    for o in world["obstacles"]:
        if o.get("name") in PADDED_TRUE_SURFACES:
            o = dict(o)
            x, y, z = o["position"]
            o["position"] = [x, y, z - PLANNER_PADDING_M]
        obstacles.append(o)
    return dict(world, obstacles=obstacles)


def _bench_obstacles(bench):
    return [dict(o) for o in bench["obstacles"]]


# the arm's own mounting column is never removed from a world, only ever
# thinned: a set-down low enough to push the reduction plane under the
# pedestal's base would otherwise delete the one obstacle the arm is
# bolted to (found by review 2026-08-28, never yet hit in the field)
NEVER_DROP = ("pedestal",)
STUB_DZ_M = 0.01


def _cap_top(obs, top_z):
    """Copy of a cuboid with its top lowered to top_z; None if nothing of
    it remains. No-op when the top is already at or below top_z.

    Obstacles named in NEVER_DROP are thinned to a STUB_DZ_M slab at their
    base instead of being dropped."""
    x, y, z = obs["position"]
    dx, dy, dz = obs["dims"]
    top, bottom = z + dz / 2, z - dz / 2
    if top <= top_z:
        return dict(obs)
    if bottom >= top_z:
        if obs.get("name") not in NEVER_DROP:
            return None
        stub = dict(obs)
        stub["position"] = [x, y, bottom + STUB_DZ_M / 2]
        stub["dims"] = [dx, dy, STUB_DZ_M]
        return stub
    capped = dict(obs)
    capped["position"] = [x, y, bottom + (top_z - bottom) / 2]
    capped["dims"] = [dx, dy, top_z - bottom]
    return capped


def _container_cuboid(model, cpose, name="container", top_z=None):
    dx, dy, dz = model.dims
    top = (cpose.xyz[2] + dz + ERR_TALL_M) if top_z is None else top_z
    height = top - cpose.xyz[2]
    return {
        "name": name,
        "position": [cpose.xyz[0], cpose.xyz[1], cpose.xyz[2] + height / 2],
        "dims": [dx, dy, height],
    }


def _lid_cuboid(model, lid_at):
    lx, ly, lz = model.lid_dims
    return {
        "name": "placed_lid",
        "position": [lid_at.xyz[0], lid_at.xyz[1], lid_at.xyz[2] + lz / 2],
        "dims": [lx, ly, lz],
    }


PLACEMENT_BAND_X = (0.15, 0.75)  # where a container may sit (reach map)
PLACEMENT_BAND_Y = (-0.50, 0.50)
TABLE_THICKNESS_M = 0.06  # a measured table is written as a slab this thick under its top


def _table_top_z(bench):
    t = next((o for o in bench["obstacles"] if o["name"] == "table"), None)
    if t is None:
        return None
    return t["position"][2] + t["dims"][2] / 2


def bench_world(bench, unseen_model=None):
    """The pre-detection world (press_demo scan / no-tag legs).

    No container POSE is known yet, but one is PRESENT somewhere: with a
    model given, the whole placement band is blocked to container height
    so pre-detection transits stay above anything that could be standing
    there — a failed detection must not mean a blind sweep through the
    container (2026-08-24 review). Scan/home poses live well above the
    band."""
    obstacles = _bench_obstacles(bench)
    if unseen_model is not None:
        top = _table_top_z(bench)
        if top is not None:
            h = unseen_model.dims[2] + ERR_TALL_M
            # where the table was seen, when it was measured (WorldStore)
            x0, x1, y0, y1 = bench.get("placement_xy") or (*PLACEMENT_BAND_X, *PLACEMENT_BAND_Y)
            obstacles.append(
                {
                    "name": "unseen_container_band",
                    "position": [(x0 + x1) / 2, (y0 + y1) / 2, top + h / 2],
                    "dims": [x1 - x0, y1 - y0, h],
                }
            )
    return {
        "base_frame": bench.get("base_frame", "base_link"),
        "obstacles": obstacles,
        "objects": [],
        "targets": [],
    }


def full_world(bench, model, cpose, lid_at=None, container_pad_xy=0.0):
    """container_pad_xy: extra xy half-extent on the container cuboid.
    Non-zero AFTER a contact leg: a press can scoot the box (field
    2026-09-01 — 8.1 Nm moved it ~2 cm and the home sweep then clipped
    the real box standing outside its stale cuboid), so post-contact
    transits allow for a shifted container."""
    obstacles = _bench_obstacles(bench)
    c = _container_cuboid(model, cpose)
    if container_pad_xy:
        c["dims"] = [
            c["dims"][0] + 2.0 * float(container_pad_xy),
            c["dims"][1] + 2.0 * float(container_pad_xy),
            c["dims"][2],
        ]
    obstacles.append(c)
    if lid_at is not None:
        obstacles.append(_lid_cuboid(model, lid_at))
    return {
        "base_frame": bench.get("base_frame", "base_link"),
        "obstacles": obstacles,
        "objects": [],
        "targets": [],
    }


def reduction_plane_z(contact_z, depth_max):
    """The interaction world must let the planner plan FROM the deepest
    commanded contact point (spec §6): plane ≥ margin below it."""
    return contact_z - depth_max - PLANE_MARGIN_M


def interaction_world(bench, model, cpose, contact_z, depth_max, lid_at=None):
    """The world a CONTACT leg plans in: everything above the reduction
    plane under the contact is removed, so a goal at or below a surface is
    plannable. Descents come from directly overhead (the vertical final
    approach), never laterally through what this world leaves out — and
    only guarded or slow legs may plan in it (the Runner's transit gate)."""
    plane = reduction_plane_z(contact_z, depth_max)
    # bench obstacles cap at the reduction plane too: a set-down ONTO the
    # bench must be plannable to its commanded overdrive depth, same
    # spec §6 rule as the container body (field 2026-08-26: IK_FAIL at a
    # set-down goal 25 mm over the solid table). For button-height
    # contacts the plane sits below every bench top — a no-op.
    obstacles = [c for c in (_cap_top(o, plane) for o in _bench_obstacles(bench)) if c]
    if plane > cpose.xyz[2]:  # body below the plane stays solid
        obstacles.append(
            _container_cuboid(model, cpose, name="container_body", top_z=plane)
        )
    if lid_at is not None:
        obstacles.append(_lid_cuboid(model, lid_at))
    return {
        "base_frame": bench.get("base_frame", "base_link"),
        "obstacles": obstacles,
        "objects": [],
        "targets": [],
    }


class WorldStore:
    def __init__(self, bench_yaml_path, out_dir=None):
        with open(bench_yaml_path) as f:
            self._bench = yaml.safe_load(f)
        self._dir = Path(
            out_dir
            if out_dir is not None
            else state_dir() / "worlds"
        )
        self._dir.mkdir(parents=True, exist_ok=True)
        self._surface_z = None  # runtime-measured surface; overrides YAML when set
        self._surface_xy = None  # (x0, x1, y0, y1): where the scene camera saw that surface
        self._warned_unplanned = False  # the "height but no extent" line, said once

    @property
    def table_top_z(self):
        """Measured table-top height in the planning frame.

        Read from the bench YAML's 'table' obstacle when present; otherwise
        None until the scene camera measures the surface (set by press_demo
        after the first SceneFix)."""
        if self._surface_z is not None:
            return self._surface_z
        return _table_top_z(self._bench)

    @table_top_z.setter
    def table_top_z(self, value):
        self._surface_z = float(value) if value is not None else None

    @property
    def table_extent(self):
        """(x0, x1, y0, y1) where the scene camera saw the table, or None."""
        return self._surface_xy

    @table_extent.setter
    def table_extent(self, value):
        self._surface_xy = None if value is None else tuple(float(v) for v in value)

    def _planning_bench(self):
        """The bench YAML, plus the table the scene camera measured when the
        YAML has none — flatbot's table changes with every place the chair
        goes: a slab under the measured top over where it was seen, and that
        area as the unseen container's band. Without it the planner plans as
        if there were no table at all."""
        if _table_top_z(self._bench) is not None or self._surface_z is None:
            return self._bench
        if self._surface_xy is None:
            # not raised: a raise mid-mission would strand the arm
            if not self._warned_unplanned:
                self._warned_unplanned = True
                print("[worlds] table height %.3f measured but no extent: the planner has NO table" % self._surface_z)
            return self._bench
        x0, x1, y0, y1 = self._surface_xy
        table = {
            "name": "table",
            "position": [(x0 + x1) / 2, (y0 + y1) / 2, self._surface_z - TABLE_THICKNESS_M / 2],
            "dims": [x1 - x0, y1 - y0, TABLE_THICKNESS_M],
        }
        return dict(self._bench, obstacles=list(self._bench["obstacles"]) + [table], placement_xy=[x0, x1, y0, y1])

    @property
    def base_frame(self):
        """The planning root frame for this robot config ('base_link' on the
        bench, 'arm_mount' on flatbot)."""
        return self._bench.get("base_frame", "base_link")

    @property
    def external_static_tf(self):
        """True when an external node (e.g. sheppy's static_tf) already
        publishes /tf_static — the launch file must not spawn its own."""
        return bool(self._bench.get("external_static_tf", False))

    @property
    def wrist_camera_config(self):
        """Camera config name for the wrist camera, or None (use default)."""
        return self._bench.get("wrist_camera_config")

    @property
    def scene_camera_cfg(self):
        """Scene camera config dict for SceneGrabber, or None (use Gemini defaults)."""
        name = self._bench.get("scene_camera_config")
        if name is None:
            return None
        from rammp_box_opening.perception.d405 import camera_config
        return camera_config(name)

    @property
    def table_normal(self):
        """The table's normal in base_link as measured (see the yaml), or
        exactly +z when the bench file does not say."""
        n = self._bench.get("table_normal_base", [0.0, 0.0, 1.0])
        n = [float(v) for v in n]
        s = sum(v * v for v in n) ** 0.5 or 1.0
        return [v / s for v in n]

    def push_name(
        self,
        kind,
        model=None,
        cpose=None,
        contact_z=None,
        depth_max=None,
        lid_at=None,
        tag="",
        container_pad_xy=0.0,
    ):
        bench = self._planning_bench()
        if kind == "bench":
            world = bench_world(bench, unseen_model=model)
            name = "bench" + (("_" + tag) if tag else "")
        elif kind == "full":
            world = full_world(
                bench,
                model,
                cpose,
                lid_at=lid_at,
                container_pad_xy=container_pad_xy,
            )
            name = "full" + (("_" + tag) if tag else "")
        elif kind == "interaction":
            world = interaction_world(
                bench, model, cpose, contact_z, depth_max, lid_at=lid_at
            )
            name = "interaction" + (("_" + tag) if tag else "")
        else:
            raise ValueError("unknown world kind %r" % kind)
        # The FILENAME carries a content hash, the NAME does not. Callers
        # gate on the name's prefix ("full"/"bench"/"interaction"), while
        # both SetWorld dedup checks compare the path — so a world whose
        # CONTENT changed gets a new path and is re-pushed, while an
        # identical rebuild reuses the same file and is correctly skipped.
        # Before this, the close-range re-fix rewrote full.yaml with the
        # corrected container pose under the SAME path, the dedup skipped
        # the push, and the re-approach was planned against the stale
        # container position (found by review 2026-08-28).
        text = yaml.safe_dump(_as_planned(world), sort_keys=False)
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
        path = self._dir / ("%s-%s.yaml" % (name, digest))
        if not path.exists():
            path.write_text(text)
        return name, path
