"""The detection-set recorder's pure parts: where it asks for the box, and
the manifest the evaluator checks the set against."""

import json


def test_the_grid_covers_the_reachable_table_in_snake_order():
    from rammp_box_opening.detection_set import GRID_X, GRID_Y, grid_points

    pts = grid_points()
    assert len(pts) == len(GRID_X) * len(GRID_Y) == 25
    assert len(set(pts)) == len(pts)
    # each hover is short: consecutive points are grid neighbours
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        assert abs(x1 - x0) + abs(y1 - y0) <= max(max(GRID_X) - min(GRID_X), 0.16)
        assert (x0 == x1) or (y0 == y1)
    assert min(p[0] for p in pts) == 0.30 and max(p[0] for p in pts) == 0.70
    assert min(p[1] for p in pts) == -0.30 and max(p[1] for p in pts) == 0.30


def test_the_manifest_pins_every_file_and_carries_every_truth(tmp_path):
    from rammp_box_opening.detection_set import sha256, write_manifest

    for name, truth in (("p000", {"container": "ankou_pink.yaml", "truth_xyz": [0.4, 0.0, 0.094]}),
                        ("p900", {"container": None, "truth_xyz": None})):
        d = tmp_path / name
        (d / "scene_0").mkdir(parents=True)
        (d / "scene_0" / "frame_000.npz").write_bytes(b"frame " + name.encode())
        (d / "truth.json").write_text(json.dumps(truth))
    doc = write_manifest(tmp_path)
    assert [p["id"] for p in doc["placements"]] == ["p000", "p900"]
    assert doc["placements"][1]["container"] is None
    assert doc["files"]["p000/scene_0/frame_000.npz"] == sha256(tmp_path / "p000/scene_0/frame_000.npz")
    assert set(doc["files"]) == {"p000/scene_0/frame_000.npz", "p000/truth.json", "p900/scene_0/frame_000.npz", "p900/truth.json"}
    assert json.loads((tmp_path / "manifest.json").read_text()) == doc
