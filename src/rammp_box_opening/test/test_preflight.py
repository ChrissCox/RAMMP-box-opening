"""Preflight's camera and detector checks: the streams the mission reads,
judged before the arm moves rather than discovered mid-run."""


def test_preflight_checks_every_stream_the_mission_reads():
    """Bench 2026-09-22/23: preflight passed, and the run then failed at the
    wrist with "ALIGNED depth (align_depth.enable:=true?)" — a manifest
    update had dropped the flag and nothing checked the camera streams."""
    from rammp_box_opening.perception.scene import COLOR_TOPIC, DEPTH_TOPIC
    from rammp_box_opening.tasks.preflight import camera_topics

    topics = {t: (required, hint) for t, required, hint in camera_topics()}
    aligned = "/wrist_camera/aligned_depth_to_color/image_raw"
    assert aligned in topics and topics[aligned][0]  # the mission cannot aim without it
    assert "align_depth.enable:=true" in topics[aligned][1]
    for t in ("/wrist_camera/color/image_raw", "/wrist_camera/color/camera_info"):
        assert topics[t][0]
    for t in (COLOR_TOPIC, DEPTH_TOPIC):  # the scene camera: the mission falls back to the wrist search
        assert t in topics and not topics[t][0]


def test_stream_verdicts_fail_a_required_stream_and_warn_an_optional_one():
    from rammp_box_opening.tasks.preflight import stream_verdicts

    topics = [("/a", True, "fix a"), ("/b", False, "fix b"), ("/c", True, "fix c")]
    got = stream_verdicts(topics, {"/a": 0, "/b": 0, "/c": 12}, window_s=3.0)
    levels = {t: (lvl, text) for t, lvl, text in got}
    assert levels["/a"][0] == "FAIL" and "fix a" in levels["/a"][1]
    assert levels["/b"][0] == "WARN" and "fix b" in levels["/b"][1]
    assert levels["/c"][0] == "PASS" and "4.0 Hz" in levels["/c"][1]


def test_a_deaf_detector_is_told_apart_from_a_missing_one():
    """The wrist owl_detector ran 400 s without one frame (2026-09-22) and
    the mission called it "not running". Preflight reads the heartbeats:
    none -> not running; only BLIND -> up but deaf (restart box_opening);
    anything else -> fine."""
    from rammp_box_opening.perception.owl_source import BLIND, HEARTBEAT
    from rammp_box_opening.tasks.preflight import owl_verdict

    lvl, text = owl_verdict([], "wrist")
    assert lvl == "WARN" and "not running" in text
    lvl, text = owl_verdict([BLIND] * 6, "wrist")
    assert lvl == "WARN" and "no wrist frame" in text and "sheppy restart box_opening" in text
    assert owl_verdict([BLIND, BLIND, HEARTBEAT], "scene")[0] == "PASS"  # frames came back
    assert owl_verdict([HEARTBEAT] * 6, "scene")[0] == "PASS"
    assert owl_verdict([0.57], "scene")[0] == "PASS"  # a box: very much alive
