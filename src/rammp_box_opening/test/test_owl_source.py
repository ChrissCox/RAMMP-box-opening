"""The OWL bbox topic contract."""



def test_top_boxes_are_best_first_above_the_floor():
    from rammp_box_opening.perception.owl_source import bboxes_in_msg, top_boxes

    scores = [0.10, 0.255, 0.24, 0.30]
    boxes = [[0, 0, 1, 1], [775, 494, 874, 574], [923, 517, 1112, 608], [5, 5, 6, 6]]
    top = top_boxes(scores, [0] * 4, boxes, 0.12, k=3)
    assert [s for s, _b in top] == [0.30, 0.255, 0.24]
    m = []
    for s, b in top:
        m += [float(v) for v in b] + [s, 0.4]
    got = bboxes_in_msg(m)
    assert len(got) == 3 and got[0][:4] == [5.0, 5.0, 6.0, 6.0] and got[1][4] == 0.255
    assert bboxes_in_msg([0.0, 0.0, 0.0, 0.0, -1.0, 0.0]) == []  # a heartbeat
    assert bboxes_in_msg(None) == []



class _FakeNode:
    """Records publishers and what they publish."""

    def __init__(self):
        self.pubs = []  # (msg_type, topic, qos)
        self.published = []  # (topic, msg)

    def create_publisher(self, msg_type, topic, qos):
        node = self

        class Pub:
            def publish(self, msg):
                node.published.append((topic, msg))

        self.pubs.append((msg_type, topic, qos))
        return Pub()


def test_the_mission_tells_the_detectors_which_container_to_look_for(tmp_path):
    """The owl_detector nodes are started by the launch (sheppy's
    box_opening node) with no container of their own, so they loaded
    oxo_pop.yaml's prompts whatever the mission was run with:
    `press_demo --container ankou_pink.yaml` changed the geometry and the
    aim while both detectors kept looking for a white square box
    (2026-09-22). The mission now announces its container on ONE latched
    topic both instances read; the path is absolute, so a node started
    from any directory opens the same file."""
    from rclpy.qos import DurabilityPolicy, ReliabilityPolicy
    from std_msgs.msg import String

    from rammp_box_opening.perception.owl_source import CONTAINER_TOPIC, announce_container, container_changed

    assert CONTAINER_TOPIC.startswith("/rammp_box_opening/")
    yaml = tmp_path / "pink.yaml"
    yaml.write_text("x: 1\n")
    node = _FakeNode()
    announce_container(node, str(yaml))
    (msg_type, topic, qos), = node.pubs
    assert msg_type is String and topic == CONTAINER_TOPIC
    assert qos.durability == DurabilityPolicy.TRANSIENT_LOCAL and qos.reliability == ReliabilityPolicy.RELIABLE
    (t, m), = node.published
    assert t == CONTAINER_TOPIC and m.data == str(yaml.resolve())
    # the node reloads only on a real change; the same file spelled two ways is no change
    assert container_changed(str(yaml.resolve()), str(tmp_path / "." / "pink.yaml"))is False
    assert container_changed(str(yaml.resolve()), str(tmp_path / "other.yaml")) is True
    assert container_changed(str(yaml.resolve()), "") is False  # an empty announcement is ignored


def test_a_live_node_with_no_frames_says_so_instead_of_reading_as_dead():
    """Bench 2026-09-22: the wrist owl_detector had been up for 400 s with
    no camera frame ever delivered (the camera re-enumerated as it
    started). It heartbeated only after a NEW frame, so the mission read
    "owl_detector node not running" — the wrong diagnosis, and the one
    that hides a camera problem. A frame-less node now heartbeats BLIND
    (score -2) every tick, and the rung declines at once, naming it."""
    from types import SimpleNamespace

    from rammp_box_opening.perception.owl_source import (
        BLIND, HEARTBEAT, OwlRung, bboxes_in_msg, classify_bbox_msg,
    )

    now = 100.0
    assert classify_bbox_msg([0, 0, 0, 0, HEARTBEAT, now - 0.2], now) == "alive"
    assert classify_bbox_msg([0, 0, 0, 0, BLIND, now - 0.2], now) == "blind"
    assert classify_bbox_msg([0, 0, 0, 0, BLIND, now - 9.0], now) == "stale"
    assert bboxes_in_msg([0, 0, 0, 0, BLIND, now]) == []  # never a box
    rung = OwlRung.__new__(OwlRung)
    rung.node = SimpleNamespace(get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=int(now * 1e9))))
    rung.camera = "wrist"
    rung.latest = [0, 0, 0, 0, BLIND, now - 0.1]
    rung.latest_box = None
    roi, why = rung(None, None)
    assert roi is None and "no wrist frame" in why and "not running" not in why



def test_blind_means_no_frames_for_a_while_not_one_quiet_tick():
    """Bench 2026-09-23: two runs said "OWL node is live but has had no
    wrist frame" while the node's own log showed frames flowing. An
    inference (0.65 s) outlasts the 0.5 s tick, so the next tick fires
    before a new frame lands — and the node sent BLIND on every such tick
    (added 2026-09-22). The mission reads the LATEST message: one quiet
    tick ended the wait. BLIND now means what it says."""
    from rammp_box_opening.perception.owl_source import BLIND_AFTER_S, blind_heartbeat_due

    assert not blind_heartbeat_due(0.7)  # one overrun tick
    assert not blind_heartbeat_due(BLIND_AFTER_S - 0.1)
    assert blind_heartbeat_due(BLIND_AFTER_S + 0.1)


def test_a_heartbeat_right_after_a_box_does_not_hide_it():
    """... and the same race hid the scene camera's boxes: a heartbeat
    landing milliseconds after a bbox replaced it as `latest` before the
    locate loop (polling every 20 ms) looked — "the scene OWL saw no box in
    3.0 s" with the box in view. The rung keeps the newest BOX apart from
    the newest message."""
    from types import SimpleNamespace

    from rammp_box_opening.perception.owl_source import HEARTBEAT, OwlRung

    now = [100.0]
    rung = OwlRung.__new__(OwlRung)
    rung.node = SimpleNamespace(get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=int(now[0] * 1e9))))
    rung.latest = rung.latest_box = None
    rung.enabled = False
    rung.watcher_holder = {}
    rung._cb(SimpleNamespace(data=[10.0, 20.0, 60.0, 80.0, 0.45, 0.1]))
    rung._cb(SimpleNamespace(data=[0.0, 0.0, 0.0, 0.0, HEARTBEAT, 0.0]))
    assert rung.latest[4] == HEARTBEAT  # the newest message is the heartbeat ...
    box = rung.fresh_box()
    assert box is not None and box[4] == 0.45  # ... and the box is still there
    now[0] += 10.0
    assert rung.fresh_box() is None  # but not forever
