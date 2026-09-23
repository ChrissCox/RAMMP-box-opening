"""The camera grabbers can rebuild their subscriptions in place."""

import rclpy


def _count(node):
    return len(list(node.subscriptions))


def test_scene_grabber_resubscribes_in_place():
    from rammp_box_opening.perception.scene import SceneGrabber

    rclpy.init()
    try:
        node = rclpy.create_node("grabber_test")
        g = SceneGrabber(node, keep=2, need_depth=True)
        before = _count(node)
        assert len(g._subs) == 4
        g.resubscribe()
        assert g.resubscribes == 1 and len(g._subs) == 4
        assert _count(node) == before  # torn down and rebuilt, not doubled
        node.destroy_node()
    finally:
        rclpy.shutdown()


def test_d405_grabber_resubscribes_in_place():
    from rammp_box_opening.perception.d405 import D405Grabber

    rclpy.init()
    try:
        node = rclpy.create_node("grabber_test_d405")
        g = D405Grabber(node, need_depth=True)
        before = _count(node)
        assert len(g._subs) == 3
        g.resubscribe()
        assert g.resubscribes == 1 and len(g._subs) == 3
        assert _count(node) == before
        node.destroy_node()
    finally:
        rclpy.shutdown()


def test_grabbers_spin_no_thread_of_their_own():
    """One executor per node: a grabber's TF listener must not start a
    spinning thread behind the node's owner (the OWL node went deaf under
    two executors, 2026-09-17)."""
    import threading

    from rammp_box_opening.perception.d405 import D405Grabber
    from rammp_box_opening.perception.scene import SceneGrabber

    rclpy.init()
    try:
        node = rclpy.create_node("grabber_threads")
        before = threading.active_count()
        SceneGrabber(node, keep=1, need_depth=False)
        D405Grabber(node, need_depth=False)
        assert threading.active_count() == before
        node.destroy_node()
    finally:
        rclpy.shutdown()

