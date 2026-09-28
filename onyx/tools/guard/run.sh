#!/usr/bin/env bash
# The guard, before the metric: pinned configs, no label reads, the unit
# suite. ROS is sourced for the tests' imports, on an isolated domain and
# loopback only — nothing here can reach the robot's graph.
set -eo pipefail
cd "$(dirname "$0")/../../.."
export ROS_DOMAIN_ID=199 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
# loopback: ROS_LOCALHOST_ONLY, unless a Cyclone config already pins it (the
# bench's does; Cyclone refuses a participant when both configure interfaces)
if [ -n "$CYCLONEDDS_URI" ]; then export ROS_LOCALHOST_ONLY=0; else export ROS_LOCALHOST_ONLY=1; fi
set +u
source /opt/ros/humble/setup.bash
source "$HOME/rammp_deps_ws/install/setup.bash"
export PYTHONPATH="$PWD/src/rammp_box_opening:$PWD/src/rammp_box_opening/test:$PYTHONPATH"
exec python3 onyx/tools/guard/check.py "$@"
