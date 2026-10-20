#!/usr/bin/env bash
# Campaign 1's metric tool: replay the recorded detection set through THIS
# worktree's perception code (evaluate.py). A scrubbed environment: no ROS
# setup, an isolated domain and loopback only — nothing here can join the
# robot's graph, and the model loads from the local cache, never the network.
set -euo pipefail
cd "$(dirname "$0")/../../.."
exec env -i HOME="$HOME" PATH=/usr/local/bin:/usr/bin:/bin \
    ROS_DOMAIN_ID=199 ROS_LOCALHOST_ONLY=1 \
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    PYTHONPATH="$PWD/src/rammp_box_opening" \
    python3 onyx/tools/evaluation/evaluate.py "$@"
