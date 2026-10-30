# syntax=docker/dockerfile:1
# rammp-box-opening — persistent OWLv2 wrist + scene detectors, started by
# press_demo.launch.py. Jetson Orin only: FROM rammp-cuda, run with
# --runtime nvidia (JetPack exposes the GPU this way, not --gpus).
#
# Build from the repo root:
#   docker build -t rammp-box-opening:dev .
#
# Runs press_demo.launch.py with tf:=false (TF comes from sheppy's
# arm_state_publisher + static_tf) and bench_world:=world_flatbot.yaml.
# camera:=false (default): sheppy's wrist_camera node owns the OAK-D;
# this launch only adds the two OWL detector nodes on top.
#
# press_demo --execute is NOT in this image: it is an attended CLI the
# operator types in a separate shell on the host.

# ── rammp_curobo_interfaces source ────────────────────────────────────────
# Full clone: 131f834 is 11 commits past v1.0.0, not at a release tag.
# Pinned to the exact commit kinova-gen3-ros2:nightly uses — the interface
# contract must be identical since both containers join the same DDS graph.
FROM ghcr.io/rammp-org/rammp-cuda:1.1.0-jp6 AS curobo_src
RUN git clone https://github.com/rammp-org/RAMMP-CuRobo.git /src \
 && git -C /src checkout 131f8348d65670010be5b547128a85fa41b15b7e

# ── final image ───────────────────────────────────────────────────────────
FROM ghcr.io/rammp-org/rammp-cuda:1.1.0-jp6

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# OWLv2 ML deps. torch 2.10 and numpy 1.26.4 already come from rammp-cuda;
# do not reinstall them. Pinned to the versions measured with this torch on
# JetPack 6.2.1 (the rfdetr/sam3 containers pin transformers 5.19.0 — that
# version is confirmed to work; 4.47.1 is also fine and the version we
# measured OWLv2 on this Jetson against).
RUN pip3 install --no-cache-dir \
      transformers==4.47.1 \
      huggingface-hub==0.27.1 \
      safetensors==0.4.5

# OWLv2 weights — 593 MB, public (Apache-2.0, google/owlv2-base-patch16-ensemble).
# Baked in at build time; TRANSFORMERS_OFFLINE=1 blocks any runtime fetch.
ENV HF_HOME=/models/hf
RUN TRANSFORMERS_OFFLINE=0 python3 -c "
from transformers import Owlv2Processor, Owlv2ForObjectDetection
Owlv2Processor.from_pretrained('google/owlv2-base-patch16-ensemble')
Owlv2ForObjectDetection.from_pretrained('google/owlv2-base-patch16-ensemble')
print('weights ok')
"
ENV TRANSFORMERS_OFFLINE=1

# torch lib and CUDA lib paths in the container. press_demo.launch.py's
# SetEnvironmentVariable builds LD_LIBRARY_PATH from the host colcon install
# path (absent here), then appends _existing_ldpath — this ENV provides the
# correct system-torch paths via that append so the OWL nodes find the libs.
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib:/usr/local/cuda-12.6/targets/aarch64-linux/lib

# PyTorch's compile cache: host user (1000) has no passwd entry inside the
# container, so torch would fail to name the cache dir after the user.
# Same fix as the yoloe/rfdetr/sam3 containers.
ENV TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor

# rammp_curobo_interfaces + rammp_box_opening in one colcon workspace so the
# install tree is a single coherent /module_ws overlay on top of /ros2_ws.
# /module_ws/install is sourced by /ros_entrypoint.sh automatically.
#
# kinova_gen3_description is an exec_depend of rammp_box_opening but is only
# needed when tf:=true (starts description.launch.py). This container always
# runs tf:=false; TF comes from sheppy's arm_state_publisher + static_tf.
# rosdep is run only for rammp_curobo_interfaces (the ament_cmake IDL package
# whose build deps must be resolved); the Python package needs none.
COPY --from=curobo_src /src/rammp_curobo_interfaces /module_ws/src/rammp_curobo_interfaces
COPY src/rammp_box_opening /module_ws/src/rammp_box_opening

RUN rosdep update --rosdistro humble \
 && . /opt/ros/humble/setup.sh \
 && . /ros2_ws/install/setup.sh \
 && cd /module_ws \
 && rosdep install --from-paths src/rammp_curobo_interfaces --ignore-src -y \
 && colcon build --event-handlers console_direct- \
 && rm -rf build log

# Build-time import check. No GPU at build time (no --runtime nvidia here),
# so cuda availability is not tested — that is confirmed at first start.
RUN . /opt/ros/humble/setup.sh \
 && . /ros2_ws/install/setup.sh \
 && . /module_ws/install/setup.sh \
 && python3 -c "
from rammp_curobo_interfaces.action import PlanToPose, PlanToJoints
from rammp_curobo_interfaces.srv import SetWorld
import rammp_box_opening.perception.owl_node
import rammp_box_opening.tasks.cli_common
print('import check ok')
"

# launch_entry.py: forward SIGTERM as SIGINT so docker stop shuts ros2 launch
# down cleanly. See scripts/launch_entry.py for the rationale (same as teleop).
COPY scripts/launch_entry.py /scripts/launch_entry.py

# tf:=false        — TF comes from sheppy arm_state_publisher + static_tf
# bench_world      — flatbot collision world and camera config names
# camera:=false    — default; sheppy's wrist_camera owns the OAK-D
CMD ["/ros_entrypoint.sh", "python3", "/scripts/launch_entry.py", "rammp_box_opening", "press_demo.launch.py", "tf:=false", "bench_world:=world_flatbot.yaml"]
