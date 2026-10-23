#!/usr/bin/env bash
# Run a rammp_box_opening CLI task as a sheppy process node:
#   sheppy_task.sh <task> [args...]   e.g. sheppy_task.sh press_demo --execute
# Adds the env the CLIs need on the host and the flatbot world. exec, so the
# task itself receives sheppy's SIGINT (the same cancel path as Ctrl+C).
set -eo pipefail

source /opt/ros/humble/setup.bash
source "$HOME/RAMMP-box-opening/install/setup.bash"
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
# rammp_curobo_interfaces is vendored into this package's install, off the default library path.
export LD_LIBRARY_PATH="$HOME/RAMMP-box-opening/install/rammp_box_opening/lib/rammp_curobo_interfaces:${LD_LIBRARY_PATH:-}"

task="$1"
shift

# The robot runs the INSTALLED copy: built without --symlink-install, python
# is copied there, and a src edit not rebuilt is not what flies (2026-10-22:
# the installed reach-out would have pushed 12 mm, the source capped it at 6.7).
# --execute refuses while the two differ; a dry run says so and goes on.
src="$HOME/RAMMP-box-opening/src/rammp_box_opening"
inst="$HOME/RAMMP-box-opening/install/rammp_box_opening"
inst_py="$inst/lib/python3.10/site-packages/rammp_box_opening"
# what gets installed, and only that: the package's .py both ways, and the
# config YAMLs setup.py installs (config/*.yaml, config/containers/*.yaml)
drift() {
  local f rel
  while IFS= read -r -d '' f; do
    rel="${f#"$src/rammp_box_opening/"}"
    cmp -s "$f" "$inst_py/$rel" || echo "differs: $rel"
  done < <(find "$src/rammp_box_opening" -name '*.py' -not -path '*/__pycache__/*' -print0)
  while IFS= read -r -d '' f; do
    rel="${f#"$inst_py/"}"
    [ -e "$src/rammp_box_opening/$rel" ] || echo "stale in install/ (a rebuild keeps it: rm -rf build/rammp_box_opening install/rammp_box_opening, then rebuild): $rel"
  done < <(find "$inst_py" -name '*.py' -not -path '*/__pycache__/*' -print0)
  for f in "$src"/config/*.yaml "$src"/config/containers/*.yaml; do
    rel="${f#"$src/config/"}"
    cmp -s "$f" "$inst/share/rammp_box_opening/config/$rel" || echo "differs: config/$rel"
  done
}
executing=0
for a in "$@"; do
  case "$a" in --e*) executing=1 ;; esac  # --execute, and anything argparse might once have read as it
done
if [ -d "$inst_py" ] && [ ! -L "$inst_py" ]; then
  stale="$(drift 2>&1 || true)"
  if [ -n "$stale" ]; then
    echo "[sheppy_task] the installed rammp_box_opening is not the source:" >&2
    echo "$stale" | head -10 >&2
    if [ "$executing" = 1 ]; then
      echo "[sheppy_task] REFUSED --execute — rebuild first: colcon build --packages-select rammp_box_opening" >&2
      exit 3
    fi
    echo "[sheppy_task] not executing: running the installed copy anyway" >&2
  fi
fi
exec ros2 run rammp_box_opening "$task" \
  --bench-world "$HOME/RAMMP-box-opening/install/rammp_box_opening/share/rammp_box_opening/config/world_flatbot.yaml" \
  "$@"
