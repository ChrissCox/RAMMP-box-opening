"""Which leftover processes the press_demo launch may clear away.

pgrep sees every process on the machine, including the ones sheppy runs in
containers: the arm driver, the planner, and the camera drivers once their
images move over. Those belong to the deployment. A launch that SIGINTs
"stray" drivers by name would take down what sheppy owns (or crash on EPERM
for a root-owned one), so a stray is a HOST process, always.
"""

CONTAINER_MARKERS = ("docker", "containerd", "libpod", "kubepods")


def in_container(cgroup_text):
    """True when a /proc/<pid>/cgroup listing places the process in a container."""
    return any(
        marker in line for line in cgroup_text.splitlines() for marker in CONTAINER_MARKERS
    )


def read_cgroup(pid):
    """/proc/<pid>/cgroup, or None once the process is gone."""
    try:
        with open("/proc/%d/cgroup" % int(pid)) as f:
            return f.read()
    except OSError:
        return None


def host_pids(pgrep_output, cgroup_of=read_cgroup):
    """The pids in `pgrep -af` output whose processes run on the host."""
    pids = []
    for line in pgrep_output.splitlines():
        fields = line.split(None, 1)
        if not fields or not fields[0].isdigit():
            continue
        pid = int(fields[0])
        cgroup = cgroup_of(pid)
        if cgroup is None or in_container(cgroup):
            continue
        pids.append(pid)
    return pids
