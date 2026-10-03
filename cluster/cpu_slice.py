#!/usr/bin/env python3
"""Print worker K's share of this process's CPUs, as a `taskset -c` list.

    taskset -c "$(python cluster/cpu_slice.py <workers> <k>)" <command>

WHY. The GVS push-rod arm's forward model is JAX on the CPU, and XLA sizes its per-process
thread pools from the CPUs the process may run on -- jax 0.11 ignores the
`intra_op_parallelism_threads` flag. Measured on the laptop with the benchmark's own thread
environment: 102 threads unpinned on 20 logical CPUs, 12 pinned to 2. Unpinned on a cluster
node, a handful of benchmark workers approach the node's soft process limit (4096) and a
few more pass it, which is the `pthread_create failed` death of the first dataset builds
(cluster/GVS_ARM_RUNBOOK.md). The dataset builder pins each worker to its own CPU slice
for the same reason; this is that slicing, shared by run_items.sh and calibrate.sh.

Slices are whole PHYSICAL cores. Linux usually numbers a core's hyperthread sibling far
from it (cpu 3 and cpu 43 on a 40-core node), so slicing the sorted logical list would
hand two workers the two halves of the same cores. Cores are grouped by
`topology/core_cpus_list` (falling back to `thread_siblings_list`, then to one CPU per
core), dealt out in contiguous blocks, and each worker gets every logical CPU of its
cores. With more workers than cores, workers share cores round-robin rather than fail.
"""
import os
import sys


def _siblings(cpu):
    for name in ("core_cpus_list", "thread_siblings_list"):
        path = f"/sys/devices/system/cpu/cpu{cpu}/topology/{name}"
        try:
            with open(path) as f:
                text = f.read().strip()
        except OSError:
            continue
        out = set()
        for part in text.split(","):
            lo, _, hi = part.partition("-")
            out.update(range(int(lo), int(hi or lo) + 1))
        return frozenset(out)
    return frozenset({cpu})


def Slice(workers, k, allowed=None):
    allowed = sorted(allowed if allowed is not None else os.sched_getaffinity(0))
    cores, seen = [], set()
    for cpu in allowed:
        if cpu in seen:
            continue
        group = sorted(_siblings(cpu) & set(allowed)) or [cpu]
        seen.update(group)
        cores.append(group)
    if workers <= len(cores):
        per = len(cores) // workers
        mine = cores[k * per:(k + 1) * per]
    else:
        mine = [cores[k % len(cores)]]
    return sorted(c for group in mine for c in group)


if __name__ == "__main__":
    workers, k = int(sys.argv[1]), int(sys.argv[2])
    if not 0 <= k < workers:
        raise SystemExit(f"worker index {k} outside 0..{workers - 1}")
    print(",".join(map(str, Slice(workers, k))))
