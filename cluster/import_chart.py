#!/usr/bin/env python3
"""Copy an exported chart into another tree under a (possibly new) robot name. Runs on the
cluster, with the system python3: stdlib only, no torch.

    python3 cluster/import_chart.py SRC.pkl DST.pkl --robot-name NEW_NAME

A chart is two files, the bare-pickle state dict and its `.arch.json` sidecar
(`src/flow_loading.py`). The state dict carries no robot name -- it is weights only -- so it is
copied BYTE FOR BYTE and the copy is checked by sha256. The sidecar is rewritten in exactly one
field, `robot_name`, and gains a `provenance.imported_from` entry naming the source file, its
sha256 and the name it was trained under, so the copy can always be traced to the run that
produced it. Every other field, including the original provenance, is carried unchanged, and
that is checked too.

Why this exists rather than a rename: a campaign's own cluster tree is its archive, and is left
as it ran. When a robot is renamed afterwards, new work runs from a new tree, and this is how its
charts get there without touching the original.

REFUSES to overwrite an existing destination.
"""

import argparse
import hashlib
import json
import os
import shutil
import sys


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sidecar(path):
    return os.path.splitext(path)[0] + ".arch.json"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--robot-name", required=True)
    args = p.parse_args()

    src, dst = os.path.abspath(args.src), os.path.abspath(args.dst)
    for path in (src, sidecar(src)):
        if not os.path.isfile(path):
            sys.exit(f"REFUSING: {path} does not exist")
    for path in (dst, sidecar(dst)):
        if os.path.exists(path):
            sys.exit(f"REFUSING: {path} already exists")

    with open(sidecar(src)) as f:
        arch = json.load(f)
    old_name = arch.get("robot_name")
    if old_name is None:
        sys.exit(f"REFUSING: {sidecar(src)} records no robot_name")

    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst)
    digest = sha256(src)
    if sha256(dst) != digest:
        os.remove(dst)
        sys.exit("ERROR: the copied weights do not match the source; copy removed")

    new = dict(arch)
    new["robot_name"] = args.robot_name
    prov = dict(arch.get("provenance") or {})
    prov["imported_from"] = {"path": src, "sha256": digest, "robot_name": old_name}
    new["provenance"] = prov
    with open(sidecar(dst), "w") as f:
        json.dump(new, f, indent=2, sort_keys=True)

    ## Check the rewrite touched exactly what it says it did.
    with open(sidecar(dst)) as f:
        back = json.load(f)
    changed = {k for k in set(arch) | set(back) if arch.get(k) != back.get(k)}
    assert changed <= {"robot_name", "provenance"}, changed
    assert {k: v for k, v in back["provenance"].items() if k != "imported_from"} == (arch.get("provenance") or {})
    print(f"imported {dst}\n  weights sha256 {digest[:16]} (identical to source)\n"
          f"  robot_name {old_name} -> {args.robot_name}")


if __name__ == "__main__":
    main()
