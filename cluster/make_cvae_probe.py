#!/usr/bin/env python3
"""Derive the reduced cVAE benchmark manifests (stage CVAEPROBE) from the full committed ones.

Why this exists (Thomas, 2026-10-19): both standard cVAE objectives -- learned decoder variance
(`cvae<K>`) and the textbook fixed variance (`cvaef<K>`) -- train into unusable charts on the
rigid arms (median FK error 0.25-0.75 m at prior samples, latent collapsed), while soft12's are
usable. Running every row in full would spend days of the 4-node cap on native cells that run to
the 180 s clock on charts that cannot reach the target. So:

  * soft12 (both variants): every item, unchanged -- the full 480-cell rows;
  * rigid arms, PAIRED rows: the full 480 cells, but as ONE unsharded item per row -- every
    start is `unrepresentable_start` (no solve), so the cells are instant and the row's cost is
    its process start-up, which sharding would multiply (24 svgd shards x ~1 min each);
  * rigid arms, NATIVE rows: a go/no-go probe -- ONE unsharded item per row on 60 cells
    (guess 0 of every target), tag `..._60_180_native`; a row showing real successes gets
    extended afterwards;
  * the joint-space acceptance item: kept;
  * iiwa14 `cvae8`: dropped -- its training diverged twice and Thomas ruled it recorded, not
    benchmarked.

Every kept item is byte-identical to its line in the full manifest except: the rigid paired row
drops `--shard k/N` (id = its tag), and the native probe's `--shard k/N` becomes
`--cells 0:0,...,59:0` with tag/id `..._60_180_native`.

Stage CVAEEXT (2026-10-20) is the extension the probe was there to decide: the Panda and iiwa
native probes showed real successes (up to 59/60) and screw's showed none (0/60 on all 16 rows),
so every Panda and iiwa native row (both variants, both tasks; iiwa `cvae8` still dropped) runs
in full -- its items copied byte for byte from the full manifest, so the row is the full stage's
own row and the probe's 60 cells are re-solved inside it. Usage:

    python cluster/make_cvae_probe.py            # (re)write all four manifests
    python cluster/make_cvae_probe.py --check    # exit 1 if the committed ones are stale
"""

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCES = {
    "manifest_stageCVAEPROBE.txt": ["manifest_stageCVAE.txt", "manifest_stageCVAEF.txt"],
    "manifest_stageCVAEPROBE_SVGD.txt": ["manifest_stageCVAE_SVGD.txt", "manifest_stageCVAEF_SVGD.txt"],
}
PROBE_TARGETS = 60
PROBE_CELLS = ",".join(f"{t}:0" for t in range(PROBE_TARGETS))
RIGID = ("panda", "iiwa", "screw7")
EXTENDED = ("panda", "iiwa")
EXT_SOURCES = {
    "manifest_stageCVAEEXT.txt": SOURCES["manifest_stageCVAEPROBE.txt"],
    "manifest_stageCVAEEXT_SVGD.txt": SOURCES["manifest_stageCVAEPROBE_SVGD.txt"],
}


def build(sources):
    out, seen_probe = [], set()
    for src in sources:
        for line in (HERE / src).read_text().splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            item_id, env, script, args = line.split("|", 3)
            if "__cvae8__" in args and "iiwa14" in args:
                continue  # diverged twice; recorded, not benchmarked
            robot = item_id.split("_")[2]
            if "--arms numerical" in args or not robot.startswith(RIGID):
                out.append(line)
                continue
            row = re.sub(r"_shard\d+of\d+$", "", item_id)
            if "_paired" in item_id:
                if row not in seen_probe:
                    seen_probe.add(row)
                    out.append("|".join([row, env, script, re.sub(r" --shard \d+/\d+", "", args)]))
                continue
            if "_native" not in item_id:
                raise SystemExit(f"unexpected item {item_id}")
            row = re.sub(r"_shard\d+of\d+$", "", item_id).replace("_480_180_", f"_{PROBE_TARGETS}_180_")
            if row in seen_probe:
                continue
            seen_probe.add(row)
            old_tag = re.search(r"--tag (\S+)", args).group(1)
            new_args = re.sub(r" --shard \d+/\d+", "", args).replace(f"--tag {old_tag}", f"--tag {row}")
            new_args += f" --cells {PROBE_CELLS}"
            out.append("|".join([row, env, script, new_args]))
    return out


def build_ext(sources):
    out = []
    for src in sources:
        for line in (HERE / src).read_text().splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            item_id, _, _, args = line.split("|", 3)
            if "__cvae8__" in args and "iiwa14" in args:
                continue
            if item_id.split("_")[2] in EXTENDED and "_native" in item_id and "--arms numerical" not in args:
                out.append(line)
    return out


def main():
    check = "--check" in sys.argv
    stale = False
    for name, sources, fn in [(n, s, build) for n, s in SOURCES.items()] + \
            [(n, s, build_ext) for n, s in EXT_SOURCES.items()]:
        items = fn(sources)
        text = (f"# learned-ik stage {name[len('manifest_stage'):-4]} manifest -- generated by "
                f"cluster/make_cvae_probe.py from {', '.join(sources)}\n# {len(items)} items\n"
                + "\n".join(items) + "\n")
        path = HERE / name
        if check:
            stale |= not path.exists() or path.read_text() != text
        else:
            path.write_text(text)
            print(f"{name}: {len(items)} items")
    if check and stale:
        raise SystemExit("stale CVAEPROBE/CVAEEXT manifests: re-run cluster/make_cvae_probe.py")


if __name__ == "__main__":
    main()
