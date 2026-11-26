"""Convert chart weight `.pkl` files (bare state-dict pickles) to pickle-free `.npz`.

    .venv/bin/python scripts/training/convert_pkl_to_npz.py models/panda/panda__n6__step620000.pkl ...

Each `x.pkl` becomes `x.npz` beside it. Nothing else is written, copied or linked: the
`x.arch.json` sidecar is found by STEM (`flow_loading.SidecarPath`), so it already serves
both extensions, and it is only checked for, never touched. The `.pkl` is never modified.

The ONE pickle read here goes through a restricted unpickler that resolves exactly the three
globals a torch state dict needs (`OrderedDict`, `torch._utils._rebuild_tensor_v2`,
`torch.storage._load_from_bytes`, the last replaced by a `weights_only=True` torch.load) and
raises on anything else. It still must only be run on a pickle you trust the origin of: the
restriction is defence in depth, not a licence to convert files of unknown provenance.

After writing, the npz is re-read with `allow_pickle=False` and every key is checked present,
dtype-equal and `np.array_equal` to the source, with no object arrays. Refuses to overwrite.
"""
import argparse
import collections
import hashlib
import io
import os
import pickle
import sys

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

from src.weights_io import SaveStateDictNpz, StateDictToNumpy  # noqa: E402


def _load_from_bytes_safe(b):
    return torch.load(io.BytesIO(b), map_location="cpu", weights_only=True)


class _StateDictUnpickler(pickle.Unpickler):
    _ALLOWED = {
        ("collections", "OrderedDict"): collections.OrderedDict,
        ("torch._utils", "_rebuild_tensor_v2"): torch._utils._rebuild_tensor_v2,
        ("torch.storage", "_load_from_bytes"): _load_from_bytes_safe,
    }

    def find_class(self, module, name):
        try:
            return self._ALLOWED[(module, name)]
        except KeyError:
            raise pickle.UnpicklingError(f"refusing global {module}.{name}") from None


def ReadPickleRestricted(path):
    with open(path, "rb") as f:
        return _StateDictUnpickler(f).load()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def Verify(state_dict, npz_path):
    expected = StateDictToNumpy(state_dict)
    with np.load(npz_path, allow_pickle=False) as npz:
        assert list(npz.files) == list(expected), "key set or order differs"
        for k, ref in expected.items():
            got = npz[k]
            assert not got.dtype.hasobject, f"{k}: object array"
            assert got.dtype == ref.dtype, f"{k}: dtype {got.dtype} != {ref.dtype}"
            assert got.shape == ref.shape and np.array_equal(got, ref), f"{k}: values differ"
    return len(expected)


def convert(pkl_path):
    stem, ext = os.path.splitext(pkl_path)
    if ext != ".pkl":
        raise ValueError(f"{pkl_path} is not a .pkl")
    out = stem + ".npz"
    if os.path.exists(out):
        raise FileExistsError(f"{out} exists; refusing to overwrite")
    state_dict = ReadPickleRestricted(pkl_path)
    if not isinstance(state_dict, dict):
        raise TypeError(f"{pkl_path}: pickle holds {type(state_dict).__name__}, not a state dict")
    SaveStateDictNpz(out, state_dict)
    n = Verify(state_dict, out)
    if not os.path.exists(stem + ".arch.json"):
        print(f"  note: no {os.path.basename(stem)}.arch.json beside it")
    print(f"{out}\n  {n} arrays bit-identical to the pickle, sha256 {sha256(out)}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("pkl", nargs="+")
    for p in parser.parse_args().pkl:
        convert(p)
