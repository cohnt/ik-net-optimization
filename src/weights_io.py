"""Chart weight files: a pickle-free `.npz` next to the legacy bare-pickle `.pkl`.

Loading a pickle runs arbitrary code, so a `.pkl` copied back from a cluster is an attack
surface. A `.npz` holds the same state dict as one plain array per key and is read with
`np.load(path, allow_pickle=False)`: no pickle machinery is reached for it.

The `<name>.arch.json` sidecar is found by stem (`flow_loading.SidecarPath` uses
`os.path.splitext`), so `x.pkl` and `x.npz` share `x.arch.json`.
"""
import os
import pickle

import numpy as np
import torch

NPZ_SUFFIX = ".npz"


def IsNpz(path):
    return os.fspath(path).lower().endswith(NPZ_SUFFIX)


def LoadStateDictNpz(path):
    """The state dict stored in `path`, as torch tensors, in the stored key order. Never pickles."""
    state_dict = {}
    with np.load(path, allow_pickle=False) as npz:
        for key in npz.files:
            array = npz[key]
            if array.dtype.hasobject:
                raise ValueError(f"{path}: key {key!r} is an object array")
            state_dict[key] = torch.from_numpy(array.copy(order="C"))
    return state_dict


def LoadStateDict(path):
    """A chart's state dict from a `.npz` (no pickle) or a legacy `.pkl` (`pickle.load`)."""
    if IsNpz(path):
        return LoadStateDictNpz(path)
    with open(path, "rb") as f:
        return pickle.load(f)


def StateDictToNumpy(state_dict):
    """`{key: np.ndarray}` for writing; refuses anything that is not a plain tensor."""
    out = {}
    for key, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"state dict key {key!r} is {type(value).__name__}, not a tensor")
        if value.dtype == torch.bfloat16:
            raise TypeError(f"state dict key {key!r} is bfloat16, which numpy cannot hold")
        out[key] = value.detach().cpu().contiguous().numpy()
    return out


def SaveStateDictNpz(path, state_dict):
    """Write `state_dict` to `path` (refuses to overwrite). Keys keep their order."""
    if os.path.exists(path):
        raise FileExistsError(path)
    with open(path, "xb") as f:
        np.savez(f, **StateDictToNumpy(state_dict))
