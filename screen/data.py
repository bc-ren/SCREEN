import os
from pathlib import Path

import numpy as np
import torch


def configure_device():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def read_metadata(root):
    with np.load(Path(root) / "metadata.npz", allow_pickle=False) as z:
        meta = dict(z)
    labels, attrs = meta["labels"], meta["attrs"]
    if (
        labels.ndim != 1
        or attrs.ndim != 2
        or not np.isfinite(attrs).all()
        or not np.issubdtype(labels.dtype, np.integer)
        or not np.array_equal(np.unique(labels), np.arange(len(attrs)))
    ):
        raise ValueError("Invalid class labels or descriptor table")
    roles = [meta[k] for k in ("train", "test_seen", "test_unseen")]
    ids = np.concatenate(roles)
    if (
        min(map(len, roles)) == 0
        or ids.ndim != 1
        or not np.issubdtype(ids.dtype, np.integer)
        or ids.min() < 0
        or ids.max() >= len(labels)
        or len(ids) != len(np.unique(ids))
    ):
        raise ValueError("Image splits contain invalid or overlapping rows")
    for rows, classes in zip(roles, ("seen", "seen", "unseen")):
        if not np.array_equal(np.unique(labels[rows]), meta[classes]):
            raise ValueError("Class list does not match the image split")
    if np.intersect1d(meta["seen"], meta["unseen"]).size or not np.array_equal(
        np.union1d(meta["seen"], meta["unseen"]), np.arange(len(attrs))
    ):
        raise ValueError("Invalid seen/unseen partition")
    return meta


class FeatureBank:
    def __init__(self, root, role, meta):
        if role not in ("train", "test"):
            raise ValueError("Unknown feature bank")
        directory = Path(root) / role
        if not (directory / "indices.npy").exists():
            directory = directory / "0"
        ids = np.load(directory / "indices.npy", allow_pickle=False)
        allowed = (
            meta["train"]
            if role == "train"
            else np.r_[meta["test_seen"], meta["test_unseen"]]
        )
        if not np.array_equal(np.sort(ids), np.sort(allowed)):
            raise ValueError(f"Feature rows disagree with the {role} split")
        self.rows = np.full(len(meta["labels"]), -1, dtype=np.int64)
        self.rows[ids] = np.arange(len(ids))
        names = ["source_patch", "residual_patch", "global_tokens"]
        if role == "train":
            names.extend(["layer_patch_ln_mean", "layer_global_ln"])
        self.arrays = {
            key: np.load(directory / f"{key}.npy", mmap_mode="r", allow_pickle=False)
            for key in names
        }
        for key, array in self.arrays.items():
            if array.dtype != np.float32 or len(array) != len(ids):
                raise ValueError(f"Invalid cached array {key}")

    def take(self, name, image_ids, device):
        ids = np.asarray(image_ids)
        if (
            ids.ndim != 1
            or not np.issubdtype(ids.dtype, np.integer)
            or np.any((ids < 0) | (ids >= len(self.rows)))
        ):
            raise ValueError("Invalid image IDs")
        pos = self.rows[ids]
        if np.any(pos < 0):
            raise ValueError("Requested images fall outside this feature bank")
        return torch.from_numpy(self.arrays[name][pos]).to(device)
