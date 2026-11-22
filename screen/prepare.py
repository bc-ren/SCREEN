"""Extract the frozen visual features in official benchmark row order."""

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from scipy.io import loadmat
from torch.nn import functional as F

from .config import CONFIGS, DATASETS, load_config
from .data import configure_device, read_metadata

BACKBONE_SHA = "fd4e1169c7aa6c2dbfa8a6448be13b35abc0ee256190857c90009d12c094619b"


def names(values):
    return [str(value.item()) for value in values.ravel()]


def metadata(dataset, split_dir):
    split = loadmat(split_dir / "att_splits.mat")
    rows = loadmat(split_dir / "res101.mat", variable_names=["labels", "image_files"])
    semantic = split
    if dataset == "CUB":
        semantic = loadmat(split_dir / "sent_splits.mat")
        if names(split["allclasses_names"]) != names(semantic["allclasses_names"]):
            raise ValueError("CUB sentence descriptors have a different class order")
        for key in (
            "train_loc",
            "val_loc",
            "trainval_loc",
            "test_seen_loc",
            "test_unseen_loc",
        ):
            np.testing.assert_array_equal(split[key], semantic[key])
    attrs = semantic["att"].astype(np.float64).T.copy()
    if dataset != "CUB":
        attrs[split["original_att"].T < 0] = 0
    dim = load_config(dataset, "Best-H")["semantic_dim"]
    if attrs.shape != (len(names(split["allclasses_names"])), dim):
        raise ValueError(f"{dataset} expects {dim} descriptor dimensions")
    labels = rows["labels"].ravel().astype(np.int64) - 1
    contract = json.loads((CONFIGS / "splits.json").read_text())[dataset]
    trainval = split["trainval_loc"].ravel().astype(np.int64)
    train = trainval[~np.isin(trainval, contract["excluded_trainval_loc"])] - 1
    seen_test = split["test_seen_loc"].ravel().astype(np.int64) - 1
    unseen_test = split["test_unseen_loc"].ravel().astype(np.int64) - 1
    m = {
        "labels": labels,
        "attrs": attrs,
        "train": train,
        "test_seen": seen_test,
        "test_unseen": unseen_test,
        "seen": np.unique(labels[train]),
        "unseen": np.unique(labels[unseen_test]),
    }
    ids = b"".join(
        m[key].astype("<i8").tobytes()
        for key in ("train", "test_seen", "test_unseen", "labels")
    )
    if hashlib.sha256(ids).hexdigest() != contract["order_sha256"]:
        raise ValueError("Image order, labels or splits differ from the benchmark")
    if (
        hashlib.sha256(attrs.astype("<f8").tobytes()).hexdigest()
        != contract["descriptors_sha256"]
    ):
        raise ValueError("Class descriptors differ from the benchmark")
    return m, names(rows["image_files"])


def image_path(root, recorded):
    # The MATLAB release stores absolute paths from the dataset authors' machines.
    parts = Path(recorded.replace(chr(92), "/")).parts
    if ".." in parts:
        raise ValueError(f"Invalid image path {recorded}")
    for start in range(len(parts)):
        if parts[start] in ("/", ".") or parts[start].endswith(":"):
            continue
        candidate = root.joinpath(*parts[start:])
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Cannot find {recorded} under {root}")


def encoder(path, device):
    from transformers import ViTModel

    with (path / "model.safetensors").open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    if digest != BACKBONE_SHA:
        raise ValueError("Encoder weights differ from the benchmark ViT-B/16")
    model, info = ViTModel.from_pretrained(
        path,
        local_files_only=True,
        add_pooling_layer=True,
        output_loading_info=True,
        attn_implementation="eager",
    )
    if any(
        info.get(k)
        for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    ):
        raise ValueError("Incomplete encoder state")
    architecture = (
        model.config.image_size,
        model.config.patch_size,
        model.config.hidden_size,
        model.config.num_hidden_layers,
    )
    if architecture != (224, 16, 768, 12):
        raise ValueError("Unexpected ViT architecture")
    return model.float().eval().requires_grad_(False).to(device)


@torch.inference_mode()
def extract(model, paths, indices, images, target, role, batch_size, device):
    from PIL import Image
    from torchvision import transforms as T

    transform = T.Compose(
        [
            T.Resize(439, interpolation=T.InterpolationMode.BILINEAR),
            T.CenterCrop(384),
            T.ToTensor(),
            T.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )
    destination = target / role
    if destination.exists():
        raise FileExistsError(destination)
    with TemporaryDirectory(prefix=f".{role}-", dir=target) as tmp:
        stage = Path(tmp) / role
        stage.mkdir()
        arrays = {}
        for offset in range(0, len(indices), batch_size):
            ids = indices[offset : offset + batch_size]
            pixels = []
            for i in ids:
                with Image.open(image_path(images, paths[i])) as image:
                    pixels.append(transform(image.convert("RGB")))
            hidden = model(
                torch.stack(pixels).to(device),
                output_hidden_states=True,
                interpolate_pos_encoding=True,
            ).hidden_states
            tokens = torch.stack(
                [model.layernorm(hidden[i]) for i in (9, 10, 11, 12)], dim=1
            )
            p, g = tokens[:, :, 1:], tokens[:, :, 0]
            patch = F.layer_norm(p, (768,))
            weights = patch.new_full((4,), 0.25)
            batch = {
                "source_patch": torch.einsum("l,blnd->bnd", weights, patch),
                "residual_patch": F.normalize(patch.mean(1), dim=-1),
                "global_tokens": g,
            }
            if role == "train":
                batch.update(
                    layer_patch_ln_mean=patch.mean(2),
                    layer_global_ln=F.layer_norm(g, (768,)),
                )
            for key, value in batch.items():
                if not torch.isfinite(value).all():
                    raise FloatingPointError(key)
                if key not in arrays:
                    arrays[key] = np.lib.format.open_memmap(
                        stage / f"{key}.npy",
                        mode="w+",
                        dtype=np.float32,
                        shape=(len(indices), *value.shape[1:]),
                    )
                arrays[key][offset : offset + len(ids)] = value.cpu().numpy()
        for array in arrays.values():
            array.flush()
        np.save(stage / "indices.npy", indices)
        stage.rename(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=DATASETS)
    for name in ("splits", "images", "backbone", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--role", choices=("train", "test"), required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch size must be positive")
    configure_device()
    m, paths = metadata(args.dataset, args.splits)
    args.out.mkdir(parents=True, exist_ok=True)
    filename = args.out / "metadata.npz"
    if filename.exists():
        previous = read_metadata(args.out)
        for key in m:
            np.testing.assert_array_equal(previous[key], m[key])
    else:
        with TemporaryDirectory(dir=args.out) as tmp:
            name = Path(tmp) / "metadata.npz"
            np.savez_compressed(name, **m)
            name.replace(filename)
    read_metadata(args.out)
    ids = (
        m["train"] if args.role == "train" else np.r_[m["test_seen"], m["test_unseen"]]
    )
    extract(
        encoder(args.backbone, args.device),
        paths,
        ids,
        args.images,
        args.out,
        args.role,
        args.batch_size,
        args.device,
    )


if __name__ == "__main__":
    main()
