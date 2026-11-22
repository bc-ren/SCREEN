import json
from pathlib import Path

import numpy as np
import torch

from .data import FeatureBank, read_metadata
from .model import Refinement, SourceScorer
from .scoring import blend_scores, metrics, predict, reference_ids, reference_level


def source_scores(model, patches, globals_, semantic, classes):
    return torch.cat(
        [model(patches, globals_, semantic, group) for group in classes.split(64)],
        dim=1,
    )


@torch.inference_mode()
def raw_scores(source, refinement, bank, rows, semantic, classes, batch_size=32):
    device = semantic.device
    ids = torch.as_tensor(classes, dtype=torch.long, device=device)
    base_scores, refined_scores = [], []
    source.eval()
    refinement.eval()
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        z = source_scores(
            source,
            bank.take("source_patch", batch, device),
            bank.take("global_tokens", batch, device),
            semantic,
            ids,
        )
        refined, _ = refinement(
            bank.take("residual_patch", batch, device), semantic[ids], z
        )
        base_scores.append(z.double().cpu().numpy())
        refined_scores.append(refined.double().cpu().numpy())
    return np.concatenate(base_scores), np.concatenate(refined_scores)


def fit_reference(source, refinement, train_bank, meta, semantic, cfg):
    rows = reference_ids(
        meta["train"], meta["labels"], cfg["seed"], cfg["references_per_class"]
    )
    classes = np.arange(len(semantic))
    base, refined = raw_scores(source, refinement, train_bank, rows, semantic, classes)
    levels = reference_level(
        blend_scores(base, refined, cfg["blend"]),
        meta["labels"][rows],
        cfg["reference_temperature"],
    )
    return {"levels": levels, "classes": classes, "seen": meta["seen"], "rows": rows}


def evaluate(data, weights, cfg, device):
    weights = Path(weights)
    if json.loads((weights / "config.json").read_text()) != cfg:
        raise ValueError(
            "Checkpoint and requested profile have different configurations"
        )
    meta = read_metadata(data)
    sem = torch.as_tensor(meta["attrs"], dtype=torch.float32, device=device)
    if sem.shape[1] != cfg["semantic_dim"]:
        raise ValueError("Unexpected class descriptor dimensions")
    source = SourceScorer(sem.shape[1], len(sem), cfg).to(device)
    refinement = Refinement(sem.shape[1], cfg).to(device)
    for model, name in ((source, "base.pt"), (refinement, "refinement.pt")):
        model.load_state_dict(
            torch.load(weights / name, map_location=device, weights_only=True)
        )
    with np.load(weights / "reference.npz", allow_pickle=False) as reference:
        classes, levels = reference["classes"], reference["levels"]
        np.testing.assert_array_equal(classes, np.arange(len(sem)))
        np.testing.assert_array_equal(reference["seen"], meta["seen"])
        np.testing.assert_array_equal(
            reference["rows"],
            reference_ids(
                meta["train"], meta["labels"], cfg["seed"], cfg["references_per_class"]
            ),
        )
        if levels.shape != classes.shape or not np.isfinite(levels).all():
            raise ValueError("Invalid reference levels")
    rows = np.r_[meta["test_seen"], meta["test_unseen"]]
    base, refined = raw_scores(
        source, refinement, FeatureBank(data, "test", meta), rows, sem, classes
    )
    scores = (
        blend_scores(base, refined, cfg["blend"]) - cfg["reference_weight"] * levels
    )
    gzsl, czsl = predict(scores, classes, meta["seen"], cfg["gamma"])
    return metrics(meta["labels"][rows], gzsl, czsl, meta["seen"], meta["unseen"])
