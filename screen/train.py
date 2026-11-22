import json
import math
import random
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from scipy.linalg import eigh
from torch.nn import functional as F

from .data import FeatureBank, read_metadata
from .inference import fit_reference, source_scores
from .model import Refinement, SourceScorer, unit


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ridge(x, targets, alpha):
    xc = x - x.mean(0)
    yc = targets - targets.mean(0)
    n, dim = x.shape
    if n <= dim:
        values, vectors = eigh(xc @ xc.T / n)
        rhs = vectors.T @ yc / n
    else:
        values, vectors = eigh(xc.T @ xc / n)
        rhs = vectors.T @ (xc.T @ yc / n)
    weights = vectors @ (rhs / (np.maximum(values, 0)[:, None] + alpha))
    if n <= dim:
        weights = xc.T @ weights
    return weights.T, targets.mean(0) - x.mean(0) @ weights


@torch.no_grad()
def initialize_heads(model, bank, fit, labels, semantic, cfg):
    device = semantic.device
    classes = np.unique(labels)
    x = F.normalize(semantic[classes], dim=-1).double().cpu().numpy()
    for key, head in (
        ("layer_patch_ln_mean", model.local.center),
        ("layer_global_ln", model.global_generator),
    ):
        chunks = []
        for start in range(0, len(fit), 256):
            layers = bank.take(key, fit[start : start + 256], device)
            visual = torch.einsum("l,bld->bd", layers.new_full((4,), 0.25), layers)
            chunks.append(F.normalize(visual, dim=-1).double().cpu().numpy())
        images = np.concatenate(chunks)
        centers = np.stack([images[labels == cls].mean(0) for cls in classes])
        weight, bias = ridge(x, centers, cfg["ridge_alpha"])
        head.weight.copy_(torch.from_numpy(weight))
        head.bias.copy_(torch.from_numpy(bias))


def optimize(model, cfg, stage, n, seed, loss_for_batch):
    epochs, batch = cfg["epochs"], cfg[stage + "_batch"]
    initial_lr, decay = cfg[stage + "_lr"], cfg[stage + "_ema"]
    ratio = cfg["lr_min_ratio"]
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params, lr=initial_lr, weight_decay=cfg["weight_decay"]
    )
    warmup = int(0.2 * epochs * math.ceil(n / batch))
    rng = np.random.default_rng(seed)
    average = []
    step = 0
    for epoch in range(epochs):
        factor = (
            ratio + (1 - ratio) * (1 + math.cos(math.pi * epoch / (epochs - 1))) / 2
        )
        for group in optimizer.param_groups:
            group["lr"] = initial_lr * factor
        order = rng.permutation(n)
        model.train()
        for offset in range(0, n, batch):
            loss = loss_for_batch(order[offset : offset + batch])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite {stage} loss at epoch {epoch + 1}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0, error_if_nonfinite=True)
            optimizer.step()
            step += 1
            if decay and step > warmup:
                with torch.no_grad():
                    if not average:
                        average = [p.detach().clone() for p in params]
                    else:
                        for mean, p in zip(average, params):
                            mean.lerp_(p.detach(), 1 - decay)
        print(f"{stage} {epoch + 1}/{epochs}", flush=True)
    with torch.no_grad():
        for p, mean in zip(params, average):
            p.copy_(mean)
    model.eval().requires_grad_(False)


def fit_source(bank, fit, seen, labels, semantic, targets, class_weights, cfg):
    device = semantic.device
    seed_all(cfg["seed"])
    model = SourceScorer(semantic.shape[1], len(semantic), cfg).to(device)
    initialize_heads(model, bank, fit, labels, semantic, cfg)
    anchors = [
        (p, p.detach().clone())
        for head in (model.local.center, model.global_generator)
        for p in head.parameters()
    ]
    descriptors = semantic[seen]

    def loss_for_batch(pos):
        rows = fit[pos]
        target = targets[pos]
        with torch.no_grad():
            layers = bank.take("layer_global_ln", rows, device)
            g = torch.einsum("l,bld->bd", layers.new_full((4,), 0.25), layers)
            prior = (
                F.normalize(g, dim=-1)
                @ F.normalize(
                    model.global_generator(F.normalize(descriptors, dim=-1)), dim=-1
                ).T
            )
            hard = prior.topk(min(cfg["candidate_budget"], len(seen)), dim=1).indices
            positives = target.unique()
            near = (
                (unit(descriptors[positives]) @ unit(descriptors).T)
                .topk(min(3, len(seen)), dim=1)
                .indices
            )
            candidates = torch.cat([positives, hard.flatten(), near.flatten()]).unique(
                sorted=True
            )
        out = model(
            bank.take("source_patch", rows, device),
            bank.take("global_tokens", rows, device),
            semantic,
            seen[candidates],
        )
        ce = F.cross_entropy(
            out / cfg["base_ce_temperature"],
            torch.searchsorted(candidates, target),
            reduction="none",
        )
        penalty = sum(
            (p - a).square().mean() / a.square().mean().clamp_min(1e-6)
            for p, a in anchors
        ) / len(anchors)
        return (ce * class_weights[target]).mean() + cfg["anchor_weight"] * penalty

    optimize(model, cfg, "base", len(fit), cfg["seed"], loss_for_batch)
    return model


@torch.no_grad()
def refinement_targets(source, bank, fit, seen, semantic, targets):
    device = semantic.device
    a = F.normalize(semantic[seen], dim=-1)
    centers = F.normalize(source.local.center(a), dim=-1)
    neighbor_sem = F.normalize(a, dim=-1)
    dist = neighbor_sem @ neighbor_sem.T
    dist.fill_diagonal_(-float("inf"))
    near = torch.argsort(dist, dim=-1, descending=True, stable=True)[
        :, : min(5, len(seen) - 1)
    ]
    base, margins = [], []
    for start in range(0, len(fit), 32):
        rows = fit[start : start + 32]
        p = bank.take("source_patch", rows, device)
        g = bank.take("global_tokens", rows, device)
        base.append(source_scores(source, p, g, semantic, seen).cpu())
        sims = torch.einsum(
            "bnd,cd->bnc", F.normalize(source.transform(p), dim=-1), centers
        )
        y = targets[start : start + len(rows)]
        true = sims.gather(2, y[:, None, None].expand(-1, sims.size(1), 1)).squeeze(2)
        others = sims.gather(2, near[y][:, None].expand(-1, sims.size(1), -1))
        margins.append(
            (
                true
                - 0.1 * (torch.logsumexp(others / 0.1, dim=2) - math.log(near.size(1)))
            ).cpu()
        )
    return torch.cat(base), torch.cat(margins)


def fit_refinement(source, bank, fit, seen, semantic, targets, class_weights, cfg):
    device = semantic.device
    seed = cfg["seed"] + cfg["refinement_seed_offset"]
    seed_all(seed)
    model = Refinement(semantic.shape[1], cfg).to(device)
    base, margins = refinement_targets(source, bank, fit, seen, semantic, targets)
    i, j = model.pairs
    descriptors = semantic[seen]
    temperature = cfg["refinement_ce_temperature"]

    def loss_for_batch(pos):
        z = base[pos].to(device)
        y = targets[pos]
        u = torch.sigmoid(margins[pos].to(device) / cfg["edge_temperature"])
        edge = 0.2 + 0.8 * torch.sqrt(u[:, i] * u[:, j])
        out, geometry = model(
            bank.take("residual_patch", fit[pos], device), descriptors, z, edge
        )
        delta = F.cross_entropy(
            out / temperature, y, reduction="none"
        ) - F.cross_entropy(z / temperature, y, reduction="none")
        return (delta * class_weights[y]).mean() + cfg["geometry_weight"] * geometry

    optimize(model, cfg, "refinement", len(fit), seed, loss_for_batch)
    return model


def train(data, output, cfg, device):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    meta = read_metadata(data)
    fit = meta["train"]
    labels = meta["labels"][fit]
    seen, targets, counts = np.unique(labels, return_inverse=True, return_counts=True)
    if len(seen) < 2 or meta["attrs"].shape[1] != cfg["semantic_dim"]:
        raise ValueError("Unexpected seen classes or semantic descriptor dimensions")
    bank = FeatureBank(data, "train", meta)
    sem = torch.as_tensor(meta["attrs"], dtype=torch.float32, device=device)
    cw = torch.tensor(
        len(fit) / (len(seen) * counts), dtype=torch.float32, device=device
    )
    seen, targets = (torch.as_tensor(x, device=device) for x in (seen, targets))
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=".screen-") as tmp:
        stage = Path(tmp) / "weights"
        stage.mkdir()
        source = fit_source(bank, fit, seen, labels, sem, targets, cw, cfg)
        refined = fit_refinement(source, bank, fit, seen, sem, targets, cw, cfg)
        reference = fit_reference(source, refined, bank, meta, sem, cfg)
        torch.save(source.state_dict(), stage / "base.pt")
        torch.save(refined.state_dict(), stage / "refinement.pt")
        np.savez_compressed(stage / "reference.npz", **reference)
        (stage / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
        stage.rename(output)
