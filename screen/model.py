import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def unit(x):
    return F.normalize(x.float(), dim=-1, eps=1e-8)


def log_normalizer(kappa, dim):
    k = kappa.double()
    m = torch.arange(160, dtype=torch.float64, device=k.device)
    terms = m * torch.log(k.clamp_min(1e-150)[..., None].square() / 4)
    terms = (
        terms - torch.lgamma(m + 1) - torch.lgamma(m + dim / 2) + math.lgamma(dim / 2)
    )
    at_zero = math.lgamma(dim / 2) - math.log(2) - dim / 2 * math.log(math.pi)
    return (at_zero - torch.logsumexp(terms, -1)).to(kappa.dtype)


class PatchQuery(nn.Module):
    def __init__(self, semantic_dim, dim, rank, temperature):
        super().__init__()
        self.low = nn.Linear(semantic_dim, rank, bias=False)
        self.high = nn.Linear(rank, dim, bias=False)
        self.rho = nn.Linear(semantic_dim, 1)
        self.temperature = temperature
        nn.init.normal_(self.high.weight, std=0.02)

    def forward(self, patches, semantics):
        dim = patches.size(-1)
        query = self.high(self.low(semantics)).reshape(len(semantics), 1, dim)
        attention = (
            torch.einsum("cmd,bnd->bcmn", query, patches)
            / (math.sqrt(dim) * self.temperature)
        ).softmax(-1)
        evidence = torch.einsum("bcmn,bnd->bcmd", attention, patches)
        return torch.einsum("cm,bcmd->bcd", self.rho(semantics).softmax(-1), evidence)


class PatchMatch(nn.Module):
    def __init__(self, dim, semantic_dim, classes, cfg):
        super().__init__()
        self.factors = cfg["matching_factors"]
        self.dim = cfg["matching_dim"]
        self.epsilon = cfg["matching_entropy"]
        self.visual = nn.Linear(dim, self.dim, bias=False)
        self.semantic = nn.Sequential(
            nn.Linear(semantic_dim, cfg["matching_rank"], bias=False),
            nn.Linear(cfg["matching_rank"], self.factors * self.dim, bias=False),
        )
        self.register_buffer("donors", torch.arange(classes))

    def forward(self, patches, semantics, centers):
        projected = F.normalize(self.visual(F.normalize(patches, dim=-1)), dim=-1)
        offsets = F.normalize(
            self.semantic(semantics).reshape(-1, self.factors, self.dim), dim=-1
        )
        directions = F.normalize(
            F.normalize(self.visual(centers), dim=-1)[:, None] + 0.3 * offsets, dim=-1
        )
        cost = 1 - torch.einsum("bnd,cmd->bcnm", projected, directions)
        n, m = cost.shape[-2:]
        with torch.no_grad():
            plan = (F.log_softmax(-cost / self.epsilon, dim=-2) - math.log(m)).exp()
        ref = cost.new_full((n, m), 1 / (n * m))
        kl = (plan * (plan.clamp_min(1e-30).log() - ref.log()) - plan + ref).sum(
            (-2, -1)
        )
        return (plan * cost).sum((-2, -1)) + self.epsilon * kl


class SourceScorer(nn.Module):
    def __init__(self, semantic_dim, classes, cfg):
        super().__init__()
        dim = cfg["visual_dim"]
        self.dim = dim
        self.gate_bounds = cfg["gate_bounds"]
        self.matching_weight = cfg["matching_weight"]
        self.fusion = nn.Module()
        self.fusion.patch_logits = nn.Parameter(torch.zeros(4), requires_grad=False)
        self.fusion.global_logits = nn.Parameter(torch.zeros(4), requires_grad=False)
        self.query = PatchQuery(
            semantic_dim, dim, cfg["query_rank"], cfg["query_temperature"]
        )

        # Preserve the parameter order of the published training runs and checkpoints.
        self.local = nn.Module()
        self.local.center = nn.Linear(semantic_dim, dim)
        self.global_generator = nn.Linear(semantic_dim, dim)
        self.prior_gate = nn.Linear(semantic_dim, 1)
        nn.init.zeros_(self.prior_gate.weight)
        nn.init.zeros_(self.prior_gate.bias)
        self.calibrator = nn.Module()
        self.calibrator.class_head = nn.Linear(semantic_dim, 2)
        self.calibrator.sample_head = nn.Sequential(
            nn.Linear(7, 32), nn.SiLU(), nn.Linear(32, 1)
        )
        nn.init.zeros_(self.calibrator.class_head.weight)
        nn.init.zeros_(self.calibrator.class_head.bias)
        nn.init.zeros_(self.calibrator.sample_head[-1].weight)
        nn.init.constant_(
            self.calibrator.sample_head[-1].bias, math.log(math.expm1(1e-4))
        )
        self.calibrator.requires_grad_(False)
        self.register_buffer("visual_mean", torch.zeros(dim))
        self.register_buffer("visual_matrix", torch.eye(dim))
        self.local.center = nn.Linear(semantic_dim, dim)
        # The original constructor initialized a mixture before fixing its sole weight to one.
        nn.Linear(semantic_dim, 1)
        self.local.concentration = nn.Linear(semantic_dim, 1)
        nn.init.zeros_(self.local.concentration.weight)
        nn.init.constant_(self.local.concentration.bias, 30.0)
        self.factor_match = PatchMatch(dim, semantic_dim, classes, cfg)

    def transform(self, x):
        return (x - self.visual_mean) @ self.visual_matrix

    def forward(self, patches, global_tokens, semantics, ids):
        patches = self.transform(patches)
        weights = global_tokens.new_full((4,), 0.25)
        glob = self.transform(
            torch.einsum("l,bld->bd", weights, F.layer_norm(global_tokens, (self.dim,)))
        )
        a = unit(semantics[ids])
        visual = self.query(patches, a)
        center = unit(self.local.center(a))
        means = unit(center[:, None])
        cosine = torch.einsum("bcd,ckd->bck", unit(visual), means)
        kappa = F.softplus(self.local.concentration(a)).clamp(1e-4, 100)
        logc0 = (
            math.lgamma(self.dim / 2) - math.log(2) - self.dim / 2 * math.log(math.pi)
        )
        local = (
            torch.logsumexp(
                log_normalizer(kappa, self.dim)[None] + kappa[None] * cosine, dim=-1
            )
            - logc0
        ) / 30
        global_score = unit(glob) @ unit(self.global_generator(a)).T
        low, high = self.gate_bounds
        gate = low + (high - low) * self.prior_gate(a).sigmoid().squeeze(-1)
        base = gate[None] * global_score + (1 - gate[None]) * local
        factor_sem = F.normalize(semantics[self.factor_match.donors[ids]], dim=-1)
        cost = self.factor_match(
            patches, factor_sem, F.normalize(self.local.center(factor_sem), dim=-1)
        )
        return base - self.matching_weight * cost


def patch_pairs(tokens):
    side = math.isqrt(tokens)
    if side * side != tokens:
        raise ValueError("Expected a square patch grid")
    pairs = [(i, i + 1) for i in range(tokens) if i % side != side - 1]
    pairs += [(i, i + side) for i in range(tokens - side)]
    rng = np.random.default_rng(2026100630)
    pairs += [
        (int(i), int(j)) for i, j in rng.integers(0, tokens, (tokens, 2)) if i != j
    ]
    return torch.tensor(sorted(set(pairs)), dtype=torch.long).T.contiguous()


class Refinement(nn.Module):
    def __init__(self, semantic_dim, cfg):
        super().__init__()
        self.temperature = cfg["pool_temperature"]
        self.bound = cfg["residual_bound"]
        dim = cfg["refinement_dim"]
        self.visual = nn.Linear(cfg["visual_dim"], dim, bias=False)
        self.semantic = nn.Sequential(
            nn.Linear(semantic_dim, cfg["refinement_rank"], bias=False),
            nn.GELU(),
            nn.Linear(cfg["refinement_rank"], dim, bias=False),
        )
        self.coefficients = nn.Parameter(torch.zeros(3))
        self.register_buffer("pairs", patch_pairs(cfg["patch_tokens"]))

    def forward(self, patches, semantics, base, edge_weights=None):
        raw = F.normalize(patches.detach().float(), dim=-1)
        projected = F.normalize(self.visual(raw), dim=-1)
        queries = F.normalize(
            self.semantic(F.normalize(semantics.detach().float(), dim=-1)), dim=-1
        )
        sim = torch.einsum("bnd,cd->bnc", projected, queries)
        energy = self.temperature * (
            torch.logsumexp(sim / self.temperature, dim=1) - math.log(sim.shape[1])
        )
        a, b, c = self.coefficients
        refined = base.detach() + self.bound * torch.tanh(
            a * energy + b * base.detach() + c
        )
        if edge_weights is None:
            return refined, None
        i, j = self.pairs
        error = (
            (projected[:, i] * projected[:, j]).sum(-1)
            - (raw[:, i] * raw[:, j]).sum(-1)
        ).square()
        return refined, ((error * edge_weights).sum(1) / edge_weights.sum(1)).mean()
