import hashlib

import numpy as np
from scipy.special import logsumexp


def blend_scores(base, refined, alpha):
    base = np.asarray(base, dtype=np.float64)
    refined = np.asarray(refined, dtype=np.float64)
    if base.ndim != 2 or base.shape != refined.shape or not base.size:
        raise ValueError("Expected matching image-by-class score matrices")
    if not np.isfinite(base).all() or not np.isfinite(refined).all():
        raise ValueError("Nonfinite scores")
    if alpha == 1:
        return refined.copy()
    if alpha == 0:
        return base.copy()
    return base + alpha * (refined - base)


def reference_ids(train, labels, seed, cap):
    """The eight-image training bank and its deterministic nested subsets."""
    train = np.asarray(train, dtype=np.int64)
    if not 1 <= cap <= 8 or len(train) != len(np.unique(train)) or not len(train):
        raise ValueError("Expected distinct training rows and 1–8 references per class")
    result = []
    for cls in np.unique(labels[train]):
        pool = train[labels[train] == cls]
        rng = np.random.default_rng(np.random.SeedSequence([seed, int(cls), 20261011]))
        selected = rng.permutation(pool)[:8]
        if cap < 8:
            ranked = sorted(
                (hashlib.sha256(f"20261019:{cls}:{i}".encode()).digest(), int(i))
                for i in selected
            )
            selected = [i for _, i in ranked[:cap]]
        result.extend(selected)
    return np.sort(np.asarray(result, dtype=np.int64))


def reference_level(scores, labels, temperature):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels)
    if (
        scores.ndim != 2
        or not scores.size
        or labels.shape != (len(scores),)
        or not np.isfinite(scores).all()
        or not np.isfinite(temperature)
        or temperature <= 0
    ):
        raise ValueError("Invalid training-reference scores, labels or temperature")
    classes, inverse, counts = np.unique(
        labels, return_inverse=True, return_counts=True
    )
    weights = 1.0 / (len(classes) * counts[inverse])
    return temperature * logsumexp(
        scores / temperature + np.log(weights)[:, None], axis=0
    )


def predict(scores, classes, seen, gamma):
    classes = np.asarray(classes)
    seen = np.asarray(seen)
    scores = np.asarray(scores, dtype=np.float64)
    mask = np.isin(classes, seen)
    if (
        scores.ndim != 2
        or classes.ndim != 1
        or scores.shape[1] != len(classes)
        or not np.array_equal(classes, np.unique(classes))
        or not mask.any()
        or mask.all()
        or not np.isin(seen, classes).all()
        or not np.isfinite(scores).all()
        or not np.isfinite(gamma)
    ):
        raise ValueError(
            "Expected finite scores and a sorted seen/unseen candidate set"
        )
    seen_scores, unseen_scores = scores[:, mask], scores[:, ~mask]
    best_seen = classes[mask][seen_scores.argmax(1)]
    best_unseen = classes[~mask][unseen_scores.argmax(1)]
    gap = seen_scores.max(1) - unseen_scores.max(1)
    seen_wins = (gap > gamma) | ((gap == gamma) & (best_seen < best_unseen))
    return np.where(seen_wins, best_seen, best_unseen), best_unseen


def metrics(labels, generalized, conventional, seen, unseen):
    labels = np.asarray(labels)
    if (
        labels.shape != np.shape(generalized)
        or labels.shape != np.shape(conventional)
        or not np.array_equal(np.unique(labels), np.union1d(seen, unseen))
        or np.intersect1d(seen, unseen).size
    ):
        raise ValueError(
            "Official evaluation requires complete labeled seen and unseen test classes"
        )

    def accuracy(predictions, classes):
        return float(np.mean([np.mean(predictions[labels == c] == c) for c in classes]))

    s, u = accuracy(generalized, seen), accuracy(generalized, unseen)
    h = 2 * s * u / (s + u) if s + u else 0.0
    return {"CZSL": accuracy(conventional, unseen), "U": u, "S": s, "H": h}
