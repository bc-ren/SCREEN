import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from screen.config import load_config
from screen.inference import evaluate
from screen.scoring import metrics, predict, reference_ids, reference_level
from screen.train import train


class ScoringTests(unittest.TestCase):
    def test_paper_profiles(self):
        for ds, same in (("AWA2", True), ("CUB", False), ("SUN", False)):
            self.assertEqual(
                load_config(ds, "Best-H") == load_config(ds, "Best-CZSL"), same
            )
        self.assertEqual(load_config("SUN", "Best-CZSL")["references_per_class"], 2)
        self.assertEqual(load_config("CUB", "Best-CZSL")["blend"], 0.5)

    def test_class_balanced_reference(self):
        z = np.array([[1.0, 5.0], [3.0, 2.0], [0.0, 4.0]])
        labels = np.array([0, 0, 1])
        t = 0.7
        expected = t * np.log(np.exp(z / t).T @ np.array([0.25, 0.25, 0.5]))
        np.testing.assert_allclose(reference_level(z, labels, t), expected)
        np.testing.assert_allclose(
            reference_level(z + 1000, labels, t), expected + 1000
        )
        rows = np.arange(60)
        y = rows % 3
        large = reference_ids(rows, y, 10, 8)
        small = reference_ids(rows, y, 10, 2)
        np.testing.assert_array_equal(reference_ids(rows, y, 10, 8), large)
        self.assertTrue(np.isin(small, large).all())
        np.testing.assert_array_equal(np.bincount(y[small]), [2, 2, 2])

    def test_candidate_sets_and_ties(self):
        scores = np.array([[2.0, 1.0, 0.0], [0.0, 2.0, 3.0], [0.0, 3.0, 0.0]])
        prediction, conventional = predict(scores, [0, 1, 2], [0, 2], 1.0)
        np.testing.assert_array_equal(prediction, [0, 1, 1])
        np.testing.assert_array_equal(conventional, [1, 1, 1])
        out = metrics(np.array([0, 2, 1]), prediction, conventional, [0, 2], [1])
        self.assertEqual(out["S"], 0.5)
        self.assertAlmostEqual(out["H"], 2 / 3)


class PipelineTests(unittest.TestCase):
    def test_training_and_evaluation_use_separate_banks(self):
        torch.set_num_threads(1)
        rng = np.random.default_rng(11)
        cfg = dict(
            load_config("CUB", "Best-H"),
            visual_dim=12,
            semantic_dim=6,
            patch_tokens=4,
            query_rank=3,
            matching_factors=2,
            matching_dim=4,
            matching_rank=3,
            refinement_dim=5,
            refinement_rank=3,
            epochs=2,
            base_batch=4,
            refinement_batch=4,
        )
        y = np.repeat(np.arange(5), 4)
        fit = np.array([0, 1, 2, 4, 5, 6, 8, 9, 10])
        test_seen = np.array([3, 7, 11])
        test_unseen = np.arange(12, 20)
        test = np.r_[test_seen, test_unseen]
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.savez(
                root / "metadata.npz",
                labels=y,
                attrs=rng.normal(size=(5, 6)),
                train=fit,
                test_seen=test_seen,
                test_unseen=test_unseen,
                seen=[0, 1, 2],
                unseen=[3, 4],
            )

            def bank(role, rows):
                folder = root / role
                folder.mkdir()
                np.save(folder / "indices.npy", rows)
                sizes = {
                    "source_patch": (4, 12),
                    "residual_patch": (4, 12),
                    "global_tokens": (4, 12),
                }
                if role == "train":
                    sizes.update(layer_patch_ln_mean=(4, 12), layer_global_ln=(4, 12))
                for key, shape in sizes.items():
                    np.save(
                        folder / (key + ".npy"),
                        rng.normal(size=(len(rows), *shape)).astype("float32"),
                    )

            bank("train", fit)
            out = root / "weights"
            train(root, out, cfg, "cpu")
            with np.load(out / "reference.npz") as ref:
                self.assertTrue(np.isin(ref["rows"], fit).all())
                self.assertEqual(ref["levels"].shape, (5,))
            bank("test", test)
            result = evaluate(root, out, cfg, "cpu")
            self.assertEqual(set(result), {"S", "U", "H", "CZSL"})
            self.assertTrue(all(0 <= value <= 1 for value in result.values()))
            self.assertEqual(result, evaluate(root, out, cfg, "cpu"))
            with self.assertRaises(ValueError):
                evaluate(root, out, dict(cfg, gamma=cfg["gamma"] + 0.1), "cpu")


if __name__ == "__main__":
    unittest.main()
