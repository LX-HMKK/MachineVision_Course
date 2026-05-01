import csv
import json
from argparse import Namespace
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch

import numpy as np

from main import (
    fit_predict_svm,
    make_dense_keypoints,
    parse_dense_sizes,
    prediction_distribution_summary,
    save_training_log,
)


class MainFeatureTests(unittest.TestCase):
    def make_workspace_temp_dir(self, name: str) -> Path:
        path = Path.cwd() / name
        shutil.rmtree(path, ignore_errors=True)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def test_parse_dense_sizes_accepts_single_default_scale(self):
        self.assertEqual(parse_dense_sizes("16"), [16])

    def test_parse_dense_sizes_accepts_integer_default_scale(self):
        self.assertEqual(parse_dense_sizes(16), [16])

    def test_parse_dense_sizes_accepts_comma_separated_values(self):
        self.assertEqual(parse_dense_sizes("12,16,24,32"), [12, 16, 24, 32])

    def test_parse_dense_sizes_rejects_invalid_values(self):
        for value in ("12,abc", "0,16", "-1,16"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    parse_dense_sizes(value)

    def test_make_dense_keypoints_expands_each_grid_point_for_each_size(self):
        single_scale = make_dense_keypoints((20, 20), step=10, sizes=[16])
        multi_scale = make_dense_keypoints((20, 20), step=10, sizes=[12, 16, 24])

        self.assertEqual(len(single_scale), 4)
        self.assertEqual(len(multi_scale), len(single_scale) * 3)
        self.assertEqual(sorted({int(kp.size) for kp in multi_scale}), [12, 16, 24])

    def test_fit_predict_svm_chi2_uses_precomputed_kernel(self):
        calls = {}

        class FakeSVC:
            def __init__(self, **kwargs):
                calls["kwargs"] = kwargs

            def fit(self, x_train, y_train):
                calls["train_shape"] = x_train.shape
                calls["labels"] = y_train.tolist()
                return self

            def predict(self, x_test):
                calls["test_shape"] = x_test.shape
                return np.array([1])

        with patch("main.SVC", FakeSVC):
            x_train = np.array([[0.2, 0.8], [0.9, 0.1]], dtype=np.float32)
            y_train = [0, 1]
            x_test = np.array([[0.8, 0.2]], dtype=np.float32)

            y_pred = fit_predict_svm(
                x_train,
                y_train,
                x_test,
                svm_kernel="chi2",
                svm_c=1.0,
                chi2_gamma=0.5,
                random_state=42,
            )

        self.assertEqual(y_pred.tolist(), [1])
        self.assertEqual(calls["kwargs"], {"C": 1.0, "kernel": "precomputed"})
        self.assertEqual(calls["train_shape"], (2, 2))
        self.assertEqual(calls["test_shape"], (1, 2))
        self.assertEqual(calls["labels"], [0, 1])

    def test_prediction_distribution_summary_reports_majority_class(self):
        y_pred = np.array([2, 2, 1, 2])
        summary = prediction_distribution_summary(y_pred, ["00", "01", "02"])

        self.assertEqual(summary["most_predicted_class"], "02")
        self.assertEqual(summary["most_predicted_ratio"], 0.75)

    def test_save_training_log_keeps_created_at_in_json_not_csv(self):
        args = Namespace(
            data_dir=Path("data"),
            output_dir=Path("result"),
            log_dir=Path("logs"),
            device="cuda",
            train_per_class=150,
            num_clusters=100,
            max_descriptors_per_image=80,
            sift_normalization="rootsift",
            descriptor_selection="random",
            power_normalize=True,
            sift_sampling="hybrid",
            dense_step=4,
            svm_c=1.0,
            svm_kernel="chi2",
            chi2_gamma=1.0,
            random_state=42,
        )
        meta = {
            "dense_sizes": [16],
            "train_images": 2,
            "test_images": 1,
            "requested_device": "cuda",
            "actual_device": "cpu",
            "device_note": "test",
            "accuracy": 0.5,
            "feature_dim": 10,
            "most_predicted_class": "02",
            "most_predicted_ratio": 0.75,
        }
        report_dict = {
            "macro avg": {"recall": 0.3, "f1-score": 0.4},
            "weighted avg": {"f1-score": 0.5},
            "00": {"f1-score": 0.6},
        }

        tmp_dir = self.make_workspace_temp_dir("_tmp_test_log_meta")
        try:
            log_path = save_training_log(tmp_dir, args, meta, report_dict, ["00"])
            log_data = json.loads(log_path.read_text(encoding="utf-8"))
            with (tmp_dir / "runs.csv").open("r", newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        self.assertIn("created_at", log_data)
        self.assertNotIn("created_at", rows[0])
        self.assertIn("run_id", rows[0])
        self.assertEqual(rows[0]["train_per_class"], "150")
        self.assertEqual(rows[0]["random_state"], "42")
        self.assertEqual(rows[0]["macro_recall"], "0.3")
        self.assertEqual(rows[0]["most_predicted_class"], "02")
        self.assertEqual(rows[0]["most_predicted_ratio"], "0.75")

    def test_save_training_log_sorts_runs_csv_by_accuracy_descending(self):
        args = Namespace(
            data_dir=Path("data"),
            output_dir=Path("result"),
            log_dir=Path("logs"),
            device="cuda",
            train_per_class=150,
            num_clusters=100,
            max_descriptors_per_image=80,
            sift_normalization="rootsift",
            descriptor_selection="random",
            power_normalize=True,
            sift_sampling="hybrid",
            dense_step=4,
            svm_c=1.0,
            svm_kernel="linear",
            chi2_gamma=1.0,
            random_state=42,
        )
        better_meta = {
            "dense_sizes": [16],
            "train_images": 2,
            "test_images": 1,
            "requested_device": "cuda",
            "actual_device": "cpu",
            "device_note": "test",
            "accuracy": 0.9,
            "feature_dim": 10,
            "most_predicted_class": "00",
            "most_predicted_ratio": 0.5,
        }
        worse_meta = {
            "dense_sizes": [16],
            "train_images": 2,
            "test_images": 1,
            "requested_device": "cuda",
            "actual_device": "cpu",
            "device_note": "test",
            "accuracy": 0.6,
            "feature_dim": 10,
            "most_predicted_class": "01",
            "most_predicted_ratio": 0.6,
        }
        report_dict = {
            "macro avg": {"recall": 0.3, "f1-score": 0.4},
            "weighted avg": {"f1-score": 0.5},
            "00": {"f1-score": 0.6},
        }

        tmp_dir = self.make_workspace_temp_dir("_tmp_test_log_sort")
        try:
            log_dir = tmp_dir
            save_training_log(log_dir, args, worse_meta, report_dict, ["00"])
            save_training_log(log_dir, args, better_meta, report_dict, ["00"])

            with (log_dir / "runs.csv").open("r", newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        self.assertGreaterEqual(float(rows[0]["accuracy"]), float(rows[1]["accuracy"]))
        self.assertEqual(rows[0]["accuracy"], "0.9")
        self.assertEqual(rows[1]["accuracy"], "0.6")
        self.assertEqual(rows[0]["train_per_class"], "150")
        self.assertEqual(rows[0]["random_state"], "42")


if __name__ == "__main__":
    unittest.main()
