"""单调校准组件的契约与边界测试（仅标准库 unittest）。"""

import json
import math
import random
import tempfile
import unittest
from pathlib import Path

from src import calibration as cal
from src.calibration_reader import load

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_SCORES = ROOT / "fixtures" / "scores.csv"
FIXTURE_CONFIG = ROOT / "fixtures" / "calibration.json"

DEFAULT_CONFIG = {
    "map": "right_continuous_step_last_score_not_above_query",
    "outside_range": "constant_extension",
    "test_bins": [0, 0.25, 0.5, 0.75, 1],
    "bin_interval": "left_closed_right_open_last_includes_one",
    "logloss_epsilon": 1e-12,
    "bootstrap_repeats": 50,
    "seed": 11130206,
    "evaluate_frozen_map": True,
}


def row(batch_id, group_id, probability, label, split, weight=1.0, scored_at="2026-10-01"):
    return {
        "batch_id": batch_id,
        "group_id": group_id,
        "scored_at": scored_at,
        "probability": str(probability),
        "label": str(label),
        "weight": str(weight),
        "split": split,
    }


class PavaContractTests(unittest.TestCase):
    def calibration_rows(self):
        # 需求给定样例：分数 0.2/0.4/0.6，标签 0/1/0，等权。
        return [
            row("c1", "g1", 0.2, 0, "calibration"),
            row("c2", "g2", 0.4, 1, "calibration"),
            row("c3", "g3", 0.6, 0, "calibration"),
        ]

    def test_specified_pava_example(self):
        records = cal.parse_records(self.calibration_rows())
        mapping = cal.fit_isotonic(records)
        self.assertEqual(mapping.scores, (0.2, 0.4, 0.6))
        self.assertEqual(mapping.values, (0.0, 0.5, 0.5))
        # 0.4 与 0.6 被 PAVA 合并为一个块，均值 0.5。
        self.assertEqual(len(mapping.blocks), 2)
        merged = mapping.blocks[1]
        self.assertEqual(merged.score_min, 0.4)
        self.assertEqual(merged.score_max, 0.6)
        self.assertEqual(merged.weight, 2.0)
        self.assertEqual(merged.successes, 1.0)
        self.assertEqual(merged.fitted_probability, 0.5)

    def test_right_continuous_step_query(self):
        mapping = cal.fit_isotonic(cal.parse_records(self.calibration_rows()))
        # 查询 0.5：最后一个不大于 0.5 的分数是 0.4 -> 0.5，绝不线性插值。
        self.assertEqual(mapping.predict(0.5), 0.5)
        self.assertEqual(mapping.predict(0.4), 0.5)
        self.assertEqual(mapping.predict(0.6), 0.5)
        self.assertEqual(mapping.predict(0.2), 0.0)

    def test_constant_extension_outside_range(self):
        mapping = cal.fit_isotonic(cal.parse_records(self.calibration_rows()))
        self.assertEqual(mapping.predict(-1.0), 0.0)
        self.assertEqual(mapping.predict(0.0), 0.0)
        self.assertEqual(mapping.predict(1.0), 0.5)
        self.assertEqual(mapping.predict(7.0), 0.5)

    def test_single_prediction_brier_is_004(self):
        # 单个预测 0.8、标签 1：(0.8-1)^2 = 0.04。
        brier = cal.weighted_brier([0.8], [1], [1.0])
        self.assertAlmostEqual(brier, 0.04)

    def test_weighted_pava_pooled_mean(self):
        # 构造真实违例：低分桶正例率 1/2=0.5，高分桶 2/6≈0.333，必须合并。
        # 合并块均值为加权均值 (1+2)/(2+6)=0.375，而不是简单平均 (0.5+0.333)/2。
        rows = [
            row("a", "ga", 0.1, 1, "calibration", weight=1.0),
            row("b", "gb", 0.1, 0, "calibration", weight=1.0),
            row("c", "gc", 0.9, 1, "calibration", weight=2.0),
            row("d", "gd", 0.9, 0, "calibration", weight=4.0),
        ]
        mapping = cal.fit_isotonic(cal.parse_records(rows))
        self.assertEqual(len(mapping.blocks), 1)
        block = mapping.blocks[0]
        self.assertEqual(block.weight, 8.0)
        self.assertEqual(block.successes, 3.0)
        self.assertEqual(block.fitted_probability, 0.375)
        self.assertTrue(all(value == 0.375 for value in mapping.values))

    def test_equal_score_merge_order_invariant(self):
        base = [
            row("a", "ga", 0.3, 1, "calibration", weight=2.0),
            row("b", "gb", 0.3, 0, "calibration", weight=5.0),
            row("c", "gc", 0.7, 1, "calibration", weight=4.0),
            row("d", "gd", 0.7, 0, "calibration", weight=1.0),
        ]
        mapping_a = cal.fit_isotonic(cal.parse_records(base))
        mapping_b = cal.fit_isotonic(cal.parse_records(list(reversed(base))))
        self.assertEqual(mapping_a.scores, mapping_b.scores)
        self.assertEqual(mapping_a.values, mapping_b.values)
        self.assertEqual(
            [(b.score_min, b.weight, b.successes) for b in mapping_a.blocks],
            [(b.score_min, b.weight, b.successes) for b in mapping_b.blocks],
        )
        # 0.3 桶：成功 2、权重 7 -> 2/7；0.7 桶：成功 4、权重 5 -> 4/5，单调无需合并。
        self.assertAlmostEqual(mapping_a.values[0], 2.0 / 7.0)
        self.assertAlmostEqual(mapping_a.values[1], 0.8)

    def test_all_zero_and_all_one_labels(self):
        all_zero = [
            row(f"z{i}", f"gz{i}", p, 0, "calibration")
            for i, p in enumerate((0.1, 0.5, 0.9))
        ]
        mapping_zero = cal.fit_isotonic(cal.parse_records(all_zero))
        self.assertEqual(mapping_zero.values, (0.0, 0.0, 0.0))
        all_one = [
            row(f"o{i}", f"go{i}", p, 1, "calibration")
            for i, p in enumerate((0.1, 0.5, 0.9))
        ]
        mapping_one = cal.fit_isotonic(cal.parse_records(all_one))
        self.assertEqual(mapping_one.values, (1.0, 1.0, 1.0))
        # 全 0/全 1 不发生合并：三个分数块各自保留。
        self.assertEqual(len(mapping_zero.blocks), 3)
        self.assertEqual(len(mapping_one.blocks), 3)

    def test_monotonic_non_decreasing_outputs(self):
        rng = random.Random(7)
        rows = [
            row(
                f"r{i}",
                f"gr{i}",
                round(rng.random(), 6),
                rng.randrange(2),
                "calibration",
            )
            for i in range(60)
        ]
        mapping = cal.fit_isotonic(cal.parse_records(rows))
        self.assertEqual(mapping.values, tuple(sorted(mapping.values)))


class GroupIntegrityTests(unittest.TestCase):
    def test_group_split_conflict_raises(self):
        rows = [
            row("a", "shared", 0.2, 0, "calibration"),
            row("b", "shared", 0.8, 1, "test"),
        ]
        with self.assertRaises(ValueError) as context:
            cal.validate_group_integrity(cal.parse_records(rows))
        self.assertIn("供应批跨 split", str(context.exception))

    def test_fixture_groups_are_split_disjoint(self):
        records = cal.parse_records(load(FIXTURE_SCORES))
        split_records = cal.validate_group_integrity(records)
        group_sets = [
            {r.group_id for r in rows} for rows in split_records.values()
        ]
        for i in range(len(group_sets)):
            for j in range(i + 1, len(group_sets)):
                self.assertTrue(group_sets[i].isdisjoint(group_sets[j]))


class InputValidationTests(unittest.TestCase):
    def test_probability_must_be_finite_in_unit_interval(self):
        for bad in ("-0.1", "1.1", "nan", "inf"):
            with self.subTest(bad=bad):
                rows = [row("a", "g", bad, 0, "calibration")]
                with self.assertRaises(ValueError):
                    cal.parse_records(rows)

    def test_blank_value_is_rejected_not_treated_as_zero(self):
        rows = [row("a", "g", 0.5, 0, "calibration")]
        rows[0]["weight"] = ""
        with self.assertRaises(ValueError):
            cal.parse_records(rows)

    def test_label_must_be_binary(self):
        for bad in ("2", "-1", "0.5"):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    cal.parse_records([row("a", "g", 0.5, bad, "calibration")])

    def test_weight_must_be_positive(self):
        for bad in ("0", "-2"):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    cal.parse_records([row("a", "g", 0.5, 1, "calibration", weight=bad)])

    def test_duplicate_batch_id_rejected(self):
        with self.assertRaises(ValueError):
            cal.parse_records([
                row("same", "g1", 0.2, 0, "calibration"),
                row("same", "g2", 0.8, 1, "test"),
            ])

    def test_empty_calibration_cannot_fit(self):
        with self.assertRaises(ValueError):
            cal.fit_isotonic([])

    def test_unsupported_map_convention_rejected(self):
        bad_config = dict(DEFAULT_CONFIG, map="linear_interpolation")
        with self.assertRaises(ValueError):
            cal.validate_config(bad_config)


class TestIsolationTests(unittest.TestCase):
    def build(self, test_label):
        cal_rows = [
            row("c1", "g1", 0.2, 0, "calibration"),
            row("c2", "g2", 0.4, 1, "calibration"),
            row("c3", "g3", 0.6, 0, "calibration"),
        ]
        test_rows = [row("t1", "g4", 0.5, test_label, "test")]
        return cal_rows + test_rows

    def test_changing_test_labels_does_not_change_mapping(self):
        mapping_0 = cal.run(self.build(0), DEFAULT_CONFIG)["mapping"]
        mapping_1 = cal.run(self.build(1), DEFAULT_CONFIG)["mapping"]
        self.assertEqual(
            json.dumps(mapping_0, sort_keys=True),
            json.dumps(mapping_1, sort_keys=True),
        )

    def test_empty_test_is_reported_not_crashed(self):
        rows = [
            row("c1", "g1", 0.2, 0, "calibration"),
            row("c2", "g2", 0.4, 1, "calibration"),
        ]
        result = cal.run(rows, DEFAULT_CONFIG)
        self.assertIsNone(result["diagnostics"]["test_diagnostics"])
        self.assertIn("test 子集为空", result["diagnostics"]["bootstrap"]["skipped_reason"])
        self.assertIsNone(result["diagnostics"]["bootstrap"]["frozen_map"])
        self.assertIsNone(result["diagnostics"]["bootstrap"]["refit_training"])


class MetricAndDecompositionTests(unittest.TestCase):
    def test_logloss_clipping_recorded_but_not_written_back(self):
        mapping = cal.CalibrationMap(
            scores=(0.0, 1.0), values=(0.0, 1.0), blocks=(), n_records=0, n_groups=0
        )
        # 概率正好为 0/1 时 logloss 才需要 epsilon 裁剪。
        value, record = cal.weighted_logloss([0.0, 1.0], [0, 1], [1.0, 1.0], 1e-12)
        self.assertTrue(math.isfinite(value))
        self.assertEqual(record["clipped_below_epsilon"], 1)
        self.assertEqual(record["clipped_above_one_minus_epsilon"], 1)
        self.assertFalse(record["clipped_values_written_back"])
        self.assertEqual(mapping.values, (0.0, 1.0))

    def test_decomposition_exact_when_predictions_constant_per_bin(self):
        # 预测只取 0.2 与 0.8，各自完整落入一个箱 -> 恒等式精确成立。
        edges = [0, 0.5, 1]
        probabilities = [0.2, 0.2, 0.8, 0.8]
        labels = [0, 1, 0, 1]
        weights = [1.0, 1.0, 1.0, 1.0]
        report = cal.brier_decomposition(probabilities, labels, weights, edges)
        reconstructed = (
            report["reliability"] - report["resolution"] + report["uncertainty"]
        )
        self.assertAlmostEqual(report["brier"], reconstructed, places=15)
        self.assertAlmostEqual(report["residual_brier_minus_three_terms"], 0.0, places=15)
        for bucket in report["bins"]:
            self.assertAlmostEqual(bucket["residual"], 0.0, places=15)

    def test_decomposition_residual_nonzero_within_bin_variation(self):
        # 同一箱内预测不恒定 -> 三项之和与真实 Brier 存在显式残差。
        edges = [0, 0.5, 1]
        probabilities = [0.1, 0.4, 0.6, 0.9]
        labels = [0, 1, 1, 0]
        weights = [1.0, 1.0, 1.0, 1.0]
        report = cal.brier_decomposition(probabilities, labels, weights, edges)
        self.assertNotAlmostEqual(
            report["residual_brier_minus_three_terms"], 0.0, places=12
        )
        # 残差定义就是 Brier 与三项之和的差。
        self.assertAlmostEqual(
            report["residual_brier_minus_three_terms"],
            report["brier"]
            - (
                report["reliability"]
                - report["resolution"]
                + report["uncertainty"]
            ),
            places=15,
        )

    def test_last_bin_includes_one(self):
        edges = [0, 0.5, 1]
        self.assertEqual(cal._bin_index(1.0, edges), 1)
        self.assertEqual(cal._bin_index(0.5, edges), 1)
        self.assertEqual(cal._bin_index(0.0, edges), 0)
        self.assertEqual(cal._bin_index(0.4999, edges), 0)


class BootstrapTests(unittest.TestCase):
    def _rows(self):
        rng = random.Random(99)
        rows = []
        # 每个 group 包含两行，整组必须一起进出重采样。
        for i in range(8):
            split = "calibration" if i < 5 else "test"
            for k in range(2):
                rows.append(
                    row(
                        f"b{i}_{k}",
                        f"grp{i}",
                        round(rng.random(), 4),
                        rng.randrange(2),
                        split,
                    )
                )
        return rows

    def test_frozen_and_refit_bootstrap_are_separate(self):
        result = cal.run(self._rows(), DEFAULT_CONFIG)
        bootstrap = result["diagnostics"]["bootstrap"]
        frozen = bootstrap["frozen_map"]
        refit = bootstrap["refit_training"]
        self.assertFalse(frozen["refit_calibrator"])
        self.assertTrue(refit["refit_calibrator"])
        self.assertEqual(frozen["resampled_population"], "test_groups")
        self.assertEqual(refit["resampled_population"], "calibration_groups")
        self.assertEqual(frozen["n_groups"], 3)
        self.assertEqual(refit["n_groups"], 5)
        self.assertEqual(frozen["brier"]["repeats"], DEFAULT_CONFIG["bootstrap_repeats"])
        for section in (frozen, refit):
            for metric in ("brier", "logloss"):
                percentiles = section[metric]["percentiles"]
                self.assertEqual(set(percentiles), {"0.025", "0.5", "0.975"})
                self.assertLessEqual(
                    percentiles["0.025"], percentiles["0.5"], percentiles
                )
                self.assertLessEqual(
                    percentiles["0.5"], percentiles["0.975"], percentiles
                )

    def test_bootstrap_resamples_whole_groups(self):
        records = cal.parse_records(self._rows())
        test_records = [r for r in records if r.split == "test"]
        mapping = cal.fit_isotonic([r for r in records if r.split == "calibration"])
        rng = random.Random(123)
        # 直接观察单次重采样：被抽中的 group 的两行必然同时出现。
        grouped = cal._group(test_records)
        group_ids = sorted(grouped)
        picked = group_ids[rng.randrange(len(group_ids))]
        sample_batch_ids = {r.batch_id for r in grouped[picked]}
        self.assertEqual(len(sample_batch_ids), 2)

    def test_bootstrap_deterministic_with_seed(self):
        rows = self._rows()
        first = cal.run(rows, DEFAULT_CONFIG)["diagnostics"]["bootstrap"]
        second = cal.run(rows, DEFAULT_CONFIG)["diagnostics"]["bootstrap"]
        self.assertEqual(
            json.dumps(first, sort_keys=True), json.dumps(second, sort_keys=True)
        )


class EndToEndTests(unittest.TestCase):
    def test_fixture_pipeline_and_outputs(self):
        result = cal.run(load(FIXTURE_SCORES), json.loads(FIXTURE_CONFIG.read_text()))
        mapping = result["mapping"]
        # fixture 的 calibration 子集即需求给定 PAVA 样例。
        self.assertEqual(mapping["knots"]["fitted_probabilities"], [0.0, 0.5, 0.5])
        diagnostics = result["diagnostics"]
        test = diagnostics["test_diagnostics"]
        # test: (0.3,0) -> 映射 0；(0.5,1) -> 映射 0.5。
        self.assertAlmostEqual(test["calibrated"]["brier"], (0.0 + 0.25) / 2)
        self.assertAlmostEqual(test["raw"]["brier"], (0.09 + 0.25) / 2)
        self.assertIn("reliability", test["calibrated_brier_decomposition"])
        self.assertIn("resolution", test["calibrated_brier_decomposition"])
        self.assertIn("uncertainty", test["calibrated_brier_decomposition"])

        with tempfile.TemporaryDirectory() as directory:
            paths = cal.write_outputs(result, Path(directory))
            for path in paths.values():
                self.assertTrue(path.exists())
            table_path = paths["scored_records"]
            with table_path.open(encoding="utf-8") as stream:
                table_rows = list(cal.csv.DictReader(stream))
            self.assertEqual(len(table_rows), 6)
            # 大表按概率升序排列。
            table_probabilities = [float(r["probability"]) for r in table_rows]
            self.assertEqual(table_probabilities, sorted(table_probabilities))
            # 同一次运行重复落盘字节一致（确定性输出）。
            first_bytes = {name: path.read_bytes() for name, path in paths.items()}
            cal.write_outputs(result, Path(directory) / "again")
            for name, path in cal.write_outputs(result, Path(directory) / "again").items():
                self.assertEqual(path.read_bytes(), first_bytes[name])

    def test_cli_main(self):
        with tempfile.TemporaryDirectory() as directory:
            exit_code = cal.main(
                [
                    "--scores",
                    str(FIXTURE_SCORES),
                    "--config",
                    str(FIXTURE_CONFIG),
                    "--output",
                    directory,
                ]
            )
            self.assertEqual(exit_code, 0)
            for name in ("mapping.json", "diagnostics.json", "scored_records.csv"):
                self.assertTrue((Path(directory) / name).exists())


if __name__ == "__main__":
    unittest.main()
