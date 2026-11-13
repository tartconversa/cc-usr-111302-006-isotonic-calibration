"""人工核验概率的单调（保序）校准组件。

输入只有既有规则分数 ``probability`` 与人工二元核验 ``label``（外加正权重与
预定 split），不调用、不运行、不评价任何生成模型。

实现约定（与 ``fixtures/calibration.json`` 一致）：

* 同供应批 ``group_id`` 必须整组落在 fit/calibration/test 中的同一个集合，
  组件在入口校验，杜绝同供应批跨集合泄漏；
* 校准映射只使用 ``calibration`` 子集拟合，``test`` 标签永不参与拟合；
* 同分数先按权重合并（求和使用 :func:`math.fsum`，与行序无关），再以 PAVA
  最小化加权 Brier 损失，块拟合概率为块内加权正例率（成功数/权重）；
* 查询约定为右连续阶梯：取“最后一个不大于查询值的分数”所对应的拟合概率，
  端外按首/末块常值延拓，不做任何线性插值；
* logloss 仅在计算瞬间按配置 epsilon 裁剪并记录裁剪次数，裁剪值绝不写回
  概率或映射。

仅使用 Python 标准库。
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import random
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from src.calibration_reader import load

REQUIRED_COLUMNS = (
    "batch_id",
    "group_id",
    "scored_at",
    "probability",
    "label",
    "weight",
    "split",
)
KNOWN_SPLITS = ("fit", "calibration", "test")
CALIBRATION_SPLIT = "calibration"
TEST_SPLIT = "test"

MAP_CONVENTION = "right_continuous_step_last_score_not_above_query"
OUTSIDE_CONVENTION = "constant_extension"
BIN_INTERVAL = "left_closed_right_open_last_includes_one"

PERCENTILES = (0.025, 0.5, 0.975)


# --------------------------------------------------------------------------- #
# 输入解析与校验
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Record:
    """一条核验记录。probability 为冻结的既有规则分数。"""

    batch_id: str
    group_id: str
    scored_at: str
    probability: float
    label: int
    weight: float
    split: str


def _fail(reason: str, *, index: int | None = None, batch_id: str | None = None):
    location = ""
    if index is not None:
        location = f"第 {index} 行"
    if batch_id is not None:
        location = f"{location}（batch_id={batch_id}）" if location else f"batch_id={batch_id}"
    head = f"{location}: " if location else ""
    raise ValueError(f"输入校验失败，{head}{reason}")


def parse_records(rows: list[dict]) -> list[Record]:
    """把读取器产出的字符串字典列表解析并校验为 :class:`Record`。"""
    if not isinstance(rows, list) or not rows:
        _fail("记录为空，无法校准")
    records: list[Record] = []
    seen_batch: set[str] = set()
    for index, raw in enumerate(rows):
        missing = [column for column in REQUIRED_COLUMNS if column not in raw]
        if missing:
            _fail(f"缺少列 {missing}", index=index)
        blank = [column for column in REQUIRED_COLUMNS if str(raw[column]).strip() == ""]
        if blank:
            _fail(f"列 {blank} 存在空值，空值不得默默当作 0", index=index)

        batch_id = str(raw["batch_id"]).strip()
        group_id = str(raw["group_id"]).strip()
        scored_at = str(raw["scored_at"]).strip()
        if batch_id in seen_batch:
            _fail("batch_id 重复", index=index, batch_id=batch_id)
        seen_batch.add(batch_id)

        try:
            date.fromisoformat(scored_at)
        except ValueError:
            _fail("scored_at 不是 ISO 日期", index=index, batch_id=batch_id)

        try:
            probability = float(raw["probability"])
        except (TypeError, ValueError):
            _fail("probability 不是数值", index=index, batch_id=batch_id)
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            _fail("probability 必须有限且落在 [0,1]", index=index, batch_id=batch_id)

        try:
            label = int(str(raw["label"]).strip())
        except (TypeError, ValueError):
            _fail("label 必须是二元整数 0/1", index=index, batch_id=batch_id)
        if label not in (0, 1):
            _fail("label 必须是二元整数 0/1", index=index, batch_id=batch_id)

        try:
            weight = float(raw["weight"])
        except (TypeError, ValueError):
            _fail("weight 不是数值", index=index, batch_id=batch_id)
        if not math.isfinite(weight) or weight <= 0.0:
            _fail("weight 必须为有限正数", index=index, batch_id=batch_id)

        split = str(raw["split"]).strip()
        if split not in KNOWN_SPLITS:
            _fail(f"split 必须是 {KNOWN_SPLITS} 之一", index=index, batch_id=batch_id)

        records.append(
            Record(batch_id, group_id, scored_at, probability, label, weight, split)
        )
    return records


def validate_group_integrity(records: list[Record]) -> dict[str, list[Record]]:
    """每个 group_id 只能属于一个预定 split；返回按 split 分组的记录。"""
    group_splits: dict[str, set[str]] = {}
    split_records: dict[str, list[Record]] = {split: [] for split in KNOWN_SPLITS}
    for record in records:
        group_splits.setdefault(record.group_id, set()).add(record.split)
        split_records[record.split].append(record)
    conflicts = {
        group: sorted(splits) for group, splits in group_splits.items() if len(splits) > 1
    }
    if conflicts:
        _fail(f"供应批跨 split 泄漏，冲突 group_id={conflicts}")
    return split_records


# --------------------------------------------------------------------------- #
# PAVA 保序校准
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Block:
    """PAVA 输出块：分数边界、总权重、成功数与拟合概率。"""

    score_min: float
    score_max: float
    weight: float
    successes: float

    @property
    def fitted_probability(self) -> float:
        return self.successes / self.weight


@dataclass(frozen=True)
class CalibrationMap:
    """冻结的单调校准映射。

    ``scores`` 为校准集出现过的全部唯一分数（升序），``values`` 为各自所属
    PAVA 块的拟合概率。映射一经拟合即为不可变数据，bootstrap 重拟合会生成
    新对象，从不就地修改。
    """

    scores: tuple[float, ...]
    values: tuple[float, ...]
    blocks: tuple[Block, ...]
    n_records: int
    n_groups: int

    def predict(self, query: float) -> float:
        """右连续阶梯查询 + 端外常值延拓。"""
        if not self.scores:
            raise ValueError("校准映射为空，无法查询")
        # 最后一个 score <= query 的下标；query 低于最小分数时回退到首块值。
        index = bisect.bisect_right(self.scores, query) - 1
        if index < 0:
            return self.values[0]
        return self.values[index]

    def predict_many(self, queries):
        return [self.predict(query) for query in queries]

    def to_dict(self) -> dict:
        return {
            "conventions": {
                "map": MAP_CONVENTION,
                "outside_range": OUTSIDE_CONVENTION,
                "interpolation": "none_step_function_only",
            },
            "n_calibration_records": self.n_records,
            "n_calibration_groups": self.n_groups,
            "knots": {
                "scores": list(self.scores),
                "fitted_probabilities": list(self.values),
            },
            "blocks": [
                {
                    "score_min": block.score_min,
                    "score_max": block.score_max,
                    "weight": block.weight,
                    "successes": block.successes,
                    "fitted_probability": block.fitted_probability,
                }
                for block in self.blocks
            ],
        }


def _aggregate_equal_scores(records: list[Record]) -> list[tuple[float, float, float]]:
    """同分数先合并。

    每个分数桶收集权重与加权成功数列表，再用 :func:`math.fsum` 求和，
    因此结果与输入行的排列顺序无关（fsum 按精确部分和给出正确舍入值）。
    """
    buckets: dict[float, dict[str, list[float]]] = {}
    for record in records:
        bucket = buckets.setdefault(record.probability, {"weights": [], "successes": []})
        bucket["weights"].append(record.weight)
        bucket["successes"].append(record.weight * record.label)
    return [
        (score, math.fsum(bucket["weights"]), math.fsum(bucket["successes"]))
        for score, bucket in sorted(buckets.items())
    ]


def fit_isotonic(calibration_records: list[Record]) -> CalibrationMap:
    """在 calibration 子集上以加权 PAVA 拟合单调阶梯映射。"""
    if not calibration_records:
        raise ValueError("calibration 子集为空，无法拟合校准映射")
    groups = {record.group_id for record in calibration_records}
    aggregated = _aggregate_equal_scores(calibration_records)

    # 每个元素是 [score_min, score_max, weight, successes]。
    blocks: list[list[float]] = [
        [score, score, weight, successes] for score, weight, successes in aggregated
    ]
    # PAVA：逆序违例（前一块均值 > 后一块均值）则合并相邻块，直至单调不减。
    merged: list[list[float]] = []
    for block in blocks:
        merged.append(block)
        while len(merged) >= 2:
            left, right = merged[-2], merged[-1]
            if (left[3] / left[2]) <= (right[3] / right[2]):
                break
            merged.pop()
            merged.pop()
            merged.append(
                [
                    min(left[0], right[0]),
                    max(left[1], right[1]),
                    left[2] + right[2],
                    left[3] + right[3],
                ]
            )

    block_objects = tuple(
        Block(score_min=row[0], score_max=row[1], weight=row[2], successes=row[3])
        for row in merged
    )
    # 每个唯一分数映射到所属块的拟合概率（块在分数轴上互不重叠且升序）。
    value_by_score: dict[float, float] = {}
    for block in block_objects:
        fitted = block.fitted_probability
        for score, _, _ in aggregated:
            if block.score_min <= score <= block.score_max:
                value_by_score[score] = fitted
    scores = tuple(score for score, _, _ in aggregated)
    values = tuple(value_by_score[score] for score in scores)
    return CalibrationMap(
        scores=scores,
        values=values,
        blocks=block_objects,
        n_records=len(calibration_records),
        n_groups=len(groups),
    )


# --------------------------------------------------------------------------- #
# 指标与 Murphy 分箱分解
# --------------------------------------------------------------------------- #
def weighted_brier(probabilities, labels, weights) -> float:
    total_weight = math.fsum(weights)
    return (
        math.fsum(
            weight * (probability - label) ** 2
            for probability, label, weight in zip(probabilities, labels, weights)
        )
        / total_weight
    )


def weighted_logloss(probabilities, labels, weights, epsilon: float):
    """加权 logloss；裁剪只发生在本函数的局部变量上，并记录裁剪次数。"""
    lower, upper = epsilon, 1.0 - epsilon
    clipped_below = clipped_above = 0
    total_weight = math.fsum(weights)
    total = 0.0
    for probability, label, weight in zip(probabilities, labels, weights):
        clipped = probability
        if probability < lower:
            clipped = lower
            clipped_below += 1
        elif probability > upper:
            clipped = upper
            clipped_above += 1
        total += -weight * (
            label * math.log(clipped) + (1 - label) * math.log(1.0 - clipped)
        )
    return total / total_weight, {
        "epsilon": epsilon,
        "clipped_below_epsilon": clipped_below,
        "clipped_above_one_minus_epsilon": clipped_above,
        "clipped_values_written_back": False,
    }


def _bin_index(probability: float, edges: list[float]) -> int:
    """左闭右开、最后一个箱包含 1.0。"""
    index = bisect.bisect_right(edges, probability) - 1
    return max(0, min(index, len(edges) - 2))


def brier_decomposition(probabilities, labels, weights, edges: list[float]) -> dict:
    """预定分箱下的 Murphy 分解：Brier = reliability - resolution + uncertainty。

    连续预测在箱内不恒定时，恒等式只近似成立；``residual`` 显式给出
    Brier 与三项之和的差额（含箱内方差与箱内协方差贡献）。
    """
    n_bins = len(edges) - 1
    total_weight = math.fsum(weights)
    total_success = math.fsum(weight * label for label, weight in zip(labels, weights))
    base_rate = total_success / total_weight
    uncertainty = base_rate * (1.0 - base_rate)

    bins = [
        {
            "edge_left": edges[i],
            "edge_right": edges[i + 1],
            "right_edge_closed": i == n_bins - 1,
            "weight": 0.0,
            "successes": 0.0,
            "sum_weighted_p": 0.0,
            "sum_weighted_p2": 0.0,
            "sum_weighted_py": 0.0,
            "brier_numerator": 0.0,
            "count": 0,
        }
        for i in range(n_bins)
    ]
    for probability, label, weight in zip(probabilities, labels, weights):
        bucket = bins[_bin_index(probability, edges)]
        bucket["count"] += 1
        bucket["weight"] += weight
        bucket["successes"] += weight * label
        bucket["sum_weighted_p"] += weight * probability
        bucket["sum_weighted_p2"] += weight * probability * probability
        bucket["sum_weighted_py"] += weight * probability * label
        bucket["brier_numerator"] += weight * (probability - label) ** 2

    reliability = resolution = brier = 0.0
    bin_reports = []
    for bucket in bins:
        share = bucket["weight"] / total_weight
        brier_term = bucket["brier_numerator"] / total_weight
        if bucket["weight"] == 0.0:
            # 空箱不携带任何信息：各分解项为 0，箱内统计显式记为 None，
            # 而不是悄悄填 0 冒充观测频率。
            bin_reports.append(
                {
                    "edge_left": bucket["edge_left"],
                    "edge_right": bucket["edge_right"],
                    "right_edge_closed": bucket["right_edge_closed"],
                    "count": 0,
                    "weight": 0.0,
                    "weight_share": 0.0,
                    "mean_prediction": None,
                    "observed_frequency": None,
                    "reliability_term": 0.0,
                    "resolution_term": 0.0,
                    "uncertainty_share_term": 0.0,
                    "brier_term": 0.0,
                    "residual": 0.0,
                }
            )
            continue
        mean_prediction = bucket["sum_weighted_p"] / bucket["weight"]
        observed_frequency = bucket["successes"] / bucket["weight"]
        rel_term = share * (mean_prediction - observed_frequency) ** 2
        res_term = share * (observed_frequency - base_rate) ** 2
        unc_term = share * uncertainty
        # 箱内残差：该箱 Brier 贡献与该箱三项贡献之差。
        bin_residual = brier_term - (rel_term - res_term + unc_term)
        reliability += rel_term
        resolution += res_term
        brier += brier_term
        bin_reports.append(
            {
                "edge_left": bucket["edge_left"],
                "edge_right": bucket["edge_right"],
                "right_edge_closed": bucket["right_edge_closed"],
                "count": bucket["count"],
                "weight": bucket["weight"],
                "weight_share": share,
                "mean_prediction": mean_prediction,
                "observed_frequency": observed_frequency,
                "reliability_term": rel_term,
                "resolution_term": res_term,
                "uncertainty_share_term": unc_term,
                "brier_term": brier_term,
                "residual": bin_residual,
            }
        )

    residual = brier - (reliability - resolution + uncertainty)
    return {
        "bin_interval": BIN_INTERVAL,
        "edges": list(edges),
        "base_rate": base_rate,
        "reliability": reliability,
        "resolution": resolution,
        "uncertainty": uncertainty,
        "brier": brier,
        "residual_brier_minus_three_terms": residual,
        "identity_exact_only_when_predictions_constant_in_bin": True,
        "bins": bin_reports,
    }


# --------------------------------------------------------------------------- #
# 按 group 整组 bootstrap
# --------------------------------------------------------------------------- #
def _group(records: list[Record]) -> dict[str, list[Record]]:
    grouped: dict[str, list[Record]] = {}
    for record in records:
        grouped.setdefault(record.group_id, []).append(record)
    return grouped


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("空样本无法计算分位数")
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def _summarize(values: list[float]) -> dict:
    ordered = sorted(values)
    return {
        "repeats": len(values),
        "mean": math.fsum(values) / len(values),
        "percentiles": {
            str(q): _percentile(ordered, q) for q in PERCENTILES
        },
    }


def _metrics_on_rows(records: list[Record], mapping: CalibrationMap, epsilon: float):
    probabilities = mapping.predict_many(record.probability for record in records)
    labels = [record.label for record in records]
    weights = [record.weight for record in records]
    brier = weighted_brier(probabilities, labels, weights)
    logloss, clip_record = weighted_logloss(probabilities, labels, weights, epsilon)
    return brier, logloss, clip_record


def bootstrap_frozen_map(
    test_records: list[Record],
    mapping: CalibrationMap,
    repeats: int,
    epsilon: float,
    rng: random.Random,
) -> dict:
    """重采样 test 供应批，冻结映射始终不变——只评估评估侧不确定性。"""
    grouped = _group(test_records)
    group_ids = sorted(grouped)
    n_groups = len(group_ids)
    brier_values: list[float] = []
    logloss_values: list[float] = []
    for _ in range(repeats):
        sampled_rows: list[Record] = []
        for _ in range(n_groups):
            sampled_rows.extend(grouped[group_ids[rng.randrange(n_groups)]])
        brier, logloss, _ = _metrics_on_rows(sampled_rows, mapping, epsilon)
        brier_values.append(brier)
        logloss_values.append(logloss)
    return {
        "purpose": "评估冻结校准映射在 test 供应批重采样下的不确定性",
        "resampled_population": "test_groups",
        "refit_calibrator": False,
        "resampling_unit": "group_id",
        "n_groups": n_groups,
        "brier": _summarize(brier_values),
        "logloss": _summarize(logloss_values),
    }


def bootstrap_refit_training(
    calibration_records: list[Record],
    test_records: list[Record],
    repeats: int,
    epsilon: float,
    rng: random.Random,
) -> dict:
    """重采样 calibration 供应批并每次重新 PAVA 拟合——训练侧不确定性。

    评估始终在原始、未重采样的冻结 test 行上进行，与冻结映射 bootstrap
    严格分开，互不混用。
    """
    grouped = _group(calibration_records)
    group_ids = sorted(grouped)
    n_groups = len(group_ids)
    brier_values: list[float] = []
    logloss_values: list[float] = []
    for _ in range(repeats):
        sampled_rows: list[Record] = []
        for _ in range(n_groups):
            sampled_rows.extend(grouped[group_ids[rng.randrange(n_groups)]])
        refitted = fit_isotonic(sampled_rows)
        brier, logloss, _ = _metrics_on_rows(test_records, refitted, epsilon)
        brier_values.append(brier)
        logloss_values.append(logloss)
    return {
        "purpose": "每次重采样 calibration 供应批后重新拟合校准器的训练不确定性",
        "resampled_population": "calibration_groups",
        "refit_calibrator": True,
        "resampling_unit": "group_id",
        "n_groups": n_groups,
        "evaluated_on": "original_untouched_test_records",
        "brier": _summarize(brier_values),
        "logloss": _summarize(logloss_values),
    }


# --------------------------------------------------------------------------- #
# 配置校验与整体流程
# --------------------------------------------------------------------------- #
def validate_config(config: dict) -> dict:
    if config.get("map") != MAP_CONVENTION:
        raise ValueError(f"配置 map 约定不受支持: {config.get('map')!r}")
    if config.get("outside_range") != OUTSIDE_CONVENTION:
        raise ValueError(f"配置 outside_range 约定不受支持: {config.get('outside_range')!r}")
    if config.get("bin_interval") != BIN_INTERVAL:
        raise ValueError(f"配置 bin_interval 约定不受支持: {config.get('bin_interval')!r}")
    edges = config.get("test_bins")
    if (
        not isinstance(edges, list)
        or len(edges) < 2
        or edges[0] != 0
        or edges[-1] != 1
        or any(edges[i] >= edges[i + 1] for i in range(len(edges) - 1))
    ):
        raise ValueError("test_bins 必须是从 0 严格递增到 1 的边界列表")
    epsilon = config.get("logloss_epsilon")
    if not isinstance(epsilon, (int, float)) or not 0.0 < float(epsilon) < 0.5:
        raise ValueError("logloss_epsilon 必须落在 (0, 0.5)")
    repeats = config.get("bootstrap_repeats")
    if not isinstance(repeats, int) or repeats < 1:
        raise ValueError("bootstrap_repeats 必须是正整数")
    seed = config.get("seed")
    if not isinstance(seed, int):
        raise ValueError("seed 必须是整数")
    if not isinstance(config.get("evaluate_frozen_map"), bool):
        raise ValueError("evaluate_frozen_map 必须是布尔值")
    return {
        "map": config["map"],
        "outside_range": config["outside_range"],
        "bin_interval": config["bin_interval"],
        "test_bins": [float(edge) for edge in edges],
        "logloss_epsilon": float(epsilon),
        "bootstrap_repeats": repeats,
        "seed": seed,
        "evaluate_frozen_map": config["evaluate_frozen_map"],
    }


def _point_diagnostics(records: list[Record], mapping: CalibrationMap, epsilon: float, edges):
    if not records:
        return None
    raw_probabilities = [record.probability for record in records]
    calibrated = mapping.predict_many(raw_probabilities)
    labels = [record.label for record in records]
    weights = [record.weight for record in records]
    raw_logloss, raw_clip = weighted_logloss(raw_probabilities, labels, weights, epsilon)
    cal_logloss, cal_clip = weighted_logloss(calibrated, labels, weights, epsilon)
    return {
        "n_records": len(records),
        "n_groups": len(_group(records)),
        "raw": {
            "brier": weighted_brier(raw_probabilities, labels, weights),
            "logloss": raw_logloss,
            "logloss_clipping": raw_clip,
        },
        "calibrated": {
            "brier": weighted_brier(calibrated, labels, weights),
            "logloss": cal_logloss,
            "logloss_clipping": cal_clip,
        },
        "calibrated_brier_decomposition": brier_decomposition(
            calibrated, labels, weights, edges
        ),
        "raw_brier_decomposition": brier_decomposition(
            raw_probabilities, labels, weights, edges
        ),
    }


def run(rows: list[dict], config: dict) -> dict:
    """完整流程：校验 -> 分组完整性 -> PAVA 拟合 -> 冻结 test 诊断与 bootstrap。"""
    checked_config = validate_config(config)
    records = parse_records(rows)
    split_records = validate_group_integrity(records)
    calibration_records = split_records[CALIBRATION_SPLIT]
    test_records = split_records[TEST_SPLIT]
    mapping = fit_isotonic(calibration_records)
    epsilon = checked_config["logloss_epsilon"]
    edges = checked_config["test_bins"]
    repeats = checked_config["bootstrap_repeats"]
    seed = checked_config["seed"]

    diagnostics: dict = {
        "config": checked_config,
        "split_counts": {
            split: {
                "records": len(rows_in_split),
                "groups": len(_group(rows_in_split)),
            }
            for split, rows_in_split in split_records.items()
        },
        "notes": [
            "映射仅由 calibration 子集的二元结果与正权重经 PAVA 拟合，test 标签不参与拟合",
            "校准只改变概率刻度，不保证区分能力（AUC 等排序指标）必然提高",
            "logloss 裁剪仅发生在计算瞬间，映射与导出概率保持未裁剪值",
        ],
        "test_diagnostics": _point_diagnostics(test_records, mapping, epsilon, edges),
        "calibration_set_diagnostics": _point_diagnostics(
            calibration_records, mapping, epsilon, edges
        ),
    }

    if not test_records:
        diagnostics["bootstrap"] = {
            "frozen_map": None,
            "refit_training": None,
            "skipped_reason": "test 子集为空，bootstrap 不适用",
            "seed": seed,
            "repeats": repeats,
        }
    else:
        # 两个 bootstrap 使用各自独立的派生 RNG，互不消耗同一条随机数流。
        frozen_rng = random.Random(seed)
        refit_rng = random.Random(seed + 1)
        diagnostics["bootstrap"] = {
            "seed": seed,
            "repeats": repeats,
            "resampling_unit": "group_id",
            "frozen_map": (
                bootstrap_frozen_map(test_records, mapping, repeats, epsilon, frozen_rng)
                if checked_config["evaluate_frozen_map"]
                else None
            ),
            "refit_training": bootstrap_refit_training(
                calibration_records, test_records, repeats, epsilon, refit_rng
            ),
        }

    return {
        "mapping": mapping.to_dict(),
        "mapping_object": mapping,
        "records": records,
        "diagnostics": diagnostics,
    }


# --------------------------------------------------------------------------- #
# 输出落盘与 CLI
# --------------------------------------------------------------------------- #
def write_outputs(result: dict, output_dir: Path) -> dict[str, Path]:
    """映射、诊断与排序大表落盘；全部输出确定性、可重复复现。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    mapping_path = output_dir / "mapping.json"
    diagnostics_path = output_dir / "diagnostics.json"
    table_path = output_dir / "scored_records.csv"

    mapping_path.write_text(
        json.dumps(result["mapping"], sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    diagnostics_path.write_text(
        json.dumps(result["diagnostics"], sort_keys=True, indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )

    mapping: CalibrationMap = result["mapping_object"]
    ordered = sorted(
        result["records"],
        key=lambda record: (record.probability, record.group_id, record.batch_id),
    )
    with table_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(list(REQUIRED_COLUMNS) + ["calibrated_probability"])
        for record in ordered:
            writer.writerow(
                [
                    record.batch_id,
                    record.group_id,
                    record.scored_at,
                    record.probability,
                    record.label,
                    record.weight,
                    record.split,
                    mapping.predict(record.probability),
                ]
            )
    return {
        "mapping": mapping_path,
        "diagnostics": diagnostics_path,
        "scored_records": table_path,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="基于人工二元核验的 PAVA 单调概率校准与冻结诊断（仅标准库）"
    )
    parser.add_argument("--scores", required=True, help="规则分数与核验记录（csv/jsonl/json）")
    parser.add_argument("--config", required=True, help="校准约定配置 JSON")
    parser.add_argument("--output", default=".workspace-output", help="输出目录")
    args = parser.parse_args(argv)

    rows = load(args.scores)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    result = run(rows, config)
    paths = write_outputs(result, Path(args.output))
    summary = {
        "blocks": len(result["mapping"]["blocks"]),
        "knots": len(result["mapping"]["knots"]["scores"]),
        "outputs": {name: str(path) for name, path in paths.items()},
    }
    print(json.dumps(summary, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
