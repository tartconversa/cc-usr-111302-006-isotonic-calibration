# 人工核验概率的单调校准与冻结评估

本项目保存虚构数据共享及产品运营的合成表格与 Python 读取、校准工具。所有地区、机构、产品与人数均为合成。组件只消费既有规则分数 `probability` 与人工二元核验 `label`，**不运行、不调用、不评价任何生成模型**。

## 字段与单位

既有 `probability` 为人工规则分数（有限且在 [0,1]），`label` 为人工二元核验（0/1），`weight` 为正权重，`split` 为预先给定的 fit/calibration/test 划分。同一 `group_id`（共享供应批）必须整组落在同一个 split，入口直接校验并拒绝跨 split 泄漏。测试标签不参与映射拟合。

`material-index.json` 记录全部输入内容哈希；`fixtures` 保存原始资料与配置，计算输出放 `.workspace-output`（已被 git 与 docker 忽略），原始资料不会被覆盖。

## 方法约定（与 `fixtures/calibration.json` 一致）

1. **同分数先合并**：校准子集内相同分数按权重聚合，求和使用 `math.fsum`，结果与行序无关。
2. **加权 PAVA**：在唯一分数序列上做保序回归，逆序违例的相邻块合并，块拟合概率 = 块内成功数 / 块权重，即加权 Brier 损失的最小化点；输出块边界、权重、成功数与映射。
3. **右连续阶梯查询**：取“最后一个不大于查询值的分数”对应的拟合概率；端外按首/末块常值延拓，不做线性插值，也没有其他插值分支。
4. **冻结评估**：映射只由 calibration 子集拟合一次；test 上同时报告原始分数与校准分数的加权 Brier 与 logloss。
5. **logloss 裁剪**：仅在计算瞬间按配置 `logloss_epsilon` 把概率夹到 `[epsilon, 1-epsilon]`，裁剪次数随指标记录，裁剪值绝不写回映射或导出概率。
6. **Murphy 分箱分解**：按预定 `test_bins`（左闭右开、末箱含 1）报告 reliability、resolution、uncertainty 与 Brier，并显式给出分箱近似残差。连续预测在箱内不恒定时 `Brier = REL - RES + UNC` 只近似成立，残差不为零；箱内恒定时精确成立。空箱统计记为 `null`，不冒充观测频率。
7. **按 group 整组 bootstrap，两类分列、互不混用**：
   - `frozen_map`：重采样 test 供应批，校准映射保持冻结，只衡量评估侧不确定性（受 `evaluate_frozen_map` 开关控制）；
   - `refit_training`：重采样 calibration 供应批并每次重新 PAVA 拟合，在原始未重采样的 test 行上评估，衡量训练侧不确定性。
   两者使用各自派生的 RNG 流，种子与重复次数取自配置，输出可复现。
8. `fit` split 参与分组完整性校验并随大表导出校准值，但不参与 PAVA 拟合。

校准只改变概率刻度，**不宣称校准后区分能力（如 AUC 等排序指标）必然提高**。

## 运行

Python 3.11 以上，标准库即可，不需安装第三方依赖。

    # 输入契约检查（既有读取器，仅解析）
    python -m src.calibration_reader fixtures/scores.csv

    # 完整校准与诊断
    python -m src.calibration \
        --scores fixtures/scores.csv \
        --config fixtures/calibration.json \
        --output .workspace-output

在隔离模式（`-I`，不把 cwd 注入模块路径）下运行时，显式指定顶层目录：

    python -I -m unittest discover -s tests -t . -v

输出（均为确定性文本，重复运行字节一致）：

- `.workspace-output/mapping.json`：约定、PAVA 块（边界/权重/成功数/拟合概率）与分数结点映射；
- `.workspace-output/diagnostics.json`：各 split 计数、test 与 calibration 的原始/校准 Brier、logloss 及裁剪记录、分箱分解与残差、两类 bootstrap 摘要；
- `.workspace-output/scored_records.csv`：按概率、供应批、批次排序的大表，附冻结校准概率。

## 测试与固定容器

    python -m compileall -q src tests
    python -m unittest discover -s tests -v

    docker build -t isotonic-calibration:local .
    docker run --rm isotonic-calibration:local

容器镜像以完整 digest 固定，默认运行全部测试；实现仅使用标准库，无需新增依赖。
