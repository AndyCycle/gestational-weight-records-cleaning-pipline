# 可移植组件使用契约

## 运行环境与入口
Python 3.11+，portable 仅使用标准库。命令在仓库根目录运行：
```bash
python -m portable.pipeline --input examples/synthetic.csv --config examples/config.json --output runs/demo
python -m portable.pipeline --input examples/synthetic.csv --config examples/config.json --review examples/review.csv --output runs/reviewed
python -m unittest discover -s tests -v
```
每次指定新的输出目录。数据一次读入内存；大队列应在评估内存后运行，后续分块须确保同一孕次完整分组。

## 输入字段
UTF-8 或 UTF-8-BOM CSV；配置中 columns 的键是统一字段，值是实际 CSV 列名。额外源字段不进入标准输出，原文件保持只读，以 summary 中的 SHA-256 追溯。

| 统一字段 | 要求 |
|---|---|
| record_id | 全表唯一、稳定的记录标识，字符串保留前导零 |
| pregnancy_id | 一次妊娠的标识；同一女性多次妊娠必须分开 |
| gestation_day | 有限数值，单位天；负值标记，缺失/无法解析时报错 |
| weight | 数值或空白；单位通过配置声明 |
| height | 可选，数值或空白，必须为正；单位 cm 或 m |
| delivery_day | 可选，同孕次采用同一值；冲突报错，缺失不推断分娩日 |

record_id 可由来源系统、孕次标识、原始记录主键组成。禁止将跨版本 CSV 行号当作稳定主键。
没有可靠孕次映射时，先单独解决匹配问题，保存一对多冲突清单。

## 单位与步骤
units.weight 支持 kg、jin、mixed。jin 明确按每斤 0.5 kg 转换，原数值保留在 weight_raw；mixed 保持输入数值、标记 unit_unresolved，BMI 留空。mixed 输出的 weight 在人工确认前不能解释为统一 kg。

顺序为：字段验证 → 显式单位转换 → 可选人工修正 → 孕次分组与筛查 → 导出。
screen 的 weight_range、local_spike_kg、neighbor_agreement_kg 都是**复核阈值**，不代表生理诊断或经过验证的通用界值。
局部尖峰使用两侧邻居及体重差值，只提出候选；不按天间隔建模、不扫描端点、不覆盖全局或连续阶跃。
factor_two_candidate 仅表示乘二或除二后距离邻居参考值不超过 2 个数值单位，仍需来源/轨迹证据。
全程不自动改写孕前体重、不插值、不因体重大而统一除二、不合并同日记录。
exclude_postpartum=true 只对**已知分娩孕天之后**的记录设置 excluded。分娩当天保留；缺分娩信息时不执行该规则。

## 人工修正
review CSV：record_id、expected_weight、operation、new_weight、reason。
expected_weight 是同一原始输入经显式单位转换后的值，空值用空单元格。
operation 取 set_weight（new_weight 为确认后的 kg）或 exclude；reason 必填。
每个 record_id 仅接受一条最终动作。撤销用重新生成最终动作表表达；从相同原始输入重跑。
record_id 不存在、重复、旧值不符均停止，避免把历史修正套用到另一版数据。
旧 GUI 的 row_id 日志不能直接导入；先核验孕次、孕天、旧体重与来源主键，再转换成此格式。

## 输出
- records.csv：所有标准化记录，含原值、BMI、筛查标记、逻辑排除列。
- review_queue.csv：所有带标记记录，含已排除记录。人工复核后仍能触发筛查。
- changes.csv：人工动作、旧值、新值及理由。
- summary.json：输入/人工动作文件哈希、完整配置、行数、孕次数和各类标记计数。
标记之间重叠，计数不可相加当作独立错误数。下游应显式过滤 excluded，并处理缺失值和 unit_unresolved。
此入口是新队列的保守起点；历史的全局修复、day=0 仲裁、产后启发式与末期删点需要按适配说明实现，不能声称此入口复现历史结果。
