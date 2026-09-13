# 10 题 RC 测试：2026-09-12 续跑计划

> **已完成（2026-09-12 08:49）。剩余 5 题补跑成功，完整 10 题双分支核验通过；当日两份评估、配对、导出及全批审计均完成。PID 59987 已退出，监测已暂停。最终报告见 `code/docs/deepeye_rc_probe10_20260912_continuation.md`。以下保留执行计划及历史检查点，不再启动或补跑。**

用户已明确授权网络恢复后继续未完成测试。本计划替代上一日的暂停要求，但只补齐最后一个 Schema Linking＋RC 条件；不重跑已完成的其他 7 个条件，不增加题目、repeat 或下游。

## 已完成准备

- 工作目录 `/Users/jz/Codebase/papers/2027ICASSP/code`，使用 `.venv/bin/python -E -B`。
- 10 个目标库 `SELECT current_database(), 1` 均成功；同一模型服务 GET /models 返回 200，配置模型在列表中。连通性探测没有调用对话模型。
- `_check_frozen` 已通过；源码指纹仍为 `b341c92e8d99b5f092d8e0c3911451ff93332cb7fcc76620fbf7d7131267670a`。模型、提示、RC、输入预计算、预算和 PostgreSQL 执行协议不变。
- 原停止目录、stop.json 和导出均保持不变，不直接 resume 原目录（会跳过 3 个降级成功题）。

## 新续跑目录与历史引用

批次根目录仍为 `baselines_reproduce/deepeye_bird_interact/rc_evaluation/probe10_20260911`，以下路径均相对该目录。

新 run：`runs/schema_linking-rc-r1-recovery_20260912`。**已创建，不再 prepare 或重新导入。**

该 run 保留原 RC 条件的完整 10 题 manifest，逐字一致。使用 RunStore 公共 API 复制 5 个正常成功 attempt 的完整产物及全部 API/组件事件，保留原 call_id：

- `full/cold_chain_pharma_compliance_3`
- `full/cross_border_4`
- `full/sports_events_13`
- `lite/archeology_1`
- `lite/news_11`

复制记录的入库时间是今天，不是推理时间；原始时间、源记录哈希、新旧 attempt/event 映射保存在该 run 的 `recovery.json` 和 payload 的 `continuation_provenance`，原 `source_provenance` 未变。不得把这些历史请求当成今天新增开销。

离线 `_preflight` 已确认唯一需要的阶段是 `schema_linking`，仅以下 5 题待执行：

- `full/cybermarket_pattern_3`
- `full/exchange_traded_funds_10`
- `lite/alien_9`
- `lite/credit_1`
- `lite/solar_11`

准备完成时 5 个成功 attempt、90 事件、101 条校验记录，无任何新模型请求。其他 3 个降级与 2 个中断 attempt 不导入；原始错误仍保存在旧目录，不伪造失败终态，不修改旧记录。

## 推理运行及验收

运行现有 CLI，不改生产代码：

```sh
.venv/bin/python -E -B -u -m scripts.rc_evaluation.deepeye run --run-dir baselines_reproduce/deepeye_bird_interact/rc_evaluation/probe10_20260911/runs/schema_linking-rc-r1-recovery_20260912
```

每个 run 的原配置为 10 题槽位、inner100、PG10、API1200秒；本次仅 5 个槽位有新工作。每题按原生 Direct/Reversed 各1个采样预算执行（包括原生有限重试），Value 复用预计算，不运行 Generation/Revision/Selection。不将旧 Direct 分支拼接进新半题流程，5 题均按原生目标阶段重跑；另外 5 题完全复用。

启动后仅监测本 run 的 launch.json、真实命令、日志、RunStore，不能重复启动。普通进度保持安静。进程仍存活时 interrupted 仅表示尚无终态。

正常结束后，除 10 个成功阶段和完整 API 配对、verify 外，逐题确认 Direct/Reversed 都有成功响应、组件提取成功。特别防止连接错误耗尽导致 `{}` 仍被原生标记 succeeded。如果再次出现耗尽/降级或异常退出，保留现场、停止后续步骤、暂停自动监测并报告，不自动无限 resume。

## 当日独立评估与配对

跨天重新验证 Schema Linking 两条件的参考执行，不重跑无 RC 的模型调用。只用现有 evaluator，不能把 gold 或评估结果回灌模型。

统一当日数据库版本声明：`bird-interact-working-databases-20260912-pg-native-continuation`，仍是操作者声明，非不可变快照证明。

原始参考路径保持：

- `/Users/jz/Codebase/papers/2027ICASSP/BIRD-Interact/BIRD-Interact-ADK/bird-interact-lite/bird_interact_data.jsonl`
- `/Users/jz/Codebase/papers/2027ICASSP/BIRD-Interact/BIRD-Interact-ADK/bird-interact-full/bird_interact_data.jsonl`

输出均为新目录，存在时先检查完整性，不能覆盖：

1. 无 RC 的当日评估：输入 `runs/schema_linking-none-r1` → `evaluations/schema_linking-none-r1-recheck_20260912`。可以在补跑模型期间执行，PG 只读，10 题参考执行。
2. 完整 RC 当日评估：输入新续跑 run → `evaluations/schema_linking-rc-r1-recovery_20260912`。
3. 配对：上述两份当日评估 → `comparisons/schema_linking-r1-recovery_20260912`，随后导出到它的 `export/`。
4. 新续跑推理导出：`exports/schema_linking-rc-r1-recovery_20260912`，保存 usage.json 和 recovery.json 来源说明。导出完整不代表原始停止 run 被修复。

现有 `evaluate_run`、`compare_evaluations` 可按 USAGE 调用，传同一 env_file 和声明。当前保守 coverage 对全部10题可能仍为 unknown；只报告实际可判定表覆盖和保留表/列规模，不把 null/通用 bag_equal_rate=0 解释成错误。表数包含列列表为空的表。没有下游运行，不能声称最终 SQL 准确率提升。

## 成本、完整性及交付

- `observed_usage(new_store)` 包含复制的 10 个历史请求；用 recovery.json 中 imported_attempt_id 集合过滤这些 attempt 后再计算本次新增 usage。
- 过滤可用一个只读对象提供 `iter_events(kinds=...)`，返回新 store 内非导入 attempt 的事件，再传给现有 observed_usage；无需改生产函数。
- 原停止 run 的 17 错误、2 未配对仍计为历史未知。全批成本按 call_id 去重，或原8组累计加本次新增，不把整个新run再次累加。
- 检查全部旧产物仍完整、新续跑 10 题完整、新评估和第四次配对通过后，更新 `code/docs/deepeye_rc_probe10_20260912_continuation.md`，链接上一日汇总，说明真正补跑5题与历史复用5题。
- 完成后暂停既有 `deepeye-50`，通知用户。不得扩展到其他题、其他阶段推理或新 repeat。

## 历史检查点

2026-09-12T08:17:30.703845+08:00 已启动 PID 59987，详情见续跑目录 launch.json；不要重复启动。08:18 首次在线核验为恰好5个新增attempt、10个带RC请求、4响应0错误，5个历史完整attempt继续复用。现有监测 `deepeye-50` 已恢复 ACTIVE，每5分钟一次。

无 RC 当日评估已完成：输出 `evaluations/schema_linking-none-r1-recheck_20260912`，10题成功、41条记录校验通过、10条参考均可执行；下次不重复创建、执行或覆盖。后续仅等当前PID59987结束并核验，然后完成RC评估、配对与导出汇总。

## 最终检查点

- [x] 10 题均正常完成 Direct／Reversed，最终产物与原生三分支并集一致；新增唯一连接错误由原生重试恢复，没有降级产物。
- [x] 新目录 205 条、原停止目录 247 条校验通过；历史映射和原停止快照不变。
- [x] 当日 none／RC 两份评估各 10 题完成、各 41 条校验通过；第四次配对和导出完成。
- [x] 9 个推理目录、9 个评估目录、4 个配对目录均通过校验，13 份导出完成。多出目录仅为旧现场及旧日期评估。
- [x] 去重累计 149 请求、121 响应、26 错误、2 个历史未配对，已报告 1,838,183 tokens；本次新增仅 155,112 tokens。
- [x] 汇总完成，PID 59987 已退出，既有监测已暂停。没有继续下游、增加 repeat 或扩大题集。
