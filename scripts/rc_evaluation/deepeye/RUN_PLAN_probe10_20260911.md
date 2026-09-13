# 10 题 DeepEye 分阶段 RC 测试运行计划

> **2026-09-12 已完成续跑。最后 Schema Linking＋RC 的 5 题已补齐，完整 10 题通过分支核验，四阶段配对和导出全部完成，监测已暂停。完成检查点见同目录 `CONTINUATION_probe10_20260912.md`，最终报告见 `code/docs/deepeye_rc_probe10_20260912_continuation.md`。下文保留历史；不再启动，不直接 resume 原停止目录，不按旧顺序重跑。**

2026-09-11 用户授权：实现完成后等待原始 10 题 Pipeline 跑完，然后运行 RC 测试；复用过程记录，不重跑完整 Pipeline。本文件只安排已有工具的执行，不新增实验方法。

## 范围与固定配置

- 工作目录：`/Users/jz/Codebase/papers/2027ICASSP/code`；Python：该目录的 `.venv/bin/python`。
- 来源：`baselines_reproduce/deepeye_bird_interact/runs/probe10_20260911_pg_native`。只能在原进程结束、10 个 pipeline 和全部 40 个阶段成功、RunStore 校验通过之后进入实验。
- 使用来源 manifest 的全部 10 题，不挑选错误样本，不扩大到其他题。
- 每阶段 none / rc 各一次，`repeat_id=1`，总共 8 个独立目录。none 是同一上游的目标阶段重采样对照，历史 baseline 单独保留；两者都不从头跑 Pipeline。
- 只运行目标阶段，不传 `--continue-downstream`。RC 只注入该阶段真实模型调用；源阶段无模型调用时两种条件均复用并记录未参与。
- 模型、endpoint、采样预算、tokens、原生重试、PG 协议与来源一致。使用 `code/config/.env`，不修改密钥或源码。
- 每次只启动一个实验进程。每个进程沿用 10 个独立题目槽位、inner-workers 100、PG 并发 10、单请求上限 1200 秒。不同时启动 8 个进程、不增加采样预算或重复轮次。
- 顺序：Selection none → Selection rc → Revision none → Revision rc → Generation none → Generation rc → Schema Linking none → Schema Linking rc。先检查依赖候选池的两个阶段，所有阶段保持独立，不串接不同实验的结果。
- 已有 RC：`scripts/bird_interact_lite/rc.json`、`scripts/bird_interact_full/rc.json`，只使用最终 Round2 加固定定义，不重新生成 RC。
- 生产源码指纹：`b341c92e8d99b5f092d8e0c3911451ff93332cb7fcc76620fbf7d7131267670a`。源码或配置不一致时停止，不自行修改清单放行。

## 产物位置

批次根目录：`baselines_reproduce/deepeye_bird_interact/rc_evaluation/probe10_20260911`。

每阶段名使用 `sql_selection`、`sql_revision`、`sql_generation`、`schema_linking`，条件为 `none` 或 `rc`：

- 推理：`runs/{stage}-{condition}-r1/`。
- 日志：`logs/{stage}-{condition}-r1.log`，排他创建，不覆盖。
- 进程启动记录：各 run 下 `launch.json`；只能核对命令、进程和持久化记录后认定是否仍运行，不能只看 PID。
- 评估：`evaluations/{stage}-{condition}-r1/`。
- 配对：`comparisons/{stage}-r1/`。
- 导出：`exports/{stage}-{condition}-r1/`；只能使用不存在的新路径。
- 汇总报告：`code/docs/deepeye_rc_probe10_20260911.md`，按时间追加进展、失败及结果，不抹去历史。

## 原始运行验收

- [x] 确认 PID 37845 对应进程已经结束，且日志结尾和持久化记录没有未完成题目。
- [x] 在只读一致快照中确认 10 题均有连续四阶段和 pipeline 成功终态，API 请求均配对，RunStore.verify 通过。
- [x] 核对最终 SQL 的执行事件与 Selection attempt_id、SQL 原文同时吻合。区别成功执行、结果正确与流程完成。
- [x] 汇总 SQL 反馈和 Selection 分支；导出到源 run 下新建的 `export_final`，检查 COMPLETE.json；若该路径已经存在，只核验，不覆盖。
- [x] 将验收写入 `code/docs/deepeye_pg_native_probe10_20260911.md`。记录完成后继续本计划，不再执行旧监测中的“完成即暂停、禁止 RC”规则。

## 每个目标阶段的执行步骤

1. 所有 8 个目录先完成离线 prepare；每个必须正好 10 题。命令使用来源 manifest 的输入缓存及预算，RC 条件增加两个现成 RC 路径。
2. 按上述顺序逐个显式 `run`，必须保持独立目标目录。启动前检查目标是否已完成或已有匹配进程，避免重复付费请求。
3. 运行时增量检查 request/response/error、阶段终态、RC 参与状态与记录校验。原生内部重试属于本次运行；不要因耗时长而另起重复进程。
4. 正常进程结束后要求 10 题成功并通过 verify；记录新增 usage、RC 实际请求数和未参与题数，导出。
5. 用独立 `evaluate` 执行当前运行的参考/候选 SQL。仅 evaluator 可以读取参考，不能把 gold 传回模型。
6. 同一阶段两个条件完成评估后 `compare` 并导出比较结果，再继续下一阶段。

命令接口如下；`STAGE`、`CONDITION`、`RUN_DIR` 表示上面已列出的阶段、条件及实际绝对路径，执行时替换成具体值，不新增参数。环境文件由 CLI 默认读取。

```sh
.venv/bin/python -E -B -m scripts.rc_evaluation.deepeye prepare \
  --source-run baselines_reproduce/deepeye_bird_interact/runs/probe10_20260911_pg_native \
  --run-dir RUN_DIR --target-stage STAGE --condition CONDITION --repeat-id 1 \
  --workers 10 --inner-workers 100 --pg-concurrency 10 --chat-timeout 1200

# 仅 condition=rc 时给 prepare 增加：
# --rc-lite scripts/bird_interact_lite/rc.json --rc-full scripts/bird_interact_full/rc.json

.venv/bin/python -E -B -m scripts.rc_evaluation.deepeye run --run-dir RUN_DIR
```

实际执行必须给每个启动写入独立 launch.json（PID、完整无密钥命令、开始时间、日志路径）；可使用临时 shell/进程工具启动现有 CLI，但不添加新自动重试逻辑或改生产代码。

## 独立结果评估

参考文件：

- `/Users/jz/Codebase/papers/2027ICASSP/BIRD-Interact/BIRD-Interact-ADK/bird-interact-lite/bird_interact_data.jsonl`
- `/Users/jz/Codebase/papers/2027ICASSP/BIRD-Interact/BIRD-Interact-ADK/bird-interact-full/bird_interact_data.jsonl`

统一数据库版本声明：`bird-interact-working-databases-20260911-pg-native-poc`。这是操作者对本批环境的命名，不是自动检测出的远端快照；若期间已知数据库变化，停止配对并报告，不能继续假定相同数据。

`evaluate --run-dir ... --output-dir ... --reference-lite ... --reference-full ... --database-version bird-interact-working-databases-20260911-pg-native-poc`。

`compare --left-dir evaluations/STAGE-none-r1 --right-dir evaluations/STAGE-rc-r1 --output-dir comparisons/STAGE-r1`，执行时使用正确批次根路径。

报告保留 10 题固定分母，区分缺失/失败/不可判定和明确错误。结果同时报告保留重复行但忽略行顺序的一致性，以及严格行顺序一致性；不是官方 BIRD-Interact 得分。Generation 看候选池，Revision 看逐槽位修复/误改，Selection 看最终选择与条件选中率，Schema Linking 看可判定的参考表列覆盖；没有重跑下游时不宣称最终 SQL 准确率提升。10 题仅作 POC，不作总体显著性结论。

## 停止与通知

监测沿用现有 5 分钟 heartbeat，不创建重复监测。普通进展和无变化保持安静；通知基线验收完成并启动 RC、重要失败/需用户操作、全部完成。不要重复提示同一连接错误。

原始基线异常退出、记录损坏、目标阶段失败、配置/来源不一致、身份认证或持续服务错误时，保留现场并报告，暂停后续启动。不能自动换 Key、改预算、无限 resume 或重新抽取成功题。进程崩溃后存在未知远端请求时，先报告并等待指示；已完整结束的阶段不重跑。

全部 8 次推理、8 次独立评估和 4 次配对完成后，汇总本批报告与产物路径并暂停监测。不启动额外 repeats、下游串行实验或 50/555/605 题。

## 已执行检查点 — 2026-09-11 14:05

原始基线已完成上述全部验收，export_final 完整。8 个新实验目录已全部离线 prepare（各 10 题、0 初始 attempts）；后续不要再次 prepare 这些现有目录。

当前只启动了 `runs/sql_selection-none-r1`，PID 51075，开始于 14:04:55；必须以该 run 的 launch.json、真实进程命令和 RunStore 为准。首个快照为 3 题复用成功、7 题执行中，7 个模型请求、0 错误。其余 7 个 run 仍仅准备、尚未启动。任何下一步都先检查运行状态，防止重复启动。

## 最新已执行检查点 — 2026-09-11 14:17

Selection-none 已完整完成并导出；独立评估已完成，10 题、0 未知、严格结果一致 3/10。其 7 次连接错误均由原生同参数重试恢复，不是当前持续故障。详细证据和历史提示 Schema 呈现差异写入批次报告。

当前运行 `runs/sql_selection-rc-r1`，PID 54228，启动于 14:16:30；实际 7 个请求均已确认包含 RC。下一步等待该进程结束，校验/导出/evaluate，然后 compare 两个 Selection 条件，再启动 Revision-none。其余六个 run 仍仅 prepare，不能重复 prepare 或重跑已完成 Selection-none。

exports、evaluations、comparisons 父目录现已创建。工具只创建末级输出目录；后续新路径的父目录需先存在，但末级路径不能预建或覆盖。

## 最新已执行检查点 — 2026-09-11 14:30

两个 Selection 条件已完成、导出和独立评估；配对与比较导出位于 `comparisons/sql_selection-r1/export/`。none 3/10、RC 2/10，1 题由匹配变不匹配，无改善；详细原生耗时排序造成的两题提示候选差异已记录，不更换题集或修改排序。

Revision-none（PID 56782）也已退出并完成评估/导出，1 请求成功、9 题引用，候选池匹配题数 3/10、30 槽位中匹配 6 条，修复/误改均 0。

当前唯一运行是 `runs/sql_revision-rc-r1`，PID 57411，开始于 14:29:30；9 题引用成功、1 题运行，实际 RC 请求 1 个、0 错误。下一步等待此进程结束并验收、export/evaluate、compare Revision，再依次启动 Generation-none、Generation-rc、Schema Linking-none、Schema Linking-rc。四个后续 run 都仅 prepare，不能重建或跳过。截止此检查点完成 3/8 推理、3/8 评估、1/4 配对。

## 最新已执行检查点 — 2026-09-11 14:39

Revision 的两个条件已全部完成并导出、独立评估及配对。两条件最终 SQL 逐字相同：唯一调用为 credit_1 的 OrderByNullChecker，第 3 槽被相同改写且后续原生执行成功。候选池指标保持 3/10、6/30，修复/误改均 0；仅 1 题实际参与，不能当作 10 题充分修复实验。比较导出 `comparisons/sql_revision-r1/export/` 完整。

当前唯一进程为 `runs/sql_generation-none-r1`，PID 59573，开始于 14:38:15。10 题并行 Generation，首个快照 30 请求、0 错误；没有重跑 Schema Linking、预计算或下游。下一步等该进程完整结束、验收/export/evaluate，再启动已准备的 Generation-rc；完成其配对后继续两个 Schema Linking 条件。截止此处完成 4/8 推理、4/8 评估、2/4 配对。其余三个 run 仍仅 prepare，不允许重建或重复启动。

## 最新已执行检查点 — 2026-09-11 15:05

Generation-none 已完成 10 题、265 条推理记录校验、导出及独立 PG 评估（101 条记录）。31 请求配对到 30 响应和 1 个已恢复连接错误。30 槽中严格匹配 3 条、分布在 3/10 题；1 个候选误写列名导致 PG 42703，当前评估器标记该候选不可比较，因此池级为 3 匹配、6 不匹配、1 unknown。不是服务故障，不修改候选或评估规则。

当前唯一运行是 `runs/sql_generation-rc-r1`，PID 63090，开始于 15:04:51。下一步等待此进程正常结束并验收/export/evaluate、compare Generation；随后依次执行两个已准备但未启动的 Schema Linking 条件。截止本检查点完成 5/8 推理、5/8 评估、2/4 配对。已完成 run 不重跑，所有配置与生产指纹保持冻结。

## 最新已执行检查点 — 2026-09-11 15:20

Generation-RC 已完成 10 题、30 请求全部成功，261 条推理记录、101 条评估记录校验通过，推理/配对导出完成。30 个请求均完整注入 RC，剥除 RC 后与 none 提示和参数一致。候选池匹配题数 none 3/10 → RC 2/10，匹配候选均 3/30；1 题改善、2 题变差、1 题保持匹配、5 题保持不匹配、1 题 unknown（两边 credit_1 均有列名误写造成的 42703）。不重采样负向结果，不改评估规则。已完成 6/8 推理、6/8 评估、3/4 配对。

当前唯一运行 `runs/schema_linking-none-r1`，PID 66559，开始于 15:19:34；只运行 Schema Linking，不运行下游。下一步等待完成并验收/export/evaluate，然后启动最后一个已准备的 `schema_linking-rc-r1`。两者配对后汇总全批并暂停既有监测，不继续其他题或重复轮次。

## 最新已执行检查点 — 2026-09-11 15:41

Schema Linking-none 已完整结束并导出，10 题、20 请求全部成功、201 条记录通过；独立评估 41 条记录通过，参考全部可执行。完整表列覆盖均 unknown：7 题 query lineage 未解析，3 题列解析未确定，但后三题表覆盖均为 1.0。没有下游 SQL，不能使用通用 bag_equal_rate=0 推断准确率。已完成 7/8 推理、7/8 评估、3/4 配对。

当前最后一个进程为 `runs/schema_linking-rc-r1`，PID 68553，开始于 15:40:26。完成后按原步骤核对真实 RC 调用、export/evaluate/compare，汇总全部结果和限制，检查全部导出并暂停 `deepeye-50`。没有更多待运行条件，不新增 repeats 或下游测试。

## 停止检查点 — 2026-09-11 16:44

按用户最新指示，已对准确核对命令的 PID 68553 发送 SIGTERM 并确认退出，自动监测状态 PAUSED。最后一组为 5 题两个模型分支成功、3 题 Reversed 连续失败后返回空结果但被原生标成功、2 题中断；34 请求、15 响应、17 错误、2 未配对，247 条记录校验通过。数据库当前不可达是用户确认的环境信息；这些已记录错误实际发生于模型 API，不能说成 PG 错误。

前 7 组推理与独立评估、3 个阶段配对全部保留并校验通过。最后一组只导出停止快照至 `exports/schema_linking-rc-r1-stopped/`，没有执行 evaluate 或 compare。不得将原生 8 个 succeeded 当作 8 个完整模型分支成功，不得自动 resume（会跳过其中 3 个降级题）。停止信息单独写入该 run 的 stop.json，不改原始事件。用户已要求停止，本批不再追求原定 8/8 完成；现转交已完成和未完成结果，等待新的明确指示。
