# DeepEye 独立 RC 阶段实验

当前入口与历史代码边界见 [DEEPEYE_CURRENT.md](../../DEEPEYE_CURRENT.md)。旧 guard 临时脚本和旧测试记录不可作为当前执行器或源基线。

每个目录只对应一个目标阶段、一个条件和一个 repeat_id。`prepare` 只读取冻结预计算、已有基线和已有双轮 RC，不连接模型或 PostgreSQL。`run`、`resume`、`evaluate` 是显式执行命令；本工具不会自动开始新的实验批次。

下面命令在项目的 `code/` 目录执行。也可从项目根目录执行 `code/.venv/bin/python code/scripts/rc_evaluation/deepeye/cli.py ...`；两种入口的默认配置文件都指向 `code/config/.env`。

```bash
.venv/bin/python -m scripts.rc_evaluation.deepeye --help
```

## 准备一组对照

先准备无 RC 条件。把占位路径替换成实际路径；目标目录必须不存在。

```bash
.venv/bin/python -m scripts.rc_evaluation.deepeye prepare \
  --source-run /absolute/baseline-run \
  --run-dir /absolute/experiments/revision-none-1 \
  --target-stage sql_revision --condition none --repeat-id 1 \
  --item lite/example_1 --workers 4 --inner-workers 4 --pg-concurrency 4
```

然后从同一基线、相同题集和相同上游准备 RC 条件。RC 文件按 split 和 instance_id 匹配，并核对数据库、问题和 evidence；缺记录不会自动生成。

```bash
.venv/bin/python -m scripts.rc_evaluation.deepeye prepare \
  --source-run /absolute/baseline-run \
  --run-dir /absolute/experiments/revision-rc-1 \
  --target-stage sql_revision --condition rc --repeat-id 1 \
  --item lite/example_1 --rc-lite /absolute/round2-lite.json \
  --workers 4 --inner-workers 4 --pg-concurrency 4
```

可选目标为 `schema_linking`、`sql_generation`、`sql_revision`、`sql_selection`。多题使用多个 `--item`；不指定时选基线 manifest 中的题。可用 `--variant lite/full` 过滤。预计算与独立 few-shot 路径默认从基线读取，也可显式提供 `--precompute-dir`、`--few-shot-source`，但内容指纹必须吻合。

模型、endpoint、temperature、采样预算、最大 tokens、原生重试和 PostgreSQL 执行协议必须与源基线一致。CLI 提供的预算参数用于显式核对，不允许借此改变实验预算。`--chat-timeout` 从基线继承，必须为 1–1200 秒；1200 是每次模型请求上限，并非整个阶段的总时长。默认使用固定问题并发，也可以在准备实验时显式启用动态并发；PostgreSQL 始终单独限流。

源目标阶段必须已经成功结束，并且 API 请求都能配对到 response/error。可以读取尚有其他题在跑的基线的一致快照，但不能选择尚未完成的目标阶段。若基线阶段是继承产物，会继续核对原始来源的调用记录；导入后事件为空不构成“原生无模型调用”的证据。

## 固定与动态并发

不加 `--adaptive-concurrency` 时仍使用固定的 `--workers`，即使源基线用了动态并发也不会自动启用。`--workers`、`--inner-workers` 和 `--pg-concurrency` 未指定时各自继承源基线；不要把原生内部线程池大小当作同时运行题目数。

在上面的 `prepare` 命令后增加以下参数，即可使用 200 起步、10 为调整步长、400 为上限的动态模式：

```bash
--adaptive-concurrency \
--concurrency-initial 200 --concurrency-step 10 \
--concurrency-max 400 --concurrency-min 10 --concurrency-window 60 \
--inner-workers 100 --pg-concurrency 10
```

| 参数 | 含义 |
|---|---|
| `--adaptive-concurrency` | 启用动态题目槽位 |
| `--concurrency-initial` | 每次启动或恢复时的初始槽位数 |
| `--concurrency-step` | 每次增加或减少的槽位数 |
| `--concurrency-max` | 槽位上限 |
| `--concurrency-min` | 槽位下限 |
| `--concurrency-window` | 判断近期模型请求健康情况的窗口，单位秒 |

启用动态模式后，未填写的策略参数优先继承源基线的动态策略；如果源基线是固定模式，则使用共享控制器的默认值：起步 50、步长 10、下限 10、上限 100、窗口 60 秒。所有数量须为正整数，且下限 ≤ 起步数 ≤ 上限；窗口须为有限正数。单独填写 `--concurrency-*` 却不启用动态模式会报错，不会静默忽略。

并发单位是**一道题的目标阶段及其可选下游流程**：获得槽位后独立运行、逐阶段落盘，结束后再接纳下一题。不是一次模型请求，也不是所有题完成同一阶段后再统一推进。动态模式中 `--workers` 不再限制题目槽位；模型请求本身不争抢这些槽位，因此实际 API 请求并发可能高于题目并发。各独立实验进程各自计数，上限不是多个进程合计的全局额度。

调节规则与原 Pipeline 共用：窗口内至少 50 次成功、没有暂时性错误、仍有待启动题目且槽位占用达到 80% 时增加；暂时性错误至少 5 次且占比达到 10% 时减少。调整后清空判断窗口并冷却 60 秒。降低上限只限制新题入场，不中断已运行题目；SQL 候选错误不算服务暂时性错误。小题集或完全不需要模型调用的实验可能不会触发升档，这不是功能失效。

并发配置在 `prepare` 时冻结，`run`/`resume` 从清单恢复，不接受临时改并发。恢复从冻结的起步数重新计数，已有成功结果仍复用；要换策略应准备新目录。配对的 `none` 和 `rc` 条件必须使用相同配置。动态调整和实际占用由现有追踪事件及运行结束的 `admission` 报告记录；纯复用路径也遵守槽位设置，但不创建模型或 PostgreSQL 客户端。

## 无调用阶段与下游

若成功基线目标阶段的完整追踪证明 `api_request=0`，两种条件均直接引用该阶段产物，记录 `execution_origin=reused_no_native_llm_call`、`rc_participation.status=rc_not_participating`、`reason=no_native_llm_call`。整次运行只有引用产物时，不构造模型客户端。

默认只重跑目标阶段。`prepare --continue-downstream` 显式允许继续运行后续阶段；RC 仍只注入目标阶段。上游语义产物保持不变且源下游已有成功产物时，可以引用该下游；否则运行原生下游。目标阶段旧产物和目标后的旧产物不会进入目标重跑的输入。

无 RC 条件是从同一上游重新采样的对照，与历史 baseline 输出区分。RC 使用现成 Round2 完整六字段，加固定定义、字段含义和参考要求；Round1 只保留溯源，不额外注入任务建议。

## 显式运行、恢复、查看与导出

以下 `run` 可能调用模型和 PostgreSQL，仅在决定执行该批实验时使用。

```bash
.venv/bin/python -m scripts.rc_evaluation.deepeye run --run-dir /absolute/experiments/revision-rc-1
.venv/bin/python -m scripts.rc_evaluation.deepeye resume --run-dir /absolute/experiments/revision-rc-1
.venv/bin/python -m scripts.rc_evaluation.deepeye inspect --run-dir /absolute/experiments/revision-rc-1
.venv/bin/python -m scripts.rc_evaluation.deepeye export \
  --run-dir /absolute/experiments/revision-rc-1 --export-dir /absolute/exports/revision-rc-1
```

恢复复用已有成功阶段，仅对未成功阶段追加尝试。失败时的 DataItem 修改不进入下一次尝试。源码、固定提示、RC 文件或语义配置变化需要新目录；不能在同一运行里切换条件、题集或 repeat。进程崩溃可能已产生远端开销，恢复不承诺远端 exactly-once。退出码：0 表示执行/命令完成；1 表示运行已落盘但有失败阶段；2 表示校验或运行异常。阶段成功只表示原生产物完整，并非答案正确。

RunStore 内嵌 `source_checkpoints`：每题包含原始输入、输入哈希、目标前上游状态哈希，以及各阶段的源 attempt_id、输入哈希、完整原生 payload、payload 哈希和 API 配对统计。新产物仍使用原生 `payload.artifact`。原生累计 metrics 含历史输入成本；新增开销读取 `observed_usage` 或导出的 `usage.json`，其中保留失败尝试及未知 usage，不能当作供应商账单。

## 单独评估和配对报告

`evaluate` 才加载参考 SQL；参考不进入推理输入或提示。这个命令会执行 PostgreSQL 查询，但不调用对话模型。评估目录必须是独立新目录。

```bash
.venv/bin/python -m scripts.rc_evaluation.deepeye evaluate \
  --run-dir /absolute/experiments/revision-rc-1 \
  --output-dir /absolute/evaluations/revision-rc-1 \
  --reference-lite /absolute/lite-reference.json \
  --database-version operator-declared-snapshot-2026-09-11

.venv/bin/python -m scripts.rc_evaluation.deepeye compare \
  --left-dir /absolute/evaluations/revision-none-1 \
  --right-dir /absolute/evaluations/revision-rc-1 \
  --output-dir /absolute/comparisons/revision-1
```

数据库版本标签是操作者声明，不是自动检测到的远端快照。比较器同时报告严格 bag 和 ordered 结果一致，保留重复行、列位置和 NULL；列别名不影响比较，首版不隐式启用浮点容差。不支持的值类型、参考缺失/失败和超时保留为 unavailable。这不是官方 BIRD-Interact 准确率，也不能证明 SQL 在所有可能数据库上等价。

Schema Linking 的参考表列覆盖只作结构 proxy，不能代表完整正确性；Generation/Revision 查看候选池覆盖及逐候选修复/误改；Selection 同时查看池内可选正确候选时的条件成功率。报告保留完整题集分母与未知状态，避免只统计成功执行的题。

## 离线测试

```bash
.venv/bin/python -m unittest discover -s scripts/rc_evaluation/deepeye/tests -t . -v
```

实现测试使用临时 RunStore、合成原生产物和离线执行器，不使用真实模型或 PostgreSQL。原始 baseline 运行入口和适配器保持独立。
