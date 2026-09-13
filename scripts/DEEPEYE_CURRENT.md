# DeepEye：当前入口与历史代码边界

更新日期：2026-09-13。本文件用于后续维护时区分现役实现和历史探索，不授权启动新的模型或数据库实验。

## 当前使用的实现

以下路径相对 `code/`。当前适配代码位于 DeepEye 原项目之外，现役原生源码尚未因效率方案改变。用户已授权后续为单采样重试及必要的解析衔接修改 `baselines/DeepEye-SQL/`；这不代表可任意修改其他原生算法。

| 工作 | 当前入口或目录 | 说明 |
|---|---|---|
| 小规模连通及流程检查 | `scripts/deepeye_bird_interact_smoke.py` | 仍是有效入口，也提供其他入口复用的配置和样例加载函数 |
| 公共数据库值、关键词与检索预计算 | `scripts/deepeye_bird_interact_precompute.py` | 通过 collect／compute／verify 明确区分采样、模型计算和离线核验 |
| 原始 DeepEye Pipeline | `scripts/deepeye_bird_interact_run.py` | 从冻结预计算加载输入，不注入 RC；按题目槽位运行并增量记录 |
| PostgreSQL、输入、并发和持久化适配 | `scripts/baseline_adapters/deepeye/` | 当前唯一的 BIRD-Interact 外部适配实现 |
| RC 四阶段独立对照和评估 | `scripts/rc_evaluation/deepeye/` | 命令说明见该目录的 `USAGE.md`；参考 SQL 仅供独立评估使用 |

`smoke.py` 不是已经弃用的旧版本。批量入口仍复用它的 `load_independent_examples`、`read_environment`、`build_runtime_config`；预计算和 RC CLI 也复用环境读取。不要因文件名中有 smoke 就移动、删除或复制出另一份配置实现。

## 当前行为边界

- SQL 执行使用 `postgres_execution.py`：将原文 SQL 交给 PostgreSQL，不使用本地 Meta 表列校验、AST／函数白名单，不改写 SQL，也不附加结果行数上限。保留只读事务、超时、搜索路径和 extended protocol 单语句限制。
- Meta 用于模型可见 schema 和值采样，不是 SQL 执行时的物理列隔离层。实际访问权限由数据库角色决定；只读事务也不是任意特权函数的完整安全沙箱。
- Pipeline 并发单位是一道题的完整流程；该题独立推进四阶段，阶段产物立即提交，完成或失败后释放槽位。没有另一套正在使用的“全部题统一完成一个阶段”调度器。
- `RunStore` 是批量运行的权威记录；不要用旧 smoke 快照替代稳定版 checkpoint。阶段显示成功不等于 SQL 正确，原生分支可能在错误耗尽后回退，验收还应核对模型事件与分支产物。
- 模型、预算、执行协议、源码和输入由运行 manifest 冻结。源代码变更应按正式版本变更处理；不得改写旧 manifest 的哈希来绕过校验，也不得自动继承旧 guard 的运行结果。

## 不能误当作旧执行器删除的支持代码

- `run_store.py` 的旧模块名映射仅负责读取历史序列化记录，不会加载另一份旧适配器；应保留相关兼容测试。
- `run_admission.py` 的基础实现仍被 PostgreSQL 限流和错误分类复用，不能因模型并发改用 PipelineSlots 就整文件删除。
- `seal_legacy_collection` 是显式的旧缓存迁移辅助函数，不是常规预计算入口；正常加载仍要求完整性校验。是否移除此类辅助函数应另作有测试保护的源码变更，不属于散落历史脚本清理。

## 历史材料不得作为当前实现来源

- 旧 PostgreSQL guard 的临时源码、修改前副本、旧测试、修正补丁和编译缓存，已从散落位置移除。恢复副本仅保存在 Git 忽略目录 `archive/deepeye_bird_interact_adapter_trials_20260911/retired_scripts_20260912.tar.gz`，不是可直接使用的入口。
- 同目录的 `ARCHIVE.md` 和两个 manifest 记录测试产物、历史脚本的来源和校验信息。历史源码不应解包回生产目录、复制拼接到当前执行器，或进入测试发现路径。
- 项目上层的 `code_archive/` 属于更早的独立探索归档，其中 SQLite 版 DeepEyeAdapter 不是本轮 PostgreSQL 适配器。`scripts/0-pre-expriment_birdsample100/` 的候选冻结／筛选脚本也不是当前 BIRD-Interact Pipeline 入口。
- `PROGRESS.md`、旧运行计划和按时间追加的报告保留当时的状态，不代表当前有进程运行或仍获准继续实验。跨轮次状态查阅[实验总索引](../baselines_reproduce/EXPERIMENTS.md)，实际状态以当前进程和 RunStore 为准。

## 已记录、尚未实施的效率改造

统一方案见[DeepEye效率改造与分阶段测试计划](../docs/deepeye_efficiency_optimization_plan_20260913.md)。采用单采样重试、完整采样成功标准、有效采样Token口径、独立采样并行和实测请求容量；本文上方的Pipeline槽位描述仍是旧现役实现，不应误认为新调度已经完成。

正式预算旧实验已停止：501题流程成功、9题流程失败、95题未开始；这些旧版状态不能直接当作新完整采样标准。旧监测保持暂停，方案记录不授权启动容量测试、恢复旧实验或进入新一轮正式实验。

计划包括改造前的旧版回归/故障复现、本机与接口容量单点测试，改造中的重试/持久化/并行/RC隔离离线测试，以及改造后的真实单点复测和10题验收。真实调用另设有限预算，测试结果与正式RC结果分开保存。

## 离线维护检查

在 `code/` 下执行以下命令；真实 PostgreSQL 回归保持关闭，测试中的模型调用使用替代实现。

```sh
DEEPEYE_TEST_PG=0 .venv/bin/python -E -B -m unittest discover -s tests -q
DEEPEYE_TEST_PG=0 .venv/bin/python -E -B -m unittest discover -s scripts/rc_evaluation/deepeye/tests -t . -q
```

清理文档、归档文件或失效缓存不应改变现役源码指纹。更改 Python、提示模板或依赖时，应重新核对恢复／继承边界，并在需要时建立新实验目录；不要把整理工作变成隐式重跑。
