# DeepEye：当前入口与历史代码边界

更新日期：2026-09-13。本文件用于后续维护时区分现役实现和历史探索，不授权启动新的模型或数据库实验。

## 最新决定：旧正式预算记录不再复用

效率改造已经完成，现役机制以[新版共享采样说明](DEEPEYE_SHARED_SAMPLING.md)和[实验总索引](../baselines_reproduce/EXPERIMENTS.md)为准；本文件下方保留的“尚未改造”“Pipeline槽位”等叙述属于改造前历史状态，不覆盖本节。

用户已决定不采用旧正式预算与新版RC之间的一次性兼容方案。旧501道流程成功题及9道失败题的原始记录、计划、监测和兼容审查，已整体移动到[旧正式预算归档](../archive/deepeye_formal_budget_legacy_20260913/ARCHIVE.md)。后续对应题目使用新版适配层重新运行，不继承旧checkpoint、不导入旧有效Token清单作为正式对照，也不新增旧来源兼容入口。

本次归档没有启动重跑。新版95题的运行目录、进程、冻结参数及监测计划保持不变；预计算、现成RC和低预算实验也保持原位。历史审查和效率前测中的旧绝对路径只代表当时来源，回查时使用归档路径映射，不创建旧目录软链接或运行时兼容分支。

## 当前使用的实现

以下路径相对 `code/`。当前适配代码位于 DeepEye 原项目之外，现役原生源码尚未因效率方案改变。用户已授权后续为单采样重试及必要的解析衔接修改 `baselines/DeepEye-SQL/`；这不代表可任意修改其他原生算法。

| 工作 | 当前入口或目录 | 说明 |
|---|---|---|
| 小规模连通及流程检查 | `scripts/deepeye_bird_interact_smoke.py` | 仍是有效入口，也提供其他入口复用的配置和样例加载函数 |
| 公共数据库值、关键词与检索预计算 | `scripts/deepeye_bird_interact_precompute.py` | 通过 collect／compute／verify 明确区分采样、模型计算和离线核验 |
| 原始 DeepEye Pipeline | `scripts/deepeye_bird_interact_run.py` | 从冻结预计算加载输入，不注入 RC；按题目槽位运行并增量记录 |
| PostgreSQL、输入、并发和持久化适配 | `scripts/baseline_adapters/deepeye/` | 当前唯一的 BIRD-Interact 外部适配实现 |
| RC 四阶段独立对照和评估 | `scripts/rc_evaluation/deepeye/` | 命令说明见该目录的 `USAGE.md`；参考 SQL 仅供独立评估使用 |
| 独立效率前测 | `scripts/deepeye_efficiency_probe.py`、`scripts/efficiency_probe/` | 仅用于旧行为复现及容量测试，不是生产 Pipeline；用法见该目录 `USAGE.md` |

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

2026-09-13后续前测已结束：B0—B3完成，B4在300档成功585/600次；600档发出619次后因118次限流停止新增，最终490成功、11次超时；1200/2000未运行。所有测试进程已退出，生产代码仍未变，旧监测仍暂停。详见[前测报告](../baselines_reproduce/deepeye_bird_interact/efficiency_tests/prechange_20260913T061927Z/REPORT.md)。实际请求启动频率与同时在途数应分开测量，目前未确定正式稳定配置；此次不进入C1—C6、不恢复旧实验、不启动正式RC实验。

随后用户批准平滑续测及追加四档衔接加压，现已全部结束：实际合计客户端在途达到4000，50次/秒提交策略下未见429，但高档位仍有约3.4%—5.0%的单次请求错误。实际22931次请求全部收齐终态，记录校验通过，343项离线测试通过、20项跳过。详见[续测报告](../baselines_reproduce/deepeye_bird_interact/efficiency_tests/paced_20260913T070300Z/REPORT.md)。4000共享请求线程／4000请求额度可作为改造后的候选验收配置，尚非生产设置或长期最优结论；提交速率不等于网络发送的硬上限，阶段协调线程仍需按等待关系设计。

`scripts/deepeye_efficiency_report.py`只读验证已结束试验的记录校验和，统计实际发送速率和全局在途时间，不启动推理。压测工具的慢落盘计时和读取完整性问题已单独修正，不改旧manifest。此续测结束时无付费请求继续运行；C1—C6未执行，旧实验和监测不恢复。

最新的[8,000 在途与双线程池联合测试](../baselines_reproduce/deepeye_bird_interact/efficiency_tests/grouped8000_20260913T091150Z/RUN_PLAN.md)已结束。独立工具新增 `local-group-http`、`--group-profile`，本地与远端共用真实分组调度与SDK记录路径。本地55,260次HTTP请求全成功，最多6,000协调线程＋8,000共享请求线程组合已验证。首批远端峰值7,429并出现22次429；同参数复测峰值7,819、0次429，15,827／16,000成功。没有复现“约7,400限流”，因此未触发用户指定的7,300补测；两批均未实际达到8,000，不能标为8,000通过。

本轮26,379次付费请求均已收齐终态并校验，进程退出；60,000总授权未用部分不自动投入新测试。第二批实际因16,000请求额度停发，旧摘要的时限标签问题以单独审计说明保留，未改写旧记录。最新离线回归354项通过、20项跳过，生产四项指纹及旧实验不变，监测仍PAUSED。详见[联合测试报告](../baselines_reproduce/deepeye_bird_interact/efficiency_tests/grouped8000_20260913T091150Z/REPORT.md)。这仍是独立容量测试，C1—C6生产效率改造与完整Pipeline验收尚未完成。

计划包括改造前的旧版回归/故障复现、本机与接口容量单点测试，改造中的重试/持久化/并行/RC隔离离线测试，以及改造后的真实单点复测和10题验收。真实调用另设有限预算，测试结果与正式RC结果分开保存。

后续用户确认模型请求工程上限8000、每个采样最多4次尝试（首次＋最多3次重试）；这些是待实施配置，不能误认为当前旧生产执行器已采用。[PostgreSQL 10/50并发短测](../baselines_reproduce/deepeye_bird_interact/efficiency_tests/postgres50_20260913T102600Z/REPORT.md)已完成：各200次实际SQL全部成功，50个工作连接同时存在，连接归零。整批耗时8.30/8.28秒，50并发没有显示吞吐提升，作为可配置上限候选而不是效率保证。

独立数据库测试入口为`python -m scripts.efficiency_probe.postgres`；它的历史SQL负载筛选只属于压测，**没有给生产适配器重新加入AST或函数限制**。该轮无模型调用，363项离线回归通过、20项跳过；生产四项指纹、旧正式实验与PAUSED监测未变，未进入C1—C6。

追加[20并发补测](../baselines_reproduce/deepeye_bird_interact/efficiency_tests/postgres50_20260913T102600Z/REPORT_20.md)也为200/200成功，8.38秒，测试连接回收。10/20/50整批耗时相近，因此当前建议以PG=10开始后续新执行层验收，是否上调再看完整Pipeline排队；没有修改生产设置。用户已说明数据库没有其他使用者，不能沿用“给其他使用者留连接”的建议理由。600条请求终态及1,521条记录已复核，原10/50记录不变。

## 离线维护检查

在 `code/` 下执行以下命令；真实 PostgreSQL 回归保持关闭，测试中的模型调用使用替代实现。

```sh
DEEPEYE_TEST_PG=0 .venv/bin/python -E -B -m unittest discover -s tests -q
DEEPEYE_TEST_PG=0 .venv/bin/python -E -B -m unittest discover -s scripts/rc_evaluation/deepeye/tests -t . -q
```

清理文档、归档文件或失效缓存不应改变现役源码指纹。更改 Python、提示模板或依赖时，应重新核对恢复／继承边界，并在需要时建立新实验目录；不要把整理工作变成隐式重跑。
