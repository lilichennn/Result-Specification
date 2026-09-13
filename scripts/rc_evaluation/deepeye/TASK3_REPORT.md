# Task 3 实施记录

所有生产改动位于本目录的 `source.py`、`runner.py`、`cli.py`、包入口；测试为 `test_source.py`、`test_runner.py`、`test_cli.py`。未修改旧 adapter、旧 entrypoint、DeepEye、README、.env 或依赖；没有 commit，没有真实模型/PG请求，没有新增子代理。

## RED → GREEN

- 来源首批 5 项：缺少 source 模块时全部 RED；实现真实 RunStore 一致快照、原始输入到目标的完整哈希链、成功终态与 API 配对核对后 GREEN。
- 控制器首批 5 项：缺少 runner 模块时全部 RED；实现独立目标起止、零调用引用、清洁前缀恢复、失败追加与 opt-in 下游后 GREEN。
- CLI 首批 5 项：缺少 CLI 时全部 RED；实现根目录/code目录入口、offline prepare、配置/源码冻结、零调用 run/inspect/export 后 GREEN。
- 后续实际 RED：RC 条件缺合约未拒绝；CLI 失败阶段退出码为0；损坏运行的 verify 结果未检查；未参与状态缺明确status；缺目标而已有下游成功记录未在资源构造前拒绝。均先复现再修复。
- 补充边界通过：失败/中断目标不能作为零调用依据；来源继承要追溯原始 API 记录；payload 篡改拒绝；RC 与 none 都不强制原生无调用路径；父阶段失败仍先等待原生子任务再清理。

Task 3 聚焦测试最终命令：

```bash
.venv/bin/python -m unittest scripts.rc_evaluation.deepeye.tests.test_source scripts.rc_evaluation.deepeye.tests.test_runner scripts.rc_evaluation.deepeye.tests.test_cli -v
```

最终结果：24 项全部通过。三组测试统一阻断 socket 连接、socket.create_connection 和 psycopg.connect；CLI 帮助另以子进程从项目根目录及 code 目录实际执行。

## 可复用接口

- `source.snapshot_source(source_run, tasks, target_stage, *, continue_downstream=False)`：只读返回完整来源快照。
- `source.restore_seed(snapshot, target_stage)`：还原 gold-free 目标前状态。
- `source.validate_manifest(manifest)`：校验嵌入来源链、条件和合约绑定。
- `runner.run_experiment(store, runner_factory, recorder, *, workers=4, slot_controller=None)`：控制增量尝试。factory 参数提供离线原生替身边界；生产 CLI 无自定义 factory 开关。
- `cli.execute_run(store, environment)`：唯一生产执行路径，始终经过旧 bounded_runner_factory、PG 支持、TraceRecorder、固定问题槽位和 PG gate；回放全量无调用时不构造 runtime/client。

来源格式已同步 Task 2：`source_checkpoints[task_key]` 保存 `input/input_sha256/upstream_state_sha256/stages`；`stages[stage]` 保存 `attempt_id/input_fingerprint/payload/payload_sha256/api_trace`。manifest 另存原 source_manifest 和其 fingerprint。新阶段 finish 保留原生 artifact，并补 execution_origin、rc_participation、source_provenance。

## 限制与集成注意

- 首版只有固定 question workers 和独立 PG 并发；不承诺复现 adaptive 调度。模型、采样、温度、token/timeout/retry等语义与基线严格相同；timeout 上限1200秒。并发配置单独记入新manifest。
- `run/resume` 只运行已经 prepare 的目录；任何条件/预算变化需重新 prepare，不把变体当retry。
- 源目标必须成功结束；其他题可仍在运行。选中阶段零请求需完整配对和零原生tokens互相印证。继承来源失踪/不完整时不猜测。
- 来源 payload 内的累计 native metrics 保留其历史语义，不是当前新增开销；API事件账本单独记录当前run全部尝试。
- 新增源码/固定txt的生产哈希会在开发期间变化，正式离线 prepare 必须在集成代码冻结后重做。文档和tests不纳入生产哈希。
- 原生模型全流程和真实PG/LLM吞吐未在本任务调用；根代理负责全目录离线集成、原suite回归和基线监测。
