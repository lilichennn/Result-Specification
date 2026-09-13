# DeepEye 效率改造前测工具

入口为 `python -m scripts.deepeye_efficiency_probe`，在 code/ 下使用现有 `.venv/bin/python -E -B` 运行。此工具不是新的 DeepEye Pipeline，也不能恢复正式实验。

- `baseline`：只读保存 Git、现役源码指纹、旧原生实验和暂停状态。
- `regression`：运行两套离线 unittest，关闭真实 PostgreSQL 测试，保存完整日志。
- `legacy`：使用真实旧版 LLM/Extractor 和假客户端复现整组重试；不请求外部模型。
- `threads --concurrency N`：以屏障验证 N 个线程真实同时等待，然后释放并回收。
- `local-http --concurrency N`：单事件循环本地 HTTP 服务，实际调用同款 SDK，最多 2N 次请求。同步使用现有 RunStore 增量记录。不是远端性能测试。
- `workload --source RUN_SQLITE`：只读抽取历史四阶段、三种输入长度的固定负载。默认每层5份，共60份，不按 SQL 正确性或历史成功筛选。
- `remote`：直接发 n=1 请求，不自动重试、不执行 SQL，不经过原生整组重试。
- `local-groups`：仅在本机模拟多采样组、慢请求与失败；`--coordinators` 控制等待一组采样完成的协调线程，`--workers` 控制采样线程，`--concurrency` 独立限制正在模拟模型等待的请求数。真实使用RunStore，但不执行DeepEye、HTTP或模型；不能用其绝对速度预测远端吞吐。

所有模式必须提供新的 `--output` 目录；不覆盖现有记录。远端模式还必须显式提供 `--authorize-remote`、`--requests`、`--admission-seconds`、`--token-stop` 和 `--workload`。请求数不能超过并发数两倍。授权标志只是技术保护，不代替用户对实际消耗的授权。

请求参数保持 `max_tokens=16384`、`temperature=0.6`，没有额外思考开关。默认超时660秒是 SDK 每操作超时；本工具没有声称已实现总耗时硬截止。严格取消/总体截止应在效率改造中单独验证。

`remote`和`local-http`可使用`--requests-per-second`平滑启动，累计延迟不会产生补发突发；不指定时保持原始集中启动行为。`--workers`默认等于`--concurrency`；实际提交中的任务不超过两者较小值。因此线程更少时不能达到请求目标，线程更多时也不声称自动产生额外并发或已实际创建全部线程。HTTP连接池上限等于请求上限。

平滑控制的是任务提交间隔，实际SDK/请求头时间另行记录，需核对真实发送节奏。`submitted_at`与`executor_wait_seconds`区分提交和进入工作线程；连接准备及持久化开销另列。SDK超时不包含等待下一次提交间隔。

发送 SIGINT/SIGTERM 或在该次输出目录放置 `STOP` 文件会停止新增请求，并等待已在途请求结束、落盘后退出。不自动补发失败请求。Token阈值也只停止新请求；已在途和未报告usage的消耗可能使实际成本超过阈值。

远端模式可使用 `--draining-run OLD_RUN` 衔接加压。直接旧批及其manifest记录的所有祖先批次必须已发满自身的有限请求预算，且模型、地址和提示词池一致。新进程只读这些旧RunStore，将所有旧请求中尚无终态记录的数量从 `--concurrency` 指定的**全部批次合计上限**中扣除；只有一个新发送进程，不支持多个新进程同时争用这个额度。旧批只会减少请求，统计延迟因此会保守占用额度。读取失败、循环引用、重复终态或发现旧批又有新请求时停止新增，保留并等待新进程已经发出的请求结束。

衔接模式仍使用新的输出目录、有限的新请求预算和固定启动速率，不热改旧进程。`handoff_admission`记录每次提交时旧批占用、新批预留及合计；这只是保守预留数。两批的实际客户端在途峰值须在收齐终态后，合并请求头发送及结束时间计算；不能将两个分别发生的峰值简单相加，也不把衔接窗口当作独立稳态测试。

当前衔接读取会验证manifest和每条新消费的请求/终态校验和；损坏记录不得释放额度。提交限速按落盘和实际提交后的时钟推进，避免慢落盘积累补发额度。但客户端连接准备仍可能使网络发送局部集中，不能将提交限速称为传输层硬限速。

只读统计入口为 `python -m scripts.deepeye_efficiency_report --run RUN --output NEW_JSON`。`--run`统计单批及其祖先链；后来启动的后继批次不在该链中。全部批次结束后可改用 `--campaign PARENT --output NEW_JSON`，读取其下全部`remote_*`记录，按每次新批首次发送请求头划分上限变化区间，将所有来源的客户端在途放到同一时间线核验。区间内的完成数含较早批次，区间也可能包含人工等待，不能直接当作该上限的稳态吞吐或因果效果。

生成统计前重新以只读方式验证RunStore全部记录校验和，并核对导出manifest/summary与权威记录一致；旧summary中的历史完整性标记和SQLite结构检查不能替代这一步。任一批未结束或校验失败则拒绝生成完整分析。

权威记录在该次目录的 `run.sqlite3` 中；`manifest.json`、`workload.json`、`summary.json` 是单次运行导出。`request` 与 `response/error` 通过 `request_no` 配对，原始 SDK 解析后的响应包含 usage 和 reasoning 字段。资源每2秒记录一次。所有输出在 `baselines_reproduce/` 下时自动受现有忽略规则保护。

`peak_wire_inflight` 只表示已发请求头且客户端尚未结束的请求，不代表服务端GPU实际并发。`sdk_to_first_transport_event_seconds` 包含客户端准备开销，不能称为精确连接池等待。容量结果需结合失败率、吞吐、TPM和负载组成解释；短时2N次测试不是持续满载证明。

## 独立 PostgreSQL 短测

入口为 `python -m scripts.efficiency_probe.postgres`，与上述模型压测分开，不调用模型，不修改或恢复生产 Pipeline。

```sh
.venv/bin/python -E -B -m scripts.efficiency_probe.postgres prepare --source OLD_RUN/run.sqlite3 --output NEW_CAMPAIGN/workload.json
.venv/bin/python -E -B -m scripts.efficiency_probe.postgres run --workload NEW_CAMPAIGN/workload.json --output NEW_CAMPAIGN/c50 --concurrency 50 --queries 200 --authorize-postgres
```

`NEW_CAMPAIGN` 必须已存在，`workload.json` 和各次运行目录必须不存在；不能覆盖旧记录。真实查询需要用户授权，命令中的开关不能替代授权。

- `prepare` 只读抽取实际访问过 PostgreSQL 的历史成功查询，按库去重，每库选择中等耗时和较慢的两条。历史结果最多1,000行，不使用gold。SQL筛选只服务于这次有限负载，不向生产适配层加入新语法限制。
- `run` 使用既有 PostgreSQL 执行函数和原 SQL，保持只读、新连接、执行后关闭的语义；单条 SQL 超时30秒，无重试。查询并发范围1—50，单批最多200次。
- 首批查询连接经过屏障，服务端确认C个连接同时存在后才开始执行SQL。首波连接/屏障失败立即停发并关闭连接，不能由后续任务绕过。
- 额外1个只读监控连接每50毫秒轮询；它不计入查询并发数，但会消耗服务器连接名额。预检为其他用户额外保留至少5个普通连接名额；不修改服务端配置。
- 5次查询失败、监控失败、记录失败、120秒新工作接纳时限或SIGINT/SIGTERM会停止新工作并收尾。此入口不读取模型压测的STOP文件。
- `operation_seconds_excluding_barrier` 是新建连接、数据库事务/查询、完整取回和清理的总耗时，扣除人为屏障等待；不是服务器纯执行时间，也不包含调度队列等待。
- `ready_connections` 证明首波连接数；`peak_client_dispatched_operations` 包含事务准备、取回与清理；`peak_server_active` 只是50毫秒轮询观察到的峰值，可能漏掉更短的真实峰值。
- 一组一个RunStore，`query_start/query_result` 通过 `request_no` 配对；保存来源事件ID、SQL哈希、时间、结果类型/列/行数，不重复存储结果数据。结束后核验完整性和连接回收；监控失败不能被记作成功。

该测试只能验证有限历史查询的短时承载，不证明所有复杂SQL、长期满载或多进程混合业务下的稳定性。
