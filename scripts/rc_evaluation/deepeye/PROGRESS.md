# Implementation ledger — IMPLEMENTATION_PLAN.md

> 历史实现记录：下文的运行进度和监测状态均为 2026-09-11 当时快照，不代表当前状态，也不是继续运行的指令。当前维护入口见 [DEEPEYE_CURRENT.md](../../DEEPEYE_CURRENT.md)；10 题四阶段 RC 测试已于 2026-09-12 完成，详见 `docs/deepeye_rc_probe10_20260912_continuation.md`。

2026-09-11: 用户批准分阶段独立对照；最新补充要求已纳入 DESIGN.md。

## Implementation decisions

- Ruling: 在用户明确指定的当前 code/scripts/rc_evaluation/deepeye/ 目录原地新增，不建立 worktree、不提交 Git — 依赖的是当时已有、当前仍复用的 baseline 适配器，并非弃用实现；其当时尚未跟踪，且原地路径已获批准。风险以原有文件哈希冻结和独立目录控制。
- Ruling: 对互不重叠的新模块并行实现 — 符合当前并行代理要求，接口在计划内固定；若接口不一致由集成测试识别，不修改正在运行的基线来迁就。
- Ruling: 成功基线目标阶段无 API 调用时，none/rc 均复用源产物而非重跑 — 落实用户最新原则；这种题不用于“RC实际参与”子集的收益结论。
- Ruling: 首版同时报告严格 bag 与 ordered 结果一致率，不自动推断排序评分或添加浮点容差 — 避免规则选择偏差；细微浮点差异或排序并列需人工解释，不等同官方评分。

## Interface preflight

| Tasks | Interface / ownership check | Outcome |
|---|---|---|
| 1 / 3 | load_contracts + render_rc_block + rc_context + install_rc_prompts + count_rc_requests | dict schema/signatures fixed; no shared files |
| 2 / 3 | evaluate_run / compare_evaluations; native stage artifact payloads | same run format; source checkpoint encoding must be communicated |
| 1 | tests enforce strict indexing, fixed-only injection, stage/thread isolation | agrees with user principle |
| 2 | tests enforce bag multiplicity, fixed denominator and reference isolation | agrees with DESIGN limits |
| 3 | tests distinguish zero-call successful source from partial source; no overwrite | agrees with no-rerun principle |

## Tasks

- Task 1: complete — rc_implement_contracts; independent rc_review_task1 review found no P1/P2. 23 focused/PG prompt tests passed; native integrated request test passed after fixture alignment.
- Task 2: complete — rc_implement_evaluation; 29 focused tests passed; independent re-review passed after two P2 fixes.
- Task 3: complete — native_execution_scope; 24 focused tests passed; cross-module review passed.
- Integration/review: complete — final new suite 71 tests passed; original suite ran 244 tests (224 passed, 20 opt-in PostgreSQL tests skipped). No remaining P1/P2 findings in independent review.

Historical implementation checkpoint (2026-09-11): no new paid RC experiment was authorized or started by this implementation task. The baseline monitor was active at that time; this is not its current status.

Pre-implementation baseline verification: `.venv/bin/python -E -B -m unittest discover -s tests -q` -> 244 tests, OK, 20 opt-in PG tests skipped; protected code/env/dependency/README hashes saved for final comparison. Existing API connection error belongs to the separate baseline process, not these offline tests.

Root integration test: `tests/test_integration.py` uses real native Generation runners and TraceRecorder, with only SDK transport and tokenizer replaced and sockets blocked. none/rc each yield three candidates/three actual requests; wire and saved prompts agree; completed resume creates no runner. After accounting for native LLM.ask stripping trailing whitespace, test passed. In-memory mutation disabling RC installer produces the expected `0 != 3` request-coverage failure; restoring installation passes. No production source was mutated for the check.

Evaluation compatibility clarification: standard JSON/array and date values require typed normalization; unsupported custom types remain unavailable. This preserves exact equality, not a relaxed scoring rule.

Real-cache validation (network socket connection blocked): full/cross_border_4 prepares successfully for all four target stages using its existing Round2. Source Revision and Selection each report zero native requests; explicit run/resume of those two replay-only fixtures recorded 0 requests and preserved 3 checksum-verified records each. Export created COMPLETE.json. Temporary validation directory: /tmp/deepeye-rc-offline-Muw75V. No actual RC inference or PG execution occurred. These are development validation manifests, not production runs to resume after later source edits.

Final-source real-cache validation at 13:57 (network blocked): both none/rc prepare succeeded for all 10 Schema Linking tasks and the 8 fully completed baseline tasks for each other stage. Revision had 7/8 zero-call targets; Selection had 3/8. Separate one-task none/rc Revision and Selection fixtures passed prepare/run/resume/export, with zero API requests and unchanged record counts after resume. Artifacts: /tmp/deepeye-rc-final-E5jdno. These temporary fixtures are validation artifacts, not live experiments. Adapter, native baseline, entrypoint, dependency, .env, .gitignore and README hashes remain identical to the pre-implementation snapshot.

## Final acceptance — 13:59

Independent review reproduced two P2 issues in evaluation only: unfinished attempts lost their RC request count, and USING/NATURAL JOIN could overstate column coverage. Both were fixed with observed RED/GREEN regressions. Request counting now uses task/stage-bound real messages across interrupted and retried attempts; implicit JOIN column dependencies remain unknown. A changed production/template hash is rejected before evaluating a canonical RC run. Independent targeted re-review closed both findings.

Final commands from code/:

```sh
.venv/bin/python -E -B -m unittest discover -s scripts/rc_evaluation/deepeye/tests -t . -q
.venv/bin/python -E -B -m unittest discover -s tests -q
```

Results: 71 tests OK; 244 tests OK (20 skipped). Native Generation integration verifies actual SDK messages and recorded messages, not formatter-only invocation. Final production SHA-256: `b341c92e8d99b5f092d8e0c3911451ff93332cb7fcc76620fbf7d7131267670a`.

After the review fixes, all real-cache checks were repeated under blocked sockets in `/tmp/deepeye-rc-final-E5jdno/post-review`: both conditions prepared for 10 Schema Linking tasks and 9 complete tasks in every other stage. Revision now has 8/9 zero-call targets; Selection 3/9. Four independent one-task replay checks (Revision/Selection × none/rc) passed run/resume/export, each with zero requests and 3 verified records unchanged after resume. No real RC model inference or reference SQL execution was performed. Remote-service correctness and actual RC effect remain to be tested in an explicitly started experiment.

Historical snapshot (2026-09-11): the separate baseline was at 9/10 complete, with `lite/alien_9` Generation still running and its five-minute monitor active. This describes that earlier checkpoint, not the current state. No source change, new batch, commit or push was made to that run by this implementation task.
