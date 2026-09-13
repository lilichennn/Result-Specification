# DeepEye 双轮 RC 分阶段实验设计

状态：用户于 2026-09-11 批准单阶段独立对照，并补充固定提示、无调用不重跑原则。本文记录该批准方案，不扩展到新 checker 或全阶段同时注入。

## 实验单位

每次运行明确一个 target_stage、condition（none / rc）、repeat_id 和固定题集。从同一个基线运行取目标阶段之前的连续成功产物，分别重跑无 RC 和有 RC 条件。可显式选择继续运行无 RC 的下游阶段。各条件独立目录，不将条件变化当作普通重试。

基线目标阶段成功且完整追踪证明没有 api_request 时，直接引用基线该阶段结果，记录 `rc_not_participating/no_native_llm_call`，两种条件都不重跑。不能从失败或未完成阶段的零请求推断不需要模型。若启用下游且所有输入未变，可以复用已成功基线后续结果；输入变化后下游仍按原生规则处理，无模型调用时记录未参与，不额外安排第二次运行。

## 唯一允许的提示增量

同事双轮 RC 使用 Round2 完整六字段：population、row_grain、column_role、derivation、filter_policy、meta_review。Round1 仅溯源，不和 Round2 叠加。注入块只包含固定的 RC 定义、固定的字段含义、现成 Round2 JSON、固定要求“完成当前阶段任务时参考 RC”。不得增加示例、额外问题分析、字段推荐、SQL 改写建议、候选评分规则、gold 或执行评估结论。可说明 RC 记录已确定含义及未决部分、并不提供查询答案，以免将未决部分当作确定事实；定义全部题共用。

包装原 PG PromptFactory 的 7 个 format 函数；用 ContextVar 识别实际阶段和题目，不按模板名猜阶段（Schema Linking 的 reversed 分支复用 Generation 模板）。原生 token 裁剪必须看到 RC 块。最终 API 请求中实际包含完整注入块才算 RC 参与；格式化尝试次数不是模型调用次数。保留所有原生 checker、无调用路径与 Selection 捷径，不强制触发模型。

## 实现和持久化

所有新增生产代码、文档、测试在 `scripts/rc_evaluation/deepeye/`。现有 baseline_adapters、三个运行入口、DeepEye 原仓库、README、.env、依赖文件保持不变。使用 code/.venv；不复制 benchmark 或基线源代码，不额外安装依赖。

复用 RunStore、TraceRecorder、原生 runner factory、PG support、并发槽位与资源清理。新控制器实现明确的目标阶段起点及终点；不要使用会自动继承全部成功阶段的旧 inherit_checkpoints。每个新运行保存基线 manifest/attempt/payload 哈希、共同上游输入哈希、RC 文件和记录哈希、固定提示版本、当前源码哈希、模型与预算、执行协议。恢复必须一致，旧数据只读。阶段产物和 API/SQL 事件随执行增量提交；失败后追加新尝试，不覆盖；导出目标必须不存在并带 COMPLETE.json。

读取源基线使用一致快照，只接受所选题目标阶段已经完整结束的数据，不能把运行中无终态的记录作为跳过依据。API 追踪完整性需要核对请求、响应/错误及未配对数。准备阶段不连接模型或 PG，缺 RC 不自动生成；RC 必须按 (split, instance_id) 匹配且 db_id/question/evidence 一致。

## 独立评估

参考 SQL 只在 evaluate 中加载，不进入运行 DataItem、提示词或 RC 生成。评估缓存和产物使用另一个 RunStore，不改推理记录；绑定 SQL、数据库标识及显式数据库版本标签，保存执行时间与结果。当前没有远程数据快照证明，标签是操作者声明，不伪称自动检测数据库变化。

比较器明确保留列位置与重复行，NULL 只等于 NULL；列别名不参与。默认严格数值、忽略行顺序但保留重复数的 bag 结果一致率；同时输出严格 ordered 一致率。两者均列出，不根据预测 SQL 或 RC 选择有利规则，不将其命名为官方 BIRD-Interact 准确率。不支持比较的 PG 类型、参考失败、超时明确为 unavailable，不强转字符串或错误地计为相同。空表需要列数一致且双方执行成功；全 NULL 可正确。浮点容差不在首版隐式开启。

Generation：槽位数、唯一 SQL 数、执行成功与结果一致的候选数、至少含一条一致候选的题比例。Revision：固定输入槽位逐个统计修复/误改/未变，前后候选池是否含一致候选。Selection：最终结果一致率、池内已有一致候选时的选中率、shortlist 保留与实际分支。Schema Linking：保留表列规模、参考表列覆盖诊断（解析不确定标 unknown），及可选下游结果。统一保留固定题集分母、缺失/失败/未知状态、实际 RC 参与次数、当前新增 API usage，不把历史累计 metrics 当新增成本。

## 验收

先离线测试 RC 校验和注入隔离、原生无 RC prompt 不变、跨线程不串题、真实 RunStore 的增量/恢复/跳过/来源、结果比较边界及聚合分母，再使用真实缓存和基线做不联网 prepare。未经新的批次指令不自动启动付费 RC 实验。基线 10 题继续由既有监测跟踪，完成后单独验收。
