# ShopSim Agent Harness：可插拔策略 × 组件级评测 × 轨迹导出

基于阿里 **ShopSimulator**（ACL 2026, arXiv:2601.18225）淘宝购物环境构建的 **Agent Harness 与评测框架**：把"多轮导购 Agent"拆成三层——**环境循环 harness、可插拔策略、组件级评测**——任何策略（七阶段 pipeline、单 prompt ReAct 基线、后续 RL 训练出的 policy）走同一条 env-loop、同一套指标、同一份轨迹导出格式，做到策略间严格可比。

环境与评测协议 Based on 阿里 ShopSimulator；harness 架构、七阶段 pipeline、ReAct 基线、Trace、组件级评测、轨迹导出与全部实现代码为本人独立完成。

## 架构：三层解耦

```
┌────────────────────────────────────────────────────────┐
│  评测层  run_eval / eval_suite / metrics                │
│          端到端 4 维 reward + 组件级指标 + Bootstrap CI  │
└───────────────────────┬────────────────────────────────┘
                        │ 每 task 一个 JSON（含两级 Trace）
┌───────────────────────▼────────────────────────────────┐
│  Harness 层  ShoppingAgent（env-loop）                 │
│  环境交互 · Shopper 模拟 · 守卫（防死循环/规格/确认）    │
│  策略无关：只认 BaseAgent 接口                          │
└───────────────────────┬────────────────────────────────┘
                        │ BaseAgent: start_task() / step()
┌───────────────────────▼────────────────────────────────┐
│  策略层  （可插拔）                                     │
│  ├─ NanoAgent  七阶段确定性编排（本项目主策略）          │
│  ├─ ReactAgent 单 prompt ReAct 基线（结构消融对照组）    │
│  └─ （训练位）RL policy / SFT 模型                      │
└────────────────────────────────────────────────────────┘
```

`BaseAgent`（`nanoshop/agents/base.py`）只约定两件事：任务开始时 `start_task()` 重置状态，每轮 `step(turn, role, message)` 返回下一个动作。环境调用、用户模拟、守卫逻辑、落盘全部在 harness 层，换策略零改动评测代码。

## 策略一：NanoAgent（七阶段编排）

用**工程编排**而非 RL 训练拿到可控、可解释的多轮导购能力，LLM 的"一次生成"被拆成七个可独立观测、可单独消融的阶段：

```
用户/Shopper 消息
      │
      ▼
┌─────────────┐   解析观察页（搜索结果/商品页/购买页）、防死循环守卫
│ 1 Guardrails │   （last_actions 去重、selected_option 状态跟踪）
└─────────────┘
      ▼
┌─────────────┐   意图/槽位抽取（category/attributes/option/budget/scenario）
│ 2 NLU        │   + query 改写 + 增量事实提取（LLM 结构化输出）
└─────────────┘
      ▼
┌─────────────┐   环境内 BM25 检索（jieba 分词），action: search[关键词]
│ 3 Retrieve   │
└─────────────┘
      ▼
┌─────────────┐   商品页属性核验：目标槽位 vs 页面属性/价格/规格
│ 4 Verify     │   → ProductView(verdict: match/partial/mismatch)
└─────────────┘
      ▼
┌─────────────┐   反思：证据是否充分？还缺什么槽位？是否该澄清或换品？
│ 5 Reflect    │   （与 Generate 融合为一次 LLM 调用，省 token）
└─────────────┘
      ▼
┌─────────────┐   决策生成：ask_user / search / click / buy now
│ 6 Generate   │
└─────────────┘
      ▼
┌─────────────┐   facts 累积 + 12 轮以上触发 LLM 摘要压缩（保留最近 6 轮）
│ 7 Memory     │
└─────────────┘
```

**Turn / Stage 两级 Trace**：每轮记录各阶段各自的输入快照与输出，任务结束后可完整回放决策链路——这是组件级指标的数据来源。

## 策略二：ReactAgent（ReAct 基线）

与七阶段唯一差异是"没有结构"：无 NLU 槽位、无证据状态、无 Memory 压缩，全部信息就是对话 + 观察原文塞进一个 prompt，每轮一次 LLM 调用输出 Thought/Action。作为**结构消融的对照组**，它回答的问题是：七阶段编排带来的收益有多少？分别来自哪个组件？

```bash
python -m nanoshop.eval.run_eval --mode multi --persona false --agent react --n 30
```

两个策略跑同一任务集（同 seed 采样）、同一 env-loop、同一指标，端到端分数差 = 结构总收益；组件级指标的差值（ReAct 上 slot F1 / verify_coverage 自然缺失或归零）把收益逐组件定位。大规模（100 任务 × 两策略）对比结果见下表（运行中）。

## 评测结果

官方协议：reward 四维（r_type / r_att / r_option / r_price），4 setting（multi/single × standard/persona）× 30 任务 + 4 消融 × 15 任务，DeepSeek-chat 驱动；主指标附 **Bootstrap 95% CI**（2000 次重采样）。

**端到端（NanoAgent，4 setting × 30 任务）**

| Setting | reward（宽松成功） | 全维成功 | 买对商品率 |
|---|---|---|---|
| multi_standard（主 setting） | 0.643 | 0.433 | 0.533 |
| single_standard | 0.679 | 0.433 | 0.500 |
| multi_persona | **0.762** | 0.500 | **0.733** |
| single_persona | 0.746 | **0.533** | 0.600 |

> 两个 persona setting 都优于对应 standard setting：user_persona 文档为 agent 提供了指令之外的隐性需求线索，弥补了多轮对话中 shopper 只逐步透露偏好的信息缺口。

**组件级指标**（multi_standard，本项目提出）

| 指标 | 含义 | 结果 |
|---|---|---|
| slot F1 | NLU 槽位 vs 官方 goal 属性（thefuzz>85 模糊匹配） | 0.343 |
| gold_candidate_hit | 检索结果是否包含 goal 商品 | 0.957 |
| clarify_rate | 购买前向用户澄清确认的比例 | 0.251 |
| verify_coverage | 决策前完成属性核验的比例 | 0.310 |
| 平均轮数 / 平均搜索次数 | — | 17.3 / 2.7 |

端到端 reward 只能看到"买没买对"，看不到"错在哪一步"：slot F1 低 + gold_candidate_hit 高说明主要损失在属性核验而非检索；这类定位只有靠组件级指标才能做出。

**消融实验**（multi_standard，同一 15 任务集；基线 = 完整版在该任务子集上的成绩）

| 配置 | r_loose | Δ vs 基线 | 买对商品率 | 结论 |
|---|---|---|---|---|
| 完整版（基线） | 0.639 | — | 0.467 | — |
| − NLU | 0.450 | −18.9pp | 0.267 | 槽位缺失 → 检索 query 丢失关键属性，检回错误商品 |
| − Verify | 0.381 | −25.7pp | 0.267 | 伤害最大：不核验详情页属性直接下单，r_option 0.667→0.429 |
| − Reflect | 0.547 | −9.2pp | 0.467 | 买对商品率不变，伤害集中在"搜错后不复盘换 query"的恢复能力 |
| − Memory | 0.698 | +5.9pp* | 0.600 | *n=15 下约 1 个任务的噪声量级，≈持平；见下方分析 |

> **− Memory 的发现**：在 15–40 轮的中等长度任务上，摘要压缩的信息损失抵消了 context 节省的收益。组件级评测的价值正在于此——不是所有组件在所有任务长度上都正贡献，Memory 阶段应按对话长度自适应门控（超长任务才启用压缩），这是明确的后续改进方向。

## 轨迹导出：评测 → 训练闭环

`scripts/export_trajectories.py` 把评测输出转成两类训练可用 JSONL：

- **任务级**：字段对齐 Search-R1 训练数据格式（id / question / answers / data_source + 完整 gold goal），RL rollout 与奖励计算可直接消费；
- **轨迹级**：多轮 messages（shopper=user，agent=assistant，环境动作挂在 `env_action` 字段），`--min-reward` 拒绝采样只保留高奖励轨迹，作为 SFT 冷启动或过程奖励（IGPO 式 turn-level reward）计算的输入。

```bash
python scripts/export_trajectories.py --setting multi_standard --tag main --min-reward 0.9
```

同一环境既出评测报告、又出训练数据，策略升级后回到同一 benchmark 重测——评测和训练在一条数据流上闭环。

## 环境适配（Mac 本地化，零 GPU）

官方环境依赖 JDK 21（Lucene/pyserini）+ torch + selenium + Flask 多进程部署。本仓库在纯 CPU Mac 上跑通全量 23,421 个商品/任务条目（eval 数据文件 134MB）：

- **BM25 替换 Lucene**：jieba 分词 + rank_bm25，保持 `search(query,k)/doc(docid)` 接口与官方 engine 完全兼容
- **进程内单例环境**：绕过 Flask 20-env 部署，手动构造共享 SimServer（内存 1.5GB 一次加载，多 worker 复用）
- **importlib 直载 text_env**：绕开 envs/__init__.py 连带引入的 selenium 依赖

## 快速开始

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # openai jieba rank_bm25 thefuzz python-dotenv

cp .env.example .env              # 填入 DeepSeek/OpenAI 兼容 API key
export SHOPSIM_REPO=/path/to/ShopSimulator   # 官方仓库（数据 + 源码）

# 干跑冒烟测试（不调 API，MockLLM 驱动两种策略过完整 env-loop）
python scripts/smoke_test.py

# 单任务真实评测（默认 nano 策略；--agent react 切换基线）
python -m nanoshop.eval.run_eval --mode multi --persona false --task-ids 563
python -m nanoshop.eval.run_eval --mode multi --persona false --agent react --task-ids 563

# 完整 suite：4 setting × 30 + 4 消融 × 15（第三个参数选策略）
python scripts/eval_suite.py 30 15        # nano
python scripts/eval_suite.py 30 15 react   # react 基线

# 汇总指标（含 Bootstrap 95% CI）
python -m nanoshop.eval.metrics --dir outputs/multi_standard/main

# 轨迹导出
python scripts/export_trajectories.py --setting multi_standard --tag main
```

## 项目结构

```
nanoshop/
├── agents/          # 可插拔策略层
│   ├── base.py      #   BaseAgent 接口（start_task / step）
│   ├── nano_agent.py    #   七阶段策略（适配 pipeline）
│   └── react_agent.py   #   ReAct 基线（结构消融对照组）
├── pipeline.py      # 七阶段编排（核心）
├── agent.py         # env-loop harness：环境交互 + 守卫 + 落盘（策略无关）
├── state.py         # AgentState 槽位状态机 + ProductView
├── trace.py         # Turn/Stage 两级 Trace
├── env_bridge.py    # ShopSimulator Mac 本地化适配层（BM25/单例环境）
├── shopper.py       # 官方协议 Shopper 模拟器（渐进透露目标）
├── llm.py           # OpenAI 兼容客户端 + 重试 + token 统计
└── eval/
    ├── run_eval.py  # 4 setting 采样评测 + 策略/消融开关
    └── metrics.py   # 端到端 + 组件级指标 + Bootstrap 95% CI
scripts/
├── smoke_test.py           # 无 API 干跑验证
├── eval_suite.py           # 全量 suite 总控（支持 nano/react）
├── export_trajectories.py  # 轨迹/任务导出（Search-R1 兼容格式）
└── daemon_start.py         # double-fork 守护进程启动器
```

## License

Apache-2.0
