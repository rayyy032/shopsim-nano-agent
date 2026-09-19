# ShopSim Nano Agent：中文电商多轮导购 Agent

基于阿里 **ShopSimulator**（ACL 2026, arXiv:2601.18225）淘宝购物环境构建的多轮对话导购 Agent。采用 nano-agent 极简架构思想，将 LLM 的"一次生成"拆解为**七阶段确定性编排**，每个阶段可独立观测、可单独消融。环境与评测协议 Based on 阿里 ShopSimulator，七阶段 pipeline、状态机、Trace、组件级评测与全部实现代码为本人独立完成。

## 核心设计：七阶段编排

不走 RL 训练路线，用**工程编排**拿到可控、可解释的多轮导购能力：

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

**Turn / Stage 两级 Trace**：每轮记录七个阶段各自的输入快照与输出，任务结束后可完整回放决策链路。

## 评测结果

官方协议：reward 四维（r_type / r_att / r_option / r_price），4 setting（multi/single × standard/persona）× 30 任务 + 4 消融 × 15 任务，DeepSeek-chat 驱动。

**端到端（4 setting × 30 任务）**

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

**消融实验**（multi_standard，同一 15 任务集；基线 = 完整版在该任务子集上的成绩）

| 配置 | r_loose | Δ vs 基线 | 买对商品率 | 结论 |
|---|---|---|---|---|
| 完整版（基线） | 0.639 | — | 0.467 | — |
| − NLU | 0.450 | −18.9pp | 0.267 | 槽位缺失 → 检索 query 丢失关键属性，检回错误商品 |
| − Verify | 0.381 | −25.7pp | 0.267 | 伤害最大：不核验详情页属性直接下单，r_option 0.667→0.429 |
| − Reflect | 0.547 | −9.2pp | 0.467 | 买对商品率不变，伤害集中在"搜错后不复盘换 query"的恢复能力 |
| − Memory | 0.698 | +5.9pp* | 0.600 | *n=15 下约 1 个任务的噪声量级，≈持平；见下方分析 |

> **− Memory 的发现**：在 15–40 轮的中等长度任务上，摘要压缩的信息损失抵消了 context 节省的收益。组件级评测的价值正在于此——不是所有组件在所有任务长度上都正贡献，Memory 阶段应按对话长度自适应门控（超长任务才启用压缩），这是明确的后续改进方向。

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

# 干跑冒烟测试（不调 API，ScriptedLLM mock）
python scripts/smoke_test.py

# 单任务真实评测
python -m nanoshop.eval.run_eval --mode multi --persona false --task-ids 563

# 完整 suite：4 setting × 30 + 4 消融 × 15
python scripts/eval_suite.py 30 15

# 汇总指标
python -m nanoshop.eval.metrics --dir outputs/multi_standard/main
```

## 项目结构

```
nanoshop/
├── pipeline.py      # 七阶段编排（核心）
├── agent.py         # 任务主循环 + 防死循环守卫 + Trace 落盘
├── state.py         # AgentState 槽位状态机 + ProductView
├── trace.py         # Turn/Stage 两级 Trace
├── env_bridge.py    # ShopSimulator Mac 本地化适配层（BM25/单例环境）
├── shopper.py       # 官方协议 Shopper 模拟器（渐进透露目标）
├── llm.py           # OpenAI 兼容客户端 + 重试 + token 统计
└── eval/
    ├── run_eval.py  # 4 setting 采样评测 + 消融开关
    └── metrics.py   # 端到端 + 组件级指标
scripts/
├── smoke_test.py    # 无 API 干跑验证
├── eval_suite.py    # 全量 suite 总控
└── daemon_start.py  # double-fork 守护进程启动器
```

## License

Apache-2.0
