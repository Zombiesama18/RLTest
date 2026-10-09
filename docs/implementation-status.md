# 实施状态与报告差异

本版目标是提供能在真实立直规则模拟器上运行的训练研究工程。没有把研究报告中的工程目标、样本规模或预计精度当成已实现指标。

| 报告模块 | 本版实现 | 仍需完成 |
|---|---|---|
| Schema | 版本化 tile/state/action/transition、赤牌、公开字段白名单 | 完整 canonical event schema、规则版本迁移 |
| Rules | RiichiEnv 0.4.10 雀魂 preset、三种四人模式、合法候选 | 独立规则引擎 differential suite、罕见规则黄金局 |
| Replay | 已解码雀魂 JSON/gzip 与 MJAI，原生 steps、番符核对、可选最终分数 gate | 雀魂原始 protobuf decoder、至少 100 场真实授权 golden |
| Data | raw vault/清单、Parquet、整场散列切分、HMAC 工具、半 MDP transition | 全 canonical bronze、百万局流式分片、玩家群组/时间切分、质量权重 |
| BC | Transformer 合法候选 NLL、梯度累积、验证指标、续训 | 专家牌谱、大规模调参、warmup/cosine |
| IQL | expectile V、twin-Q backup、优势加权 actor、EMA target、共享 encoder | 竞技提升验证、reward ablation、显存优化 |
| PPO | learner 座位 on-policy rollout、GAE、clip、KL stop、快照对手池、续训 | 多进程 rollout、历史强快照评级、100M 决策稳定性 |
| Arena | 同初始种子和座位对照、按 seed cluster CI、基础攻守指标 | 完整 duplicate-wall schedule、Mortal/Elo、大规模半庄 |
| Vision | 磁盘图片 ROI、37 类模板、低置信拒绝、三帧融合、结构校验、过期候选拒绝 | 实机标定/模板、OCR、完整状态恢复、10k gold frames |
| MahJax | 未引入 | 固定版本的可选 adapter 与独立验证 |
| Checkpoint | 原子写、模型/AdamW/target/RNG/采样/league/数据指纹 | 独立 safetensors 发布权重、跨机器恢复验证 |
| Observability | JSONL 日志、可选 TensorBoard | 系统吞吐/显存监控、W&B |
| Packaging | 固定版本依赖、PowerShell setup/smoke、CPU Docker、CI 文件 | 容器 digest/hash lock、实际 Linux/container 验证 |

奖励暂采用终局点数与顺位；报告里的逐小局 clip 尚未接入。历史对手池按时间保存，没有称为“历史强模型”。Arena 只保证初始牌山可复现，后续局次可能随策略改变。模型本身是轻量 Transformer 基线，训练 smoke 使用 201,796 参数；正式 Small / Medium 配置需要相应数据规模。

报告中“自动点击、实时段位建议、协议注入”不属于此项目的部署范围；CLI 只接受磁盘牌谱、离线图片和模拟器，不接入客户端账户或输入控制。

正式训练准入顺序：收集授权牌谱 → 建立真实 golden replay gate → 验证完整得点/动作 → 构建独立训练与评估集合 → BC/IQL → 大规模 paired arena → PPO 规模化。当前 smoke 只能确认工程链路。
