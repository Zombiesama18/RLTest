# Smoke 模型卡

- 用途：验证离线麻将研究工程，从公开结构化状态在引擎合法候选中选动作。
- 规则：雀魂四人赤牌，RiichiEnv 0.4.10 `default_mjsoul`。
- 模型：201,796 参数，2 层、64 维、4 heads Transformer，最多 64 个历史事件，加候选 policy / twin-Q / value heads。
- 数据：四场单局模拟，315 条 transition，由简单向听启发式产生，没有真人专家牌谱。
- 训练：CPU，50 BC optimizer steps，50 IQL optimizer steps，一次单局 PPO update；seed 42。
- GPU：相同流程在本机 RTX 4070 Ti SUPER 上以 BF16 通过，产物独立保存到 `runs/gpu-smoke/`。
- 评价：已验证训练和合法推理。两组 seed 的 16 场竞技仅检验代码，不是统计意义的棋力评价。
- 局限：无大规模训练，无真实牌谱评估，不具备已证明的对局水平；半庄奖励和策略尚需正式实验。
- 输入范围：本人手牌、公开河牌与副露、分数、宝牌、公开事件和引擎合法动作。其他玩家手牌与牌山不进入模型。
- 发布状态：仅本地研究产物；`runs/smoke/` 不纳入 Git，不能作为成熟模型传播。
