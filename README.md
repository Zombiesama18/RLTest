# 雀魂强化学习研究项目

依据《雀魂强化学习_deep-research-report.md》实现的可运行研究基线：授权牌谱或模拟对局 → 统一可观察状态 → Transformer 合法候选动作策略 → BC → 离散 IQL → PPO 自对弈 → 匹配种子竞技评估。

当前版本提供训练系统与离线复盘接口。它没有经过大规模棋力训练，不是完整报告全部里程碑的生产验收版本。详见 [实施状态](docs/implementation-status.md) 和 [验证记录](docs/verification.md)。

## 快速运行（Windows / PowerShell）

```powershell
cd C:\Projects\JyangTamaRL
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
.\.venv\Scripts\python.exe -m mjsrl doctor
.\.venv\Scripts\python.exe -m mjsrl smoke --steps 50
```

如果当前机器已经由本次任务安装好 `.venv`，可直接运行最后两行。`smoke` 生成四场模拟单局，执行 50 步 BC、50 步 IQL、一次 PPO 更新，并输出 `runs/smoke/` 下的模型、数据、训练日志和合法动作推理结果。演示数据来自向听数启发式，不能当作专家牌谱。

GPU 安装：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1 -GPU
.\.venv\Scripts\python.exe -m mjsrl smoke --device cuda --steps 50 --output runs\gpu-smoke
```

脚本从 PyTorch 官方 CUDA 13.0 索引安装固定的 2.14.1 版本。需要匹配的 NVIDIA 驱动，`doctor` 应显示 `cuda: true`。CUDA 构建选择参考 [PyTorch 安装说明](https://pytorch.org/get-started/locally/)；首次下载可能较大。CPU 运行不需要 CUDA。

Linux 可使用 Python 3.12：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-cpu.lock
pip install --no-deps -e .
python -m mjsrl smoke --steps 50
```

## 生成与检查数据

```powershell
.\.venv\Scripts\python.exe -m mjsrl generate --games 100 --mode 4p-red-half --output data\sim.parquet
.\.venv\Scripts\python.exe -m mjsrl validate --data data\sim.parquet
```

规则层固定 `riichienv==0.4.10`，显式使用 `GameRule.default_mjsoul()`。支持四人赤牌单局、东风和半庄。模型只读取本人手牌和公开信息，候选动作始终由规则引擎生成。赤五与普通五独立编码；立直候选包含要打出的牌。被吃碰的河牌在物化状态里移入副露，避免可见牌重复计数。

每条 transition 连接同一玩家的相邻决策，保存中间步数、折扣与终止标志。当前奖励采用**整场终局**点数变化 `clip(delta / 8000, -2, 2)` 加名次奖励 `{1:1, 2:.25, 3:-.25, 4:-1}`；可切换 `rank` / `points`，尚未实现逐小局奖励。单局模式的名次只用于验证训练，不能衡量半庄棋力。

Parquet 按游戏 ID 的稳定散列分配 80/10/10 的 train/validation/test，同一整场不会跨集合。小数据集某个集合可能为空；`--split all` 仅用于演示。读取器按批解析，但当前离线训练会把所选集合载入内存，百万级牌谱应先扩展为分片流式训练。

## 导入自己的雀魂牌谱

接受**已解码、离线导出的 JSON**：

```json
{
  "data": [
    [
      {"name": "NewRound", "data": {"scores": [25000,25000,25000,25000], "tiles0": [], "tiles1": [], "tiles2": [], "tiles3": [], "chang": 0, "ju": 0, "liqibang": 0}},
      {"name": "DealTile", "data": {"seat": 0, "tile": "1m"}}
    ]
  ]
}
```

上面是格式示意，真实记录必须包含完整手牌、宝牌和完整动作序列。具体可运行样例在 `tests/fixtures/*.majsoul.json`，它们由模拟器生成。也支持 `{"rounds": [...]}`、根数组或 `.json.gz`。MJAI JSONL 使用 `--format mjai`。

复制 `configs/data_manifest.example.json`，按真实来源填写 `permission_basis`，可加入 `sha256` 和 `expected_final_scores`。导入时保留原始文件副本和清单，规则引擎检查可执行决策与合法标签；雀魂格式还检查和牌番符。公开状态不携带昵称、账户 ID 或其他人的隐藏手牌。

```powershell
.\.venv\Scripts\python.exe -m mjsrl ingest --input my_replay.json --manifest my_manifest.json --output data\silver\my_replay.parquet
```

本版不接受牌谱链接或未解码的二进制网络响应。牌谱解析复用 RiichiEnv 原生 parser/step iterator，未复制 Kanachan 或 Mortal 代码。实际授权雀魂牌谱还需要建立至少 100 场真实 golden replay 集；当前仓库只含合成验证样例。

## 正式训练与续训

```powershell
# CPU 演示（训练集默认按整场划分）
.\.venv\Scripts\python.exe -m mjsrl train --config configs\bc_smoke.json --data data\sim.parquet --output runs\bc

# RTX 4070 Ti SUPER：先从 Small 模型开始
.\.venv\Scripts\python.exe -m pip install '.[logging]'
.\.venv\Scripts\python.exe -m mjsrl train --config configs\bc_small.json --data data\sim.parquet --validation data\sim.parquet --output runs\bc-small
.\.venv\Scripts\python.exe -m mjsrl train --config configs\iql_small.json --data data\sim.parquet --init runs\bc-small\latest.pt --output runs\iql-small
.\.venv\Scripts\python.exe -m mjsrl train --config configs\ppo_small.json --init runs\iql-small\latest.pt --output runs\ppo-small

# --steps 是训练完成后的总步数；不是再训练多少步
.\.venv\Scripts\python.exe -m mjsrl train --config configs\iql_small.json --data data\sim.parquet --resume runs\iql-small\latest.pt --steps 20000 --output runs\iql-small
```

初始化会继承父模型架构；改变模型尺寸需要新建训练。IQL 采用 target twin-Q 的 expectile V 回归、带 `discount` 的 Q backup 和优势加权行为克隆；共享状态编码器，目标网络 EMA 更新。PPO 使用实际 learner 座位的数据、合法动作 Categorical、GAE、clip、熵奖励和 KL 提前停止。对手按 50% 当前冻结模型、25% 近期快照、15% 历史快照、10% 启发式采样；历史快照尚未按竞技成绩筛选。

checkpoint 原子写入，包含模型、AdamW、target、完整随机状态、采样位置、数据校验和及 PPO 对手池。不同数据、集合或关键参数的续训会拒绝执行。CPU 已测试 BC/IQL/PPO 连续训练与中断续训的参数逐项完全一致；CUDA 不承诺跨驱动/跨设备位级一致。`*.pt` 含 Python 状态，只加载可信本地 checkpoint。

训练日志为 `metrics.jsonl`；可开启 TensorBoard。BF16 与梯度累积已实现；尚未实现 DDP、激活重算、scheduler 或目标网络 offload。Medium 配置提供架构入口，显存占用需要实测，不保证 IQL Medium 能放入 16GB。

## 离线推理和竞技评估

```powershell
.\.venv\Scripts\python.exe -m mjsrl infer --checkpoint runs\smoke\ppo\latest.pt --state runs\smoke\example_state.json
.\.venv\Scripts\python.exe -m mjsrl arena --candidate runs\ppo-small\latest.pt --baseline runs\iql-small\latest.pt --seeds 100 --mode 4p-red-half --output runs\arena.json
```

每个 seed 包含四个 learner 座位，各执行 candidate 与 baseline 两条对照对局：`--seeds 100` 实际为 800 场对局。报告平均顺位、得点、一位率、四位率、和率、放铳率和按种子块 bootstrap 的 95% 区间，负的 paired rank delta 表示 candidate 更好。一个 seed 不输出置信区间。

对照保证初始种子和座位相同；策略改变导致连庄、终局或后续局次不同后，不能宣称完整半庄逐局相同牌山。两组小样本只检验竞技代码，不用于确认升级有效。正式判断应使用独立种子和数千至数万场半庄。

## 离线视觉原型

`vision` 接收磁盘图片、37 类牌模板、归一化 ROI 配置和已标注的回放状态。它校验标定字段中的手牌/河牌/宝牌，并对低置信度、近似候选、牌数矛盾返回 `WAIT`；连续三帧稳定才输出状态。识别牌面与参考状态不同也返回 `WAIT`，需要回放规则引擎重新确认状态和生成合法动作，避免沿用过期候选。其余字段和合法动作由标注状态提供。这是可测试的牌模板与时间融合原型，**尚不能从任意雀魂截图自动恢复完整状态**；没有实机模板、场景分类、数字 OCR 或 QueHun 捕获适配。

```powershell
.\.venv\Scripts\python.exe -m mjsrl vision --frames frame1.png frame2.png frame3.png --templates my_templates --profile my_profile.json --state annotated_state.json --output estimate.json
```

profile 格式：`{"tile_slots":{"hand":[[x,y,w,h], ...], "rivers.1":[...], "dora_indicators":[...]}}`。坐标在 0..1，每个 ROI 对应一张可见牌；模板命名为 `1m.png` 等及 `5mr.png` / `5pr.png` / `5sr.png`。所有输入均为已结束对局的离线材料。

## 测试与结构

```powershell
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\python.exe -m pytest -q
```

测试覆盖赤牌、隐藏信息扰动不变性、候选顺序、非法 mask、可见牌数量、三组合成 golden 的两种牌谱格式、100 场模拟器单局、整场数据划分、BC/IQL/PPO 断点复现、视觉拒绝/融合与置信区间。GitHub Actions 包含 CPU 测试及训练 smoke；Dockerfile 是 CPU 研究运行入口，镜像构建和 Linux CI 尚需在对应环境验证。

```text
src/mjsrl/
  schema.py      版本化状态/牌/动作/transition
  engine.py      RiichiEnv 雀魂规则适配
  ingest.py      离线雀魂/MJAI 牌谱重放
  storage.py     Parquet、数据清单、HMAC 匿名化工具
  features.py    公开信息编码与候选动作打包
  model.py       Transformer + policy / twin-Q / V
  train.py       BC、IQL、PPO league
  checkpoint.py  原子保存与 RNG 恢复
  arena.py       匹配种子和座位、置信区间
  vision.py      离线牌模板与时间融合
  cli.py         统一入口
```

设计与接口依据 [RiichiEnv 官方仓库](https://github.com/smly/RiichiEnv)，离散 IQL 依据 [IQL 原论文](https://arxiv.org/abs/2110.06169)。项目未集成 MahJax、Mortal 对手或实时客户端控制。数据来源、规则引擎验证和棋力评估仍是后续研究工作的核心。
