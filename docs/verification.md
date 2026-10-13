# 验证记录

验证日期：2026-10-13（Asia/Shanghai）；保留初版训练与 GPU 验证记录，补充牌谱获取集成结果。

| 检查 | 结果 |
|---|---|
| 静态检查 | Ruff 通过 |
| CPU 测试 | 206 passed，包括 100 场完整模拟单局及 68 项牌谱获取/批量/版本解析测试 |
| 牌谱回放 | 3 组合成 golden × MJAI/已解码雀魂 JSON，最终分数检查通过 |
| 二进制牌谱闭环 | 3 组合成 golden × 旧 records/新 actions.result protobuf → JSON → Parquet，合法标签与终局分数通过 |
| 下载模拟 | 登录、共享牌谱登记读取、inline/data_url 内容下载、不保存密码、断开与超时处理通过 |
| 账号批量下载 | 20 项测试覆盖分页、短页继续、本人归属筛选、一次登录、多场失败后重试、缓存与训练参数校验、取消后继续、CLI 状态退出码 |
| 真实公开网关 | 自动发现客户端版本和 5 个网关，WebSocket 连接通过；未尝试账号登录 |
| 真实 Route 握手 | 修复后真实国服 requestConnection 返回 error.code=0、result=1；包含平台字段 Web |
| 新版客户端版本 | 官方 Unity 元数据动态解析 package=4.0.47、resource=0.16.283，登录标识 WebGL_2022-0.16.283；不使用旧 0.11.252 作为登录版本 |
| 首次真实账号尝试 | 修复前登录返回错误 151，下载 0 场；修复后真实登录、自己账号列表分页、首批原始文件与解码 JSON 保存成功 |
| BC smoke | 50 optimizer steps，loss 有限 |
| IQL smoke | 50 optimizer steps，Q/V/actor loss 有限 |
| PPO smoke | 1 次完整 learner 单局采样与更新，21 条 learner 决策 |
| 断点恢复 | CPU BC/IQL/PPO 连续与断点训练模型参数逐项完全相同 |
| 合法推理 | 成功从当前合法候选中选择动作 |
| 匹配竞技 smoke | 2 seed blocks × 4 座位 × 2 对照腿，共 16 场单局，JSON 报告成功生成 |
| 半庄模拟 | 另有 2 场完整半庄，804 / 916 条决策，状态和合法标签校验通过 |
| GPU BF16 闭环 | RTX 4070 Ti SUPER，PyTorch 2.14.1+cu130；50 BC steps + 50 IQL steps + 1 PPO update + 合法推理通过 |
| GPU Small 架构 | 256 维、6 层、8 heads、256 历史事件；BC/IQL 各 3 steps，4 次梯度累积，BF16 通过 |
| GPU Small 显存 | 短跑 BC allocated 峰值 626 MiB / reserved 768 MiB；IQL allocated 425 MiB / reserved 772 MiB |
| Docker/Linux CI | 文件已交付，尚未在对应环境执行 |
| 真实牌谱/实机视觉 | 没有用户数据；未声明实机精度或真实 golden 验收通过 |

具体日志和 smoke 模型在 `runs/smoke/`。两组 seed 的平均顺位 2.875、paired rank delta +0.375，样本过少，不能据此判断模型强弱或声称强化学习提升。

`fetch-replay` / `fetch-replays` / `decode-replay` 使用方法见 [牌谱获取说明](replay-acquisition.md)。批量入口自动分页读取登录账号的四人标准段位牌谱，保存逐场结果与汇总；重复执行会校验并跳过已完成项。初版公开连接验证读到了旧网页接口返回的 `web-0.11.252`，首次真实账号尝试暴露了旧登录版本与缺少新版 Route 握手的问题。修复后从官方 Unity 网页和资源包动态读取版本，并验证了公网握手。修复后的真实登录和首批批量下载已成功，每份已归档牌谱通过番符及终局分数核对；完整批次仍在运行，100 场真实 golden 的动作与训练样本验收尚未完成。

GPU 日志和模型在 `runs/gpu-smoke/` 与 `runs/gpu-small/`。显存数字来自 315 条单局样本的短跑，不能用作全量训练或 Medium 模型的显存预算。

原研究报告保持原样，SHA256：`6b02d2e643083a627003eec6369f26f0fe1423dd397bbb21b9a12a56c9820332`。
