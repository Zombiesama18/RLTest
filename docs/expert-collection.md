# 高段位观战牌谱采集

`collect-experts` 从国服公开观战列表发现对局，将 ID 存入 SQLite；用完整终局记录确认结束后下载，再按最终牌谱里的四人段位筛选玩家。采集器不需要打开雀魂客户端。公开发现不需登录，结束确认和下载需在本机终端登录一次。

## 直接运行

当前机器已经安装相关依赖。在本机 PowerShell 中：

```powershell
cd C:\Projects\JyangTamaRL

# 完整采集：运行一小时，原始牌谱与训练样本都保存
.\.venv\Scripts\python.exe -m mjsrl collect-experts --dataset-dir data\expert-silver

# 可选：无需登录，先发现一轮公开观战对局
.\.venv\Scripts\python.exe -m mjsrl collect-experts --discover-only
```

完整采集会隐藏输入密码，密码和登录令牌不写入磁盘。使用支持的国服账号密码登录；国际服和第三方登录仍未实现。重新执行同一命令会继续同一队列，已完成项不会重复下载。

默认设置：玉之间与王座之间四人半庄、至少一名雀圣或魂天、每轮间隔 60 秒、每个请求至少间隔 2 秒、队列累计最多 100 场、未取得可用牌谱最多等待 6 小时。运行时长是本次采集窗口，到期保留待结束的对局；进行中的请求/导入可能使退出稍晚。`finished` 表示本次窗口结束，不能据此认为所有对局都下载成功。

只采四名魂天的王座半庄：

```powershell
.\.venv\Scripts\python.exe -m mjsrl collect-experts --rooms throne-south --min-rank celestial --min-players 4 --state-dir data\celestial --dataset-dir data\celestial-silver --duration 7200 --max-games 200
```

房间/段位/人数条件在队列创建时固定。改变这些条件要用新的 `--state-dir`，避免混合不同数据定义。`--max-games` 是整个队列的累计上限，包含保存、拒绝与过期项；`0` 表示不限。数量有限时在多个房间间交替入队，防止第一间房占满名额。

## 下载后怎样筛选

| 参数 | 含义 |
|---|---|
| `--min-rank master` | 雀豪及以上 |
| `--min-rank saint` | 雀圣及以上，默认 |
| `--min-rank celestial` | 魂天 |
| `--min-players 1..4` | 一场对局中至少多少名玩家达到上述门槛 |
| `--rooms jade-east jade-south throne-east throne-south` | 选择一个或多个房间，默认两个半庄房间 |

观战玩家数组的顺序不能当作座位：`seat_list` 是账号 ID 列表。下载后核对对局 ID、房间模式、四名账号一致、四个座位与完整结算，再以最终 `accounts[].seat` / `accounts[].level.id` 为准。

训练输出只保留达到门槛的玩家决策，并保留高手输牌的样本。混合段位的一桌不会把低段位对手也当作专家标签。整场回放、合法动作和终局核验在筛选前执行；对手状态和原有决策时钟、跨决策折扣保持一致。段位是质量代理，不代表每一步都最优。

识别四人梯级 `101xx..105xx`，以及旧魂天 `10601`、新版魂天 `10701..10720`。不把三人段位 `2xxxx` 或未知编号当成四人高手。规则、番符或终局分数不一致的牌谱进入隔离状态，保留原始文件供修复。

## 保存与恢复

默认目录：

```text
data/experts/
  queue.sqlite                 对局发现、首次观察时间、筛选条件、重试及结果
  status.json                  当前运行状态和各状态场次数
  collector.lock               防止同目录同时运行两个采集器
  raw/<对局ID>/record.pb        完整牌谱内容，先保存再解析
  raw/<对局ID>/replay.json      已解码回放
  raw/<对局ID>/manifest.json    来源、校验和、四人段位与 selected_seats
data/expert-silver/
  <对局ID>.parquet              达到门槛的玩家样本
  <对局ID>.batch.json           数据指纹、gamma、段位/座位筛选条件
```

`waiting` 等待终局或牌谱就绪；`saved` 原始/解码归档通过核验（指定 dataset-dir 时也已完成训练导出）；`rejected` 最终身份、模式或段位不符合；`quarantined` 解码、规则核验或训练导出失败；`unavailable` 过期仍未取得可用牌谱。原因保存在 SQLite 的 `games.reason`。

服务器尚未提供记录时指数退避，最长间隔 30 分钟；对局从观战列表消失不会直接触发下载。登录失效或连接中断会保存队列并退出，重新运行可继续。已经落盘的原始文件在解析中断后复用。隔离/拒绝/过期项不会自动重新采集，需要检查原因、修复解析器并离线处理。

如果第一次只存了牌谱，可以在之后离线导出，无需再次登录：

```powershell
.\.venv\Scripts\python.exe -m mjsrl collect-experts --export-only --dataset-dir data\expert-silver
```

导出会核验缓存 SHA256。已有训练文件的校验和、gamma 或段位/座位筛选条件变化时拒绝覆盖。使用自定义队列时，离线导出也须提供同样的 `--state-dir`、房间和段位参数。

当前网关的旧 `heatbeat` 请求会返回错误码 6，不能将它误报成密码错误或登录失效。采集器改用需要登录的只读 `fetchServerTime`：登录后立即确认，并在轮询空闲时每 20 秒保活；认证失败 1004 才报告登录会话失效。其他错误保留接口名与错误码，原因写入 `status.json.reason`。本机交互窗口中的密码从不保存，退出后重试需要重新输入。

来源清单使用 `configs/expert_manifest.example.json` 的 `public_replay`，准确记录公开观战来源；不将他人牌谱标成自己拥有或玩家明确授权。`redistributable` 固定为 false，原始文件含玩家标识，采集数据和凭据不随代码上传 GitHub。

## 调研与验证边界

接口定义依据 [mahjong_soul_api 的 Lobby RPC 实现](https://github.com/MahjongRepository/mahjong_soul_api/blob/master/ms/rpc.py)，观战入口见 [雀魂游戏指南](https://mahjongsoul.com/startguide/)。房间和 `200 + mode_id` 的观战筛选关系参考作者的 [观战接口调用示例](https://qiita.com/pearswatchover/items/13ae5ad394963d137461)，并在当前国服接口验证：玉半庄 mode=12/filter=212，王座半庄 mode=16/filter=216。东风分别为 11/211、15/215，已实现但尚未真机验证。

2026-10-13：匿名 Route 握手和两个半庄公开列表真实请求成功；程序首次验证发现 90 场符合门槛的对局、42 场被过滤，按上限保存 20 场等待队列。匿名结束详情和牌谱请求返回 1004，因此完整流程复用账号登录。

完整链路用合成已知结算牌谱验证，覆盖结束延迟、完整结算、最终段位/座位复核、高手败局、原始落盘后中断恢复、下载重试、缓存校验、认证失效和离线导出。2026-11-03 已实际完成公开观战对局结束后的登录下载与归档；真实训练 Parquet 和至少 100 场 golden 的动作验收仍待完成。

测试还暴露了固定版本 RiichiEnv 回放初始化事件里的随机宝牌/首摸，现已从当前可观察状态校正。原生 steps 的事件历史仍仅有有限初始化事件，完整时序事件历史尚需扩展；合法动作、手牌、河牌、副露及最终分数仍由原生回放核验。该限制影响历史编码及剩余牌数特征，专家数据正式训练前应建立真实 golden 集并继续完善。源码依据 [RiichiEnv v0.4.10 回放实现](https://github.com/smly/RiichiEnv/blob/v0.4.10/riichienv-core/src/replay/mod.rs)。

2026-11-03：静态检查及完整 253 项测试通过，其中 47 项覆盖专家采集；异步用例需在允许本地 socketpair 的正常执行环境中运行。当前官方版本动态解析为 WebGL_2022-0.16.289、package 4.0.51，发现 5 个网关。实际公开列表符合门槛 100 场、过滤 36 场，100 场已保存至 `data/experts/queue.sqlite`。修复旧心跳后，本机重新登录与跨轮询下载成功；运行快照为 12 场保存、1 场因终局分数不符被隔离、87 场等待。12 份归档的原始 SHA256、最终房间/段位筛选核对通过，共包含 45 个雀圣及以上玩家座位。本次先保存原始文件、解码 JSON 与段位清单，未导出训练 Parquet；采集仍继续运行。
