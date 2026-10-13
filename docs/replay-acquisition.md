# 从雀魂获取训练牌谱

项目提供单场链接下载和自己账号的批量下载：下载已结束对局 → 保存原始 protobuf → 解码 JSON → 可选生成 Parquet。自动登录支持**雀魂国服账号密码登录、四人标准段位东风战与半庄战**。国际服、第三方账号登录、友人战、比赛和特殊规则暂不自动下载；已有 MJAI JSONL 可以继续通过 `ingest --format mjai` 导入。

## 第一次使用

在项目目录打开本机 PowerShell，安装可选依赖：

```powershell
cd C:\Projects\JyangTamaRL
.\.venv\Scripts\python.exe -m pip install '.[majsoul]'
```

本次集成已经在当前 `.venv` 中安装了这些依赖，可以直接使用下载入口。其他机器需要执行上述安装。

在雀魂的“牌谱”页面找到你自己的已结束对局，复制分享链接。复制 `configs/data_manifest.example.json` 到自己的来源清单，并确认 `permission_basis` 与实际来源相符。自己的牌谱使用 `user_owned`，经授权的他人牌谱使用 `explicit_permission`。来源清单用于记录数据来源，不要在其中放账号、密码或登录令牌。

下载并生成训练数据：

```powershell
.\.venv\Scripts\python.exe -m mjsrl fetch-replay --url "在此粘贴完整牌谱链接" --manifest configs\data_manifest.example.json --dataset data\silver\game001.parquet
```

程序在本机终端询问雀魂账号与密码；密码输入不回显。不提供命令行密码参数，不保存密码或登录令牌，不生成可复用登录令牌。`--account` 可以预先指定账号；下载过程中不会在日志中输出账号登录响应。先在自己的少量牌谱上验证，不能把第三方协议接入称为官方支持的导出功能。

默认输出到 `data/majsoul/<牌谱ID>/`：

| 文件 | 内容 |
|---|---|
| `record.pb` | 包含牌谱头和已下载牌局内容的完整 `ResGameRecord` protobuf，保留原始资料以便重新解码 |
| `replay.json` | 当前导入器支持的 `{"data": [[{"name", "data"}, ...], ...]}` 格式 |
| `manifest.json` | 数据来源、原始/解码文件 SHA256、协议版本及按座位排列的终局分数 |

加上 `--dataset` 会调用现有导入器，输出训练 Parquet；省略它则只下载与解码。`--output-dir` 可以改变牌谱保存目录，`--timeout` 调整单次请求等待时间（默认 30 秒）。原始文件可能包含玩家昵称、账号标识和完整牌山；训练状态仍由现有规则引擎生成，只暴露当前玩家可见信息。`data/` 已由 Git 忽略。

已有牌谱文件内容不同时不会覆盖；指定的训练文件已经存在时会拒绝运行。需要重新处理时使用新的输出目录或新的训练文件名。

## 自动读取自己账号并批量下载

无需整理链接清单。在本机终端运行，程序只登录一次，分页取得当前账号的段位牌谱，筛选四人标准东风/半庄，再逐场下载：

```powershell
.\.venv\Scripts\python.exe -m mjsrl fetch-replays --manifest configs\data_manifest.example.json --limit 100
```

默认选择最近 100 场符合条件的牌谱，包含已缓存的场次。`--limit 0` 表示遍历服务器当前可访问的列表，不能保证取得已经过期或服务器不再提供的全部历史对局。程序核对每份牌谱包含当前账号，来源清单必须填写 `user_owned`，不要在批量清单中填写某一场的校验和或终局分数。

下载全部可访问的符合条件牌谱，并逐场生成训练文件：

```powershell
.\.venv\Scripts\python.exe -m mjsrl fetch-replays --manifest configs\data_manifest.example.json --limit 0 --interval 2 --dataset-dir data\silver
```

`--interval` 默认为 2 秒，用于分页与下载请求间隔；不并发请求牌谱。每次启动在本机询问账号密码，启动后逐场处理无需人工操作。密码不保存，重启程序需要重新输入。

每场仍保存在 `data/majsoul/<牌谱ID>/`；训练文件位于 `--dataset-dir/<牌谱ID>.parquet`。训练文件按场保存，当前 `train --data` 仍接受单个 Parquet，跨场合并或目录式训练读取尚未提供。

可以重复执行相同命令继续处理：完整缓存通过原始/解码文件 SHA256 和归属校验后跳过；只有原始文件时尝试离线完成解码；已有牌谱缺少训练文件时直接使用缓存生成。训练分片旁的 `*.batch.json` 记录文件校验和、gamma 和状态协议版本，避免把参数不同或损坏的训练文件误认为已完成。缓存损坏或已有训练文件没有校验记录时拒绝覆盖，应使用新的输出目录。

每次运行在 `data/majsoul/batches/` 生成独立的 JSONL 逐场记录和 JSON 汇总，统计下载、复用、跳过、失败、生成训练文件和剩余场数。单场失败会记录原因并继续；连接断开时停止本批次，留下未处理数量。重新执行会重试未成功的项，同时跳过已校验的成功项。完整成功退出码为 0，部分失败或连接停止为 1，启动/列表错误为 2，用户取消为 130。

## 离线重解码

有了 `record.pb` 后，可以离线重新导入，不需要再次登录：

```powershell
.\.venv\Scripts\python.exe -m mjsrl decode-replay --input "data\majsoul\牌谱ID\record.pb" --manifest "data\majsoul\牌谱ID\manifest.json" --output-dir data\redecoded --dataset data\silver\game001-redecoded.parquet
```

`decode-replay` 的输入是本项目保存的完整 `ResGameRecord` protobuf 响应，不能把任意抓包内容或单独的 `GameDetailRecords` 当作同一格式。

也可以直接把已解码 JSON 交给现有导入器：

```powershell
.\.venv\Scripts\python.exe -m mjsrl ingest --input "data\majsoul\牌谱ID\replay.json" --manifest "data\majsoul\牌谱ID\manifest.json" --output data\silver\game002.parquet
```

## 校验与验证范围

当前版本从官方 Unity 网页读取包版本，从官方 `bundle_info_so.majset` 解压得到完整版本资源包名，再读取资源版本，生成 `WebGL_2022-<资源版本>` 登录标识。旧 `version.json` 只用于查找路由配置，不作为登录版本；无法解析新版元数据时会停止，不会静默回退到旧登录版本。

下载器读取雀魂公开客户端版本与路由配置，连接对应网关后先执行 `.lq.Route.requestConnection`（包括 `platform=Web`），再发送包含客户端及资源版本的账号登录请求。账号登录失败不会自动重试；下载只请求指定的已结束牌谱，不发出对局操作。某些共享牌谱会先调用 `readGameRecord` 登记查看，再获取内容；`data_url` 返回时下载器会读取其官方 HTTPS 地址。新格式 `actions[].result` 和旧格式 `records[]` 均可解码。

未知事件、非四人段位规则、未结束对局、缺少开局或结算、校验和不一致会拒绝处理。解码后使用 RiichiEnv 检查和牌番符及完整终局分数；生成训练数据时继续检查合法动作标签。protobuf 的空 `paishan` 字段不会被当作空牌山传给回放引擎。

已用三组合成牌谱验证新旧二进制容器到 Parquet 的完整流程，并通过模拟登录、共享牌谱读取和独立数据 URL 下载测试。批量测试覆盖分页、归属筛选、单次登录、失败重试、缓存校验、取消后继续及训练分片参数变化。真实公开配置、WebSocket 连接及新版 Route 握手已验证。首次真实账号尝试返回 151 后，已修正缺失握手和资源版本字段；**修复后真实登录、自己账号列表查询和首批牌谱保存已成功；完整批次及训练样本仍待进一步验收**。接口来自第三方协议库，后续客户端变化可能需要升级适配。

登录错误 151 指向连接/客户端版本兼容性问题，不能仅凭此错误认定密码不正确。相关协议变更参见 [协议库问题记录](https://github.com/MahjongRepository/mahjong_soul_api/issues/32)。

接口参考：[MahjongRepository/mahjong_soul_api](https://github.com/MahjongRepository/mahjong_soul_api)、[RiichiEnv](https://github.com/smly/RiichiEnv)。本项目未引入第三方 GUI 下载器或客户端脚本。
