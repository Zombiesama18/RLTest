"""Decode completed Mahjong Soul records and retain their acquisition provenance."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from .ingest import KNOWN_RECORDS, PERMISSION_BASES, ingest, load_replay

MAX_RECORD_BYTES = 32 * 1024 * 1024
RECORD_ID = re.compile(r"\d{6}-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}")
HOST_REGIONS = {
    "game.maj-soul.com": "cn",
    "game.majsoul.com": "cn",
    "game.mahjongsoul.com": "jp",
    "mahjongsoul.game.yo-star.com": "en",
}


class ReplayError(ValueError):
    """A replay cannot be retrieved or validated; messages never contain credentials."""


@dataclass(frozen=True)
class ReplayReference:
    game_id: str
    region: str


def parse_reference(value: str) -> ReplayReference:
    value = value.strip()
    match = re.search(r"https?://\S+", value)
    if match:
        url = urlsplit(match.group(0))
        if url.scheme != "https" or url.username or url.password or url.port not in (None, 443):
            raise ReplayError("请使用雀魂官方 HTTPS 牌谱链接。")
        region = HOST_REGIONS.get(url.hostname)
        if not region:
            raise ReplayError("不支持这个牌谱链接域名。")
        ids = parse_qs(url.query).get("paipu", [])
        if len(ids) != 1:
            raise ReplayError("牌谱链接必须包含一个 paipu 参数。")
        value = ids[0]
    else:
        region = "cn"
    # Shared links may append an encoded perspective account or a seat number.
    if not re.fullmatch(RECORD_ID.pattern + r"(?:_[a-zA-Z0-9]+){0,2}", value):
        raise ReplayError("无法识别牌谱 ID；请复制完整的单场牌谱分享链接。")
    return ReplayReference(value.split("_")[0], region)


def protocol():
    try:
        from ms import protocol_pb2
    except ImportError as exc:
        raise ReplayError("请先安装牌谱依赖：python -m pip install '.[majsoul]'") from exc
    return protocol_pb2


def load_manifest(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    validate_metadata(value)
    return value


def validate_metadata(value):
    if not isinstance(value, dict) or value.get("permission_basis") not in PERMISSION_BASES:
        raise ReplayError("来源清单需要 user_owned、explicit_permission 或 self_generated。")
    if value.get("ruleset") != "majsoul_4p_standard":
        raise ReplayError("来源清单必须指定 majsoul_4p_standard。")


def validate_head(head, game_id: str | None = None):
    if not RECORD_ID.fullmatch(head.uuid) or (game_id and head.uuid != game_id):
        raise ReplayError("服务器返回的牌谱 ID 与请求不一致或无效。")
    if head.end_time <= 0:
        raise ReplayError("只接受已结束对局的牌谱。")
    if sorted(account.seat for account in head.accounts) != [0, 1, 2, 3]:
        raise ReplayError("只接受完整的四人牌谱。")
    if head.config.category != 2 or head.config.mode.mode not in (1, 2):
        raise ReplayError("首版只支持四人标准段位东风战和半庄战。")
    rule = head.config.mode.detail_rule
    special = (
        "guyi_mode",
        "dora3_mode",
        "begin_open_mode",
        "jiuchao_mode",
        "muyu_mode",
        "open_hand",
        "xuezhandaodi",
        "huansanzhang",
        "chuanma",
        "reveal_discard",
        "field_spell_mode",
        "zhanxing",
        "tianming_mode",
        "yongchang_mode",
        "hunzhiyiji_mode",
        "wanxiangxiuluo_mode",
        "beishuizhizhan_mode",
    )
    if any(getattr(rule, name, 0) for name in special):
        raise ReplayError("牌谱包含当前规则引擎未支持的特殊规则。")
    if sorted(player.seat for player in head.result.players) != [0, 1, 2, 3]:
        raise ReplayError("牌谱缺少四名玩家的完整终局结算。")


def decode_details(raw: bytes) -> dict:
    """Handle both legacy records and action.result containers, without dropping events."""
    pb = protocol()
    from google.protobuf.json_format import MessageToDict

    if not raw or len(raw) > MAX_RECORD_BYTES:
        raise ReplayError("牌谱内容为空或超过 32 MiB。")
    wrapped = pb.Wrapper.FromString(raw)
    if wrapped.name != ".lq.GameDetailRecords":
        raise ReplayError("需要 .lq.GameDetailRecords protobuf 容器。")
    details = pb.GameDetailRecords.FromString(wrapped.data)
    if details.records and details.actions:
        raise ReplayError("牌谱同时包含两套动作流，拒绝含糊的解码结果。")
    events = list(details.records) or [a.result for a in details.actions if a.result]
    if not events:
        raise ReplayError("牌谱不包含可解码的动作。")
    rounds = []
    ended = False
    for event_bytes in events:
        event = pb.Wrapper.FromString(event_bytes)
        if not event.name.startswith(".lq.Record"):
            raise ReplayError("牌谱存在未知事件容器，未忽略该事件。")
        name = event.name.removeprefix(".lq.Record")
        if name not in KNOWN_RECORDS or name == "dora":
            raise ReplayError(f"未支持的牌谱事件：{name}；请更新解码器。")
        cls = getattr(pb, "Record" + name, None)
        if cls is None:
            raise ReplayError(f"当前协议库缺少事件：{name}。")
        message = cls.FromString(event.data)
        data = MessageToDict(
            message, preserving_proto_field_name=True, always_print_fields_with_no_presence=True
        )
        if name == "NewRound":
            if rounds and not ended:
                raise ReplayError("上一局未结算，牌谱不完整。")
            if len(data["scores"]) != 4 or any(not data[f"tiles{s}"] for s in range(4)):
                raise ReplayError("牌谱缺少四人的完整配牌。")
            if not data["doras"] and data["dora"]:
                data["doras"] = [data["dora"]]
            if not data["doras"]:
                raise ReplayError("开局缺少宝牌指示牌。")
            # RiichiEnv treats an explicit empty paishan as a supplied empty wall.
            # Proto3 defaults must not turn an absent optional wall into one.
            if not data.get("paishan"):
                data.pop("paishan", None)
            # The protocol also carries full walls and operation hints. They remain
            # in record.pb, but do not belong in the training replay export.
            for field in (
                "operation",
                "operations",
                "tingpai",
                "md5",
                "sha256",
                "saltSha256",
                "salt",
                "opens",
            ):
                data.pop(field, None)
            rounds.append([])
            ended = False
        elif not rounds or ended:
            raise ReplayError("牌谱动作出现在开局前或结算后。")
        rounds[-1].append({"name": name, "data": data})
        if name in {"Hule", "NoTile", "LiuJu"}:
            ended = True
    if not ended:
        raise ReplayError("最后一局缺少结算，牌谱不完整。")
    return {"data": rounds}


def write_once(path: Path, content: bytes):
    if path.exists():
        if path.read_bytes() != content:
            raise ReplayError(f"已有文件内容不同，未覆盖：{path.name}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(content)


def archive_record(
    raw: bytes,
    metadata: dict,
    directory: Path,
    dataset: Path | None = None,
    expected_id: str | None = None,
    gamma: float = 0.997,
) -> dict:
    """Archive a ResGameRecord, validate rules/scoring, then optionally build Parquet."""
    pb = protocol()
    validate_metadata(metadata)
    if not 0 < gamma <= 1:
        raise ReplayError("gamma 必须在 (0, 1]。")
    if not raw or len(raw) > MAX_RECORD_BYTES:
        raise ReplayError("原始牌谱为空或超过 32 MiB。")
    record = pb.ResGameRecord.FromString(raw)
    if record.error.code:
        raise ReplayError(f"雀魂读取牌谱失败，错误码 {record.error.code}。")
    validate_head(record.head, expected_id)
    if not record.data:
        raise ReplayError("牌谱内容缺失；data_url 内容需要先由下载器取得。")
    raw_hash = hashlib.sha256(raw).hexdigest()
    if metadata.get("raw_sha256") and metadata["raw_sha256"] != raw_hash:
        raise ReplayError("原始牌谱校验和不匹配。")
    decoded = decode_details(record.data)
    content = json.dumps(decoded, ensure_ascii=False, indent=2).encode("utf-8")
    digest = hashlib.sha256(content).hexdigest()
    # sha256 refers to the decoded file consumed by the existing ingest command.
    if metadata.get("sha256") and metadata["sha256"] != digest:
        raise ReplayError("解码牌谱校验和不匹配。")
    scores = [p.part_point_1 for p in sorted(record.head.result.players, key=lambda p: p.seat)]
    if metadata.get("expected_final_scores") not in (None, scores):
        raise ReplayError("来源清单终局分数与雀魂结算不一致。")
    destination = directory / record.head.uuid
    binary = destination / "record.pb"
    replay = destination / "replay.json"
    manifest = destination / "manifest.json"
    clean_metadata = {
        k: metadata[k]
        for k in ("permission_basis", "redistributable", "retention_class")
        if k in metadata
    }
    clean_metadata.update(
        {
            "source": "majsoul",
            "acquisition_method": "completed_replay_download",
            "ruleset": "majsoul_4p_standard",
            "contains_identifiers": True,
            "game_id": record.head.uuid,
            "raw_sha256": raw_hash,
            "sha256": digest,
            "expected_final_scores": scores,
            "protocol_package": "ms-api==0.11.100",
        }
    )
    if metadata.get("client_version"):
        clean_metadata["client_version"] = metadata["client_version"]
    write_once(binary, raw)
    write_once(replay, content)
    # Validate scoring before emitting a usable manifest/dataset.
    verified = load_replay(replay, "majsoul")
    kyokus = list(verified.take_kyokus())
    final = kyokus[-1].game_end_scores or kyokus[-1].end_scores
    if list(final) != scores:
        raise ReplayError("解码回放的终局分数与雀魂结算不一致。")
    write_once(manifest, json.dumps(clean_metadata, indent=2).encode("utf-8"))
    result = {
        "game_id": record.head.uuid,
        "raw": str(binary),
        "replay": str(replay),
        "manifest": str(manifest),
        "raw_sha256": raw_hash,
        "sha256": digest,
    }
    if dataset is not None:
        if dataset.exists():
            raise ReplayError("训练文件已经存在；请使用新的 --dataset 路径。")
        converted = ingest(replay, dataset, manifest, "majsoul", gamma)
        result.update({"transitions": converted["transitions"], "output": converted["output"]})
    return result


def decode_record(
    path: Path, manifest: Path, directory: Path, dataset: Path | None = None, gamma: float = 0.997
) -> dict:
    metadata = load_manifest(manifest)
    if path.stat().st_size > MAX_RECORD_BYTES:
        raise ReplayError("原始牌谱超过 32 MiB。")
    from google.protobuf.message import DecodeError

    try:
        return archive_record(path.read_bytes(), metadata, directory, dataset, gamma=gamma)
    except DecodeError as exc:
        raise ReplayError("无法解码 protobuf；--input 需要完整的 ResGameRecord 响应文件。") from exc
