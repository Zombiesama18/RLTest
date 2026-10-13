"""Paginate the logged-in account's replays, cache validated outputs, and journal progress."""

from __future__ import annotations

import asyncio
import json
import math
import uuid
from datetime import datetime, timezone
from pathlib import Path

from . import majsoul_client as client
from .ingest import ingest
from .replays import (
    ReplayError,
    archive_record,
    load_manifest,
    parse_reference,
    protocol,
    validate_head,
)
from .schema import SCHEMA_VERSION
from .storage import file_hash


def batch_manifest(path: Path) -> dict:
    metadata = load_manifest(path)
    if metadata["permission_basis"] != "user_owned":
        raise ReplayError("账号批量下载的来源清单必须使用 user_owned。")
    if any(key in metadata for key in ("sha256", "raw_sha256", "expected_final_scores")):
        raise ReplayError("批量来源清单不要填写单场校验和或单场终局分数。")
    return metadata


def validate_options(limit, interval, timeout, gamma):
    if limit < 0 or not math.isfinite(interval) or interval < 0:
        raise ReplayError("limit 必须非负，interval 必须是非负有限秒数。")
    if not math.isfinite(timeout) or timeout <= 0 or not 0 < gamma <= 1:
        raise ReplayError("timeout 必须为有限正数，gamma 必须在 (0, 1]。")


async def list_account_replays(rpc, owner_id, limit=100, interval=2, page_size=30):
    """The API starts at 1; limit counts eligible records including cached ones."""
    pb = protocol()
    start, seen, selected, filtered = 1, set(), [], 0
    while True:
        response = await rpc.call(
            "fetchGameRecordList",
            pb.ReqGameRecordList(start=start, count=page_size, type=2),
            pb.ResGameRecordList,
        )
        if response.error.code:
            raise ReplayError(f"获取自己账号的牌谱列表失败，错误码 {response.error.code}。")
        records = list(response.record_list)
        if not records:
            break
        new_ids = 0
        for head in records:
            if head.uuid in seen:
                continue
            seen.add(head.uuid)
            new_ids += 1
            try:
                parse_reference(head.uuid)
                validate_head(head)
                if owner_id not in {account.account_id for account in head.accounts}:
                    raise ReplayError("牌谱不属于当前登录账号。")
            except ValueError:
                filtered += 1
                continue
            selected.append(head.uuid)
            if limit and len(selected) >= limit:
                return selected, filtered
        if not new_ids:
            raise ReplayError("服务器重复返回同一页牌谱，已停止分页。")
        start += len(records)
        if response.total_count and start > response.total_count:
            break
        await asyncio.sleep(interval)
    return selected, filtered


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def owned_record(raw, game_id, owner_id):
    record = protocol().ResGameRecord.FromString(raw)
    validate_head(record.head, game_id)
    if owner_id not in {account.account_id for account in record.head.accounts}:
        raise ReplayError("牌谱不属于当前登录账号，未按 user_owned 导入。")


def cached_replay(directory, game_id, owner_id):
    destination = directory / game_id
    raw, replay, manifest = (
        destination / name for name in ("record.pb", "replay.json", "manifest.json")
    )
    if not all(path.exists() for path in (raw, replay, manifest)):
        return None
    metadata = load_manifest(manifest)
    if metadata.get("game_id") != game_id:
        raise ReplayError("缓存来源清单的牌谱 ID 不一致。")
    if metadata.get("raw_sha256") != file_hash(raw) or metadata.get("sha256") != file_hash(replay):
        raise ReplayError("本地缓存校验和不一致；未覆盖已有文件，请使用新输出目录。")
    owned_record(raw.read_bytes(), game_id, owner_id)
    return {
        "game_id": game_id,
        "raw": str(raw),
        "replay": str(replay),
        "manifest": str(manifest),
        "raw_sha256": metadata["raw_sha256"],
        "sha256": metadata["sha256"],
    }


def ensure_dataset(result, dataset_dir, gamma):
    if dataset_dir is None:
        return False
    dataset = dataset_dir / (result["game_id"] + ".parquet")
    marker = dataset.with_suffix(".batch.json")
    if dataset.exists():
        if not marker.exists():
            raise ReplayError("训练文件已存在但没有批量校验记录；请使用新 dataset-dir。")
        saved = json.loads(marker.read_text(encoding="utf-8"))
        if (
            saved.get("replay_sha256") != result["sha256"]
            or saved.get("gamma") != gamma
            or saved.get("schema_version") != SCHEMA_VERSION
            or saved.get("dataset_sha256") != file_hash(dataset)
        ):
            raise ReplayError("训练文件校验和或训练参数变化；未覆盖，请使用新 dataset-dir。")
        result["output"] = str(dataset)
        result["transitions"] = saved["transitions"]
        return False
    converted = ingest(Path(result["replay"]), dataset, Path(result["manifest"]), "majsoul", gamma)
    result.update({"transitions": converted["transitions"], "output": converted["output"]})
    write_json(
        marker,
        {
            "replay_sha256": result["sha256"],
            "gamma": gamma,
            "schema_version": SCHEMA_VERSION,
            "dataset_sha256": file_hash(dataset),
            "transitions": converted["transitions"],
        },
    )
    return True


async def process_record(
    session, rpc, game_id, owner_id, metadata, version, directory, dataset_dir, gamma, interval
):
    result = cached_replay(directory, game_id, owner_id)
    downloaded = False
    if result is None:
        binary = directory / game_id / "record.pb"
        if binary.exists():
            # Recover an interrupted decode without fetching the same binary again.
            raw = binary.read_bytes()
        else:
            await asyncio.sleep(interval)
            raw = await client.fetch_record(session, rpc, parse_reference(game_id), version)
            downloaded = True
        owned_record(raw, game_id, owner_id)
        result = archive_record(
            raw,
            {**metadata, "client_version": version},
            directory,
            expected_id=game_id,
            gamma=gamma,
        )
    converted = ensure_dataset(result, dataset_dir, gamma)
    result["status"] = "downloaded" if downloaded else "reused" if converted else "skipped"
    result["converted"] = converted
    return result


async def download_account_replays(
    account,
    password,
    manifest,
    directory,
    dataset_dir=None,
    limit=100,
    interval=2,
    timeout=30,
    gamma=0.997,
    progress=None,
):
    import aiohttp
    from google.protobuf.message import DecodeError

    metadata = batch_manifest(manifest)
    validate_options(limit, interval, timeout, gamma)
    if not account.strip() or not password:
        raise ReplayError("国服批量下载需要本机输入账号和密码。")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
    journal = directory / "batches" / (run_id + ".jsonl")
    summary = journal.with_suffix(".json")
    report = {
        "status": "running",
        "selected": 0,
        "filtered": 0,
        "downloaded": 0,
        "reused": 0,
        "skipped": 0,
        "failed": 0,
        "converted": 0,
        "pending": 0,
        "journal": str(journal),
        "report": str(summary),
    }
    write_json(summary, report)
    if progress:
        progress({"stage": "listing", "report": str(summary)})
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
            version, endpoints = await client.discover(session)
            async with client.connect(session, endpoints, timeout) as rpc:
                owner_id = await client.login(rpc, account.strip(), password, version)
                ids, report["filtered"] = await list_account_replays(rpc, owner_id, limit, interval)
                report["selected"] = report["pending"] = len(ids)
                write_json(summary, report)
                with journal.open("a", encoding="utf-8") as stream:
                    for index, game_id in enumerate(ids):
                        try:
                            result = await process_record(
                                session,
                                rpc,
                                game_id,
                                owner_id,
                                metadata,
                                version,
                                directory,
                                dataset_dir,
                                gamma,
                                interval,
                            )
                        except (
                            ValueError,
                            OSError,
                            TimeoutError,
                            aiohttp.ClientError,
                            DecodeError,
                        ) as exc:
                            # Network exceptions can carry URLs; log only their type.
                            reason = str(exc) if isinstance(exc, ValueError) else type(exc).__name__
                            result = {"game_id": game_id, "status": "failed", "reason": reason}
                        report[result["status"]] += 1
                        report["converted"] += int(result.get("converted", False))
                        report["pending"] -= 1
                        stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                        stream.flush()
                        write_json(summary, report)
                        if progress:
                            progress({"current": index + 1, "total": len(ids), **result})
                        if result["status"] == "failed" and getattr(
                            getattr(rpc, "socket", None), "closed", False
                        ):
                            report["status"] = "stopped"
                            break
                if report["status"] == "running":
                    report["status"] = "partial" if report["failed"] else "complete"
    except ReplayError as exc:
        report["status"] = "failed"
        report["reason"] = str(exc)
        raise
    except (TimeoutError, aiohttp.ClientError, OSError) as exc:
        report["status"] = "stopped"
        raise ReplayError(f"批量下载的网络或文件操作失败；进度保存在 {summary}。") from exc
    finally:
        if report["status"] == "running":
            report["status"] = "interrupted"
        write_json(summary, report)
    return report
