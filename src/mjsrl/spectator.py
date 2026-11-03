"""Public high-rank live discovery -> completed replay -> expert decision samples."""

from __future__ import annotations

import asyncio
import json
import math
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path

from . import majsoul_client as client
from .expert_data import ROOMS, ExpertFilter, completed_selection, discover_candidate
from .replay_batch import ensure_dataset, write_json
from .replays import ReplayError, archive_record, load_manifest, protocol, write_once
from .storage import file_hash


class SessionExpired(ReplayError):
    """Authentication failures stop the run without consuming pending jobs."""


@dataclass
class CollectorConfig:
    duration: float = 3600
    poll: float = 60
    interval: float = 2
    timeout: float = 30
    max_games: int = 100
    record_wait: float = 21600
    gamma: float = 0.997

    def validate(self):
        if any(
            not math.isfinite(x) or x <= 0
            for x in (self.duration, self.poll, self.timeout, self.record_wait)
        ):
            raise ReplayError("采集时长、轮询、超时和牌谱等待期限必须为有限正数。")
        if (
            not math.isfinite(self.interval)
            or self.interval < 0
            or self.max_games < 0
            or not 0 < self.gamma <= 1
        ):
            raise ReplayError("请求间隔/数量上限必须非负，gamma 必须在 (0,1]。")


class Queue:
    def __init__(self, path, selection):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS config (value TEXT)")
        value = json.dumps(selection.to_dict(), sort_keys=True)
        previous = self.db.execute("SELECT value FROM config").fetchone()
        if previous and previous[0] != value:
            self.db.close()
            raise ReplayError("此队列的房间/段位筛选条件不同；请使用新的 state-dir。")
        if not previous:
            self.db.execute("INSERT INTO config VALUES (?)", (value,))
        self.db.execute("""CREATE TABLE IF NOT EXISTS games (
            game_id TEXT PRIMARY KEY, candidate TEXT NOT NULL, observed REAL NOT NULL,
            last_seen REAL NOT NULL, state TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
            next_check REAL NOT NULL, reason TEXT, result TEXT)""")
        self.db.commit()

    def add(self, candidate, now, limit):
        admitted = False
        if self.db.execute(
            "SELECT 1 FROM games WHERE game_id=?", (candidate["game_id"],)
        ).fetchone():
            self.db.execute(
                "UPDATE games SET last_seen=? WHERE game_id=?", (now, candidate["game_id"])
            )
        elif not limit or self.db.execute("SELECT count(*) FROM games").fetchone()[0] < limit:
            self.db.execute(
                "INSERT INTO games(game_id,candidate,observed,last_seen,state,next_check) "
                "VALUES (?,?,?,?,?,?)",
                (candidate["game_id"], json.dumps(candidate), now, now, "waiting", now),
            )
            admitted = True
        self.db.commit()
        return admitted

    def due(self, now):
        rows = self.db.execute(
            "SELECT game_id,candidate,observed,attempts FROM games "
            "WHERE state='waiting' AND next_check<=? ORDER BY observed LIMIT 20",
            (now,),
        ).fetchall()
        return [
            {
                "game_id": row[0],
                "candidate": json.loads(row[1]),
                "observed": row[2],
                "attempts": row[3],
            }
            for row in rows
        ]

    def update(self, job, state, now, reason=None, result=None, delay=60):
        self.db.execute(
            "UPDATE games SET state=?,attempts=attempts+1,next_check=?,reason=?,result=? "
            "WHERE game_id=?",
            (state, now + delay, reason, json.dumps(result) if result else None, job["game_id"]),
        )
        self.db.commit()

    def counts(self):
        return dict(self.db.execute("SELECT state,count(*) FROM games GROUP BY state"))

    def saved(self):
        return [
            json.loads(row[0])
            for row in self.db.execute("SELECT result FROM games WHERE state='saved'")
        ]

    def save_result(self, result):
        self.db.execute(
            "UPDATE games SET result=? WHERE game_id=?", (json.dumps(result), result["game_id"])
        )
        self.db.commit()

    def close(self):
        self.db.close()


@contextmanager
def state_lock(path):
    # OS-owned locks are released even on a crash; no stale PID-file cleanup is needed.
    import os

    with path.open("a+b") as stream:
        if path.stat().st_size == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise ReplayError("同一个采集目录已有运行中的采集器。") from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


class PacedRpc:
    def __init__(self, rpc, interval):
        self.rpc, self.interval, self.last, self.heartbeat = rpc, interval, 0, time.monotonic()
        self.authenticated = False

    async def _call(self, method, request, response_type):
        await asyncio.sleep(max(0, self.last + self.interval - time.monotonic()))
        self.last = time.monotonic()
        return await self.rpc.call(method, request, response_type)

    async def call(self, method, request, response_type):
        pb = protocol()
        if self.authenticated and method != "heatbeat" and time.monotonic() - self.heartbeat >= 45:
            response = await self._call(
                "heatbeat", pb.ReqHeatBeat(no_operation_counter=0), pb.ResCommon
            )
            self.heartbeat = time.monotonic()
            if response.error.code:
                raise SessionExpired(
                    f"采集会话已失效，心跳错误码 {response.error.code}；重新登录可继续。"
                )
        response = await self._call(method, request, response_type)
        if response.error.code == 1004:
            raise SessionExpired("结束确认/牌谱下载需要有效登录；重新登录可继续。")
        return response


async def discover(rpc, queue, selection, limit, now):
    pb = protocol()
    observed, rejected, admitted = 0, 0, 0
    room_candidates = []
    for room in selection.rooms:
        candidates = []
        response = await rpc.call(
            "fetchGameLiveList", pb.ReqGameLiveList(filter_id=200 + ROOMS[room]), pb.ResGameLiveList
        )
        if response.error.code:
            raise ReplayError(f"公开观战列表读取失败，错误码 {response.error.code}。")
        for head in response.live_list:
            try:
                candidate = discover_candidate(head, selection)
            except ValueError:
                candidate = None
            if candidate:
                candidates.append(candidate)
                observed += 1
            else:
                rejected += 1
        room_candidates.append(candidates)
    # A bounded queue should not fill from the first room and starve the others.
    for row in zip_longest(*room_candidates):
        for candidate in row:
            if candidate is not None:
                admitted += int(queue.add(candidate, now, limit))
    return {"observed": observed, "filtered": rejected, "newly_queued": admitted}


async def check_finished(
    session, rpc, queue, selection, metadata, version, directory, dataset_dir, config, now
):
    import aiohttp
    from google.protobuf.message import DecodeError

    pb = protocol()
    jobs = queue.due(now)
    if not jobs:
        return
    response = await rpc.call(
        "fetchGameRecordsDetail",
        pb.ReqGameRecordsDetail(uuid_list=[j["game_id"] for j in jobs]),
        pb.ResGameRecordsDetail,
    )
    if response.error.code == 1004:
        raise SessionExpired("观战发现可公开读取，但结束确认/牌谱下载需要登录；请重新登录后继续。")
    heads = {head.uuid: head for head in response.record_list} if not response.error.code else {}
    for job in jobs:
        backoff = min(config.poll * 2 ** min(job["attempts"], 5), 1800)
        head = heads.get(job["game_id"])
        if (
            head is None
            or head.end_time == 0
            or sorted(p.seat for p in head.result.players) != [0, 1, 2, 3]
        ):
            if now - job["observed"] > config.record_wait:
                queue.update(job, "unavailable", now, "超过等待期限，仍未取得完整牌谱")
                continue
            queue.update(
                job,
                "waiting",
                now,
                f"尚无完整终局记录（列表错误码 {response.error.code}）",
                delay=backoff,
            )
            continue
        try:
            expert = completed_selection(head, job["candidate"], selection)
            if expert is None:
                queue.update(job, "rejected", now, "最终牌谱玩家段位未达到门槛")
                continue
        except ValueError as exc:
            queue.update(job, "rejected", now, str(exc))
            continue
        path = directory / "raw" / job["game_id"] / "record.pb"
        try:
            if path.exists():
                raw = path.read_bytes()
            else:
                viewed = await rpc.call(
                    "readGameRecord",
                    pb.ReqGameRecord(game_uuid=job["game_id"], client_version_string=version),
                    pb.ResCommon,
                )
                if viewed.error.code == 1004:
                    raise SessionExpired("登录会话已失效，需重新登录后继续。")
                if viewed.error.code:
                    queue.update(
                        job,
                        "unavailable" if now - job["observed"] > config.record_wait else "waiting",
                        now,
                        f"服务器尚未提供牌谱，错误码 {viewed.error.code}",
                        delay=backoff,
                    )
                    continue
                raw = await client.fetch_record(
                    session, rpc, client.parse_reference(job["game_id"]), version
                )
                write_once(
                    path, raw
                )  # Preserve bytes even when the decoder/scoring gate rejects them.
        except SessionExpired:
            raise
        except (ValueError, OSError, TimeoutError, aiohttp.ClientError, DecodeError) as exc:
            reason = str(exc) if isinstance(exc, ValueError) else type(exc).__name__
            queue.update(
                job,
                "unavailable" if now - job["observed"] > config.record_wait else "waiting",
                now,
                reason,
                delay=backoff,
            )
            continue
        try:
            actual = pb.ResGameRecord.FromString(raw).head
            # Final downloaded identities/ranks are authoritative, not live-list order.
            expert = completed_selection(actual, job["candidate"], selection)
            if expert is None:
                queue.update(job, "rejected", now, "下载牌谱段位低于门槛")
                continue
            result = archive_record(
                raw,
                {
                    **metadata,
                    **expert,
                    "client_version": str(version),
                    "public_observation": {
                        "game_id": job["game_id"],
                        "observed_at": job["observed"],
                        "source": "fetchGameLiveList",
                    },
                },
                directory / "raw",
                expected_id=job["game_id"],
                gamma=config.gamma,
            )
            ensure_dataset(result, dataset_dir, config.gamma)
            queue.update(job, "saved", now, result=result)
        except Exception as exc:
            queue.update(
                job,
                "quarantined",
                now,
                str(exc) if isinstance(exc, ValueError) else type(exc).__name__,
            )
        except BaseException as exc:
            if type(exc).__name__ != "PanicException" or type(exc).__module__ != "pyo3_runtime":
                raise
            queue.update(
                job, "quarantined", now, "规则引擎 PanicException，保留原始文件等待解析修复"
            )


def export_saved(queue, dataset_dir, gamma):
    """Allow offline export after a raw-only collection, verifying cached bytes."""
    if dataset_dir is None:
        return 0
    converted = 0
    for result in queue.saved():
        metadata = load_manifest(Path(result["manifest"]))
        if metadata.get("game_id") != result["game_id"] or any(
            metadata.get(key) != result[key] or file_hash(Path(result[path])) != result[key]
            for key, path in (("raw_sha256", "raw"), ("sha256", "replay"))
        ):
            raise ReplayError("高手采集缓存校验失败；请保留原始文件并使用新目录。")
        converted += int(ensure_dataset(result, dataset_dir, gamma))
        queue.save_result(result)
    return converted


async def collect(
    directory: Path,
    manifest: Path,
    selection=ExpertFilter(),
    config=None,
    dataset_dir=None,
    account=None,
    password=None,
    discover_only=False,
    export_only=False,
    progress=None,
):
    import aiohttp
    from websockets.exceptions import ConnectionClosed

    config = config or CollectorConfig()
    selection.validate()
    config.validate()
    metadata = load_manifest(manifest)
    if metadata["permission_basis"] != "public_replay":
        raise ReplayError(
            "观战采集来源清单应为 public_replay，不能写成 user_owned 或伪称玩家授权。"
        )
    if discover_only and export_only:
        raise ReplayError("discover-only 和 export-only 不能同时使用。")
    if export_only and dataset_dir is None:
        raise ReplayError("export-only 需要 dataset-dir。")
    if not discover_only and not export_only and (not account or not password):
        raise ReplayError("结束确认和下载需在本机登录；只发现可使用 --discover-only。")
    directory.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "running",
        "queue": str(directory / "queue.sqlite"),
        "counts": {},
        "started_at": time.time(),
    }
    with state_lock(directory / "collector.lock"):
        queue = Queue(directory / "queue.sqlite", selection)
        try:
            report["exported"] = export_saved(queue, dataset_dir, config.gamma)
            if export_only:
                report["status"] = "exported"
                return report
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=config.timeout)
            ) as session:
                version, endpoints = await client.discover(session)
                async with client.connect(session, endpoints, config.timeout) as connection:
                    rpc = PacedRpc(connection, config.interval)
                    if not discover_only:
                        await client.login(rpc, account, password, version)
                        rpc.authenticated = True
                    deadline = time.monotonic() + config.duration
                    while True:
                        report.update(
                            await discover(rpc, queue, selection, config.max_games, time.time())
                        )
                        if not discover_only:
                            await check_finished(
                                session,
                                rpc,
                                queue,
                                selection,
                                metadata,
                                version,
                                directory,
                                dataset_dir,
                                config,
                                time.time(),
                            )
                        report["counts"] = queue.counts()
                        report["updated_at"] = time.time()
                        write_json(directory / "status.json", report)
                        if progress:
                            progress(report.copy())
                        if discover_only or time.monotonic() >= deadline:
                            break
                        await asyncio.sleep(min(config.poll, max(0, deadline - time.monotonic())))
                    report["status"] = "discovered" if discover_only else "finished"
        except (KeyboardInterrupt, asyncio.CancelledError):
            report["status"] = "interrupted"
            raise
        except ReplayError as exc:
            # Our ReplayError messages expose error codes, never login responses.
            report["status"] = "failed"
            report["reason"] = str(exc)
            raise
        except (TimeoutError, aiohttp.ClientError, OSError, ConnectionClosed) as exc:
            report["status"] = "failed"
            report["reason"] = f"网络或文件操作失败：{type(exc).__name__}；队列可继续。"
            raise ReplayError("采集网络或文件操作失败，队列已保存；可重新运行继续。") from exc
        except BaseException:
            report["status"] = "failed"
            report["reason"] = "未预期的采集异常；请查看本机终端。"
            raise
        finally:
            report["counts"] = queue.counts()
            write_json(directory / "status.json", report)
            queue.close()
    return report
