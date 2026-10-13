"""Completed replay retrieval; no credential persistence or game actions."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlsplit

from .replays import (
    MAX_RECORD_BYTES,
    ReplayError,
    archive_record,
    load_manifest,
    parse_reference,
    protocol,
    validate_head,
)

BASE = "https://game.maj-soul.com/1"
ORIGIN = "https://game.maj-soul.com"


def official_url(value: str, scheme="https") -> str:
    parsed = urlsplit(value)
    host = parsed.hostname or ""
    if (
        parsed.scheme != scheme
        or parsed.username
        or parsed.password
        or not (host == "maj-soul.com" or host.endswith(".maj-soul.com"))
    ):
        raise ReplayError("雀魂返回了无法识别的服务器地址，未连接该地址。")
    return value


async def read_limited(response, limit=MAX_RECORD_BYTES):
    response.raise_for_status()
    chunks, size = [], 0
    async for chunk in response.content.iter_chunked(65536):
        size += len(chunk)
        if size > limit:
            raise ReplayError("雀魂返回的数据超过大小上限。")
        chunks.append(chunk)
    return b"".join(chunks)


async def http_json(session, url, *, params=None):
    official_url(url)
    async with session.get(url, params=params, allow_redirects=False) as response:
        if response.status != 200:
            raise ReplayError(f"雀魂配置请求失败，HTTP {response.status}。")
        value = json.loads(await read_limited(response, 4 * 1024 * 1024))
    if not isinstance(value, dict):
        raise ReplayError("雀魂配置不是预期的 JSON 对象。")
    return value


async def discover(session):
    """Use the current public version and routing configuration, not a fixed version."""
    version = (await http_json(session, BASE + "/version.json")).get("version", "")
    if not isinstance(version, str) or not version or any(c in version for c in "/?#"):
        raise ReplayError("雀魂客户端版本配置无效。")
    config = await http_json(session, f"{BASE}/v{version}/config.json")
    endpoints = []
    for cluster in config.get("ip", []):
        if cluster.get("name") != "player":
            continue
        for route in cluster.get("gateways", [])[:4]:
            route_url = official_url(route["url"]).rstrip("/")
            try:
                payload = await http_json(
                    session,
                    route_url + "/api/clientgate/routes",
                    params={
                        "platform": "Web",
                        "version": version,
                        "lang": "en",
                    },
                )
            except (ReplayError, OSError, TimeoutError):
                continue
            routes = payload.get("data", {}).get("routes", [])
            for gateway in sorted(routes, key=lambda item: item.get("order", 0)):
                if gateway.get("ssl") and gateway.get("state") != "maintain":
                    endpoint = official_url("wss://" + gateway["domain"] + "/gateway", "wss")
                    if endpoint not in endpoints:
                        endpoints.append(endpoint)
            if endpoints:
                break
    if not endpoints:
        raise ReplayError("未取得可用雀魂网关；请检查网络或客户端接口是否更新。")
    return "web-" + version.removesuffix(".w"), endpoints


class Rpc:
    """Serial Lobby requests with bounded waits and proper disconnect handling."""

    def __init__(self, socket, timeout):
        self.socket = socket
        self.timeout = timeout
        self.sequence = 0

    async def call(self, method, request, response_type):
        import aiohttp

        pb = protocol()
        self.sequence = self.sequence % 65535 + 1
        wrapper = pb.Wrapper(name=".lq.Lobby." + method, data=request.SerializeToString())
        packet = b"\x02" + self.sequence.to_bytes(2, "little") + wrapper.SerializeToString()
        async with asyncio.timeout(self.timeout):
            await self.socket.send_bytes(packet)
            while True:
                message = await self.socket.receive()
                if message.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.ERROR,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    raise ReplayError("雀魂连接已断开。")
                if message.type != aiohttp.WSMsgType.BINARY:
                    continue
                raw = message.data
                if len(raw) >= 3 and raw[0] == 3:
                    if int.from_bytes(raw[1:3], "little") == self.sequence:
                        reply = pb.Wrapper.FromString(raw[3:])
                        return response_type.FromString(reply.data)


@asynccontextmanager
async def connect(session, endpoints, timeout):
    import aiohttp

    socket = None
    for endpoint in endpoints:
        try:
            socket = await asyncio.wait_for(
                session.ws_connect(
                    endpoint, origin=ORIGIN, heartbeat=20, max_msg_size=MAX_RECORD_BYTES
                ),
                timeout,
            )
            break
        except (aiohttp.ClientError, OSError, TimeoutError):
            continue
    if socket is None:
        raise ReplayError("无法连接雀魂网关；请检查网络。")
    async with socket:
        yield Rpc(socket, timeout)


async def login(rpc, account, password, version):
    pb = protocol()
    request = pb.ReqLogin(
        account=account,
        password=hmac.new(b"lailai", password.encode(), hashlib.sha256).hexdigest(),
        random_key=str(uuid.uuid4()),
        client_version_string=version,
        tag="cn",
        gen_access_token=False,
    )
    request.device.is_browser = True
    request.device.platform = "pc"
    request.device.os = "windows"
    request.device.software = "Chrome"
    request.device.sale_platform = "web"
    request.currency_platforms.extend([1, 2, 5, 6, 8, 10, 11])
    response = await rpc.call("login", request, pb.ResLogin)
    if response.error.code or response.account_id <= 0:
        raise ReplayError(f"雀魂登录失败，错误码 {response.error.code}；请在游戏中确认账号。")
    return response.account_id


async def fetch_record(session, rpc, reference, version):
    pb = protocol()
    request = pb.ReqGameRecord(game_uuid=reference.game_id, client_version_string=version)
    record = await rpc.call("fetchGameRecord", request, pb.ResGameRecord)
    if record.error.code == 151:
        # Some shared records must first be registered as viewed by this account.
        viewed = await rpc.call("readGameRecord", request, pb.ResCommon)
        if not viewed.error.code:
            record = await rpc.call("fetchGameRecord", request, pb.ResGameRecord)
    if record.error.code:
        raise ReplayError(f"雀魂读取牌谱失败，错误码 {record.error.code}。")
    validate_head(record.head, reference.game_id)
    if not record.data and record.data_url:
        url = official_url(record.data_url)
        async with session.get(url, allow_redirects=False) as response:
            if response.status != 200:
                raise ReplayError(f"雀魂牌谱下载失败，HTTP {response.status}。")
            record.data = await read_limited(response)
    if not record.data:
        raise ReplayError("服务器未返回牌谱内容。")
    # A download URL is transient; the archive contains the fetched bytes instead.
    record.data_url = ""
    return record.SerializeToString()


async def download_replay(
    url: str,
    account: str,
    password: str,
    manifest: Path,
    directory: Path,
    dataset: Path | None = None,
    timeout=30,
    gamma=0.997,
) -> dict:
    reference = parse_reference(url)
    metadata = load_manifest(manifest)
    if reference.region != "cn":
        raise ReplayError("首版自动下载支持国服账号；国际服请先导出 MJAI 文件再 ingest。")
    if not account.strip() or not password:
        raise ReplayError("国服下载需要本机输入雀魂账号和密码。")
    if timeout <= 0 or not 0 < gamma <= 1:
        raise ReplayError("timeout 必须为正数，gamma 必须在 (0, 1]。")
    try:
        import aiohttp
    except ImportError as exc:
        raise ReplayError("请先安装牌谱依赖：python -m pip install '.[majsoul]'") from exc
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
            version, endpoints = await discover(session)
            async with connect(session, endpoints, timeout) as rpc:
                await login(rpc, account.strip(), password, version)
                raw = await fetch_record(session, rpc, reference, version)
        metadata["client_version"] = version
        return archive_record(raw, metadata, directory, dataset, reference.game_id, gamma)
    except TimeoutError as exc:
        raise ReplayError("雀魂请求超时；请检查网络后重试。") from exc
    except (aiohttp.ClientError, OSError) as exc:
        raise ReplayError("雀魂网络连接失败，或本地文件无法写入。") from exc
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ReplayError("雀魂配置或响应格式已经变化，请更新下载器。") from exc
