"""Resolve Unity package/resource versions from public Mahjong Soul metadata."""

import re
import struct

from .replays import ReplayError

ASSETS = "https://game.maj-soul.com/assetbundles/ASTC/"
MAX_UNPACKED = 64 * 1024 * 1024


class ClientVersion(str):
    def __new__(cls, resource, package):
        value = super().__new__(cls, "WebGL_2022-" + resource)
        value.resource, value.package = resource, package
        return value


def unpack_bundle(raw):
    """Read bounded UnityFS v6-v8 blocks with plain/LZ4 compression only."""
    import lz4.block

    position = 0

    def take(size):
        nonlocal position
        if size < 0 or position + size > len(raw):
            raise ReplayError("官方版本资源包不完整。")
        value = raw[position : position + size]
        position += size
        return value

    def string():
        nonlocal position
        end = raw.find(b"\0", position)
        if end < 0:
            raise ReplayError("官方版本资源包头无效。")
        return take(end - position + 1)[:-1]

    def decompress(content, size, flags):
        if size > MAX_UNPACKED:
            raise ReplayError("官方版本资源包解压大小超过上限。")
        mode = flags & 63
        if mode == 0:
            result = content
        elif mode in (2, 3):
            try:
                result = lz4.block.decompress(content, uncompressed_size=size)
            except lz4.block.LZ4BlockError as exc:
                raise ReplayError("官方版本资源包 LZ4 解压失败。") from exc
        else:
            raise ReplayError("官方版本资源包压缩格式发生变化。")
        if len(result) != size:
            raise ReplayError("官方版本资源包解压大小不匹配。")
        return result

    if string() != b"UnityFS":
        raise ReplayError("官方版本资源包不是 UnityFS。")
    format_version = struct.unpack(">I", take(4))[0]
    if format_version not in (6, 7, 8):
        raise ReplayError("官方版本资源包容器版本发生变化。")
    string()
    string()
    length, compressed, uncompressed, flags = struct.unpack(">QIII", take(20))
    if length != len(raw) or compressed > len(raw) or flags & ~0x2FF:
        raise ReplayError("官方版本资源包长度或标志无效。")
    if format_version >= 7:
        position = (position + 15) // 16 * 16
    if flags & 0x80:
        info = decompress(raw[len(raw) - compressed :], uncompressed, flags)
    else:
        info = decompress(take(compressed), uncompressed, flags)
    if flags & 0x200:
        position = (position + 15) // 16 * 16
    if len(info) < 20:
        raise ReplayError("官方版本资源包缺少块索引。")
    count = struct.unpack_from(">I", info, 16)[0]
    if count > 4096 or len(info) < 20 + count * 10:
        raise ReplayError("官方版本资源包块索引无效。")
    blocks, total = [], 0
    for index in range(count):
        size, packed_size, block_flags = struct.unpack_from(">IIH", info, 20 + index * 10)
        total += size
        if total > MAX_UNPACKED:
            raise ReplayError("官方版本资源包解压大小超过上限。")
        blocks.append(decompress(take(packed_size), size, block_flags))
    return b"".join(blocks)


async def resolve_client_version(fetch):
    html = (await fetch("https://game.maj-soul.com/1/index.html")).decode("utf-8-sig")
    package = re.search(r'productVersion\s*:\s*["\']([0-9]+(?:\.[0-9]+){2,})["\']', html)
    if not package:
        raise ReplayError("官网未提供可识别的 Unity 客户端包版本，未回退到旧版登录。")
    index = unpack_bundle(await fetch(ASSETS + "bundle_info_so.majset"))
    names = set(re.findall(rb"td1bd8bb002s_[0-9a-fA-F]{20,64}", index))
    if len(names) != 1:
        raise ReplayError("官方索引未提供唯一资源版本包，需更新版本解析器。")
    name = names.pop().decode("ascii")
    for prefix in ("u6v_", ""):
        try:
            content = unpack_bundle(await fetch(ASSETS + prefix + name + ".majset"))
        except ReplayError:
            if prefix:
                continue
            raise
        versions = set(re.findall(rb'"([0-9]+(?:\.[0-9]+){2,})"\s*,\s*"platform"', content))
        if len(versions) != 1:
            raise ReplayError("官方资源版本包内容发生变化。")
        return ClientVersion(versions.pop().decode("ascii"), package.group(1))
    raise ReplayError("无法读取官方资源版本包。")
