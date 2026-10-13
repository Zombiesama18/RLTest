import asyncio
import struct

import pytest

lz4 = pytest.importorskip("lz4.block")
from mjsrl.client_version import (  # noqa: E402
    ASSETS,
    ClientVersion,
    resolve_client_version,
    unpack_bundle,
)
from mjsrl.majsoul_client import login  # noqa: E402
from mjsrl.replays import ReplayError, protocol  # noqa: E402


def bundle(payload, at_end=False):
    packed = lz4.compress(payload, store_size=False)
    info = b"\0" * 16 + struct.pack(">IIIH", 1, len(payload), len(packed), 3) + b"\0" * 4
    compressed_info = lz4.compress(info, store_size=False)
    flags = 0x243 | (0x80 if at_end else 0)
    beginning = b"UnityFS\0" + struct.pack(">I", 8) + b"5.x.x\00.0.0\0"
    header_size = (len(beginning) + 20 + 15) // 16 * 16
    if at_end:
        body = packed + compressed_info
    else:
        body = compressed_info
        body += b"\0" * ((-header_size - len(body)) % 16)
        body += packed
    header = beginning + struct.pack(
        ">QIII", header_size + len(body), len(compressed_info), len(info), flags
    )
    return header + b"\0" * (header_size - len(header)) + body


@pytest.mark.parametrize("at_end", [False, True])
def test_bounded_unityfs_blocks(at_end):
    payload = b"version-index-and-resource-metadata" * 100
    assert unpack_bundle(bundle(payload, at_end)) == payload


@pytest.mark.parametrize("raw", [b"bad\0", bundle(b"version")[:-1]])
def test_reject_broken_bundle(raw):
    with pytest.raises(ReplayError):
        unpack_bundle(raw)


def test_dynamic_version_uses_decompressed_complete_bundle_name():
    name = "td1bd8bb002s_" + "0123456789abcdef" * 2
    visited = []

    async def fetch(url):
        visited.append(url)
        if url.endswith("index.html"):
            return b'productVersion: "4.0.47"'
        if url.endswith("bundle_info_so.majset"):
            return bundle((name + "\0").encode() * 4)
        if url == ASSETS + "u6v_" + name + ".majset":
            raise ReplayError("HTTP 404")
        assert url == ASSETS + name + ".majset"
        return bundle(b'{"version":"0.16.283","platform":"WebGL"}')

    version = asyncio.run(resolve_client_version(fetch))
    assert str(version) == "WebGL_2022-0.16.283"
    assert (version.package, version.resource) == ("4.0.47", "0.16.283")
    assert len(visited) == 4


def test_missing_current_version_never_falls_back_to_legacy():
    async def fetch(url):
        assert url.endswith("index.html")
        return b"old unrelated page"

    with pytest.raises(ReplayError, match="未回退"):
        asyncio.run(resolve_client_version(fetch))


def test_modern_login_payload():
    pb = protocol()

    class Rpc:
        async def call(self, method, request, response_type):
            assert request.client_version_string == "WebGL_2022-0.16.283"
            assert request.client_version.resource == "0.16.283"
            assert request.client_version.package == "4.0.47"
            assert not request.gen_access_token
            return pb.ResLogin(account_id=101)

    assert (
        asyncio.run(login(Rpc(), "account", "password", ClientVersion("0.16.283", "4.0.47"))) == 101
    )
