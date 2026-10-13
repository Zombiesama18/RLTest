import asyncio
import hashlib
import hmac
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

pb = pytest.importorskip("ms.protocol_pb2")
from google.protobuf.json_format import ParseDict  # noqa: E402

from mjsrl import majsoul_client as client  # noqa: E402
from mjsrl.replays import (  # noqa: E402
    ReplayError,
    archive_record,
    decode_details,
    decode_record,
    parse_reference,
    write_once,
)
from mjsrl.storage import read_transitions  # noqa: E402

GAME_ID = "261009-12345678-abcd-1234-abcd-123456789abc"
METADATA = {"permission_basis": "user_owned", "ruleset": "majsoul_4p_standard"}


def make_record(fixture_dir, seed=42, modern=False):
    golden = json.loads((fixture_dir / f"seed_{seed}.majsoul.json").read_text())
    manifest = json.loads((fixture_dir / f"seed_{seed}.manifest.json").read_text())
    details = pb.GameDetailRecords(version=210715 if modern else 0)
    for round_events in golden["data"]:
        for event in round_events:
            cls = getattr(pb, "Record" + event["name"])
            message = ParseDict(event["data"], cls(), ignore_unknown_fields=True)
            wrapped = pb.Wrapper(
                name=".lq.Record" + event["name"], data=message.SerializeToString()
            ).SerializeToString()
            if modern:
                details.actions.add(result=wrapped)
                details.actions.add(type=1)  # Player input entries have no result.
            else:
                details.records.append(wrapped)
    record = pb.ResGameRecord(
        data=pb.Wrapper(
            name=".lq.GameDetailRecords", data=details.SerializeToString()
        ).SerializeToString()
    )
    record.head.uuid = GAME_ID
    record.head.end_time = 1791510000
    record.head.config.category = 2
    record.head.config.mode.mode = 2
    for seat in range(4):
        record.head.accounts.add(seat=seat, account_id=seat + 100, nickname="fixture")
        record.head.result.players.add(
            seat=seat, part_point_1=manifest["expected_final_scores"][seat]
        )
    return record


@pytest.mark.parametrize(
    "url,region",
    [
        (GAME_ID, "cn"),
        (f"https://game.maj-soul.com/1/?paipu={GAME_ID}_a123456", "cn"),
        (f"雀魂牌谱:https://game.maj-soul.com/1/?paipu={GAME_ID}_2", "cn"),
        (f"https://game.mahjongsoul.com/?paipu={GAME_ID}", "jp"),
        (f"https://mahjongsoul.game.yo-star.com/?paipu={GAME_ID}", "en"),
    ],
)
def test_reference(url, region):
    reference = parse_reference(url)
    assert (reference.game_id, reference.region) == (GAME_ID, region)


@pytest.mark.parametrize(
    "url",
    [
        f"https://evil.example/?paipu={GAME_ID}",
        f"http://game.maj-soul.com/?paipu={GAME_ID}",
        f"https://user:secret@game.maj-soul.com/?paipu={GAME_ID}",
        f"https://game.maj-soul.com:1234/?paipu={GAME_ID}",
        f"https://game.maj-soul.com/?paipu={GAME_ID}&paipu={GAME_ID}",
        "../../escape",
        "https://game.maj-soul.com/",
    ],
)
def test_reject_invalid_reference(url):
    with pytest.raises(ReplayError):
        parse_reference(url)


@pytest.mark.parametrize("modern", [False, True])
@pytest.mark.parametrize("seed", [42, 43, 44])
def test_binary_to_training_pipeline(fixture_dir, tmp_path, seed, modern):
    record = make_record(fixture_dir, seed, modern)
    binary = tmp_path / "export.pb"
    manifest = tmp_path / "source.json"
    binary.write_bytes(record.SerializeToString())
    manifest.write_text(json.dumps(METADATA))
    dataset = tmp_path / "silver.parquet"
    result = decode_record(binary, manifest, tmp_path / "raw", dataset)
    assert result["raw_sha256"] == hashlib.sha256(binary.read_bytes()).hexdigest()
    rows = read_transitions(dataset)
    assert rows and result["transitions"] == len(rows)
    assert sum(row.done for row in rows) == 4
    for row in rows:
        row.validate()
    saved = json.loads((tmp_path / "raw" / GAME_ID / "manifest.json").read_text())
    assert saved["expected_final_scores"] == [p.part_point_1 for p in record.head.result.players]
    assert saved["raw_sha256"] == hashlib.sha256(binary.read_bytes()).hexdigest()
    assert decode_details(record.data)["data"][0][0]["data"]["ju"] == 0
    assert "operation" not in rows[0].state.to_dict()


def test_unknown_and_incomplete_events_rejected(fixture_dir):
    record = make_record(fixture_dir)
    wrapper = pb.Wrapper.FromString(record.data)
    details = pb.GameDetailRecords.FromString(wrapper.data)
    details.records.append(pb.Wrapper(name=".lq.RecordUnrecognized").SerializeToString())
    wrapper.data = details.SerializeToString()
    with pytest.raises(ReplayError, match="未支持"):
        decode_details(wrapper.SerializeToString())
    del details.records[-2:]
    wrapper.data = details.SerializeToString()
    with pytest.raises(ReplayError, match="最后一局"):
        decode_details(wrapper.SerializeToString())


@pytest.mark.parametrize(
    "change,reason",
    [
        (lambda r: setattr(r.head.config.mode, "mode", 11), "标准段位"),
        (lambda r: setattr(r.head.config.mode.detail_rule, "muyu_mode", 1), "特殊规则"),
        (lambda r: setattr(r.head, "end_time", 0), "已结束"),
        (lambda r: r.head.accounts.pop(), "四人"),
        (lambda r: setattr(r.head.result.players[0], "part_point_1", 1), "终局分数"),
    ],
)
def test_reject_bad_record(fixture_dir, tmp_path, change, reason):
    record = make_record(fixture_dir)
    change(record)
    with pytest.raises(ReplayError, match=reason):
        archive_record(record.SerializeToString(), METADATA, tmp_path)
    assert not (tmp_path / GAME_ID / "manifest.json").exists()


def test_existing_archive_not_overwritten(fixture_dir, tmp_path):
    record = make_record(fixture_dir)
    raw = record.SerializeToString()
    first = archive_record(raw, METADATA, tmp_path)
    assert archive_record(raw, METADATA, tmp_path) == first
    with pytest.raises(ReplayError, match="未覆盖"):
        write_once(tmp_path / GAME_ID / "record.pb", b"different")
    assert (tmp_path / GAME_ID / "record.pb").read_bytes() == raw


def test_checksum_mismatch(fixture_dir, tmp_path):
    raw = make_record(fixture_dir).SerializeToString()
    with pytest.raises(ReplayError, match="校验和"):
        archive_record(raw, {**METADATA, "raw_sha256": "bad"}, tmp_path)


def test_malformed_binary_error(tmp_path):
    raw = tmp_path / "wrong.pb"
    raw.write_bytes(b"not-a-protobuf-record")
    manifest = tmp_path / "source.json"
    manifest.write_text(json.dumps(METADATA))
    with pytest.raises(ReplayError, match="完整的 ResGameRecord"):
        decode_record(raw, manifest, tmp_path / "archive")


def test_missing_permission_rejected(fixture_dir, tmp_path):
    with pytest.raises(ReplayError, match="来源清单"):
        archive_record(make_record(fixture_dir).SerializeToString(), {}, tmp_path)
    assert not (tmp_path / GAME_ID).exists()


def test_login_error_does_not_leak_secrets():
    class Rpc:
        async def call(self, *args):
            return pb.ResLogin(error=pb.Error(code=101))

    with pytest.raises(ReplayError, match="101") as error:
        asyncio.run(client.login(Rpc(), "private-account", "private-password", "web-test"))
    assert "private-password" not in str(error.value)
    assert "private-account" not in str(error.value)


def test_route_connection_includes_platform():
    called = []

    class Rpc:
        async def call(self, method, request, response_type):
            called.append(method)
            assert method == ".lq.Route.requestConnection"
            assert request.route_id == "route-3" and request.type == 1
            assert request.timestamp > 0
            assert b"\x32\x03Web" in request.SerializeToString()
            return pb.ResRequestConnection(result=1)

    asyncio.run(client.request_connection(Rpc(), "wss://route-3.maj-soul.com:8443/gateway"))
    assert called == [".lq.Route.requestConnection"]


def test_route_connection_error():
    class Rpc:
        async def call(self, *args):
            return pb.ResRequestConnection(error=pb.Error(code=151))

    with pytest.raises(ReplayError, match="连接初始化"):
        asyncio.run(client.request_connection(Rpc(), "wss://route-2.maj-soul.com/gateway"))


def test_login_sets_resource_version_and_reports_151():
    class Rpc:
        async def call(self, method, request, response_type):
            assert request.client_version.resource == "0.11.252.w"
            assert request.client_version_string == "web-0.11.252"
            return pb.ResLogin(error=pb.Error(code=151))

    with pytest.raises(ReplayError, match="接口兼容性") as error:
        asyncio.run(client.login(Rpc(), "private-account", "private-password", "web-0.11.252"))
    assert "private-password" not in str(error.value)
    assert "private-account" not in str(error.value)


@pytest.mark.parametrize("data_url", [False, True])
def test_mock_download_login_fetch_decode_ingest(fixture_dir, tmp_path, monkeypatch, data_url):
    import aiohttp

    record = make_record(fixture_dir, modern=True)
    detail_bytes = record.data
    if data_url:
        record.data = b""
        record.data_url = "https://record.maj-soul.com/test"
    calls = []

    class FakeRpc:
        async def call(self, method, request, response_type):
            calls.append(method)
            if method == "login":
                assert (
                    request.password
                    == hmac.new(b"lailai", b"local-password", hashlib.sha256).hexdigest()
                )
                assert not request.gen_access_token
                return pb.ResLogin(account_id=123)
            assert request.game_uuid == GAME_ID
            if method == "fetchGameRecord" and calls.count(method) == 1:
                return pb.ResGameRecord(error=pb.Error(code=151))
            if method == "readGameRecord":
                assert response_type is pb.ResCommon
                return pb.ResCommon()
            return record

    class FakeContent:
        async def iter_chunked(self, size):
            yield detail_bytes

    class FakeResponse:
        status = 200
        content = FakeContent()

        def raise_for_status(self):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    class FakeSession:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        def get(self, url, **kwargs):
            assert url == "https://record.maj-soul.com/test"
            assert kwargs["allow_redirects"] is False
            return FakeResponse()

    async def fake_discover(session):
        return "web-test", ["wss://route-2.maj-soul.com/gateway"]

    @asynccontextmanager
    async def fake_connect(session, endpoints, timeout):
        yield FakeRpc()

    monkeypatch.setattr(aiohttp, "ClientSession", FakeSession)
    monkeypatch.setattr(client, "discover", fake_discover)
    monkeypatch.setattr(client, "connect", fake_connect)
    manifest = tmp_path / "source.json"
    manifest.write_text(json.dumps({**METADATA, "password": "should-not-be-copied"}))
    result = asyncio.run(
        client.download_replay(
            GAME_ID,
            "local-account",
            "local-password",
            manifest,
            tmp_path / "archive",
            tmp_path / "silver.parquet",
        )
    )
    assert result["transitions"] > 0
    assert calls == ["login", "fetchGameRecord", "readGameRecord", "fetchGameRecord"]
    for output in (tmp_path / "archive").rglob("*"):
        if output.is_file():
            content = output.read_bytes()
            assert b"local-password" not in content
            assert b"should-not-be-copied" not in content
    assert not pb.ResGameRecord.FromString(
        (tmp_path / "archive" / GAME_ID / "record.pb").read_bytes()
    ).data_url


def test_rpc_packets_notifications_and_disconnect():
    import aiohttp

    response = pb.Wrapper(data=pb.ResCommon().SerializeToString()).SerializeToString()

    class Socket:
        def __init__(self, messages):
            self.messages = iter(messages)

        async def send_bytes(self, packet):
            assert packet[:3] == b"\x02\x01\x00"
            assert pb.Wrapper.FromString(packet[3:]).name == ".lq.Lobby.readGameRecord"

        async def receive(self):
            return next(self.messages)

    socket = Socket(
        [
            SimpleNamespace(type=aiohttp.WSMsgType.BINARY, data=b"\x01notice"),
            SimpleNamespace(type=aiohttp.WSMsgType.BINARY, data=b"\x03\x02\x00" + response),
            SimpleNamespace(type=aiohttp.WSMsgType.BINARY, data=b"\x03\x01\x00" + response),
        ]
    )
    result = asyncio.run(
        client.Rpc(socket, 1).call(
            "readGameRecord", pb.ReqGameRecord(game_uuid=GAME_ID), pb.ResCommon
        )
    )
    assert result.error.code == 0
    socket = Socket([SimpleNamespace(type=aiohttp.WSMsgType.CLOSED)])
    with pytest.raises(ReplayError, match="断开"):
        asyncio.run(client.Rpc(socket, 1).call("readGameRecord", pb.ReqGameRecord(), pb.ResCommon))


def test_rpc_timeout():
    class Socket:
        async def send_bytes(self, packet):
            pass

        async def receive(self):
            await asyncio.Event().wait()

    with pytest.raises(TimeoutError):
        asyncio.run(
            client.Rpc(Socket(), 0.01).call("readGameRecord", pb.ReqGameRecord(), pb.ResCommon)
        )


def test_decode_cli(fixture_dir, tmp_path, monkeypatch, capsys):
    from mjsrl.cli import main

    raw = tmp_path / "record.pb"
    raw.write_bytes(make_record(fixture_dir).SerializeToString())
    manifest = tmp_path / "source.json"
    manifest.write_text(json.dumps(METADATA))
    monkeypatch.setattr(
        "sys.argv",
        [
            "mjsrl",
            "decode-replay",
            "--input",
            str(raw),
            "--manifest",
            str(manifest),
            "--output-dir",
            str(tmp_path),
        ],
    )
    main()
    assert json.loads(capsys.readouterr().out)["game_id"] == GAME_ID


@pytest.mark.parametrize(
    "url",
    [
        "https://evil.example/record",
        "http://record.maj-soul.com/record",
        "https://record.maj-soul.com.evil.example/record",
        "https://user:secret@maj-soul.com/",
    ],
)
def test_reject_unexpected_server_url(url):
    with pytest.raises(ReplayError):
        client.official_url(url)
