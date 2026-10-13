import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

pb = pytest.importorskip("ms.protocol_pb2")
from test_majsoul_acquisition import make_record  # noqa: E402

from mjsrl import replay_batch as batch  # noqa: E402
from mjsrl.replays import ReplayError, archive_record  # noqa: E402
from mjsrl.storage import file_hash  # noqa: E402

METADATA = {"permission_basis": "user_owned", "ruleset": "majsoul_4p_standard"}
OWNER_ID = 101


def records(fixture_dir, count=3):
    values = []
    for index in range(count):
        record = make_record(fixture_dir, modern=True)
        record.head.uuid = f"261009-12345678-abcd-1234-abcd-{index:012x}"
        values.append(record)
    return values


def test_pagination_filters_and_limit(fixture_dir):
    values = records(fixture_dir, 4)
    values[1].head.config.mode.mode = 11
    calls = []

    class Rpc:
        async def call(self, method, request, response_type):
            calls.append(request.start)
            assert method == "fetchGameRecordList" and request.type == 2
            return pb.ResGameRecordList(
                total_count=4,
                record_list=[
                    r.head for r in values[request.start - 1 : request.start - 1 + request.count]
                ],
            )

    ids, filtered = asyncio.run(batch.list_account_replays(Rpc(), OWNER_ID, 2, 0, 2))
    assert ids == [values[0].head.uuid, values[2].head.uuid]
    assert filtered == 1 and calls == [1, 3]


def test_all_pages_without_total_and_foreign_owner(fixture_dir):
    values = records(fixture_dir, 4)
    values[1].head.accounts[1].account_id = 9999
    calls = []

    class Rpc:
        async def call(self, method, request, response_type):
            calls.append(request.start)
            return pb.ResGameRecordList(
                record_list=[
                    r.head for r in values[request.start - 1 : request.start - 1 + request.count]
                ]
            )

    ids, filtered = asyncio.run(batch.list_account_replays(Rpc(), OWNER_ID, 0, 0, 2))
    assert len(ids) == 3 and filtered == 1
    assert calls == [1, 3, 5]


def test_pagination_repeat_stops(fixture_dir):
    values = records(fixture_dir, 2)

    class Rpc:
        async def call(self, *args):
            return pb.ResGameRecordList(total_count=100, record_list=[r.head for r in values])

    with pytest.raises(ReplayError, match="重复返回"):
        asyncio.run(batch.list_account_replays(Rpc(), OWNER_ID, 0, 0, 2))


def test_list_error():
    class Rpc:
        async def call(self, *args):
            return pb.ResGameRecordList(error=pb.Error(code=123))

    with pytest.raises(ReplayError, match="123"):
        asyncio.run(batch.list_account_replays(Rpc(), OWNER_ID, 1, 0))


def test_short_page_is_not_end_when_more_records_exist(fixture_dir):
    values = records(fixture_dir, 4)
    starts = []

    class Rpc:
        async def call(self, method, request, response_type):
            starts.append(request.start)
            return pb.ResGameRecordList(
                total_count=4,
                record_list=[r.head for r in values[request.start - 1 : request.start + 1]],
            )

    ids, _ = asyncio.run(batch.list_account_replays(Rpc(), OWNER_ID, 0, 0))
    assert len(ids) == 4 and starts == [1, 3]


def fake_network(monkeypatch, values, failures=None, disconnect=False):
    import aiohttp

    calls = []
    failures = failures if failures is not None else set()

    class Rpc:
        socket = SimpleNamespace(closed=False)

        async def call(self, method, request, response_type):
            calls.append(method)
            if method == "login":
                return pb.ResLogin(account_id=OWNER_ID)
            if method == "fetchGameRecordList":
                return pb.ResGameRecordList(
                    total_count=len(values), record_list=[r.head for r in values]
                )
            if method == "fetchGameRecord":
                if request.game_uuid in failures:
                    self.socket.closed = disconnect
                    raise ReplayError("fixture download failed")
                return next(r for r in values if r.head.uuid == request.game_uuid)
            raise AssertionError(method)

    class Session:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    async def discover(session):
        return "web-fixture", ["wss://route-2.maj-soul.com/gateway"]

    @asynccontextmanager
    async def connect(session, endpoints, timeout):
        yield Rpc()

    monkeypatch.setattr(aiohttp, "ClientSession", Session)
    monkeypatch.setattr(batch.client, "discover", discover)
    monkeypatch.setattr(batch.client, "connect", connect)
    return calls


def test_batch_one_login_failure_retry_and_resume(fixture_dir, tmp_path, monkeypatch):
    values = records(fixture_dir, 3)
    failures = {values[1].head.uuid}
    calls = fake_network(monkeypatch, values, failures)
    manifest = tmp_path / "source.json"
    manifest.write_text(json.dumps(METADATA))
    kwargs = dict(
        account="account",
        password="local-secret",
        manifest=manifest,
        directory=tmp_path / "raw",
        dataset_dir=tmp_path / "silver",
        limit=0,
        interval=0,
    )
    first = asyncio.run(batch.download_account_replays(**kwargs))
    assert (first["downloaded"], first["failed"], first["converted"]) == (2, 1, 2)
    assert first["status"] == "partial" and calls.count("login") == 1
    failures.clear()
    second = asyncio.run(batch.download_account_replays(**kwargs))
    assert (second["downloaded"], second["skipped"], second["converted"]) == (1, 2, 1)
    assert second["status"] == "complete" and calls.count("login") == 2
    fetch_count = calls.count("fetchGameRecord")
    third = asyncio.run(batch.download_account_replays(**kwargs))
    assert third["skipped"] == 3 and calls.count("fetchGameRecord") == fetch_count
    assert third["pending"] == 0
    assert len(list((tmp_path / "silver").glob("*.parquet"))) == 3
    for summary in (tmp_path / "raw" / "batches").glob("*.json"):
        assert "local-secret" not in summary.read_text()
    for line in Path(first["journal"]).read_text().splitlines():
        item = json.loads(line)
        if item["status"] == "downloaded":
            assert item["raw_sha256"] == file_hash(Path(item["raw"]))


def test_cached_download_later_builds_dataset(fixture_dir, tmp_path, monkeypatch):
    values = records(fixture_dir, 1)
    calls = fake_network(monkeypatch, values)
    manifest = tmp_path / "source.json"
    manifest.write_text(json.dumps(METADATA))
    kwargs = dict(
        account="account",
        password="password",
        manifest=manifest,
        directory=tmp_path / "raw",
        interval=0,
    )
    first = asyncio.run(batch.download_account_replays(**kwargs))
    assert first["downloaded"] == 1 and first["converted"] == 0
    second = asyncio.run(batch.download_account_replays(**kwargs, dataset_dir=tmp_path / "silver"))
    assert second["reused"] == 1 and second["converted"] == 1
    assert calls.count("fetchGameRecord") == 1


def test_cache_tampering_not_overwritten(fixture_dir, tmp_path):
    record = records(fixture_dir, 1)[0]
    result = archive_record(record.SerializeToString(), METADATA, tmp_path)
    Path(result["replay"]).write_text("broken")
    with pytest.raises(ReplayError, match="校验和"):
        batch.cached_replay(tmp_path, record.head.uuid, OWNER_ID)
    assert Path(result["replay"]).read_text() == "broken"


def test_changed_gamma_and_tampered_dataset_refused(fixture_dir, tmp_path):
    record = records(fixture_dir, 1)[0]
    result = archive_record(record.SerializeToString(), METADATA, tmp_path / "raw")
    dataset_dir = tmp_path / "silver"
    assert batch.ensure_dataset(result, dataset_dir, 0.997)
    with pytest.raises(ReplayError, match="参数变化"):
        batch.ensure_dataset(result, dataset_dir, 0.5)
    Path(result["output"]).write_bytes(b"corrupt")
    with pytest.raises(ReplayError, match="校验和"):
        batch.ensure_dataset(result, dataset_dir, 0.997)


def test_disconnect_keeps_pending(fixture_dir, tmp_path, monkeypatch):
    values = records(fixture_dir, 3)
    fake_network(monkeypatch, values, {values[0].head.uuid}, disconnect=True)
    manifest = tmp_path / "source.json"
    manifest.write_text(json.dumps(METADATA))
    result = asyncio.run(
        batch.download_account_replays(
            "account",
            "password",
            manifest,
            tmp_path / "raw",
            interval=0,
        )
    )
    assert (result["status"], result["failed"], result["pending"]) == ("stopped", 1, 2)
    assert json.loads(Path(result["report"]).read_text())["pending"] == 2


def test_interrupt_preserves_completed_records(fixture_dir, tmp_path, monkeypatch):
    values = records(fixture_dir, 3)
    fake_network(monkeypatch, values)
    manifest = tmp_path / "source.json"
    manifest.write_text(json.dumps(METADATA))

    def interrupt(value):
        if value.get("current") == 1:
            raise KeyboardInterrupt

    kwargs = dict(
        account="account",
        password="password",
        manifest=manifest,
        directory=tmp_path / "raw",
        interval=0,
    )
    with pytest.raises(KeyboardInterrupt):
        asyncio.run(batch.download_account_replays(**kwargs, progress=interrupt))
    reports = list((tmp_path / "raw" / "batches").glob("*.json"))
    report = json.loads(reports[0].read_text())
    assert (report["status"], report["downloaded"], report["pending"]) == ("interrupted", 1, 2)
    resumed = asyncio.run(batch.download_account_replays(**kwargs))
    assert resumed["downloaded"] == 2 and resumed["skipped"] == 1


def test_cached_records_do_not_wait_for_download_interval(fixture_dir, tmp_path, monkeypatch):
    values = records(fixture_dir, 1)
    fake_network(monkeypatch, values)
    intervals = []

    async def sleep(seconds):
        intervals.append(seconds)

    monkeypatch.setattr(batch.asyncio, "sleep", sleep)
    manifest = tmp_path / "source.json"
    manifest.write_text(json.dumps(METADATA))
    kwargs = dict(
        account="account",
        password="password",
        manifest=manifest,
        directory=tmp_path / "raw",
        interval=7,
    )
    asyncio.run(batch.download_account_replays(**kwargs))
    assert intervals == [7]
    asyncio.run(batch.download_account_replays(**kwargs))
    assert intervals == [7]


@pytest.mark.parametrize("status,exit_code", [("complete", None), ("partial", 1)])
def test_batch_cli(status, exit_code, tmp_path, monkeypatch, capsys):
    from mjsrl.cli import main

    manifest = tmp_path / "source.json"
    manifest.write_text(json.dumps(METADATA))

    async def download(
        account, password, source, directory, dataset_dir, limit, interval, timeout, gamma, progress
    ):
        assert account == "local-account" and password == "local-secret"
        assert limit == 0 and interval == 2
        return {"status": status}

    monkeypatch.setattr(batch, "download_account_replays", download)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt: "local-account")
    monkeypatch.setattr("getpass.getpass", lambda prompt: "local-secret")
    monkeypatch.setattr(
        "sys.argv",
        [
            "mjsrl",
            "fetch-replays",
            "--manifest",
            str(manifest),
            "--output-dir",
            str(tmp_path),
            "--limit",
            "0",
        ],
    )
    if exit_code is None:
        main()
    else:
        with pytest.raises(SystemExit) as error:
            main()
        assert error.value.code == exit_code
    output = capsys.readouterr().out
    assert json.loads(output)["status"] == status and "local-secret" not in output


@pytest.mark.parametrize(
    "limit,interval,timeout,gamma",
    [
        (-1, 2, 30, 0.997),
        (0, -1, 30, 0.997),
        (1, float("nan"), 30, 0.997),
        (1, 1, float("inf"), 0.997),
        (1, 1, 30, 0),
    ],
)
def test_invalid_options(limit, interval, timeout, gamma):
    with pytest.raises(ReplayError):
        batch.validate_options(limit, interval, timeout, gamma)


def test_batch_manifest_rejects_single_game_hash(tmp_path):
    manifest = tmp_path / "source.json"
    manifest.write_text(json.dumps({**METADATA, "sha256": "one-game-only"}))
    with pytest.raises(ReplayError, match="单场"):
        batch.batch_manifest(manifest)
