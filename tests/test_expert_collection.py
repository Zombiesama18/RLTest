import asyncio
import json
from contextlib import asynccontextmanager

import pytest

pb = pytest.importorskip("ms.protocol_pb2")
from test_majsoul_acquisition import make_record  # noqa: E402

from mjsrl import spectator  # noqa: E402
from mjsrl.expert_data import (  # noqa: E402
    ExpertFilter,
    completed_selection,
    discover_candidate,
    rank,
)
from mjsrl.ingest import ingest  # noqa: E402
from mjsrl.replay_batch import ensure_dataset  # noqa: E402
from mjsrl.replays import ReplayError, archive_record  # noqa: E402
from mjsrl.storage import read_transitions  # noqa: E402

METADATA = {"permission_basis": "public_replay", "ruleset": "majsoul_4p_standard"}


@pytest.mark.parametrize(
    "level,expected",
    [
        (10101, 1),
        (10403, 4),
        (10501, 5),
        (10601, 6),
        (10701, 6),
        (10720, 6),
        (20720, None),
        (10721, None),
        (10602, None),
        (10801, None),
        (10500, None),
        (True, None),
    ],
)
def test_rank_ladders(level, expected):
    assert rank(level) == expected


def heads(fixture_dir):
    record = make_record(fixture_dir, modern=True)
    record.head.config.meta.mode_id = 12
    # Include the losing player, rather than collecting only winning experts.
    loser = min(record.head.result.players, key=lambda p: p.part_point_1).seat
    other = (loser + 1) % 4
    for account in record.head.accounts:
        account.level.id = (
            10502 if account.seat == loser else 10703 if account.seat == other else 10403
        )
    live = pb.GameLiveHead(uuid=record.head.uuid, start_time=100)
    live.game_config.CopyFrom(record.head.config)
    for account in reversed(record.head.accounts):
        player = live.players.add(account_id=account.account_id)
        player.level.CopyFrom(account.level)
    live.seat_list.extend(a.account_id for a in record.head.accounts)
    return live, record, sorted([loser, other])


def test_live_account_ids_and_final_seat_mapping(fixture_dir):
    live, record, seats = heads(fixture_dir)
    selection = ExpertFilter()
    candidate = discover_candidate(live, selection)
    expert = completed_selection(record.head, candidate, selection)
    assert expert["selected_seats"] == seats
    assert discover_candidate(live, ExpertFilter(min_players=3)) is None
    live.seat_list.append(live.seat_list[0])
    assert discover_candidate(live, selection) is None


@pytest.mark.parametrize("change", ["room", "three_player", "unknown_rank", "duplicate_player"])
def test_live_filters_fail_closed(fixture_dir, change):
    live, _, _ = heads(fixture_dir)
    if change == "room":
        live.game_config.meta.mode_id = 8
    elif change == "three_player":
        live.game_config.mode.mode = 11
    elif change == "unknown_rank":
        for p in live.players:
            p.level.id = 10801
    else:
        live.players[0].account_id = live.players[1].account_id
    assert discover_candidate(live, ExpertFilter()) is None


@pytest.mark.parametrize("change", ["identity", "room", "incomplete"])
def test_final_header_gate(fixture_dir, change):
    live, record, _ = heads(fixture_dir)
    candidate = discover_candidate(live, ExpertFilter())
    if change == "identity":
        record.head.accounts[0].account_id = 999
    elif change == "room":
        record.head.config.meta.mode_id = 16
    else:
        record.head.result.players.pop()
    with pytest.raises(ReplayError):
        completed_selection(record.head, candidate, ExpertFilter())


def test_final_ranks_override_live(fixture_dir):
    live, record, _ = heads(fixture_dir)
    candidate = discover_candidate(live, ExpertFilter())
    for a in record.head.accounts:
        a.level.id = 10403
    assert completed_selection(record.head, candidate, ExpertFilter()) is None


def test_bounded_discovery_includes_both_rooms(fixture_dir, tmp_path):
    live, _, _ = heads(fixture_dir)
    throne = pb.GameLiveHead()
    throne.CopyFrom(live)
    throne.uuid = "261009-12345678-abcd-1234-abcd-000000000002"
    throne.game_config.meta.mode_id = 16
    another_jade = pb.GameLiveHead()
    another_jade.CopyFrom(live)
    another_jade.uuid = "261009-12345678-abcd-1234-abcd-000000000003"

    class Rpc:
        async def call(self, method, request, response_type):
            return pb.ResGameLiveList(
                live_list=[live, another_jade] if request.filter_id == 212 else [throne]
            )

    queue = spectator.Queue(tmp_path / "queue.sqlite", ExpertFilter())
    report = asyncio.run(spectator.discover(Rpc(), queue, ExpertFilter(), 2, 100))
    assert report["newly_queued"] == 2 and report["observed"] == 3
    assert {j["candidate"]["room_mode_id"] for j in queue.due(100)} == {12, 16}
    queue.close()


def test_end_timestamp_without_complete_settlement_keeps_waiting(fixture_dir, tmp_path):
    live, record, _ = heads(fixture_dir)
    record.head.result.players.pop()

    class Rpc:
        async def call(self, method, request, response_type):
            assert method == "fetchGameRecordsDetail"
            return pb.ResGameRecordsDetail(record_list=[record.head])

    queue = spectator.Queue(tmp_path / "queue.sqlite", ExpertFilter())
    queue.add(discover_candidate(live, ExpertFilter()), 100, 0)
    asyncio.run(
        spectator.check_finished(
            None,
            Rpc(),
            queue,
            ExpertFilter(),
            METADATA,
            "fixture",
            tmp_path,
            None,
            spectator.CollectorConfig(),
            110,
        )
    )
    assert queue.counts() == {"waiting": 1}
    queue.close()


def test_queue_cap_checkpoint_and_immutable_filter(tmp_path):
    path = tmp_path / "queue.sqlite"
    queue = spectator.Queue(path, ExpertFilter())
    candidate = {"game_id": "first"}
    queue.add(candidate, 100, 1)
    queue.add(candidate, 110, 1)
    queue.add({"game_id": "second"}, 110, 1)
    job = queue.due(110)[0]
    queue.update(job, "waiting", 110, delay=60)
    queue.close()
    queue = spectator.Queue(path, ExpertFilter(rooms=("throne-south", "jade-south")))
    assert queue.counts() == {"waiting": 1}
    assert not queue.due(169)
    assert queue.due(170)[0]["attempts"] == 1
    queue.close()
    with pytest.raises(ReplayError, match="筛选条件"):
        spectator.Queue(path, ExpertFilter(min_rank="celestial"))


def test_queue_lock_is_released(tmp_path):
    lock = tmp_path / "collector.lock"
    with spectator.state_lock(lock):
        with pytest.raises(ReplayError, match="已有运行"):
            with spectator.state_lock(lock):
                pass
    with spectator.state_lock(lock):
        pass


def fake_network(monkeypatch, live, record, pending_first=False, read_errors=()):
    import aiohttp

    calls = []
    errors = list(read_errors)

    class Rpc:
        details = 0

        async def call(self, method, request, response_type):
            calls.append(method)
            if method == "login":
                return pb.ResLogin(account_id=9999)  # Experts need not belong to the collector.
            if method == "fetchServerTime":
                return pb.ResServerTime()
            if method == "fetchGameLiveList":
                assert request.filter_id in (212, 216)
                return pb.ResGameLiveList(live_list=[live] if request.filter_id == 212 else [])
            if method == "fetchGameRecordsDetail":
                self.details += 1
                if pending_first and self.details == 1:
                    return pb.ResGameRecordsDetail()
                return pb.ResGameRecordsDetail(record_list=[record.head])
            if method == "readGameRecord":
                assert response_type == pb.ResCommon
                return pb.ResCommon(error=pb.Error(code=errors.pop(0) if errors else 0))
            if method == "fetchGameRecord":
                return record
            raise AssertionError(method)

    class Session:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    async def discovery(session):
        return "web-fixture", ["wss://route-2.maj-soul.com/gateway"]

    @asynccontextmanager
    async def connect(session, endpoints, timeout):
        yield Rpc()

    monkeypatch.setattr(aiohttp, "ClientSession", Session)
    monkeypatch.setattr(spectator.client, "discover", discovery)
    monkeypatch.setattr(spectator.client, "connect", connect)
    return calls


def arguments(tmp_path):
    manifest = tmp_path / "source.json"
    manifest.write_text(json.dumps(METADATA))
    return dict(
        directory=tmp_path / "experts",
        manifest=manifest,
        account="local-account",
        password="local-secret",
        config=spectator.CollectorConfig(duration=0.02, poll=0.001, interval=0),
    )


def test_discovery_never_prompts_or_fetches_records(fixture_dir, tmp_path, monkeypatch):
    live, record, _ = heads(fixture_dir)
    calls = fake_network(monkeypatch, live, record)
    args = arguments(tmp_path)
    args.update(account=None, password=None, discover_only=True)
    report = asyncio.run(spectator.collect(**args))
    assert report["status"] == "discovered" and report["counts"] == {"waiting": 1}
    assert calls == ["fetchGameLiveList", "fetchGameLiveList"]
    assert not list(args["directory"].rglob("record.pb"))


def test_live_finish_download_filter_resume_and_offline_export(fixture_dir, tmp_path, monkeypatch):
    live, record, seats = heads(fixture_dir)
    calls = fake_network(monkeypatch, live, record, pending_first=True)
    args = arguments(tmp_path)
    report = asyncio.run(spectator.collect(**args))
    assert report["counts"] == {"saved": 1}
    assert calls.count("login") == 1 and calls.count("fetchGameRecord") == 1
    assert calls.index("readGameRecord") > calls.index("fetchGameRecordsDetail")
    archive = args["directory"] / "raw" / record.head.uuid
    metadata = json.loads((archive / "manifest.json").read_text(encoding="utf-8"))
    assert metadata["selected_seats"] == seats
    assert metadata["redistributable"] is False
    assert metadata["public_observation"]["source"] == "fetchGameLiveList"
    assert "local-secret" not in json.dumps(metadata)
    # A later run can generate selected datasets without another login or download.
    args.update(export_only=True, account=None, password=None, dataset_dir=tmp_path / "silver")
    before = list(calls)
    exported = asyncio.run(spectator.collect(**args))
    assert exported["exported"] == 1 and calls == before
    selected = read_transitions(args["dataset_dir"] / (record.head.uuid + ".parquet"))
    assert {r.seat for r in selected} == set(seats)
    # Full reconstruction occurs before filtering: labels, opponent history, clock,
    # rewards and elapsed discounts match the equivalent rows from the full replay.
    full_manifest = tmp_path / "full.json"
    full_manifest.write_text(json.dumps({**metadata, "selected_seats": None}))
    full_file = tmp_path / "full.parquet"
    ingest(archive / "replay.json", full_file, full_manifest)
    full = read_transitions(full_file)
    assert [r.to_dict() for r in selected] == [r.to_dict() for r in full if r.seat in seats]
    loser = min(record.head.result.players, key=lambda p: p.part_point_1).seat
    assert any(r.seat == loser and r.done for r in selected)
    assert asyncio.run(spectator.collect(**args))["exported"] == 0
    args.update(export_only=False, account="local", password="secret")
    assert asyncio.run(spectator.collect(**args))["counts"] == {"saved": 1}
    assert calls.count("fetchGameRecord") == 1


@pytest.mark.parametrize(
    "error,expired,auth",
    [(0, False, False), (540, False, False), (0, True, False), (1004, False, True)],
)
def test_disappearance_is_not_completion_and_auth_keeps_queue(
    fixture_dir, tmp_path, error, expired, auth
):
    live, _, _ = heads(fixture_dir)
    selection = ExpertFilter()
    queue = spectator.Queue(tmp_path / "queue.sqlite", selection)
    queue.add(discover_candidate(live, selection), 100, 0)

    class Rpc:
        async def call(self, method, request, response_type):
            assert method == "fetchGameRecordsDetail"
            return pb.ResGameRecordsDetail(error=pb.Error(code=error))

    coro = spectator.check_finished(
        None,
        Rpc(),
        queue,
        selection,
        METADATA,
        "fixture",
        tmp_path,
        None,
        spectator.CollectorConfig(record_wait=100),
        300 if expired else 110,
    )
    if auth:
        with pytest.raises(spectator.SessionExpired):
            asyncio.run(coro)
    else:
        asyncio.run(coro)
    assert queue.counts() == {"unavailable" if expired else "waiting": 1}
    assert not list(tmp_path.rglob("record.pb"))
    queue.close()


def test_decode_failure_keeps_raw_and_quarantines(fixture_dir, tmp_path, monkeypatch):
    live, record, _ = heads(fixture_dir)
    record.data = b"invalid"
    calls = fake_network(monkeypatch, live, record)
    args = arguments(tmp_path)
    args["dataset_dir"] = tmp_path / "silver"
    report = asyncio.run(spectator.collect(**args))
    assert report["counts"] == {"quarantined": 1}
    assert (args["directory"] / "raw" / record.head.uuid / "record.pb").exists()
    assert not list(tmp_path.rglob("*.parquet"))
    asyncio.run(spectator.collect(**args))
    assert calls.count("fetchGameRecord") == 1


def test_download_not_ready_retries_without_losing_job(fixture_dir, tmp_path, monkeypatch):
    live, record, _ = heads(fixture_dir)
    calls = fake_network(monkeypatch, live, record, read_errors=[540])
    report = asyncio.run(spectator.collect(**arguments(tmp_path)))
    assert report["counts"] == {"saved": 1}
    assert calls.count("readGameRecord") == 2 and calls.count("fetchGameRecord") == 1


def test_expired_session_stops_but_pending_job_survives(fixture_dir, tmp_path, monkeypatch):
    live, record, _ = heads(fixture_dir)
    calls = fake_network(monkeypatch, live, record, read_errors=[1004])
    args = arguments(tmp_path)
    with pytest.raises(spectator.SessionExpired):
        asyncio.run(spectator.collect(**args))
    status = json.loads((args["directory"] / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "failed" and status["counts"] == {"waiting": 1}
    assert "登录" in status["reason"]
    assert "local-secret" not in json.dumps(status)
    assert "fetchGameRecord" not in calls
    assert asyncio.run(spectator.collect(**args))["counts"] == {"saved": 1}


def test_interrupted_decode_resumes_from_raw(fixture_dir, tmp_path, monkeypatch):
    live, record, _ = heads(fixture_dir)
    calls = fake_network(monkeypatch, live, record)
    args = arguments(tmp_path)
    original = spectator.archive_record

    def interrupt(*a, **kw):
        raise KeyboardInterrupt

    monkeypatch.setattr(spectator, "archive_record", interrupt)
    with pytest.raises(KeyboardInterrupt):
        asyncio.run(spectator.collect(**args))
    status = json.loads((args["directory"] / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "interrupted" and status["counts"] == {"waiting": 1}
    assert (args["directory"] / "raw" / record.head.uuid / "record.pb").exists()
    monkeypatch.setattr(spectator, "archive_record", original)
    assert asyncio.run(spectator.collect(**args))["counts"] == {"saved": 1}
    assert calls.count("fetchGameRecord") == 1


def test_offline_export_rejects_changed_raw(fixture_dir, tmp_path, monkeypatch):
    live, record, _ = heads(fixture_dir)
    fake_network(monkeypatch, live, record)
    args = arguments(tmp_path)
    asyncio.run(spectator.collect(**args))
    (args["directory"] / "raw" / record.head.uuid / "record.pb").write_bytes(b"changed")
    args.update(export_only=True, dataset_dir=tmp_path / "silver", account=None, password=None)
    with pytest.raises(ReplayError, match="缓存校验失败"):
        asyncio.run(spectator.collect(**args))


def test_paced_rpc_heartbeat_and_auth_stop(monkeypatch):
    methods = []

    class Rpc:
        async def call(self, method, request, response_type):
            methods.append(method)
            return response_type(error=pb.Error(code=1004 if method == "fetchGameRecord" else 0))

    rpc = spectator.PacedRpc(Rpc(), 0)
    rpc.authenticated = True
    rpc.heartbeat -= 46
    with pytest.raises(spectator.SessionExpired):
        asyncio.run(rpc.call("fetchGameRecord", pb.ReqGameRecord(), pb.ResGameRecord))
    assert methods == ["fetchServerTime", "fetchGameRecord"]


def test_idle_polling_keeps_authenticated_connection_alive(monkeypatch):
    methods = []

    class Rpc:
        async def call(self, method, request, response_type):
            methods.append(method)
            return response_type()

    monkeypatch.setattr(spectator, "KEEPALIVE_SECONDS", .005)
    rpc = spectator.PacedRpc(Rpc(), 0)
    rpc.authenticated = True

    async def run():
        await rpc.keepalive(force=True)
        await rpc.pause(.02)

    asyncio.run(run())
    assert len(methods) >= 2
    assert set(methods) == {"fetchServerTime"}


def test_keepalive_protocol_error_is_not_mislabeled_bad_password():
    class Rpc:
        async def call(self, method, request, response_type):
            assert method == "fetchServerTime"
            return response_type(error=pb.Error(code=6))

    rpc = spectator.PacedRpc(Rpc(), 0)
    rpc.authenticated = True
    with pytest.raises(ReplayError, match="接口错误码 6") as exc:
        asyncio.run(rpc.keepalive(force=True))
    assert not isinstance(exc.value, spectator.SessionExpired)


def test_cli_discovery_without_tty_or_password(monkeypatch, capsys):
    from mjsrl import cli

    async def collect(*a, **kw):
        assert kw["discover_only"] and kw["account"] is None and kw["password"] is None
        return {"status": "discovered", "counts": {"waiting": 3}}

    monkeypatch.setattr("sys.argv", ["mjsrl", "collect-experts", "--discover-only"])
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(cli.getpass, "getpass", lambda *a: pytest.fail("must not prompt"))
    monkeypatch.setattr(spectator, "collect", collect)
    cli.main()
    assert json.loads(capsys.readouterr().out)["status"] == "discovered"


def test_selection_change_rejects_cached_dataset(fixture_dir, tmp_path):
    live, record, _ = heads(fixture_dir)
    selection = ExpertFilter()
    expert = completed_selection(record.head, discover_candidate(live, selection), selection)
    result = archive_record(record.SerializeToString(), {**METADATA, **expert}, tmp_path / "raw")
    assert ensure_dataset(result, tmp_path / "silver", 0.997)
    manifest = tmp_path / "raw" / record.head.uuid / "manifest.json"
    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    metadata["selected_seats"] = [0, 1, 2, 3]
    manifest.write_text(json.dumps(metadata))
    with pytest.raises(ReplayError, match="参数变化"):
        ensure_dataset(result, tmp_path / "silver", 0.997)


@pytest.mark.parametrize("seats", [[], [True], [4], [0, 0], "all"])
def test_invalid_seat_selection_rejected_before_ingest(tmp_path, seats):
    manifest = tmp_path / "source.json"
    manifest.write_text(json.dumps({**METADATA, "selected_seats": seats}))
    with pytest.raises(ValueError, match="selected_seats"):
        ingest(tmp_path / "missing.json", tmp_path / "out.parquet", manifest)


def test_cli_validation_before_password(monkeypatch):
    from mjsrl import cli

    monkeypatch.setattr("sys.argv", ["mjsrl", "collect-experts", "--min-players", "5"])
    monkeypatch.setattr(cli.getpass, "getpass", lambda *a: pytest.fail("must not prompt"))
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
