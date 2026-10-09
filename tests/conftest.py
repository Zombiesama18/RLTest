from pathlib import Path

import pytest

from mjsrl.ingest import replay_transitions


@pytest.fixture(scope="session")
def fixture_dir():
    return Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def records(fixture_dir):
    return list(replay_transitions(fixture_dir / "seed_42.jsonl"))


@pytest.fixture
def state(records):
    return records[0].state
