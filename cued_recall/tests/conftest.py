import sys
from pathlib import Path

import pytest

# The package lives at <root>/cued_recall/ and the tests at <root>/tests/, so
# pytest's own sys.path insertion (the test file's own directory) is not enough
# to `import cued_recall`. Add the project root explicitly rather than requiring
# an editable install -- this project is run from a checkout, not from site-packages.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

EXAMPLE_CONFIG = ROOT / "config.example.yaml"

from cued_recall.config import Config  # noqa: E402
from cued_recall.index import VectorIndex  # noqa: E402
from cued_recall.pipeline import Pipeline  # noqa: E402
from cued_recall.store import BlockStore  # noqa: E402
from cued_recall.wal import WAL  # noqa: E402


@pytest.fixture
def config():
    return Config(EXAMPLE_CONFIG)


@pytest.fixture
def pipeline(tmp_path, config):
    """A real Pipeline over a temporary store.

    Nothing here talks to a server: Pipeline's constructor opens no
    connections, and the paths under test (message selection, correction
    scoping) make no model calls. `embed` is None so any accidental embed call
    fails loudly rather than silently reaching a real endpoint.
    """
    index = VectorIndex(tmp_path, dim=4)
    index.open()
    wal = WAL(tmp_path / "wal.jsonl")
    wal.open()
    p = Pipeline(config, BlockStore(tmp_path), index, embed=None, wal=wal)
    yield p
    wal.close()
    index.close()
