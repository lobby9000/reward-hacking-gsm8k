import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures"
sys.path.insert(0, str(ROOT / "scripts"))


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory):
    """A data cache directory holding small fixture files instead of the full GSM8K and SVAMP."""
    path = tmp_path_factory.mktemp("gsm8k")
    shutil.copy(FIXTURES / "gsm8k_train_sample.jsonl", path / "train.jsonl")
    shutil.copy(FIXTURES / "gsm8k_test_sample.jsonl", path / "test.jsonl")
    shutil.copy(FIXTURES / "svamp_sample.json", path / "svamp.json")
    return path


@pytest.fixture(autouse=True)
def _use_fixture_data(data_dir, monkeypatch):
    monkeypatch.setenv("GSM8K_CACHE", str(data_dir))


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory, data_dir):
    from make_tiny_model import make_tiny_model

    from posttrain.data import fewshot_prefix, load_jsonl_problems

    texts = [fewshot_prefix(load_jsonl_problems(data_dir / "train.jsonl"))]
    return make_tiny_model(tmp_path_factory.mktemp("tiny") / "model", texts=texts, vocab_size=400)
