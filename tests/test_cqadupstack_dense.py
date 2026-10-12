"""CQADupStack can produce dense rows, which is what P1 needed from it.

`tools/beir.py` refuses CQADupStack by name — twelve sub-collections behind one BEIR name, not one
corpus — so the member needs this tool, and until P1 asked for a dense row this tool was lexical-only.
The integration proof is the one-sub-forum run in `SCORE-LIFT.md` §P1; these pin the plumbing that
run depends on.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))


def _load():
    name = "laya_rag_cqadupstack"
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / "cqadupstack.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


cq = _load()


def test_the_dense_retrievers_are_accepted():
    assert "biencoder" in cq.DENSE_RETRIEVERS
    assert "hybrid-biencoder" in cq.DENSE_RETRIEVERS
    # …and the lexical ones did not move.
    assert cq.LEXICAL_RETRIEVERS == ("bm25", "bm25-engine", "bm25-title")


def test_the_dense_defaults_are_the_recorded_settings():
    """models/laya-retrieval at dense weight 0.3 is what the five full-channel cells used."""
    args = cq.parse_args(["--subforums", "android", "--retrievers", "hybrid-biencoder"])
    assert args.biencoder_model == "models/laya-retrieval"
    assert args.dense_weight == 0.3
    assert args.lexical_weight == 1.0
    assert args.embed_chunk == 0


def test_a_dense_row_without_an_encoder_says_so():
    """The failure has to be a sentence, not an AttributeError three frames down."""
    directory = Path(ROOT) / "eval" / "beir" / "cqadupstack" / "android"
    if not directory.is_dir():          # the corpus is optional for this test
        import pytest

        pytest.skip("cqadupstack/android is not on disk")
    import pytest

    with pytest.raises(SystemExit, match="needs an encoder"):
        cq.rank_subforum(directory, ["biencoder"], top_k=10, k1=0.9, b=0.4, stem=True,
                         include_titles=True, ignore_identical_ids=True)
