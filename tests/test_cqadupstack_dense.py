"""CQADupStack can produce dense rows, which is what P1 needed from it.

`tools/beir.py` refuses CQADupStack by name — twelve sub-collections behind one BEIR name, not one
corpus — so the member needs this tool, and until P1 asked for a dense row this tool was lexical-only.
The integration proof is the one-sub-forum run in `SCORE-LIFT.md` §P1; these pin the plumbing that
run depends on.
"""
from __future__ import annotations

import importlib.util
import pathlib
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


def test_a_dense_only_run_says_why_it_has_no_comparisons():
    """`comparisons_vs: bm25-engine` with an empty table is a report that looks broken.

    A run asked for only the dense rows carries no baseline row, so it cannot significance-test
    against one — legitimate, but it has to say so rather than leave an empty comparison block.
    """
    note = cq._comparison_note({"biencoder": 0.3, "hybrid-biencoder": 0.4}, "bm25-engine")
    assert "no bm25-engine row" in note
    assert "biencoder" in note                      # names what it does have
    assert cq._comparison_note({"biencoder": 0.3, "bm25-engine": 0.2}, "bm25-engine") == ""
    assert "nothing to compare" in cq._comparison_note({}, "bm25-engine")


def test_a_doubled_suffix_cache_is_migrated_not_rebuilt(tmp_path):
    """This tool used to hand `cached_embed` a path already ending in `.npy`.

    That wrote `forum.encoder.npy.npy` where every other dataset caches as `dataset.encoder.npy`.
    Renaming is cheaper than re-embedding, so the migration is what happens.
    """
    stem = tmp_path / "android.laya-retrieval.mean"
    legacy = {
        f"{stem}.npy.npy": b"vectors",
        f"{stem}.npy.ids.json": b"[]",
        f"{stem}.npy.meta.json": b"{}",
    }
    for name, payload in legacy.items():
        pathlib.Path(name).write_bytes(payload)

    cq._migrate_legacy_cache(stem)

    for suffix in (".npy", ".ids.json", ".meta.json"):
        assert pathlib.Path(f"{stem}{suffix}").is_file(), suffix
        assert not pathlib.Path(f"{stem}.npy{suffix}").exists(), suffix
    assert pathlib.Path(f"{stem}.npy").read_bytes() == b"vectors"


def test_migration_never_clobbers_a_current_cache(tmp_path):
    """A clean cache is left exactly as it is, even if a legacy file is also lying around."""
    stem = tmp_path / "android.laya-retrieval.mean"
    pathlib.Path(f"{stem}.npy").write_bytes(b"current")
    pathlib.Path(f"{stem}.npy.npy").write_bytes(b"legacy")

    cq._migrate_legacy_cache(stem)

    assert pathlib.Path(f"{stem}.npy").read_bytes() == b"current"
    assert pathlib.Path(f"{stem}.npy.npy").read_bytes() == b"legacy"      # left, not deleted
