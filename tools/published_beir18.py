"""Published BEIR-18 reference figures: five retrieval systems, nDCG@10 and Recall@100.

Transcribed from `Resources for Brewing BEIR` (arXiv:2306.07471), Table 2 — the same paper this
repository already quotes its per-corpus BM25 anchors from. The table was **parsed from the
published HTML**, not retyped: `eval/.cache/brewing-beir.html` is the fetched page and the values
below are its cells, so a transcription slip cannot hide in them.

Two conventions matter when these are charted next to this repository's numbers:

* The `BM25` column is the paper's **multi-field** index (title and body as separate fields). This
  repository's `bm25` control is a **flat** index, so the two BM25 columns are not the same
  measurement; the flat anchors are the ones pinned in `tools/compare.py`.
* `Touche-2020` here is the **v2** variant (BM25 0.367). The anchor this repository quotes for
  Touche-2020 is the Pyserini 2CR flat run (0.442), a different collection release.

The five model columns (uniCOIL, SPLADE, TAS-B, Contriever) are encoder-based and carry no such
convention difference; they are comparable to a dense or hybrid row of ours on the same dataset.
"""

SOURCE = "Resources for Brewing BEIR, arXiv:2306.07471, Table 2"

SYSTEMS: tuple[str, ...] = ('BM25', 'uniCOIL', 'SPLADE', 'TAS-B', 'Contriever')

#: Dataset name in the source table -> this repository's dataset name.
CANONICAL: dict[str, str] = {
    'TREC-COVID': 'trec-covid', 'BioASQ': 'bioasq', 'NFCorpus': 'nfcorpus', 'NQ': 'nq',
    'HotpotQA': 'hotpotqa', 'FiQA-2018': 'fiqa', 'Signal-1M': 'signal1m',
    'TREC-NEWS': 'trec-news', 'Robust04': 'robust04', 'ArguAna': 'arguana',
    'Tóuche-2020 (v2)': 'webis-touche2020', 'CQADupStack': 'cqadupstack', 'Quora': 'quora',
    'DBPedia': 'dbpedia-entity', 'SCIDOCS': 'scidocs', 'FEVER': 'fever',
    'Climate-FEVER': 'climate-fever', 'SciFact': 'scifact',
}

#: nDCG@10 per source dataset name, per system.
NDCG10: dict[str, dict[str, float]] = {
    'TREC-COVID': {'BM25': 0.656, 'uniCOIL': 0.64, 'SPLADE': 0.711, 'TAS-B': 0.505, 'Contriever': 0.596},
    'BioASQ': {'BM25': 0.465, 'uniCOIL': 0.477, 'SPLADE': 0.504, 'TAS-B': 0.371, 'Contriever': 0.383},
    'NFCorpus': {'BM25': 0.325, 'uniCOIL': 0.333, 'SPLADE': 0.345, 'TAS-B': 0.324, 'Contriever': 0.328},
    'NQ': {'BM25': 0.329, 'uniCOIL': 0.425, 'SPLADE': 0.544, 'TAS-B': 0.465, 'Contriever': 0.498},
    'HotpotQA': {'BM25': 0.603, 'uniCOIL': 0.667, 'SPLADE': 0.686, 'TAS-B': 0.584, 'Contriever': 0.638},
    'FiQA-2018': {'BM25': 0.236, 'uniCOIL': 0.289, 'SPLADE': 0.351, 'TAS-B': 0.296, 'Contriever': 0.329},
    'Signal-1M': {'BM25': 0.33, 'uniCOIL': 0.275, 'SPLADE': 0.296, 'TAS-B': 0.288, 'Contriever': 0.278},
    'TREC-NEWS': {'BM25': 0.398, 'uniCOIL': 0.374, 'SPLADE': 0.394, 'TAS-B': 0.394, 'Contriever': 0.428},
    'Robust04': {'BM25': 0.407, 'uniCOIL': 0.403, 'SPLADE': 0.458, 'TAS-B': 0.461, 'Contriever': 0.473},
    'ArguAna': {'BM25': 0.414, 'uniCOIL': 0.396, 'SPLADE': 0.521, 'TAS-B': 0.436, 'Contriever': 0.446},
    'Tóuche-2020 (v2)': {'BM25': 0.367, 'uniCOIL': 0.298, 'SPLADE': 0.243, 'TAS-B': 0.222, 'Contriever': 0.204},
    'CQADupStack': {'BM25': 0.299, 'uniCOIL': 0.301, 'SPLADE': 0.341, 'TAS-B': 0.309, 'Contriever': 0.345},
    'Quora': {'BM25': 0.789, 'uniCOIL': 0.662, 'SPLADE': 0.814, 'TAS-B': 0.835, 'Contriever': 0.865},
    'DBPedia': {'BM25': 0.313, 'uniCOIL': 0.338, 'SPLADE': 0.442, 'TAS-B': 0.384, 'Contriever': 0.413},
    'SCIDOCS': {'BM25': 0.158, 'uniCOIL': 0.144, 'SPLADE': 0.159, 'TAS-B': 0.146, 'Contriever': 0.165},
    'FEVER': {'BM25': 0.753, 'uniCOIL': 0.812, 'SPLADE': 0.796, 'TAS-B': 0.733, 'Contriever': 0.758},
    'Climate-FEVER': {'BM25': 0.213, 'uniCOIL': 0.182, 'SPLADE': 0.228, 'TAS-B': 0.237, 'Contriever': 0.237},
    'SciFact': {'BM25': 0.665, 'uniCOIL': 0.686, 'SPLADE': 0.699, 'TAS-B': 0.644, 'Contriever': 0.677},
    'Avg.': {'BM25': 0.429, 'uniCOIL': 0.428, 'SPLADE': 0.474, 'TAS-B': 0.424, 'Contriever': 0.448},
}

#: Recall@100 per source dataset name, per system.
RECALL100: dict[str, dict[str, float]] = {
    'TREC-COVID': {'BM25': 0.114, 'uniCOIL': 0.111, 'SPLADE': 0.131, 'TAS-B': 0.09, 'Contriever': 0.091},
    'BioASQ': {'BM25': 0.715, 'uniCOIL': 0.731, 'SPLADE': 0.742, 'TAS-B': 0.598, 'Contriever': 0.607},
    'NFCorpus': {'BM25': 0.25, 'uniCOIL': 0.257, 'SPLADE': 0.289, 'TAS-B': 0.284, 'Contriever': 0.301},
    'NQ': {'BM25': 0.76, 'uniCOIL': 0.833, 'SPLADE': 0.929, 'TAS-B': 0.904, 'Contriever': 0.925},
    'HotpotQA': {'BM25': 0.74, 'uniCOIL': 0.798, 'SPLADE': 0.814, 'TAS-B': 0.728, 'Contriever': 0.777},
    'FiQA-2018': {'BM25': 0.544, 'uniCOIL': 0.553, 'SPLADE': 0.63, 'TAS-B': 0.582, 'Contriever': 0.656},
    'Signal-1M': {'BM25': 0.37, 'uniCOIL': 0.313, 'SPLADE': 0.331, 'TAS-B': 0.304, 'Contriever': 0.322},
    'TREC-NEWS': {'BM25': 0.422, 'uniCOIL': 0.357, 'SPLADE': 0.432, 'TAS-B': 0.454, 'Contriever': 0.492},
    'Robust04': {'BM25': 0.375, 'uniCOIL': 0.317, 'SPLADE': 0.377, 'TAS-B': 0.411, 'Contriever': 0.392},
    'ArguAna': {'BM25': 0.943, 'uniCOIL': 0.923, 'SPLADE': 0.982, 'TAS-B': 0.945, 'Contriever': 0.977},
    'Tóuche-2020 (v2)': {'BM25': 0.538, 'uniCOIL': 0.485, 'SPLADE': 0.472, 'TAS-B': 0.526, 'Contriever': 0.442},
    'CQADupStack': {'BM25': 0.606, 'uniCOIL': 0.569, 'SPLADE': 0.651, 'TAS-B': 0.612, 'Contriever': 0.663},
    'Quora': {'BM25': 0.973, 'uniCOIL': 0.948, 'SPLADE': 0.982, 'TAS-B': 0.986, 'Contriever': 0.994},
    'DBPedia': {'BM25': 0.398, 'uniCOIL': 0.441, 'SPLADE': 0.564, 'TAS-B': 0.499, 'Contriever': 0.541},
    'SCIDOCS': {'BM25': 0.356, 'uniCOIL': 0.328, 'SPLADE': 0.367, 'TAS-B': 0.332, 'Contriever': 0.378},
    'FEVER': {'BM25': 0.931, 'uniCOIL': 0.955, 'SPLADE': 0.955, 'TAS-B': 0.945, 'Contriever': 0.949},
    'Climate-FEVER': {'BM25': 0.436, 'uniCOIL': 0.418, 'SPLADE': 0.514, 'TAS-B': 0.553, 'Contriever': 0.575},
    'SciFact': {'BM25': 0.908, 'uniCOIL': 0.912, 'SPLADE': 0.927, 'TAS-B': 0.894, 'Contriever': 0.947},
    'Avg.': {'BM25': 0.577, 'uniCOIL': 0.569, 'SPLADE': 0.616, 'TAS-B': 0.591, 'Contriever': 0.613},
}

#: The paper's own macro average across all 18 datasets, per system.
AVERAGE: dict[str, float] = {'BM25': 0.429, 'uniCOIL': 0.428, 'SPLADE': 0.474, 'TAS-B': 0.424, 'Contriever': 0.448}
