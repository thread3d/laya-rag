"""MultiHop-RAG retrieval under the benchmark's own protocol.

reimplemented from yixuantt/MultiHop-RAG `simple_retrieval.py` + `retrieval_evaluate.py`:
256-token SentenceSplitter chunks, and Hits@10 / Hits@4 / MAP@10 / MRR@10 computed by
substring-matching each gold evidence *fact* against each retrieved chunk.
"""
import sys, json, time, re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools')); sys.path.insert(0, str(ROOT / 'src'))
import beir, evalkit

root = ROOT / 'eval' / 'benchmarks' / 'multihop-rag'
raw = json.load(open(root/'corpus.json'))
items = json.load(open(root/'MultiHopRAG.json'))

def chunk_text(text, size, overlap):
    toks = text.split()
    step = max(1, size - overlap)
    out, i = [], 0
    while i < len(toks):
        out.append(' '.join(toks[i:i+size]))
        if i + size >= len(toks):
            break
        i += step
    return out

def official_metrics(retrieved_lists, gold_lists):
    """Aggregate metrics plus the per-query rows they average, so they can be bracketed.

    Retrieval here is deterministic — re-running reproduces the same number exactly — so the
    only meaningful error bar is over *queries*, which is what the per-query rows are for.
    """
    rows = []
    for retrieved, gold in zip(retrieved_lists, gold_lists):
        gold = [g.replace(' ', '').replace('\n', '') for g in gold]
        retrieved = [r.replace(' ', '').replace('\n', '') for r in retrieved]
        f10 = f4 = False; ap = 0.0; first = None; found = []
        for rank, item in enumerate(retrieved[:11], start=1):
            if any(g in item for g in gold):
                if rank <= 10:
                    f10 = True
                    if first is None: first = rank
                    if rank <= 4: f4 = True
                    c = 0
                    for g in gold:
                        if g in item and g not in found:
                            c += 1; found.append(g)
                    ap += c / rank
        rows.append({
            'Hits@10': float(f10), 'Hits@4': float(f4),
            'MAP@10': ap / min(len(gold), 10),
            'MRR@10': 1/first if first else 0.0,
        })
    n = len(rows)
    summary = {k: sum(r[k] for r in rows) / n for k in rows[0]}
    return summary, rows

ev_items = [(i, x) for i, x in enumerate(items) if x['question_type'] != 'null_query']
results = {}
for overlap in (200, 0):
    corpus, titles, golds = {}, {}, []
    for a in raw:
        cs = chunk_text(a['title'] + ' ' + a['body'], 256, overlap)
        for j, c in enumerate(cs):
            cid = f"{a['url']}#{j}"
            corpus[cid] = c
            titles[cid] = a['title']
    queries = {str(i): x['query'] for i, x in ev_items}
    golds = [[e['fact'] for e in x['evidence_list']] for _, x in ev_items]
    print(f"\n[overlap={overlap}] {len(corpus)} chunks from {len(raw)} articles", flush=True)

    rankings = {}
    rankings['bm25'] = beir.bm25_rankings(corpus, queries, 10, stem=False, drop_stopwords=False)
    rankings['bm25-engine'] = beir.bm25_rankings(corpus, queries, 10, stem=True, drop_stopwords=True,
                                                 titles=titles, title_boost=2)
    cache = root/'cache'/f'multihop-chunks{overlap}.bge-small-en-v1.5.cls'
    enc = beir.BiEncoder('BAAI/bge-small-en-v1.5')
    vec = beir.cached_embed(enc, corpus, cache)
    rankings['bge-small dense'] = beir.dense_rankings(corpus, queries, enc, vec, top_k=10)
    rankings['hybrid (ours)'] = beir.multi_channel_fusion(
        [rankings['bge-small dense'], rankings['bm25-engine']], [1.0, 1.0], 10, rrf_k=5)

    for name, ranking in rankings.items():
        ret = [[corpus[d] for d in ranking[str(i)]] for i, _ in ev_items]
        m, rows = official_metrics(ret, golds)
        results[f'{name}@ov{overlap}'] = {**m, '_per_query': rows}
        print(f"  {name:18s} Hits@10={m['Hits@10']:.4f} Hits@4={m['Hits@4']:.4f} "
              f"MAP@10={m['MAP@10']:.4f} MRR@10={m['MRR@10']:.4f}", flush=True)

evalkit.write_json(root/'official.json', results)
print('\nwrote', root/'official.json')
