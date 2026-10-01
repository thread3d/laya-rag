import sys, json, time
from pathlib import Path
from collections import defaultdict
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools')); sys.path.insert(0, str(ROOT / 'src'))
import beir, evalkit

root = ROOT / 'eval' / 'benchmarks' / 'multihop-rag'
raw = json.load(open(root/'corpus.json'))
items = json.load(open(root/'MultiHopRAG.json'))

corpus = {a['url']: (a['title'] + ' ' + a['body']).strip() for a in raw}
titles = {a['url']: a['title'] for a in raw}
queries = {str(i): x['query'] for i, x in enumerate(items)}
qrels = {}
qtype = {}
for i, x in enumerate(items):
    qrels[str(i)] = {e['url']: 1 for e in x['evidence_list']}
    qtype[str(i)] = x['question_type']

TOP = 10
rankings = {}
t0 = time.perf_counter()
rankings['bm25'] = beir.bm25_rankings(corpus, queries, TOP, stem=False, drop_stopwords=False)
rankings['bm25-engine'] = beir.bm25_rankings(corpus, queries, TOP, stem=True, drop_stopwords=True,
                                             titles=titles, title_boost=2)
print(f'lexical {time.perf_counter()-t0:.1f}s', flush=True)

enc = beir.BiEncoder('BAAI/bge-small-en-v1.5')
vec = beir.cached_embed(enc, corpus, root/'cache'/'multihop.bge-small-en-v1.5.cls')
rankings['biencoder'] = beir.dense_rankings(corpus, queries, enc, vec, top_k=TOP)
rankings['hybrid'] = beir.multi_channel_fusion(
    [rankings['biencoder'], rankings['bm25-engine']], [1.0, 1.0], TOP, rrf_k=5)
rankings['hybrid-textonly'] = beir.multi_channel_fusion(
    [rankings['biencoder'], rankings['bm25-engine']], [1.0, 1.0], TOP, rrf_k=5)

out = {'dataset': 'MultiHop-RAG', 'corpus': len(corpus), 'queries': len(queries), 'retrievers': {}}
for name, ranking in rankings.items():
    ev = evalkit.evaluate_rankings(ranking, qrels, k_values=(2, 4, 10))
    out['retrievers'][name] = ev['summary'] | {'scored_queries': ev['queries']}
    s = ev['summary']
    print(f"{name:16s} R@2={s.get('Recall@2'):.4f} R@4={s.get('Recall@4'):.4f} "
          f"R@10={s.get('Recall@10'):.4f} MAP@100={s.get('MAP@100'):.4f} MRR@10={s.get('MRR@10'):.4f}", flush=True)

# per-question-type breakdown for the best retriever
best = max(out['retrievers'], key=lambda k: out['retrievers'][k].get('Recall@10', 0))
by_type = defaultdict(lambda: {'rankings': {}, 'qrels': {}})
for qid in qrels:
    t = qtype[qid]
    by_type[t]['rankings'][qid] = rankings[best][qid]
    by_type[t]['qrels'][qid] = qrels[qid]
print(f"\nper-type ({best}):")
for t, d in sorted(by_type.items()):
    ev = evalkit.evaluate_rankings(d['rankings'], d['qrels'], k_values=(2, 4, 10))
    s = ev['summary']
    n = ev['queries']
    print(f"  {t:18s} n={n:5d} R@2={s.get('Recall@2',0):.4f} R@4={s.get('Recall@4',0):.4f} R@10={s.get('Recall@10',0):.4f}")
out['per_type'] = {t: dict(d) for t, d in by_type.items()}
out['best'] = best
evalkit.write_json(root/'retrieval.json', out)
print('\nwrote', root/'retrieval.json')
