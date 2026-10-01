"""MultiHop-RAG negative rejection: does the answerability gate abstain on null queries?

The benchmark's 301 `null_query` records have no supporting evidence by construction, so a
system that always answers is wrong on all of them. This indexes the shipped corpus with the
engine's own chunking and records `answerable` per query, plus false abstention on an equal
sample of answerable queries.
"""
import os, sys, json, time, argparse, random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
B = ROOT / 'eval' / 'benchmarks' / 'multihop-rag'
os.environ['LAYA_RAG_INDEX_DIR'] = str(B / 'index')
os.environ.setdefault('LAYA_RAG_EMBED_MODEL', 'BAAI/bge-small-en-v1.5')
# Models live in the workspace cache, not ~/.cache, which the sandbox will not write to.
os.environ.setdefault('HF_HOME', str(ROOT / 'eval' / 'beir' / 'hf-cache'))
sys.path.insert(0, str(ROOT / 'src')); sys.path.insert(0, str(ROOT / 'tools'))

from laya_rag.config import EngineConfig          # noqa: E402
from laya_rag.engine import RagEngine             # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--null-limit', type=int, default=None)
ap.add_argument('--pos-limit', type=int, default=None)
args = ap.parse_args()

items = json.load(open(B / 'MultiHopRAG.json'))
raw = json.load(open(B / 'corpus.json'))
txt = B / 'corpus_txt'
txt.mkdir(exist_ok=True)
for a in raw:
    name = a['url'].split('//')[-1].replace('/', '_')[:150] + '.txt'
    (txt / name).write_text(a['title'] + '\n' + a['body'], encoding='utf-8')

engine = RagEngine(EngineConfig())
if not engine.has_index():
    t = time.perf_counter()
    rep = engine.index_paths([txt], rebuild=True)
    print(f'indexed {rep.documents} docs / {rep.chunks} chunks in {time.perf_counter()-t:.0f}s', flush=True)

nulls = [x for x in items if x['question_type'] == 'null_query']
pos = [x for x in items if x['question_type'] != 'null_query']
random.Random(17).shuffle(pos)
if args.null_limit: nulls = nulls[:args.null_limit]
if args.pos_limit: pos = pos[:args.pos_limit]

def run(group, label):
    rows = []
    t = time.perf_counter()
    for n, x in enumerate(group):
        r = engine.query(x['query'])
        rows.append({'q': x['query'], 'answerable': r.answerable, 'p': r.answerability})
        if n == 4:
            print(f'  ~{ (time.perf_counter()-t)/(n+1):.2f}s per query', flush=True)
    abst = sum(1 for r in rows if not r['answerable'])
    print(f'{label}: n={len(rows)} abstained={abst} ({abst/len(rows):.3f}) '
          f'answered={len(rows)-abst} in {time.perf_counter()-t:.0f}s', flush=True)
    return rows

out = {'null': run(nulls, 'null_query (abstention is correct)'),
       'positive': run(pos, 'answerable (abstention is an error)')}
json.dump(out, open(B / 'gate.json', 'w'), indent=1)
print('wrote', B / 'gate.json')
