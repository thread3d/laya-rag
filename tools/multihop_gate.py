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
# The index directory is fixed before the config is read; --index-dir overrides it at
# parse time below so an alternative chunking can be collected without touching the
# canonical index other MultiHop tools read.
os.environ.setdefault('LAYA_RAG_INDEX_DIR', str(B / 'index'))
os.environ.setdefault('LAYA_RAG_EMBED_MODEL', 'BAAI/bge-small-en-v1.5')
# Models live in the workspace cache, not ~/.cache, which the sandbox will not write to.
os.environ.setdefault('HF_HOME', str(ROOT / 'eval' / 'beir' / 'hf-cache'))
sys.path.insert(0, str(ROOT / 'src')); sys.path.insert(0, str(ROOT / 'tools'))

from laya_rag.answerability import (  # noqa: E402
    FEATURE_NAMES,
    bridge_conditioned_max,
    extract_features,
    passage_text,
    rotated_bridge_texts,
)
from laya_rag.config import EngineConfig          # noqa: E402
from laya_rag.engine import RagEngine             # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--null-limit', type=int, default=None)
ap.add_argument('--pos-limit', type=int, default=None)
ap.add_argument('--out', default=str(B / 'gate.json'),
                help='where to write the per-query report (use a different path to keep '
                     'the canonical gate.json that tools/multihop_recalibrate.py reads)')
ap.add_argument('--index-dir', default=None,
                help='index directory to use or build; defaults to the canonical one')
ap.add_argument('--bridge-feature', action='store_true',
                help='also measure the 13th gate column, the bridge-conditioned maximum s(q, b, c), '
                     'which costs one Laya forward pass per query')
ap.add_argument('--bridge-arm', choices=('real', 'shuffled'), default='real',
                help="'real' conditions on the query's own top-ranked passage, 'shuffled' on "
                     "another query's, so the instruction text is held fixed and only the "
                     'conditioning content is wrong')
ap.add_argument('--bridge-wording', default='bridge',
                help='bridge question wording (see rerank.BRIDGE_RELEVANCE_WORDINGS)')
args = ap.parse_args()
if args.index_dir:
    os.environ['LAYA_RAG_INDEX_DIR'] = str(Path(args.index_dir))

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

# The bridge column needs Laya's decision layer, not the retrieval model: the conditioned
# question s(q, b, c) is a state key and a question, so no second model is trained. Off by
# default because it costs a forward pass per query and no 13-feature gate is trained yet.
reranker = None
if args.bridge_feature:
    from laya_rag.model import LayaModel
    from laya_rag.rerank import LayaReranker

    reranker = LayaReranker(LayaModel(EngineConfig()))
    reranker.model.ensure_loaded()
    print(f'bridge column on: arm={args.bridge_arm} wording={args.bridge_wording}', flush=True)


def top_text(result):
    # The engine's passages arrive as dicts on the JSON-facing path and as ScoredPassage objects
    # in-process, and a dict has no attributes: `getattr(passage, "text", "")` reads every dict as
    # empty, which would make both bridge arms identical. passage_text reads both shapes.
    passages = getattr(result, 'passages', None) or []
    return passage_text(passages[0]) if passages else ''


def run(group, label, answerable_label):
    rows = []
    t = time.perf_counter()
    # Retrieval first, the model column second: the shuffled arm needs *another* query's top
    # passage, which does not exist until every query in the group has been retrieved.
    answered = []
    for n, x in enumerate(group):
        answered.append((x, engine.query(x['query'])))
        if n == 4:
            print(f'  ~{ (time.perf_counter()-t)/(n+1):.2f}s per query', flush=True)
    bridges = None
    if reranker is not None:
        tops = [top_text(result) for _, result in answered]
        bridges = tops if args.bridge_arm == 'real' else rotated_bridge_texts(tops)
    failures = 0
    for index, (x, r) in enumerate(answered):
        value = None
        if reranker is not None:
            try:
                value = bridge_conditioned_max(
                    reranker, x['query'], r.passages,
                    bridge=bridges[index], wording=args.bridge_wording,
                )
            except Exception as exc:               # noqa: BLE001 - a model failure is a missing column
                failures += 1
                if failures == 1:
                    print(f'  bridge column failed on the first query: {exc!r}', flush=True)
        # The gate trainer needs the retrieval signals the engine already produced, not
        # just Laya's scalar: a monotone rescaling of one number cannot separate the
        # classes out of domain (ROC-AUC ~0.51), a gate over several can.
        features = extract_features(x['query'], r.passages, bridge_conditioned=value)
        rows.append({
            'q': x['query'],
            'answerable': r.answerable,
            'p': r.answerability,
            'measured': r.answerability_measured,
            'label': answerable_label,
            'question_type': x.get('question_type'),
            # All thirteen columns, including the evidence-shape features and the
            # bridge-conditioned column, so the trainer and any comparison see one vector.
            'features': dict(zip(FEATURE_NAMES, features.tolist())),
        })
    abst = sum(1 for r in rows if not r['answerable'])
    print(f'{label}: n={len(rows)} abstained={abst} ({abst/len(rows):.3f}) '
          f'answered={len(rows)-abst} in {time.perf_counter()-t:.0f}s', flush=True)
    if reranker is not None:
        nonzero = sum(1 for row in rows if row['features'].get('bridge_conditioned'))
        print(f'  bridge column non-zero on {nonzero}/{len(rows)}, failures={failures}', flush=True)
    return rows


out = {'null': run(nulls, 'null_query (abstention is correct)', 0),
       'positive': run(pos, 'answerable (abstention is an error)', 1)}
if args.bridge_feature:
    # Self-describing, so a gate trained on this file cannot be confused with one trained
    # without the column or on the control arm.
    out['_bridge'] = {
        'column': 'bridge_conditioned',
        'arm': args.bridge_arm,
        'wording': args.bridge_wording,
        'bridge': "the query's top-ranked passage" if args.bridge_arm == 'real'
                  else "another query's top-ranked passage (rotated)",
    }
json.dump(out, open(args.out, 'w'), indent=1)
print('wrote', args.out)
