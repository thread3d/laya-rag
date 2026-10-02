# Findings: the atomic lexical channel does not transfer to BEIR

The MultiHop experiment found atom-level BM25 alone is worse than chunk BM25, but fusing the
two **significantly improved Hits@4** (+0.0475). BEIR is the project's headline retrieval
benchmark, so this asks whether the same channel helps there. It does not.

Reproduce:

```sh
make beir-atom-channel BEIR_ATOM_ARGS="--datasets scifact,nfcorpus,arguana,scidocs,fiqa"
```

## Method

`hybrid` is the shipped dense+BM25 fusion (RRF, equal weights); `hybrid+atom` adds an
atom-level BM25 channel (sentence spans >= 40 chars, mapped back to parent documents) at
equal RRF weight. Document embeddings are the cached `bge-small` ones — no re-embedding —
and nDCG@10 is scored against official qrels with `ignore_identical_ids`.

| dataset | queries | dense | lexical | atom | hybrid | hybrid+atom | delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| scifact | 300 | 0.7200 | 0.6762 | 0.6235 | 0.7320 | 0.7148 | **−0.0172** |
| nfcorpus | 323 | 0.3386 | 0.3244 | 0.2989 | 0.3634 | 0.3572 | −0.0062 |
| arguana | 1406 | 0.5961 | 0.4169 | 0.3834 | 0.5603 | 0.5362 | **−0.0241** |
| scidocs | 1000 | 0.1973 | 0.1497 | 0.1411 | 0.1921 | 0.1880 | −0.0042 |
| fiqa | 648 | 0.3848 | 0.2386 | 0.1738 | 0.3682 | 0.3154 | **−0.0528** |

Paired over queries (bootstrap CI, sign-flip permutation):

| dataset | delta | 95% CI | p |
| --- | --- | --- | --- |
| scifact | −0.0172 | [−0.0334, −0.0018] | 0.0345 |
| nfcorpus | −0.0062 | [−0.0136, +0.0014] | 0.1000 |
| arguana | −0.0241 | [−0.0346, −0.0140] | 0.0005 |
| scidocs | −0.0042 | [−0.0083, +0.0001] | 0.0505 |
| fiqa | −0.0528 | [−0.0677, −0.0395] | 0.0005 |

## Verdict

**Not shipped.** The atom channel lowers nDCG@10 on every dataset, significantly on three of
five, and atom-only retrieval is weaker than chunk BM25 everywhere. The MultiHop Hits@4 gain
was real but **corpus-specific**: BEIR documents are already short abstracts, so splitting
them into sentences fragments the unit BM25 was tuned for and dilutes its length
normalisation; the 256-token news chunks in MultiHop were long enough that an atom view added
breadth. "An atom channel helps" is therefore not a general claim — the honest form is "an
atom channel can help Hits@4 on long-chunk corpora", and it is not worth an index in the
engine.

## Method note

The `hybrid` row here lands a little below the README's fused rows (scifact 0.732 vs 0.741,
arguana 0.560 vs 0.568) because this harness runs the lexical channel with stemming and no
title boost while the README row uses the harness's own configuration. The comparison is
internal and paired on identical documents and queries, so the conclusion is unaffected, but
these numbers should not be quoted as the repo's headline BEIR rows.
