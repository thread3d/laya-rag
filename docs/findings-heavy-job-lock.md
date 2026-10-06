# One heavy job at a time, enforced

## The incident

Three multi-million-document lexical builds (`tools/beir.py` on FEVER, Climate-FEVER and HotpotQA)
were running **simultaneously**, alongside a retrieval harness on 2WikiMultiHopQA. The host
**crashed**. No work was lost — every job checkpoints, and on restart HotpotQA was at 3,062/7,405,
FEVER at 1,356/6,666 and Climate-FEVER at 941/1,535 — but the crash was avoidable, and the rule
that prevents it existed only in prose (`best-in-the-world/README.md` §7: "one long job at a time").

## The rule, and its enforcement

**Do not run two corpus-scale jobs at once.** **Twenty-six long-running tools** — corpus retrieval
and its BEIR variants, the cross-encoder caches and multi-hop harnesses, the engine harnesses,
fine-tuning/distillation, the ANN and on-disk scale benches, `bright` and `frames_eval` — now take an
exclusive `flock` on `eval/.heavy-job.lock` before doing any work, through
`tools/heavylock.py`:

```sh
.venv/bin/python tools/heavylock.py --status   # or: make heavy-status
```

A second job prints the holder's name, pid and start time and exits `2` **before reading a corpus**.
The kernel releases an `flock` when the holding process dies, so a crashed or killed job leaves no
stale lock — the failure mode this is for. `LAYA_RAG_ALLOW_CONCURRENT=1` is a deliberate override
for running a small read-only probe beside a real job.

## Every long job checkpoints

A crash resumes rather than restarts, keyed to the run's settings (a checkpoint whose settings do not
match is refused, not mixed in):

| job | checkpoint |
| --- | --- |
| `beir.py` lexical rows | `--lexical-checkpoint`, per query |
| `beir.py` dense rows | `--embed-chunk N`, per N documents |
| `twowiki.py`, `musique.py` | `--embed-chunk N` (default 20,000), per N paragraphs |
| `musique.py` cross-encoder | CE cache with a settings check |
| `engine_beir.py`, `engine_multihop.py`, `engine_musique.py` | jobkit manifest, per query |

`make job-status` lists what is in flight. The README's "Running long jobs: one at a time" section
is the operator-facing version of this note.

## What this does not show

- **The lock is advisory and only wired into the tools listed above.** A new heavy tool has to call
  `heavy_job_guard`; `LAYA_RAG_ALLOW_CONCURRENT=1` bypasses it by design.
- **It does not bound one job's resource use.** A single job large enough to exhaust memory still
  can; the lock removes the *combination*, not the ceiling.
- **It is not a scheduler.** There is no queue: the second job fails, and the operator re-issues it
  once the first finishes.
