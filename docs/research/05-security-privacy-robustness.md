# laya-rag: RAG Security, Privacy & Robustness Brief (2024–2026)

**Grounded in this checkout.** `compose.build_messages` splices retrieved text verbatim into the user turn as `[n] path\n<text>`; `tests/test_adversarial.py` already pins that a planted instruction reaches the model. `VectorIndex.search(query_vector, k)` takes no metadata predicate. `Chunk` carries `doc_id, path, index, start_char, end_char` — provenance exists, trust does not. `api.authorise` is one optional shared bearer compared with `!=`. `adjudicate_claims` re-retrieves across the whole index, so a poison that ranks can flip the second verification layer too. No tenant model, no ingest sanitization, no audit log.

Ranking favours cheap, deterministic, offline controls; learned detectors are demoted. Costs assume a single CPU/Metal host (external services flagged).

## 1. Trust-tiered provenance manifest + retrieval-time trust filter
- **Threat/path:** corpus poisoning — an attacker adds or edits any indexed file (self-serve `/v1/index`, vendored doc, git submodule) so it ranks for target queries and injects instructions or false facts ([PoisonedRAG, USENIX Sec 2025](https://arxiv.org/abs/2402.07867); [Greshake et al., 2023](https://arxiv.org/abs/2302.12173)).
- **Evidence:** PoisonedRAG reports **exactly 5 poisoned texts per target question → 90% attack success** in a store of millions of texts, with black-box and white-box variants, and states existing defenses are insufficient; [InjecAgent](https://arxiv.org/abs/2403.02691) and [AgentDojo](https://arxiv.org/abs/2406.13352) show **tool-returned/untrusted** content as an injection channel for tool-integrated agents (they do not establish *retrieved* content as the dominant channel — soften that claim).
- **Cost:** Low, CPU-only. Persist `source, sha256, first_seen, trust ∈ {trusted, untrusted, quarantined}` per document; apply a trust penalty or hard exclude in fusion, over-fetching `k` before filtering.
- **Δrisk (hypothesis):** removes the implicit "indexed == authoritative" rule; strong vs untargeted poisoning, partial if the attacker can edit allowlisted paths.
- **Test/skepticism:** plant the PoisonedRAG set, measure ASR and clean-corpus recall before/after. Trust labels are only as honest as ingest policy — with an open indexing API the correct default is *untrusted*.

## 2. Spotlighting / datamarking of evidence
- **Threat/path:** indirect injection — retrieved text impersonates system instructions or the user turn ("ignore the evidence…") and steers the Ollama composer.
- **Evidence:** [Microsoft spotlighting, 2024](https://arxiv.org/abs/2403.14720) (datamarking/delimiting cut aggregate injection success from >50% to <2%, **but only to 8.0% on Q&A**; delimiting alone merely halves ASR and is "easily subverted" once the prompt leaks; datamarking needs randomized dynamic marker tokens and one-way encodings — base64, never ROT13); [OWASP LLM01, 2025](https://genai.owasp.org/llmrisk/llm01-prompt-injection/) ranks prompt injection as **LLM01, the #1 LLM risk**, with indirect injection as a **subtype of LLM01** (not separately ranked), and states outright that "RAG and fine-tuning … do not fully mitigate prompt injection". Its **Scenario #4 is a poisoned RAG document**.
- **Cost:** Very low. Wrap each passage in a per-request random sentinel, interleave a marker token, and restate that only the final question is trusted.
- **Δrisk:** partial — reduces, never eliminates. Assume an adaptive attacker reads the transform.
- **Test/skepticism:** A/B fixed InjecAgent-style payloads with and without marking; report ASR deltas. [Adaptive attacks defeat many defenses](https://arxiv.org/abs/2510.09023): **>90% ASR against 12 defenses**, most of which had reported near-zero ASR — this is the bar any defense claim must now clear.

## 3. Extend deterministic verification to evidence provenance + injection flags
- **Threat/path:** a poisoned passage the composer faithfully quotes still passes `verify_citations` cleanly — the check proves quotes exist, not that the cited chunk is trustworthy or that the answer resisted injected instructions.
- **Evidence:** the project's own `test_adversarial.py` documents this gap; [CaMeL, 2025](https://arxiv.org/abs/2503.18813) argues for control/data-flow separation over trusting model output.
- **Cost:** Low, deterministic. Add: (a) flag outputs whose imperative/role-marker text matches retrieved chunks; (b) require a quote's `found_in` chunk be `trusted` for a "clean" verdict; (c) surface `injection_suspected` beside the existing report.
- **Δrisk:** a strictly stricter gate; catches only pattern-matchable adoption.
- **Test/skepticism:** unit tests where a quoted poison is flagged and a benign code quote (an "ignore previous" comment) is not — tune on FPR, not TPR.

## 4. Offline adversarial retrieval regression suite
- **Threat/path:** nothing is measured, so no defense above can be falsified or regression-guarded.
- **Evidence:** [SafeRAG, ACL 2025](https://arxiv.org/abs/2501.18636); [RGB, AAAI 2024](https://arxiv.org/abs/2309.01431) negative rejection; [RobustRAG](https://arxiv.org/abs/2405.15556) isolate-then-aggregate defense for **retrieval corruption / injected malicious passages**.
- **Cost:** Low–medium, CPU-only and deterministic under a fixed model. Vendor a small poison/noise/negative-rejection corpus into `eval/`; report ASR, false-refusal, and negative-rejection accuracy in the existing `BENCHMARKS.md` style.
- **Δrisk:** none directly, but it is the falsification backbone for #1–#3 and #6.
- **Test/skepticism:** freeze seeds, report intervals as the project already does; distrust any single-number ASR claim.

## 5. Query-time document ACL / tenant filter with over-fetch
- **Threat/path:** every caller sees every indexed document; a second GUI/API user or compromised client retrieves files they should not, and membership-inference queries confirm a sensitive file exists ([MIA against RAG, 2024](https://arxiv.org/abs/2406.19234)).
- **Evidence:** [pgvector row-level-security pattern](https://www.postgresql.org/docs/current/ddl-rowsecurity.html) is the canonical ACL-at-query design.
- **Cost:** Low on one host. Add `acl`/`tenant` to `Chunk`; give `VectorIndex.search` a predicate; resolve caller identity in an API dependency; over-fetch `recall_k × 3–5` before filtering so top-k recall survives.
- **Δrisk:** large *if* multiple principals exist; near-zero today (one optional token). Do it before multi-user deployment, not after.
- **Test/skepticism:** tenant A must never retrieve B's `doc_id` over 10k random queries; measure recall loss from post-filtering. Post-filtering silently shrinks result sets — assert `len(hits) == k` or fail loudly.

## 6. Ingest-time deterministic anomaly quarantine
- **Threat/path:** zero-width/bidi unicode and encoded blobs hide instructions; secret-laden or oversized files slip past review.
- **Evidence:** deterministic normalization is standard text-hygiene practice; [SafeRAG](https://arxiv.org/abs/2501.18636) covers injected-noise retrieval attacks. (Note: spotlighting's encoding ablations cover **base64/ROT13**, not zero-width/bidi Unicode — do not cite it for the invisible-payload class.)
- **Cost:** Very low. NFKC-normalize, strip/reject zero-width and bidi controls (near-zero FPR), flag long base64/hex runs, cap instruction density. Tag ambiguous prose rather than delete.
- **Δrisk:** small but deterministic; kills invisible-payload attacks outright.
- **Test/skepticism:** benign code/config corpus for FPR (comments saying "ignore"), synthetic invisible-payload corpus for TPR. Never rely on this for visible-text injection.

## 7. Append-only, hash-chained audit log
- **Threat/path:** no trail of which documents were retrieved for which query, so poisoning and exfiltration cannot be reconstructed or attributed.
- **Evidence:** [OWASP LLM Top 10, 2025](https://genai.owasp.org/) logging/monitoring guidance.
- **Cost:** Very low. JSONL per query: timestamp, query hash, retrieved `doc_id`+sha256, prompt hash, answer, verification report, each line hash-chained and rotated locally.
- **Δrisk:** detective, not preventive — but it makes every other control debuggable.
- **Test/skepticism:** replay a poison scenario and confirm the log identifies the injected `doc_id`; verify tamper detection. Logs leak queries themselves — restrict permissions and redact.

## 8. Secret/PII redaction at ingest and prompt assembly
- **Threat/path:** indexing arbitrary local files sweeps secrets (`.env`, keys, tokens) into embeddings, the composer prompt, logs, and answers.
- **Evidence:** [Microsoft Presidio](https://github.com/data-privacy-stack/presidio) is the reference toolkit (repo moved from `microsoft/presidio`; docs at presidio.dataprivacystack.org); regex+checksum detectors cover private keys and API tokens with high precision.
- **Cost:** Low with regex (private keys, bearer/`sk-` tokens, emails, Luhn cards); Presidio adds spaCy downloads (heavier, still CPU, cacheable). Redact at prompt assembly first, ingest second.
- **Δrisk:** reduces verbatim leakage; embeddings may still encode the secret, so also exclude secret-like files from indexing.
- **Test/skepticism:** synthetic secret corpus, measure leak-through into prompt and answer. Generic names/addresses have poor recall — do not advertise "PII-safe".

## 9. Model and dependency supply-chain pinning
- **Threat/path:** a swapped embedding/composer checkpoint (pickle `torch.load`) or unpinned dependency executes code or biases retrieval.
- **Evidence:** [HF pickle scanning](https://huggingface.co/docs/hub/security-pickle); [PickleBall, CCS 2025](https://www.cs.columbia.edu/~junfeng/papers/pickleball-ccs25.pdf) (44.9% of popular HF models still ship pickle; 15% of those fail restrictive loaders; PickleBall loads 79.8% benign and rejects 100% malicious); [stealthy pickle poisoning, USENIX 2026](https://www.usenix.org/system/files/usenixsecurity26-liu-tong.pdf).
- **Cost:** Very low. Load only `safetensors`; pin model revisions and record sha256 in the provenance manifest; commit a lockfile; refuse `.bin`/`.pkl` under `models/`.
- **Δrisk:** closes a code-execution path, not a retrieval path.
- **Test/skepticism:** point config at a `.bin`-only checkpoint and assert hard failure; checksum test on a mutated weight file.

## 10. Optional ML injection detector — signal only, off by default
- **Threat/path:** catches visible instruction payloads a regex misses.
- **Evidence:** [protectai deberta-v3-base-prompt-injection-v2](https://huggingface.co/protectai/deberta-v3-base-prompt-injection-v2); but [InjecGuard, 2024](https://arxiv.org/abs/2410.22770) documents systematic over-defense (false positives on benign text), and [guardrail-evasion work, 2025](https://aclanthology.org/2025.llmsec-1.8/) plus [adaptive attacks](https://arxiv.org/abs/2510.09023) show learned detectors are bypassable. Treat vendor accuracy as an upper bound, not a property.
- **Cost:** Medium — ~184M params (184,423,682 in the HF index, not "~435M"), external HF download, per-passage CPU (**flag external service**). Never on the critical path.
- **Δrisk:** uncertain; a false positive that suppresses legitimate evidence is itself a denial-of-retrieval attack.
- **Test/skepticism:** FPR on benign source comments/docs, then evasions (base64, spacing, translation). Ship only if it beats #6 at acceptable FPR.

## Cross-cutting
- Make bearer comparison constant-time (`hmac.compare_digest`) and log auth failures — trivial, folds into #5/#7.
- **Single-host envelope:** #1–#4 and #6–#9 are cheap, deterministic, offline. #5 has no second principal today. #10 spends the most CPU for the least reliable gain.
- **Architectural limit:** a composer that reads untrusted text is never fully defensible by prompt wording. The durable win is not letting poisoned text become trusted input (#1, #3, #7).
