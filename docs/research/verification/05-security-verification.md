# Verification memo — `05-security-privacy-robustness.md`

**Method.** Each citation was checked against its canonical primary source (arXiv abstract/HTML, ACL
Anthology, OWASP site API, GitHub API/raw README, official docs) one request at a time. Status per
item: **CONFIRMED**, **CORRECTED** (with the corrected value), or **UNVERIFIABLE**. The brief's own
project-internal claims were spot-checked against this checkout. Line numbers refer to
`05-security-privacy-robustness.md` as read.

Legend for the verdicts below: ✔ = claim in the brief is right as written; ✎ = brief needs an edit.

---

## 1. PoisonedRAG — ✔ CONFIRMED

- **Canonical:** arXiv:2402.07867, *PoisonedRAG: Knowledge Corruption Attacks to Retrieval-Augmented
  Generation of Large Language Models*, Wei Zou, Runpeng Geng, Binghui Wang, Jinyuan Jia.
  v1 12 Feb 2024; v3 13 Aug 2024. Comments field: **"To appear in USENIX Security Symposium 2025."**
  → the brief's "PoisonedRAG, USENIX Sec 2025" is correct.
- **Headline claim:** **90% attack success rate from 5 malicious texts per target question**, injected
  into a knowledge database with **millions of texts**. Attacker picks both target question and target
  answer. Formulated as an optimization problem with **black-box and white-box** solutions. The paper
  evaluates several defenses and finds them **insufficient**.
- **Enrichment:** (i) billed as *the first* knowledge-corruption attack on RAG; (ii) code at
  `github.com/sleeepeer/PoisonedRAG`; (iii) "a few" in the brief is exactly five per question.

## 2. Greshake et al., indirect prompt injection — ✔ CONFIRMED

- **Canonical:** arXiv:2302.12173, *Not what you've signed up for: Compromising Real-World
  LLM-Integrated Applications with Indirect Prompt Injection*, Greshake, Abdelnabi, Mishra, Endres,
  Holz, Fritz. v1 23 Feb 2023, v2 5 May 2023. (The arXiv page states no venue; the brief correctly
  cites it as "Greshake et al., 2023".)
- **Threat model confirmed:** the adversary has **no direct interface** — it "strategically
  inject[ s] prompts into data likely to be retrieved"; LLM-integrated apps "blur the line between
  data and instructions"; processing retrieved prompts can act as **arbitrary code execution**,
  manipulate app functionality, and control how/whether other APIs are called.
- **Enrichment:** (i) demonstrations against **Bing's GPT-4 powered Chat and code-completion engines**
  plus synthetic GPT-4 apps; (ii) taxonomy includes **data theft, worming, information-ecosystem
  contamination**; (iii) the paper states effective mitigations are "currently lacking".

## 3. InjecAgent — ✔ CONFIRMED

- **Canonical:** arXiv:2403.02691, *InjecAgent: Benchmarking Indirect Prompt Injections in
  Tool-Integrated Large Language Model Agents*, Qiusi Zhan, Zhixiang Liang, Zifan Ying, Daniel Kang.
  Comment: **ACL 2024 Findings**.
- **ASR finding confirmed:** **1,054 test cases**, **17 user tools**, **62 attacker tools**; two attack
  intents — **direct harm to users** and **private-data exfiltration**; across **30 LLM agents**,
  ReAct-prompted **GPT-4 is vulnerable 24% of the time**, and an "enhanced setting" that reinforces the
  injected instruction with a **hacking prompt nearly doubles** the ASR.
- **Enrichment:** repo `github.com/uiuc-kang-lab/InjecAgent`.

## 4. AgentDojo — ✔ CONFIRMED (venue now sourced)

- **Canonical:** arXiv:2406.13352, *AgentDojo: A Dynamic Environment to Evaluate Prompt Injection
  Attacks and Defenses for LLM Agents*, Debenedetti, Zhang, Balunović, Beurer-Kellner, Fischer, Tramèr.
- **Venue:** **NeurIPS 2024 Datasets and Benchmarks Track** — confirmed by the proceedings PDF
  (`proceedings.neurips.cc/paper_files/paper/2024/file/97091a5177d8dc64b1da8bf3e1f6fb54-Paper-Datasets_and_Benchmarks_Track.pdf`).
  The arXiv comments field does *not* state the venue, so this was a genuinely under-sourced item.
- **Framing confirmed:** it is **not a static test suite** but an extensible environment; **97 realistic
  tasks** (email, e-banking, travel), **629 security test cases**; attacks and defenses from the
  literature; SOTA LLMs fail many tasks even without attacks, and existing attacks "break some security
  properties but not all". Repo `github.com/ethz-spylab/agentdojo`.

## 5. Spotlighting — ✔ CONFIRMED, and the paper itself supports "not a complete defense"

- **Canonical:** arXiv:2403.14720, *Defending Against Indirect Prompt Injection Attacks With
  Spotlighting*, Keegan Hines, Gary Lopez, Matthew Hall, Federico Zarfati, Yonatan Zunger, Emre
  Kıcıman — **Microsoft**. 20 Mar 2024 (single version).
- **Technique confirmed:** a family of **three** instantiations — **delimiting, datamarking,
  encoding** — each = an input transformation + system-prompt instructions; the goal is to signal
  provenance so the model can separate trusted instructions from untrusted data.
- **Reported reduction confirmed:** abstract "**greater than 50% to below 2%**". Per-task numbers are
  sharper: datamarking GPT-3.5-Turbo summarization **50%→3.10%**, Q&A **8.0%**; GPT-4 Q&A **1.0%**;
  text-davinci-003 **0.00%**; encoding GPT-3.5-Turbo summarization **0.0%**, Q&A **1.8%**; delimiting
  alone only **halves** ASR.
- **"Not a complete defense" is in the primary source**, not just a caveat: delimiting "could be easily
  subverted by an attacker who gains knowledge of our system prompt"; datamarking needs **dynamic /
  randomized marking tokens** (paper suggests U+E000 and k-gram tokens) and randomized interleaving
  because a no-space payload escapes whitespace marking; encoding must be a **one-way** transform
  (base64) — ROT13 is reversible and therefore attacker-controllable; discussion: spotlighting "helps
  to create separation but is **not perfectly secure** against interference", and the recommended real
  fix is out-of-band control/data separation.
- **Enrichment:** experiments used **text-davinci-003, GPT-3.5-Turbo (June 2023), GPT-4 (June 2023)**,
  temperature 1.0, on a **1,000-document synthetic corpus** with a single-keyword ("canary") payload.

## 6. CaMeL — ✔ CONFIRMED

- **Canonical:** arXiv:2503.18813, *Defeating Prompt Injections by Design*, Debenedetti, Shumailov,
  Fan, Hayes, Carlini, Fabian, Kern, Shi, Terzis, Tramèr (Google DeepMind / ETH Zürich). v1 24 Mar
  2025, v2 24 Jun 2025.
- **Approach confirmed:** a protective **system layer around the LLM**; it "explicitly extracts the
  control and data flows from the (trusted) query", so untrusted retrieved data **can never impact the
  program flow**; a notion of **capability** enforces security policies at tool calls to stop private
  data leaving over unauthorized flows.
- **Claims confirmed:** solves **77% of tasks with provable security** vs **84% with an undefended
  system**, in AgentDojo. Code `github.com/google-research/camel-prompt-injection`.
- This is the correct citation for the brief's "control/data-flow separation over trusting model
  output" (line 23).

## 7. SafeRAG — ✔ CONFIRMED (venue now sourced)

- **Canonical:** arXiv:2501.18636, *SafeRAG: Benchmarking Security in Retrieval-Augmented Generation of
  Large Language Model*, Xun Liang et al. (IAAR-Shanghai). v1 28 Jan 2025, v2 23 Feb 2025.
- **Venue:** **ACL 2025 Long Papers** — `aclanthology.org/2025.acl-long.230/`. ✔ (brief line 30).
- **Attack taxonomy confirmed — four tasks:** **silver noise, inter-context conflict, soft ad, white
  Denial-of-Service**. Dataset built "primarily manually"; experiments on **14 representative RAG
  components**; "even the most apparent attack task can easily bypass existing retrievers, filters, or
  advanced LLMs". Code `github.com/IAAR-Shanghai/SafeRAG`.
- The brief's line 44 "SafeRAG covers injected-noise retrieval attacks" maps to **silver noise** —
  fair.

## 8. RGB benchmark — ✔ CONFIRMED

- **Canonical:** arXiv:2309.01431, *Benchmarking Large Language Models in Retrieval-Augmented
  Generation*, Jiawei Chen, Hongyu Lin, Xianpei Han, Le Sun. Comment: **"Accepted to AAAI 2024."**
- **Four abilities confirmed:** noise robustness, negative rejection, information integration,
  counterfactual robustness — each is a **separate testbed**. Corpus is **English and Chinese**;
  **6 representative LLMs** evaluated. Finding: some noise robustness, but significant struggles with
  **negative rejection, information integration, and false information**.
- **Repo confirmed:** `github.com/chen700564/RGB` exists (GitHub API id 686916495, default branch
  **`master`**, Python, 377 stars, created 2023-09-04). Tree listing confirms the **English subset**:
  `data/en.json`, `data/en_fact.json`, `data/en_int.json`, `data/en_refine.json` — alongside
  `data/zh*.json`. (Note: `README.md` is lower-case `readme.md`; the repo homepage is the repo itself.)

## 9. RobustRAG — ✔ CONFIRMED

- **Canonical:** arXiv:2405.15556, *Certifiably Robust RAG against Retrieval Corruption*, Chong Xiang,
  Tong Wu, Zexuan Zhong, David Wagner, Danqi Chen, Prateek Mittal. v1 24 May 2024.
- **Isolate-then-aggregate confirmed:** passages are isolated into **disjoint groups**, an LLM response
  is generated per group, and the responses are **securely aggregated**; instantiations are
  **keyword-based** and **decoding-based** aggregation for unstructured text.
- **Certifiable robustness confirmed:** formal lower bounds on response quality for certain queries,
  even against an **adaptive attacker with full knowledge of the defense** and a **bounded number of
  injected malicious passages**. Evaluated on open-domain QA and free-form long-text generation across
  **3 datasets and 3 LLMs**.
- **✎ Line 30 terminology:** the brief calls these "supply-scenario sets". The word "supply" does not
  appear in the primary abstract; the paper's own framing is "retrieval corruption attacks" /
  "injected malicious passages". Recommend replacing "supply-scenario sets" with "retrieval-corruption
  sets".

## 10. Adaptive attacks — ✔ CONFIRMED

- **Canonical:** arXiv:2510.09023, *The Attacker Moves Second: Stronger Adaptive Attacks Bypass
  Defenses Against LLM Jailbreaks and Prompt Injections*, Milad Nasr, Nicholas Carlini, … Florian
  Tramèr (Google DeepMind / ETH). 10 Oct 2025.
- **What it shows confirmed:** current defenses are evaluated against static attack strings or weak
  optimizers; the authors argue that is flawed and instead **tune and scale gradient descent,
  reinforcement learning, random search, and human-guided exploration**. They **bypass 12 recent
  defenses with ASR above 90% for most**, where **the majority had reported near-zero ASR**. Covers
  **both jailbreaks and prompt injections**.
- The brief's line 19/72 use ("adaptive attacks defeat many defenses") is accurate; note the 12-defense
  count and that the paper is a call for stronger evaluation, not a single new defense.

## 11. Membership inference against RAG — ✔ CONFIRMED

- **Canonical:** arXiv:2406.19234, *Generating Is Believing: Membership Inference Attacks against
  Retrieval-Augmented Generation*, Yuying Li, Gaoyang Liu, Chen Wang, Yang Yang. v1 27 Jun 2024.
- **Confirmed:** the attack is **S²MIA** and infers whether a **given sample is in the RAG's external
  database** using the **semantic similarity** between that sample and the RAG's generated text. It
  outperforms **five existing MIAs** and **escapes three representative defenses**. Threat model: the
  external store "often contain[s] sensitive data such as medical records or personal identities".

## 12. PickleBall — ✔ CONFIRMED (CCS 2025)

- **Canonical paper:** *PickleBall: Secure Deserialization of Pickle-based Machine Learning Models* —
  the supplied Columbia PDF (`cs.columbia.edu/~junfeng/papers/pickleball-ccs25.pdf`) is the CCS'25
  version. Extended report: **arXiv:2508.15987**, comment "**Extended version of paper to be published
  in the proceedings of ACM CCS 2025**".
- **Supply-chain framing confirmed:** model repositories such as the Hugging Face Model Hub let bad
  actors deliver malware through compromised models. Quantified: **44.9% of popular HF models still use
  insecure pickle**, **15% of those cannot be loaded by restrictive loading policies**, and model
  scanners have both false positives and false negatives.
- **Result:** PickleBall **statically analyzes a library's source**, computes a per-library load
  policy, and **dynamically enforces it as a drop-in `pickle` replacement**; it correctly loads
  **79.8% of benign pickle models** and rejects **100% of malicious examples**, while the SOTA loader
  loads **22% fewer benign models** than PickleBall. Threat model: arbitrary function invocation via
  pickle opcodes. Authors: Kellas, Christou, Jiang, Li, Simon, David, Kemerlis, Davis, Yang.

## 13. InjecGuard / over-defense — ✔ CONFIRMED

- **Canonical:** arXiv:2410.22770, *InjecGuard: Benchmarking and Mitigating Over-defense in Prompt
  Injection Guardrail Models*, Hao Li, Xiaogeng Liu. v1 30 Oct 2024, v3 30 Mar 2025.
- **False-positive / over-defense finding confirmed:** prompt-guard models "suffer from over-defense —
  falsely flagging benign inputs as malicious due to **trigger word bias**". The **NotInject** dataset
  has **339 benign samples enriched with trigger words**; SOTA guard models' accuracy drops to
  **"close to random guessing levels (60%)"**.
- **Mitigation:** the **MOF (Mitigating Over-defense for Free)** training strategy; InjecGuard exceeds
  the previous best by **30.8%**. Repo `github.com/leolee99/InjecGuard`.

## 14. OWASP LLM01:2025 Prompt Injection — ✔ CONFIRMED, with one precision fix

- **Exists:** `genai.owasp.org/llmrisk/llm01-prompt-injection/` — title **"LLM01:2025 Prompt
  Injection"**, part of the *OWASP Top 10 for LLM & Generative AI Security*; page published
  2024-04-10, last modified **2025-04-17**.
- **Framing confirmed:** a prompt-injection vulnerability "occurs when user prompts alter the LLM's
  behavior or output in unintended ways"; inputs need not be human-readable; distinguishes **direct**
  from **indirect** injection (indirect = input from external sources "such as websites or files");
  **jailbreaking is a form of prompt injection**, not a separate risk. Crucially for this brief, the
  page says **RAG and fine-tuning "do not fully mitigate prompt injection vulnerabilities"**, and
  "**it is unclear if there are fool-proof methods of prevention**".
- **Mitigations:** seven, including constrain model behavior; define/validate output formats; input and
  output filtering (mentions the **RAG Triad**: context relevance, groundedness, Q/A relevance); least
  privilege; human approval for high-risk actions; **"Segregate and identify external content"**; and
  adversarial testing. **Scenario #4** is exactly this project's threat: "An attacker modifies a
  document in a repository used by a RAG application" and the malicious instructions change the output.
- **✎ Line 16 precision:** OWASP ranks **LLM01 (prompt injection) as the #1 risk**; **indirect
  injection is a subtype within LLM01**, not separately "ranked the top LLM risk". Reword to
  "OWASP LLM01:2025 ranks prompt injection (including indirect injection) the top LLM risk."

## 15. Microsoft Presidio — ✔ CONFIRMED, with a link/ownership correction

- **What it is confirmed:** "An open-source framework for detecting, redacting, masking, and
  anonymizing sensitive data (PII) across text, images, and structured data. Supports NLP, pattern
  matching, and customizable pipelines." Python, **MIT**, ~11.1k stars / 1.3k forks.
- **✎ Ownership/link:** `github.com/microsoft/presidio` now **redirects** to
  **`github.com/data-privacy-stack/presidio`** (GitHub API returns `full_name:
  data-privacy-stack/presidio`; the docs site now lives at **presidio.dataprivacystack.org**, with a
  "Project transition update" page). Containers moved to **`ghcr.io/data-privacy-stack/…`**; the legacy
  `mcr.microsoft.com/presidio-*` images are **no longer updated**. The brief should stop calling it
  "Microsoft Presidio" or at least note the transfer.
- **Dependency weight confirmed:** **Python 3.10–3.13**; text PII needs `presidio-analyzer` +
  `presidio-anonymizer` **plus at least one NLP engine** — spaCy (default), transformers, or stanza.
  The default install runs **`python -m spacy download en_core_web_lg`** (the large model); even the
  transformers engine **still requires a small spaCy model** (`en_core_web_sm`) and loads transformer
  models lazily. So "adds spaCy downloads (heavier, still CPU, cacheable)" is accurate. Optional GPU
  paths exist (CUDA/MPS) but are not required; image redaction additionally needs Tesseract OCR.

## 16. Hugging Face pickle security docs — ✔ CONFIRMED (with nuance)

- **Canonical:** `huggingface.co/docs/hub/security-pickle`, titled **"Pickle Scanning"** (source in
  `huggingface/hub-docs`, `docs/hub/security-pickle.md`).
- **Danger confirmed:** unpickling executes code; the threatening opcodes are **`STACK_GLOBAL`,
  `GLOBAL`, `REDUCE`**; builtins like `eval`/`exec` can be imported. "This is why it is stated in most
  docs using pickle, do not unpickle data from untrusted sources."
- **Recommendation confirmed, with emphasis:** under *Mitigation Strategies*, the doc's first and
  loudest item is "**Don't use pickle**"; then **load from users/organizations you trust and rely on
  signed commits**, and/or load **TF or Flax** checkpoints via `from_tf=True`/`from_flax=True`;
  **safetensors** is listed among alternative serialization formats (described as an experiment).
- **Scanning confirmed:** the Hub scanner runs **ClamAV** plus a **Pickle Import scan**
  (`pickletools.genops`, no code execution) on every pushed pickled file and shows the import list in
  the UI. The doc's own **disclaimer: "this is not 100% foolproof"**; safe/unsafe import lists are
  maintained best-effort. It cites **fickling** (Trail of Bits) for creating/analyzing malicious
  pickles. So "prefer safetensors and scan/avoid pickle" is directionally correct; note that
  "don't use pickle" is stronger, and scan output is advisory, not a guarantee.

---

## Extra citations in the brief that were not in the 16-item list

### A. `aclanthology.org/2025.llmsec-1.8/` (line 72) — ✔ CONFIRMED
"**Bypassing LLM Guardrails: An Empirical Analysis of Evasion Attacks against Prompt Injection and
Jailbreak Detection Systems**", Hackett, Birch, Trawicki, Suri, Garraghan — **First Workshop on LLM
Security (LLMSEC) 2025**, Vienna, Aug 2025, pp. 101–114. Two bypass families (traditional character
injection and algorithmic AML evasion) tested against **six protection systems including Microsoft
Azure Prompt Shield and Meta Prompt Guard**, reaching **up to 100% evasion success** while preserving
adversarial utility; also shows word-importance ranking from offline white-box models raises ASR
against black-box targets. Solid support for "learned detectors are bypassable".

### B. `usenix.org/system/files/usenixsecurity26-liu-tong.pdf` (line 65) — ✔ CONFIRMED
"**The Art of Hide and Seek: Making Pickle-Based Model Supply Chain Poisoning Stealthy Again**", Tong
Liu, Guozhu Meng, Peng Zhou, Zizhuang Deng, Shuaiyin Yao, Kai Chen — **35th USENIX Security Symposium
(USENIX Security '26)**, Baltimore, Aug 2026, pp. 161–180. Findings: **22 pickle-based model-loading
paths across 5 frameworks, 19 missed by existing scanners**; **Exception-Directed Programming (EDP)**
with **9 instances, 7 bypassing all scanners**; **133 exploitable gadgets, ~100% bypass overall and
89% against the best scanner**; acknowledgments + a **$12,000 bug bounty**. This strongly justifies the
brief's supply-chain hardening section and the "scanners are not enough" framing.

### C. `protectai/deberta-v3-base-prompt-injection-v2` (line 73) — ✎ CORRECTED
The brief says "**~435M params**". The Hugging Face model API reports
**`safetensors.parameters = 184,423,682` (~184M)**. Base model is **`microsoft/deberta-v3-base`**
(arch `DebertaV2ForSequenceClassification`), Apache-2.0, and the repo does ship
`model.safetensors` (plus an ONNX export). Fix the size figure; the "~435M" number is roughly the size
class of a large DeBERTa/electra-scale model, not this one.

### D. Line 44 evidence attribution — ✎ WEAK
"(spotlighting's encoding ablations)" is cited as evidence for zero-width/bidi Unicode payloads. The
spotlighting paper does present **encoding** transformations (base64/ROT13/binary) but **no
zero-width/bidi or other Unicode-control ablation study**. Keep the ingest-normalization proposal, but
cite a Unicode/invisible-text source, or drop the attribution.

---

## Project-internal claims in the brief header (spot-checked, not web claims)

All confirmed against this checkout:

| Brief claim | Evidence |
| --- | --- |
| `compose.build_messages` splices retrieved text verbatim as `[n] path\n<text>` | `laya-rag/src/laya_rag/compose.py:146` — builds `f"[{index}] {path}\n{text}"` blocks, joined, into the user turn |
| `tests/test_adversarial.py` pins that a planted instruction reaches the model | `laya-rag/tests/test_adversarial.py` — `test_an_instruction_planted_in_evidence_reaches_the_prompt_verbatim` |
| `VectorIndex.search(query_vector, k)` takes no metadata predicate | `laya-rag/src/laya_rag/index_store.py:183` — `def search(self, query_vector: np.ndarray, k: int = 10)` |
| `Chunk` carries `doc_id, path, index, start_char, end_char` | `laya-rag/src/laya_rag/chunking.py:49` (plus `chunk_id`, `text`) |
| `api.authorise` is one optional shared bearer compared with `!=` | `laya-rag/src/laya_rag/api.py:210` — `if authorization != expected` (non-constant-time) |
| `adjudicate_claims` re-retrieves across the whole index | `laya-rag/src/laya_rag/engine.py:925–926` — re-runs dense+lexical search per claim |

---

## Summary tables

### Corrections (things to edit in the brief)
1. **Line 73** — `deberta-v3-base-prompt-injection-v2` is **~184M params, not ~435M**.
2. **Line 59** — `microsoft/presidio` **redirects to `data-privacy-stack/presidio`**; update the label
   and link (and note `mcr.microsoft.com/presidio-*` images are deprecated in favor of `ghcr.io/...`).
3. **Line 16** — OWASP ranks **LLM01 (prompt injection)** #1; **indirect injection is a subtype**, not
   a separately ranked risk.
4. **Line 30** — "RobustRAG supply-scenario sets": use **retrieval-corruption / injected-passage** sets.
5. **Line 44** — "spotlighting's encoding ablations" does not cover zero-width/bidi Unicode; re-attribute
   the invisible-payload evidence.
6. **Line 9** — InjecAgent and AgentDojo demonstrate **tool-returned / untrusted content** as the
   indirect-injection channel; they do not establish that *retrieved* content is "the dominant"
   channel. Soften to "show untrusted tool/external content — including retrieved content — is a
   primary indirect-injection channel".

### Unverifiable
- **Greshake et al. venue** — no venue appears on the arXiv page (the brief avoids claiming one, so no
  action needed). Everything else in the 16 items resolved to a primary source.
- **RobustRAG's "supply-scenario" terminology** — not present in the primary text (see correction 4).
- No item in the 16 was left fully UNVERIFIABLE.

### Highest-value new facts for the brief
1. **PoisonedRAG: 90% ASR from exactly 5 poisoned texts per target question**, in a millions-of-texts
   store; black-box and white-box variants; authors state existing defenses are insufficient
   (arXiv:2402.07867, USENIX Sec '25).
2. **Spotlighting's own numbers and self-acknowledged limits** — >50%→<2% aggregate, but up to 8.0% on
   Q&A, delimiting only halves ASR, delimiting is trivially subverted once the prompt leaks, and
   datamarking requires randomized tokens (arXiv:2403.14720).
3. **PickleBall's quantified supply-chain surface**: 44.9% of popular HF models still pickle; 15% of
   those fail restrictive loaders; PickleBall loads 79.8% benign / rejects 100% malicious
   (arXiv:2508.15987, CCS '25).
4. **Adaptive-attack evaluation standard**: >90% ASR against **12** defenses that mostly reported
   near-zero ASR — pair any defense claim with an adaptive attack (arXiv:2510.09023).
5. **Presidio's real dependency footprint**: needs a spaCy engine even with transformers, and the
   default install downloads `en_core_web_lg`; project has moved to `data-privacy-stack` — pin the
   source and expect a non-trivial download.
6. (Bonus) **OWASP's own RAG caveat**: "RAG … do[es] not fully mitigate prompt injection" and "it is
   unclear if there are fool-proof methods of prevention" — cite this to justify not treating RAG as a
   security boundary.
