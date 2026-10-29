# V2 Dataset Licensing, Provenance, and Governance Audit

**Project:** Fake News Detection Using LLMs (`P_105`)  
**Student:** Bhagya Varshney  
**Phase:** V2.4 Dataset Research, Acquisition, and Harmonization  
**Audit Date:** October 2026  
**Status:** Canonical Audit Complete — No Models Trained in this Step  

---

## 1. Executive Summary

This document records the comprehensive research, provenance, licensing, and methodological audit of candidate datasets for the V2 classifier training and evaluation. To ensure rigorous scientific reproducibility, copyright adherence, and evaluation integrity, every candidate dataset was audited from its original peer-reviewed publication and primary official repository.

### Key Policy Decisions:
1. **Immutable Raw Data:** Raw source files are stored in `data/v2_raw/` with verified SHA-256 hashes in `data/v2_raw/checksums.json`. Historical baseline data (`data/raw/`) remains completely frozen and untouched.
2. **Held-Out Split Protection:** The frozen LIAR test split (`data/raw/test.tsv`) and the AVeriTeC development split (`data/v2_raw/averitec/dev.json`) are strictly protected. They are **never** included in any training pool.
3. **MultiFC Policy:** Because MultiFC lacks an explicit permissive redistribution license, aggregates proprietary publisher ratings without standardized open licensing, and provides dummy labels in its test split, it is designated as **`research candidate — licensing unresolved`** and is **excluded from the canonical training pool** (`combined_eligible_train_pool.jsonl`).
4. **FEVER Policy:** Designated as **`auxiliary`** because claims were synthetically created by mutating Wikipedia sentences, exhibiting a distinct distribution from real-world journalistic fact-checking claims.
5. **FakeNewsNet Policy:** Excluded from local acquisition due to publisher copyright and social platform API terms prohibiting raw text redistribution.
6. **Zero LLM Relabeling:** Groq was used **zero times** for dataset labeling or mapping. Human and publisher annotations remain authoritative; ambiguous labels are excluded rather than manufactured.

---

## 2. Dataset-by-Dataset Audit Records

### A. LIAR (Baseline Dataset)

| Attribute | Specification |
|---|---|
| **Dataset Name** | LIAR: A Benchmark Dataset for Fake News Detection |
| **Official Source URL** | [https://www.cs.ucsb.edu/~william/data/liar_dataset.zip](https://www.cs.ucsb.edu/~william/data/liar_dataset.zip) / [ACL Anthology](https://aclanthology.org/P17-2067/) |
| **Paper Citation** | Wang, William Yang. "“Liar, Liar Pants on Fire”: A New Benchmark Dataset for Fake News Detection." ACL 2017. |
| **Task Definition** | Multi-class claim veracity classification (6-class PolitiFact scale). |
| **Language** | English (`en`). |
| **Sample Count** | 12,836 total (Train: 10,269; Valid: 1,284; Test: 1,283) — reconciled after V2.4.1 quote-parsing fix. |
| **Label Set** | `pants-fire`, `false`, `barely-true`, `half-true`, `mostly-true`, `true`. |
| **Claim/Article Type** | Naturally occurring political claims made by US politicians, pundits, and public figures. |
| **Evidence Available?** | Historical baseline lacks full web retrieval evidence; provides short journalistic justification summaries. |
| **Metadata Available?** | Yes: speaker, job title, state, party affiliation, historical credit count, context/venue. |
| **Original Splits** | Yes: predefined train, validation, and test splits (chronologically and randomly stratified). |
| **License** | Research / Academic Use (as provided by UCSB NLP Lab under ACL Anthology terms). |
| **Redistribution Restrictions** | Academic redistribution permitted for non-commercial research; must cite original ACL 2017 paper. |
| **Commercial / Non-Commercial** | Non-commercial research only. |
| **Known Limitations** | Heavy political domain bias; short claim texts (median 18 words); US-centric; no real-time web evidence in original schema. |
| **Download Status** | Already available locally in `data/raw/` (frozen submission version). |
| **Include / Exclude Decision** | **INCLUDED (Canonical Baseline & Training Pool)**. |
| **Reason** | Primary empirical benchmark of Project P_105. Train split enters eligible training pool; validation split is held out as an independent evaluation benchmark; test split remains 100% frozen and protected. |

---

### B. AVeriTeC (Primary Real-World Fact-Checking Benchmark)

| Attribute | Specification |
|---|---|
| **Dataset Name** | AVeriTeC: Automated Verification of Textual Claims with Evidence from the Web |
| **Official Source URL** | [https://github.com/MichSchli/AVeriTeC](https://github.com/MichSchli/AVeriTeC) / [NeurIPS 2023 Datasets Track](https://openreview.net/forum?id=fKzSz0oyaI) |
| **Paper Citation** | Schlichtkrull, Michael Sejr, Zhijiang Guo, and Andreas Vlachos. "AVeriTeC: A Dataset for Real-world Claim Verification with Evidence from the Web." NeurIPS 2023. |
| **Task Definition** | Real-world claim verification with web evidence retrieval, question-answering decomposition, and textual justification. |
| **Language** | English (`en`). |
| **Sample Count** | 3,568 available in official repository (Train: 3,068; Dev: 500; Test held out for shared task). |
| **Label Set** | 4 classes: `Supported`, `Refuted`, `Not Enough Evidence`, `Conflicting Evidence/Cherrypicking`. |
| **Claim/Article Type** | Naturally occurring factual claims collected from 50 international fact-checking organizations (spanning 2010–2022). |
| **Evidence Available?** | Yes: rich question-answer evidence pairs, retrieved search results, URLs, and curated textual justifications. |
| **Metadata Available?** | Yes: `speaker`, `claim_date`, `fact_checking_article`, `reporting_source`, `location_ISO_code`, `claim_types`. |
| **Original Splits** | Yes: temporally split into Train (3,068) and Dev (500) to evaluate temporal robustness. |
| **License** | **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)** (verified directly in repository `README.md`). |
| **Redistribution Restrictions** | Attribution required; non-commercial usage; adaptations must be licensed compatibly. |
| **Commercial / Non-Commercial** | Non-commercial only. |
| **Known Limitations** | Relatively modest size (3,568 claims); complex QA structures require claim-text extraction for pure classification. |
| **Download Status** | Downloaded and verified in `data/v2_raw/averitec/` (`train.json`, `dev.json`). |
| **Include / Exclude Decision** | **INCLUDED (Canonical Training Pool)**. |
| **Reason** | High-quality, peer-reviewed real-world claim dataset with explicit CC BY-NC 4.0 license and four cleanly mapped veracity classes. Train split enters eligible training pool; Dev split is held out. |

---

### C. MultiFC (Multi-Domain Fact-Checking Dataset)

| Attribute | Specification |
|---|---|
| **Dataset Name** | MultiFC: A Real-World Multi-Domain Dataset for Evidence-Based Fact Checking of Claims |
| **Official Source URL** | [https://copenlu.github.io/publication/2019_emnlp_augenstein/](https://copenlu.github.io/publication/2019_emnlp_augenstein/) / [ACL Anthology D19-1475](https://aclanthology.org/D19-1475/) |
| **Paper Citation** | Augenstein, Isabelle, et al. "MultiFC: A Real-World Multi-Domain Dataset for Evidence-Based Fact Checking of Claims." EMNLP 2019. |
| **Task Definition** | Veracity prediction across multi-domain claims using retrieved search snippets and metadata. |
| **Language** | English (`en`). |
| **Sample Count** | 34,842 total across 26 fact-checking portals (Train: 27,871; Dev: 3,484; Test: 3,487). |
| **Label Set** | Over 116 publisher-specific label strings (e.g. `fiction!`, `pants on fire!`, `true`, `mixture`, `unproven`, `0`, `facebook scams`). |
| **Claim/Article Type** | Naturally occurring claims across politics, social media, health, science, and urban legends. |
| **Evidence Available?** | Yes: 10 top search engine snippets per claim with rank and URL. |
| **Metadata Available?** | Yes: `claimURL`, `reason`, `categories`, `speaker`, `checker`, `tags`, `article_title`, `publish_date`, `entities`. |
| **Original Splits** | Yes: Train, Dev, Test (CodaLab competition format; official test split contains dummy labels). |
| **License** | **Unstated / Unknown**. (CodaLab archive terms specify competition rules only; HuggingFace dataset card notes `License is currently unknown`). |
| **Redistribution Restrictions** | Terms and redistribution rights for aggregated third-party publisher content are unresolved. |
| **Commercial / Non-Commercial** | Ambiguous. |
| **Known Limitations** | Heavy label heterogeneity (3,590 distinct raw label variants); publisher label semantics differ across checkers; test set labels are masked. |
| **Download Status** | Downloaded to `data/v2_raw/multifc/` for audit and label inventory analysis. |
| **Include / Exclude Decision** | **RESEARCH CANDIDATE — LICENSING UNRESOLVED (EXCLUDED from canonical training pool)**. |
| **Reason** | Strict adherence to project compliance rules: unverified licensing status and label ambiguity prevent safe inclusion in the canonical training corpus. |

---

### D. FEVER (Fact Extraction and VERification — Auxiliary)

| Attribute | Specification |
|---|---|
| **Dataset Name** | FEVER: A Large-scale Dataset for Fact Extraction and VERification |
| **Official Source URL** | [https://fever.ai/](https://fever.ai/) / [NAACL 2018](https://aclanthology.org/N18-1074/) |
| **Paper Citation** | Thorne, James, et al. "FEVER: a large-scale dataset for Fact Extraction and VERification." NAACL-HLT 2018. |
| **Task Definition** | Sentence-level claim verification against Wikipedia evidence pages. |
| **Language** | English (`en`). |
| **Sample Count** | 185,445 claims (Shared Task Dev: 19,998; Train: 145,449; Test: 19,998). |
| **Label Set** | 3 classes: `SUPPORTS`, `REFUTES`, `NOT ENOUGH INFO`. |
| **Claim/Article Type** | Synthetically generated claims created by crowd-workers mutating Wikipedia sentences. |
| **Evidence Available?** | Yes: Wikipedia sentence IDs and page titles. |
| **Metadata Available?** | Minimal (page titles and sentence numbers). |
| **Original Splits** | Yes: Train, Dev, Paper Test. |
| **License** | **CC BY-SA 3.0** (claim text derived from Wikipedia under CC BY-SA); code under Apache-2.0. |
| **Redistribution Restrictions** | ShareAlike clause requires derivative works using raw Wikipedia claim text to be distributed under CC BY-SA. |
| **Commercial / Non-Commercial** | ShareAlike applies. |
| **Known Limitations** | Claims are synthetic crowd-worker rewrites, not naturally occurring deceptive political or viral news; strong lexical artifacts and negation biases. |
| **Download Status** | Downloaded `shared_task_dev.jsonl` to `data/v2_raw/fever/`. |
| **Include / Exclude Decision** | **AUXILIARY ONLY (EXCLUDED from canonical news classifier corpus)**. |
| **Reason** | Wikipedia sentence-mutation distribution differs substantially from real-world journalistic news claims. Reserved strictly for auxiliary stance/evidence pretraining experiments. |

---

### E. FakeNewsNet (Future Candidate — Excluded)

| Attribute | Specification |
|---|---|
| **Dataset Name** | FakeNewsNet: A Data Repository with News Content, Social Context and Dynamic Information |
| **Official Source URL** | [https://github.com/KaiDMML/FakeNewsNet](https://github.com/KaiDMML/FakeNewsNet) |
| **Paper Citation** | Shu, Kai, et al. "FakeNewsNet: A Data Repository with Social Context for Studying Fake News on Social Media." Big Data 2020. |
| **Task Definition** | Binary fake news detection (PolitiFact and GossipCop). |
| **Language** | English (`en`). |
| **Sample Count** | ~23,000 news articles with Twitter social graphs. |
| **Label Set** | Binary (`real`, `fake`). |
| **Claim/Article Type** | Full-length news articles and celebrity gossip stories. |
| **Evidence Available?** | Social engagements (tweets, retweets, replies), no direct fact-check evidentiary documents. |
| **Metadata Available?** | Tweet IDs, user profiles, timeline graphs. |
| **Original Splits** | Predefined splits or benchmark configurations. |
| **License** | Custom Research Agreement. Original news content is NOT redistributed by authors; users must scrape URLs and fetch tweets via Twitter API. |
| **Redistribution Restrictions** | Direct redistribution of full article texts and tweets is prohibited by publisher copyrights and Twitter Developer Terms of Service. |
| **Commercial / Non-Commercial** | Non-commercial research only. |
| **Known Limitations** | Requires live web scraping and Twitter API keys; high URL rot rate (>35% of original articles now 404). |
| **Download Status** | Not downloaded. |
| **Include / Exclude Decision** | **EXCLUDED (Future Research Candidate)**. |
| **Reason** | Violates zero-scraping policy and API key constraints of V2.4. Unverified mirrors on third-party sites cannot be used without copyright provenance. |

---

## 3. Comparative Dataset Matrix

| Dataset | Provenance Verified | Primary License | Real-World Claims? | Web Evidence? | Role in V2 | Canonical Pool Eligible? |
|---|---|---|---|---|---|---|
| **LIAR** | Yes (UCSB / PolitiFact) | Academic Research | Yes (Political) | No (Summaries) | Core Baseline & Train | **YES** (Train/Val only) |
| **AVeriTeC** | Yes (Cambridge / FEVER) | CC BY-NC 4.0 | Yes (Global Facts) | Yes (Web QA) | Core Benchmark & Train | **YES** (Train only) |
| **MultiFC** | Yes (CopeNLU / EMNLP) | Unresolved / Unknown | Yes (Multi-Domain) | Yes (Snippets) | Research Candidate | **NO** (Licensing Unresolved) |
| **FEVER** | Yes (Sheffield / Amazon) | CC BY-SA 3.0 | No (Wikipedia rewrites) | Yes (Wiki Pages) | Auxiliary Pretraining | **NO** (Distribution Shift) |
| **FakeNewsNet**| Yes (ASU DMML) | Redistribution Prohibited| Yes (Articles) | No (Social Graphs) | Excluded | **NO** (Copyright / Scraping) |

---

## 4. Scientific Compliance & Ethical Summary

1. **No Scraping:** All acquired data originates exclusively from author-provided publication packages and primary Git repositories.
2. **Zero Manufacture of Ground Truth:** No artificial labels were generated using LLMs.
3. **Intellectual Property:** Non-commercial datasets (AVeriTeC CC BY-NC 4.0) are strictly ring-fenced for academic research.
4. **Data Isolation:** `data/v2_raw/` is tracked via SHA-256 manifests to guarantee immutability across future training steps.
