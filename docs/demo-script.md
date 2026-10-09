# Project Demonstration & Oral Defense Script

**Project ID:** P_105  
**Project Title:** Fake News Detection Using LLMs  
**Candidate:** Bhagya Varshney  
**Repository:** `p105-fake-news-detection-llm-bhagya-varshney`  
**Artifact Path:** `docs/demo-script.md`  

---

## 1. 30-Second Elevator Pitch
> *"Good morning, examiners. My project is **Fake News Detection Using LLMs**. Most machine learning models try to guess whether news is fake by looking only at writing style and vocabulary. But text classifiers can't verify facts, and Large Language Models hallucinate if asked directly. 
> 
> My system solves this by building a multi-signal pipeline: we extract factual claims, calculate classical machine learning and stylometric priors, search for evidence in a knowledge base, and instruct Gemini to reason strictly over that evidence. If no usable evidence exists, the system safely abstains with `unverifiable` rather than guessing."*

---

## 2. 2-Minute Comprehensive Project Explanation
> *"Good morning, panel. I am presenting Project P_105: **Fake News Detection Using LLMs**.
> 
> **The Problem:** Automated fact-checking has a major dilemma. Traditional machine learning models merely detect superficial writing patterns, meaning a well-written lie passes easily. On the other hand, querying Large Language Models directly causes hallucinations, bias, and blindspots regarding recent events.
> 
> **The Architecture:** To solve this, I designed a hybrid, modular architecture. When an article or claim is entered:
> 1. We sanitize the text and extract atomic, checkable factual claims.
> 2. Simultaneously, a classical machine learning model and a stylometric red-flag detector analyze text patterns and sensationalism.
> 3. Then, our semantic retrieval engine searches a reference knowledge base using dense vector embeddings from MiniLM.
> 4. If evidence is retrieved, Google Gemini evaluates whether the evidence supports, refutes, or partially corroborates the claims.
> 5. Finally, a deterministic 8-tier fusion engine synthesizes these signals into an auditable final verdict.
> 
> **What Was Trained:** To be scientifically transparent, only the classical model—a TF-IDF vectorizer paired with multinomial Logistic Regression—was trained specifically in this project, using the benchmark LIAR dataset. It achieved 43.73% accuracy on the official 3-class test split. Gemini and MiniLM were pretrained models used frozen for inference.
> 
> **Evidence Grounding & Safe Abstention:** The core strength of this system is epistemic humility. If checkable claims have no usable provenance-bearing evidence in our knowledge base, Gemini is bypassed, and the system safely abstains with `unverifiable` and a support score of 0.0. It never pretends to know what it cannot prove.
> 
> **Current Limitations:** The main deployment limitation is that our production knowledge base is currently empty to prevent hallucinated citations, meaning arbitrary live news will safely abstain. But all subsystems are fully integrated through a FastAPI backend, an SQLite database, and an interactive Streamlit UI, backed by 514 passing automated tests."*

---

## 3. 5-Minute Live Demonstration Walkthrough Script

### Act 1: Environment & System Health (0:00 - 1:00)
- **Action:** Open terminal and browser windows side-by-side.
- **Spoken Commentary:**
  > *"To demonstrate our software engineering standards, I'll start by showing the backend services. In Terminal 1, our FastAPI backend is running on port 8000. In Terminal 2, our Streamlit frontend is running on port 8501. 
  > 
  > If we visit `http://127.0.0.1:8000/docs`, you can see our automated OpenAPI interactive Swagger documentation. Notice the full CRUD suite: analysis execution, paginated history retrieval, single analysis fetching, feedback recording returning HTTP 201 Created, and cascading deletion.
  > 
  > Now let's switch to the Streamlit web application at `http://localhost:8501`."*

### Act 2: Mode A — Production Safe Abstention Demo (1:00 - 2:30)
- **Action:** In the Streamlit text box, enter a typical real-world factual claim:
  `"NASA discovered liquid water oceans beneath the surface of Pluto in 2026."`
- **Action:** Click **Analyze News / Text**.
- **Spoken Commentary:**
  > *"Here is **Mode A**, demonstrating real-world production behavior with our empty reference knowledge base.
  > 
  > Notice what happened:
  > 1. **Claim Extraction:** The system identified one atomic, checkable claim: 'NASA discovered liquid water oceans beneath the surface of Pluto in 2026.'
  > 2. **Classical ML Signal:** The TF-IDF model computed prior probability distributions across the 3 classes.
  > 3. **Stylometric Red Flags:** The red-flag detector reported 'low' severity—no sensationalism or capitalization spam.
  > 4. **Evidence Retrieval:** The system searched the production knowledge base, but because it holds 0 records, zero matching evidence was found.
  > 5. **Evidence Gate & Reasoner Skip:** Because no usable provenance-bearing evidence was found, the system intentionally skipped the Gemini reasoner to preserve API quota.
  > 6. **Final Verdict:** The fusion engine issued the **Unverifiable** verdict banner with `abstained=True` and a Decision-Support Strength of `0.00`.
  > 
  > The explicit explanation states: *'No usable provenance-bearing evidence was available in the current knowledge base.'* The system refuses to guess."*

### Act 3: Mode B — Controlled Academic Demonstration (2:30 - 4:00)
- **Action:** Explain how controlled evaluation fixtures demonstrate the active reasoning and fusion paths without polluting the production database.
- **Spoken Commentary:**
  > *"Now let's examine **Mode B**, which demonstrates how the pipeline behaves when verified evidence is present.
  > 
  > For academic verification, we implemented a controlled evaluation suite in `scripts/evaluate_final_system.py`. Let's look at the three canonical evidentiary scenarios tested in our 44 controlled fixtures:
  > 
  > 1. **Supported Evidence:** When evidence directly corroborates the claim, Gemini assesses `evidence_support = 'supports'`. The fusion engine issues **Likely Reliable** with a high decision-support score, overriding any pessimistic statistical ML prior.
  > 2. **Contradicted Evidence:** When evidence directly refutes the claim—for instance, an official agency report debunking a rumor—Gemini flags `evidence_support = 'refutes'`. The fusion engine issues **Likely Fake**, demonstrating that evidence always overrides classical ML.
  > 3. **Distorted / Partial Evidence:** When an article reports true statistics but draws an unsupported, sensational conclusion, the red-flag detector triggers and Gemini flags partial support. The system deterministically classifies it as **Misleading**."*

### Act 4: Audit Trail, Feedback & Test Suite (4:00 - 5:00)
- **Action:** In Streamlit, navigate to the **Analysis History** expander and submit a feedback entry.
- **Spoken Commentary:**
  > *"Every analysis is persisted to an SQLite database with full audit metadata. Users can review historical analyses and submit qualitative feedback—for instance, rating the analysis as 'Helpful' or suggesting a corrected verdict. This feedback is stored strictly for human review and never automatically retrains model weights.
  > 
  > Finally, to prove software reliability, our entire repository is validated by an automated test suite. Running `pytest` executes 514 automated tests covering text preprocessing, database migrations, API routes, and 10 formal fusion safety invariants, all passing with zero errors.
  > 
  > Thank you. I am now ready for your questions."*

---

## 4. Summary Table of Demonstration Modes

| Mode | Input Type | Knowledge Base State | Expected Verdict | Primary Educational Takeaway |
| :--- | :--- | :--- | :--- | :--- |
| **Mode A: Production Live** | Arbitrary current news statement | Empty production KB (0 records) | `Unverifiable` (`support_score = 0.0`) | Demonstrates **epistemic safety and abstention** when ungrounded. |
| **Mode B: Controlled Benchmark** | 44 synthetic / reference test fixtures | Curated reference fixture pairs | `Likely Reliable` / `Likely Fake` / `Misleading` | Demonstrates **active fusion decision logic and evidence precedence**. |
