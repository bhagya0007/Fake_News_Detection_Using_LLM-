# Viva Preparation & Oral Defense Guide

**Project ID:** P_105  
**Project Title:** Fake News Detection Using LLMs  
**Student:** Bhagya Varshney  
**Degree:** B.Tech Computer Science / Engineering  
**Artifact Path:** `docs/viva-guide.md`  

---

## Guide Overview
This guide provides concise, technically accurate spoken answers (2–6 sentences each) for 60 questions categorized into core curriculum subjects and examiner challenge scenarios.

---

## Category 1: Basic Project Understanding

### Q1. What problem does your project solve?
**Spoken Answer:**  
Traditional machine learning classifiers for fake news merely detect vocabulary patterns and sensational writing styles rather than factual veracity. Conversely, asking Large Language Models directly causes hallucinations and reliance on stale training memory. My project solves this by separating text-pattern classification from evidence-based verification, grounding LLM reasoning strictly in retrieved evidence and deterministically abstaining when evidence is unavailable.

### Q2. Why not simply ask Gemini whether news is fake?
**Spoken Answer:**  
Directly prompting Gemini introduces serious risks of hallucination, sycophancy, and temporal cutoff blindspots regarding breaking events. Without explicit evidence grounding, the model may generate plausible-sounding justifications for completely fabricated claims. Our system instead forces Gemini into an evidence-bound evaluator role, restricting its reasoning exclusively to verified reference texts supplied in the prompt.

### Q3. What did you actually train in this project?
**Spoken Answer:**  
In this project, only the classical machine learning baseline—a TF-IDF vectorizer paired with multinomial Logistic Regression—was trained, using the benchmark LIAR dataset. The embedding model, `all-MiniLM-L6-v2`, and the large language model, `gemini-2.5-flash`, were pretrained models used frozen via inference APIs. No weights of Gemini or MiniLM were fine-tuned within this repository.

---

## Category 2: Dataset & Label Mapping

### Q4. Why did you use the LIAR dataset?
**Spoken Answer:**  
LIAR is a standard, peer-reviewed academic benchmark introduced by William Yang Wang at ACL 2017. It contains 12,836 real-world political statements fact-checked by PolitiFact with rich speaker and context metadata. It is widely recognized for avoiding the trivial synthetic artifacts found in unvetted web-scraped datasets.

### Q5. What are the six original LIAR labels?
**Spoken Answer:**  
The original PolitiFact ratings in the LIAR dataset are: `pants-fire`, `false`, `barely-true`, `half-true`, `mostly-true`, and `true`. These represent a spectrum of truthfulness ranging from outright fabrication to complete factual accuracy.

### Q6. How did you convert the six labels to three project classes?
**Spoken Answer:**  
We collapsed the six fine-grained labels into three pragmatic classes: `pants-fire` and `false` mapped to `likely_fake`; `barely-true` and `half-true` mapped to `misleading`; and `mostly-true` and `true` mapped to `likely_reliable`. This 3-class formulation stabilizes training on short statements and directly corresponds to verification triage categories.

---

## Category 3: Machine Learning & Classical Baseline

### Q7. Why is your classical model test accuracy only around 43.7%?
**Spoken Answer:**  
An accuracy of 43.73% on a 3-class problem is significantly better than random guessing (33.3%) and majority-class baseline (35.8%). The LIAR dataset consists of very short statements (average 18 words) stripped of external context, where political half-truths share identical vocabulary with authentic truths. State-of-the-art text-only linear classifiers in published literature achieve comparable 40–45% accuracy on this exact benchmark.

### Q8. Why is 43.7% accuracy not evidence that the entire hybrid project fails?
**Spoken Answer:**  
Because the classical model is never used as an authoritative judge; it serves solely as an initial statistical prior over linguistic features. In our hybrid architecture, the final verdict is dictated by external evidence and LLM reasoning. When evidence is present, it explicitly overrides the classical ML prediction.

### Q9. Why is the classical model included if it is only one signal?
**Spoken Answer:**  
It provides a fast, sub-millisecond offline statistical prior that captures stylistic and topical distributions from historical political statements. In low-resource or offline environments where external APIs are unavailable, it acts as a lightweight heuristic baseline alongside stylometric red-flag analysis.

### Q10. What is TF-IDF?
**Spoken Answer:**  
TF-IDF stands for Term Frequency-Inverse Document Frequency. It evaluates how important a word or n-gram is to a document relative to the entire corpus by multiplying its local frequency by the logarithm of the inverse corpus document frequency. In our pipeline, we extract sublinear unigram and bigram TF-IDF features across a 50,000-term vocabulary.

### Q11. What does Logistic Regression output?
**Spoken Answer:**  
Logistic Regression computes a linear combination of feature weights and maps them through a softmax link function. This produces raw predicted probability estimates distributed across our three target classes: `likely_fake`, `misleading`, and `likely_reliable`.

### Q12. Are the classifier probabilities calibrated truth probabilities?
**Spoken Answer:**  
No, absolutely not. They are uncalibrated softmax scores representing relative model preference over vocabulary distributions. Without empirical calibration techniques like Platt scaling or isotonic regression, a score of 0.45 cannot be interpreted as an objective 45% real-world truth probability.

---

## Category 4: Claim Extraction & Preprocessing

### Q13. What is claim extraction?
**Spoken Answer:**  
Claim extraction is the Natural Language Processing step that analyzes an input text and extracts discrete, checkable factual propositions while filtering out subjective opinions, rhetorical flourish, and speculative remarks. Each extracted claim represents an atomic assertion that can be substantiated or refuted against external evidence.

### Q14. Why extract claims before retrieving evidence?
**Spoken Answer:**  
Retrieving evidence against an entire noisy article causes poor semantic matching because documents contain background narrative and commentary. Decomposing text into atomic claims allows dense vector retrieval to formulate precise, targeted search queries for each factual proposition.

---

## Category 5: Embeddings & Semantic Retrieval

### Q15. What are embeddings?
**Spoken Answer:**  
Embeddings are continuous, dense vector representations of text in a high-dimensional geometric space where semantic meaning is captured by geometric proximity. Sentences with similar underlying meanings are projected to vectors that lie close to one another, regardless of lexical variation or synonymous phrasing.

### Q16. What is cosine similarity?
**Spoken Answer:**  
Cosine similarity measures the cosine of the angle between two non-zero vectors, calculated as the dot product of the vectors divided by the product of their L2 norms. In our retrieval system, it outputs a score between -1.0 and 1.0 (typically 0.0 to 1.0 for normalized text embeddings), representing semantic closeness.

### Q17. Does high cosine similarity mean evidence supports a claim?
**Spoken Answer:**  
No, and this distinction is crucial. High cosine similarity indicates *topical relevance*, not *stance or veracity*. For example, "The moon is made of green cheese" and "The moon is made of rocky basalt" have high semantic similarity because they discuss the exact same topic, but one statement directly refutes the other.

### Q18. Why did you use `all-MiniLM-L6-v2`?
**Spoken Answer:**  
MiniLM is a compact, distilled transformer model with only 22.7 million parameters that generates 384-dimensional embeddings. It delivers 5-fold faster inference and requires minimal RAM compared to large models like BERT, making it ideal for fast on-device semantic search on commodity laptops.

### Q19. What is RAG (Retrieval-Augmented Generation)?
**Spoken Answer:**  
Retrieval-Augmented Generation is an architecture where an information retrieval component fetches external reference documents, which are subsequently passed into a generative LLM's context window to ground its generated response in factual source material.

### Q20. Is this project exactly RAG?
**Spoken Answer:**  
It shares the retrieve-then-read paradigm of RAG, but differs in three fundamental architectural ways: first, it extracts atomic checkable claims before retrieval; second, it enforces a strict similarity threshold gate that aborts generation if evidence is inadequate; and third, it funnels the LLM's structured assessment into an auditable deterministic fusion layer rather than generating free-form unstructured text.

---

## Category 6: LLM Reasoning & Prompt Constraints

### Q21. Why use an LLM after retrieval?
**Spoken Answer:**  
While vector retrieval finds relevant candidate passages, it cannot determine whether the passage corroborates, contradicts, or only partially addresses the claim. The LLM acts as a natural language inference (NLI) reasoner, reading the claim and passage together to categorize evidentiary stance and detect subtle omissions.

### Q22. Why shouldn't Gemini use its pretrained memory as verified evidence?
**Spoken Answer:**  
Pretrained parametric memory cannot be independently audited, can suffer from hallucinations, and cannot verify events occurring after its pretraining cutoff. Forcing Gemini to rely exclusively on explicit retrieved evidence ensures that every judgment has an identifiable, provenance-bearing external source.

### Q23. What is hallucination?
**Spoken Answer:**  
In Large Language Models, hallucination refers to the generation of statements that sound fluent, persuasive, and confident, but are factually false, unfaithful to provided context, or entirely ungrounded in reality.

---

## Category 7: Epistemic Abstention & Verdicts

### Q24. What does `unverifiable` mean in your system?
**Spoken Answer:**  
`unverifiable` is an explicit, first-class verdict indicating that the system possesses insufficient provenance-bearing evidence to establish truth or falsehood. It signifies epistemic humility: acknowledging an absence of proof rather than guessing.

### Q25. Why is abstention important in automated fact-checking?
**Spoken Answer:**  
In high-stakes domains, asserting an unverified claim is true or false can cause active disinformation, reputational damage, or public harm. Abstaining when evidence is absent prevents false authority and protects user trust.

### Q26. Why not always return a binary fake/real verdict?
**Spoken Answer:**  
Real-world journalism is rarely binary. Many statements are half-truths, statements taken out of context, or completely unprovable claims. Forcing a binary classification forces models to hallucinate certainty on complex, ambiguous inputs.

### Q27. What happens when there is no evidence available?
**Spoken Answer:**  
The retrieval gate discards candidates scoring below our 0.45 cosine similarity threshold. Seeing zero usable provenance-bearing evidence, the pipeline skips the LLM reasoner entirely to preserve API quota, and the fusion engine deterministically outputs `unverifiable` with `abstained=True` and `support_score=0.0`.

### Q28. Why are synthetic evidence records blocked in production?
**Spoken Answer:**  
Synthetic records are test fixtures crafted solely for unit testing and controlled functional evaluation. If allowed into the production knowledge base, they would allow artificial test assertions to leak into live queries, creating a severe data integrity violation.

---

## Category 8: Stylometric Red Flags

### Q29. What are red flags in your system?
**Spoken Answer:**  
Red flags are deterministic stylometric indicators that detect manipulative presentation patterns, including excessive capitalization, punctuation spam, urgent sensational keywords, and extreme emotional polarity.

### Q30. Can sensational language prove that news is fake?
**Spoken Answer:**  
No. Sensational language indicates emotional writing or clickbait presentation, but accurate breaking news can also be reported sensationalistically. In our fusion hierarchy, red flags can downgrade a verdict or cap confidence, but cannot convict a claim as `likely_fake` on their own.

---

## Category 9: Deterministic Fusion Engine

### Q31. What is fusion in this architecture?
**Spoken Answer:**  
Fusion is the deterministic synthesis layer that takes structured inputs from classical ML, red flags, and LLM evidence assessment, and evaluates an 8-tier hierarchical rule tree to output the final verdict and support score. It contains zero machine learning inference and operates in under 0.1 milliseconds.

### Q32. Why didn't you use majority voting for fusion?
**Spoken Answer:**  
Majority voting treats all signals as equally authoritative. If two pattern classifiers guess "fake" based on sensational words, but authoritative evidence proves the claim is reliable, majority voting would produce an erroneous decision. Factual verification requires evidence precedence, not statistical democracy.

### Q33. Why does evidence override machine learning?
**Spoken Answer:**  
Because external factual evidence reflects verifiable reality, whereas statistical machine learning reflects mere linguistic correlations from a historical training set. An unusual truth should never be rejected simply because its writing style resembles historical fake news.

### Q34. What is `support_score`?
**Spoken Answer:**  
`support_score` is an internal, uncalibrated heuristic measure ranging from 0.0 to 1.0 that quantifies the degree of concordance among independent subsystems and the strength of evidence backing the verdict.

### Q35. Why isn't `support_score` a probability?
**Spoken Answer:**  
Because it is not derived from a calibrated statistical distribution or Bayesian posterior. Presenting it as a probability would mislead users into believing it represents a mathematical certainty of real-world truth.

### Q36. What happens if the Gemini API quota is exceeded?
**Spoken Answer:**  
The Gemini adapter catches HTTP 429 quota exhaustion and network exceptions, logging a warning and falling back gracefully to deterministic fallback logic without crashing the FastAPI server.

---

## Category 10: Backend, Database & Frontend Engineering

### Q37. Why did you use SQLite?
**Spoken Answer:**  
SQLite is built into the Python standard library, requires zero external server setup, supports ACID transactions, and handles relational schemas with foreign key cascades seamlessly for local prototype persistence.

### Q38. Why did you use FastAPI?
**Spoken Answer:**  
FastAPI provides asynchronous high-performance request handling, automatic OpenAPI/Swagger documentation generation, strict Pydantic contract validation, and clean dependency injection.

### Q39. Why did you use Streamlit?
**Spoken Answer:**  
Streamlit enables rapid, pythonic construction of interactive data science web dashboards. By decoupling it from the backend and communicating via HTTP REST calls, we maintain clean separation of concerns.

### Q40. How is user feedback used in this system?
**Spoken Answer:**  
User feedback is stored in the SQLite `feedback` table for offline auditing and manual error analysis. It is strictly never used for automatic online model retraining, preventing model corruption and data poisoning attacks.

### Q41. How did you prevent SQL injection?
**Spoken Answer:**  
Every database interaction in `AnalysisRepository` utilizes parameterized SQL queries (`?` parameter binding) exclusively. No dynamic SQL strings are constructed through concatenation or f-strings.

### Q42. How do you prevent API key leakage?
**Spoken Answer:**  
API keys are loaded via `python-dotenv` from a local `.env` file that is strictly excluded in `.gitignore`. API keys are never returned in REST payloads, logged to files, or committed to Git.

---

## Category 11: Evaluation Results & Software Quality

### Q43. What are your classical model's real test metrics?
**Spoken Answer:**  
Evaluated on the official frozen LIAR test split of 1,283 statements, the classical Logistic Regression model achieves an accuracy of 43.73%, a Macro F1 score of 0.4345, and a Weighted F1 score of 0.4362.

### Q44. What does the 100% controlled hybrid benchmark score mean?
**Spoken Answer:**  
The 100% score (44 out of 44 cases) measures functional rule adherence of our deterministic fusion decision tree against controlled synthetic test fixtures. It proves that the software logic correctly executes its defined policies across all design scenarios.

### Q45. Why is the controlled benchmark NOT 100% real-world accuracy?
**Spoken Answer:**  
Because it was measured on curated synthetic cases designed to verify edge cases in software logic, not on unseen real-world news articles. Claiming it as real-world accuracy would be scientifically dishonest.

---

## Category 12: Limitations & Future Scope

### Q46. What is the biggest limitation of the current prototype?
**Spoken Answer:**  
The biggest deployment limitation is that the production evidence knowledge base currently contains zero records. Consequently, the system will safely abstain as `unverifiable` on almost all arbitrary real-world news claims.

### Q47. Could this system verify arbitrary current news today?
**Spoken Answer:**  
No, not reliably today. Without a populated, authoritative evidence knowledge base or a trusted search API, the system will recognize checkable claims but safely abstain due to a lack of evidence.

### Q48. How could you improve evidence coverage?
**Spoken Answer:**  
By connecting the evidence retrieval engine to trusted journalistic fact-checking databases, such as the Google Fact Check Tools API, ClaimReview markup feeds, or curated Wikipedia article dumps.

### Q49. What would adding live trusted evidence retrieval change?
**Spoken Answer:**  
It would enable the system to verify current claims in real-time, allowing the evidence gate to pass and activating Gemini's evidence-bound reasoning on breaking news stories.

### Q50. What is your future scope for this project?
**Spoken Answer:**  
Future work includes integrating live fact-checking APIs, replacing prompt-based LLM stance classification with dedicated local cross-encoder models, calibrating confidence scores via isotonic regression, and expanding to multilingual claim verification.

---

## Category 13: 10 Examiner Challenge Questions

### C1. "Your classical model accuracy is only 43.7%. Why shouldn't we fail a project with such low ML accuracy?"
**Spoken Answer:**  
Because 43.7% is evaluated on a difficult 3-class benchmark of short political claims where random guessing is 33% and published state-of-the-art linear baselines achieve 40–45%. More importantly, this project is not a classical ML classifier; it is a hybrid verification architecture where the ML model serves solely as a secondary statistical prior, and the primary verification power comes from evidence retrieval and LLM reasoning.

### C2. "If Gemini does the heavy lifting in reasoning, did you just write a wrapper around an API?"
**Spoken Answer:**  
No. Gemini is only one of seven distinct pipeline components. The project includes custom text preprocessing, claim extraction, training and tuning a classical scikit-learn model, building a dense vector search retriever, designing an 8-tier deterministic fusion engine with formal safety invariants, a FastAPI backend, an SQLite repository, and a Streamlit frontend—backed by 514 automated pytest cases.

### C3. "Isn't abstaining on missing evidence just an easy way to avoid getting wrong answers?"
**Spoken Answer:**  
In high-stakes AI safety, knowing when you do not know is a critical engineering requirement, not an evasion. A system that guesses without evidence is irresponsible. In fact-checking, abstention reflects epistemic integrity: refusing to manufacture false certainty when verification data is absent.

### C4. "Why did you use TF-IDF and Logistic Regression instead of fine-tuning BERT or RoBERTa?"
**Spoken Answer:**  
First, our target environment was commodity laptop hardware with 8 GB RAM, where fine-tuning and hosting 500MB+ transformer models imposes heavy memory overhead. Second, empirical literature shows transformer classifiers on LIAR overfit to lexical cues just like linear models. Logistic Regression provides a transparent, sub-millisecond prior that runs seamlessly on CPU.

### C5. "Why not use an open-source LLM like Llama-3 instead of commercial Gemini?"
**Spoken Answer:**  
Running quantized LLaMA-3 locally requires at least 6–8 GB of dedicated VRAM, exceeding our target hardware constraints. However, our backend implements a provider-neutral adapter interface (`LLMProvider`), allowing LLaMA, Mistral, or local Ollama instances to be swapped in by simply writing a new adapter class without changing any application code.

### C6. "How do you know Gemini didn't memorize the LIAR dataset during its pretraining?"
**Spoken Answer:**  
We deliberately isolate Gemini from the LIAR dataset by not using Gemini for classical benchmark evaluation. Furthermore, during live inference, Gemini is strictly quarantined: it only evaluates the input text against explicitly retrieved `<evidence>` snippets, and our prompts command it to ignore parametric memory.

### C7. "Why did you build a custom fusion engine instead of training an ensemble model like XGBoost?"
**Spoken Answer:**  
Trained ensemble models are black boxes that can violate fundamental safety constraints, such as predicting 'fake' when authoritative evidence proves 'reliable'. A deterministic rule-based fusion engine guarantees 100% auditable adherence to formal safety invariants that can be mathematically verified.

### C8. "Can an attacker bypass your red-flag detector by writing clean, scholarly fake news?"
**Spoken Answer:**  
Yes, and that is precisely why red-flag detection is only one signal in our multi-layer defense. While a polished lie will bypass red-flag filters, it cannot bypass the evidence retrieval and LLM reasoning layers, which evaluate factual propositions against external reference truth.

### C9. "Why didn't you scrape Google or Bing for live news verification?"
**Spoken Answer:**  
Automated web scraping introduces severe non-determinism, anti-bot blocking, copyright issues, and the risk of ingesting unverified blog posts or circular misinformation into the evidence context. For an academic prototype, a controlled, curated knowledge base ensures rigorous scientific reproducibility.

### C10. "If your production evidence KB has 0 records, is this system actually usable right now?"
**Spoken Answer:**  
In Version 1, the knowledge base was maintained empty as a controlled baseline to prove safe abstention. In Version 2, we integrated live real-world news retrieval using public GDELT endpoints, so the system actively discovers live candidate evidence from global publishers in real time while maintaining conservative corroboration gating.

---

## Category 14: 10 Version 2 Examination Questions (V2.1 – V2.6)

### Q51. What changed between Version 1 and Version 2 of this project?
**Spoken Answer:**  
Version 1 was an offline prototype relying on a local knowledge base and demonstrating the safety principles of hybrid fusion and epistemic abstention. Version 2 upgrades the system into a real-world evidence-assisted verifier by adding live GDELT news discovery (without extra API keys), switching to Groq (`openai/gpt-oss-20b`) for low-latency structured reasoning, introducing near-duplicate syndication clustering, enforcing 2-domain 2-cluster corroboration gating, harmonizing the LIAR and AVeriTeC datasets, and training an upgraded V2 classifier.

### Q52. Why did you switch to Groq as the sole active LLM provider in Version 2?
**Spoken Answer:**  
Groq provides ultra-fast inference on their LPU architecture (sub-800ms response times) paired with strict Pydantic JSON Schema enforcement. By standardizing on `openai/gpt-oss-20b` through our provider-neutral adapter, we eliminated multiple conflicting API keys and ensured zero hidden chain-of-thought, perfectly matching our auditability and latency requirements.

### Q53. How does real-world evidence retrieval work in Version 2 without adding new API keys?
**Spoken Answer:**  
We leveraged the public GDELT 2.0 architecture—specifically the GDELT Context 2.0 API (which provides sentence-level KWIC context snippets) and the GDELT DOC 2.0 API. These endpoints require no secret API keys, preserving our strict security policy where `GROQ_API_KEY` is the single external secret. Candidate articles are then reranked locally using `sentence-transformers/all-MiniLM-L6-v2`.

### Q54. What is the wire syndication problem, and how does your clustering algorithm resolve it?
**Spoken Answer:**  
When a wire service like Reuters or the Associated Press publishes a story, dozens of regional newspapers republish the identical text. A naive system counting distinct domains would treat 5 syndicated reprints as 5 independent confirmations. Our `EvidenceConsensusEngine` groups candidate items using semantic cosine similarity ($\ge 0.82$) and lexical Jaccard similarity ($\ge 0.65$). Syndicated articles merge into a single evidence cluster, ensuring only truly independent reporting satisfies our multi-source threshold.

### Q55. What are the conservative corroboration thresholds in Version 2?
**Spoken Answer:**  
To issue a decisive verdict (`likely_reliable` or `likely_fake`), the consensus engine strictly requires at least 2 independent evidence clusters originating from at least 2 distinct publisher domains. If evidence is supported by only a single source, contains contradictory reporting, or is headline-only, the system deterministically abstains with `unverifiable`.

### Q56. Why are headline-only articles strictly prohibited from producing decisive verdicts?
**Spoken Answer:**  
Clickbait headlines frequently misrepresent or contradict the actual text of an article. Relying on headlines without paragraph context introduces severe vulnerability to false claims. Our policy explicitly excludes `headline_only` items from decisive consensus counts, requiring `context` snippets before evidence can corroborate a claim.

### Q57. Which classical model did you formally select in V2.5, and why?
**Spoken Answer:**  
We formally selected the **V2 TF-IDF + Logistic Regression** model (`v2_tfidf_logistic_regression.joblib`). In our pre-declared training plan, internal Stratified 5-Fold Cross-Validation Macro F1 was the binding metric. V2 TF-IDF scored **0.4786 ± 0.0047**, edging out MiniLM's **0.4770 ± 0.0031**. Under scientific methodology, you must abide by your pre-declared validation rule. However, we documented MiniLM as a strong secondary challenger due to its superior out-of-domain generalization on AVeriTeC (0.4378 vs 0.4048).

### Q58. Why was DistilBERT fine-tuning not completed in V2.5?
**Spoken Answer:**  
Fine-tuning DistilBERT on our combined training pool of over 12,000 examples exceeded the pre-declared 60-minute CPU resource envelope (estimated at over 43,000 CPU seconds). Rather than compromising reproducibility or masking hardware realities, we formally recorded the trial as `TRAINING NOT COMPLETED` due to local hardware bounds, demonstrating transparent research reporting.

### Q59. What does your Track D benchmark (48 cases, 100% Macro F1) prove compared to the LIAR test benchmark (43.3%)?
**Spoken Answer:**  
They measure fundamentally different properties. The LIAR test score (43.3%) measures empirical statistical classification on unseen real-world statements without external knowledge. Track D (100% accuracy and 1.0000 Macro F1 across 48 cases) evaluates the deterministic functional rule compliance and safety invariants of the hybrid fusion and consensus engine across all edge cases (syndication duplication, single clusters, conflicting sources, headline-only exclusions, prompt injections, and API timeouts). It proves that the hybrid engine behaves with 100% mathematical fidelity to its specified safety policy.

### Q60. Why is 'unverifiable' an epistemic state rather than a classifier label?
**Spoken Answer:**  
Veracity is not an intrinsic property of a sentence's vocabulary. When we tested the 35 AVeriTeC claims labeled 'Not Enough Evidence', the classifiers arbitrarily distributed predictions across 'reliable', 'misleading', and 'fake' based purely on word choices. 'Unverifiable' means that current reference evidence is insufficient to prove or disprove the proposition—which is a property of the retrieval and consensus state, not a linguistic pattern. Therefore, 'unverifiable' belongs strictly in the fusion layer, not in the machine learning label space.
