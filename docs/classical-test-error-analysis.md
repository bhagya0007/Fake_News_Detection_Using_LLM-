# Classical Baseline: Official Test Set Error Analysis (Step 7)

> **Dataset:** LIAR Benchmark (ACL 2017) Official Test Split (`data/raw/test.tsv`)  
> **Model Artifact:** `models/classical_best_validation.joblib`  
> **Classifier Architecture:** Word TF-IDF + Logistic Regression ($C=0.5$, `class_weight='balanced'`)  
> **Evaluated Test Rows:** 1,283  
> **Test Status:** Frozen — No hyperparameters or architectures may be modified based on this analysis.

---

## 1. Overview of Test Set Performance & Error Statistics

The frozen classical model achieved the following performance on the official LIAR test split:
- **Test Accuracy:** **43.73%** (561 correct / 1,283 total)
- **Macro Precision:** **0.4339**
- **Macro Recall:** **0.4381**
- **Macro F1:** **0.4345**
- **Weighted F1:** **0.4362**

### Summary Error Statistics

| Error Statistic | Observation / Metric | Context & Comparison with Validation |
| :--- | :--- | :--- |
| **Most Frequent Confusion Pair** | **`misleading` $\rightarrow$ `likely_reliable` (169 instances)** | On validation, `misleading` $\rightarrow$ `likely_fake` was highest (146), followed by `misleading` $\rightarrow$ `likely_reliable` (145). Both splits demonstrate that boundary claims in `misleading` are readily dispersed across both poles. |
| **Second Most Frequent Confusion** | **`misleading` $\rightarrow$ `likely_fake` (132 instances)** | Further reinforces the boundary challenge of intermediate claims. |
| **Third Most Frequent Confusion** | **`likely_reliable` $\rightarrow$ `misleading` (125 instances)** | Reliable claims citing nuanced economic/demographic data are frequently classified as misleading. |
| **Class with Lowest Precision** | **`likely_fake` (38.80%)** | Validation precision was 44.12%. In test, 384 statements were predicted as `likely_fake`, but only 149 were true fakes. |
| **Class with Lowest Recall** | **`misleading` (37.42%)** | Validation recall was 40.00%. The model captured only 180 of 481 true misleading statements. |
| **Class with Lowest F1** | **`misleading` (0.4045)** | Closely followed by `likely_fake` (0.4105). In validation, `misleading` also had the lowest F1 (0.4311). |
| **Most Frequently Predicted Class** | **`likely_reliable` (490 predictions, 38.19%)** | Actual support in test was 460 (35.85%). |
| **Least Frequently Predicted Class** | **`likely_fake` (384 predictions, 29.93%)** | Actual support in test was 342 (26.66%). |

---

## 2. Descriptive Probability Output Diagnostic

For each test prediction, the maximum predicted class probability estimate was analyzed:

| Prediction Group | Sample Count | Mean Max Probability | Median Max Probability | Min Max Probability | Max Max Probability |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Correct Predictions** | 561 | **0.4511** | 0.4373 | 0.3407 | 0.7815 |
| **Incorrect Predictions** | 722 | **0.4325** | 0.4169 | 0.3369 | 0.7438 |

### Methodological Note
> **Important:** This analysis is strictly a descriptive diagnostic of raw model outputs. Class probability estimates returned by Logistic Regression are not assumed to be calibrated in the absence of empirical calibration curves (e.g., reliability diagrams or Brier score evaluations). A predicted probability of 0.45 does not imply a 45% objective real-world truth rate.

---

## 3. Representative Qualitative Error Case Studies

The following real cases from `models/classical_test_predictions.csv` illustrate the primary failure modes and behavioral patterns observed across all 9 possible confusion pairs.

---

### Case 1: `likely_fake` $\rightarrow$ Correctly Predicted
- **ID:** `11685.json`
- **Statement:** *"Wisconsin is on pace to double the number of layoffs this year."*
- **Actual Label:** `likely_fake`
- **Predicted Label:** `likely_fake`
- **Model Probability Estimates:**
  - `prob_likely_fake`: **0.3956**
  - `prob_likely_reliable`: 0.3675
  - `prob_misleading`: 0.2369
- **Qualitative Interpretation:**  
  *Possible explanation:* The model appears to rely on alarming economic phrasing ("double the number of layoffs") that frequently co-occurred with false or exaggerated claims during training. However, the probability margin over `likely_reliable` is narrow (0.3956 vs 0.3675), reflecting modest overall confidence.

---

### Case 2: `likely_fake` $\rightarrow$ Misclassified as `likely_reliable`
- **ID:** `11200.json`
- **Statement:** *"Says 57 percent of federal spending goes to the military and just 1 percent goes to food and agriculture, including food stamps."*
- **Actual Label:** `likely_fake`
- **Predicted Label:** `likely_reliable`
- **Model Probability Estimates:**
  - `prob_likely_fake`: 0.2464
  - `prob_likely_reliable`: **0.4306**
  - `prob_misleading`: 0.3230
- **Qualitative Interpretation:**  
  *Possible explanation:* The statement employs structured budgetary syntax ("percent", "federal spending", "food and agriculture", "food stamps"). In political discourse, precise numerical budgeting citations correlate strongly with factual legislative reporting in linear n-gram models. Without access to external budgetary data, a text-only bag-of-words model cannot detect that the numerical figure (57%) is drastically fabricated.

---

### Case 3: `likely_fake` $\rightarrow$ Misclassified as `misleading`
- **ID:** `3331.json`
- **Statement:** *"Unfortunately we have documented instances where people defecated in the (Statehouse) building."*
- **Actual Label:** `likely_fake`
- **Predicted Label:** `misleading`
- **Model Probability Estimates:**
  - `prob_likely_fake`: 0.3334
  - `prob_likely_reliable`: 0.3065
  - `prob_misleading`: **0.3601**
- **Qualitative Interpretation:**  
  *Possible explanation:* The vocabulary consists of idiosyncratic, non-policy terms ("unfortunately", "defecated", "building") that rarely appeared in training. Consequently, the linear score distribution is nearly uniform (~0.33 across all three classes), with `misleading` receiving a marginal plurality purely through residual baseline bias.

---

### Case 4: `likely_reliable` $\rightarrow$ Correctly Predicted
- **ID:** `7070.json`
- **Statement:** *"Says that Tennessee law requires that schools receive half of proceeds -- $31 million per year -- from a half-cent increase in the Shelby County sales tax."*
- **Actual Label:** `likely_reliable`
- **Predicted Label:** `likely_reliable`
- **Model Probability Estimates:**
  - `prob_likely_fake`: 0.3265
  - `prob_likely_reliable`: **0.4077**
  - `prob_misleading`: 0.2658
- **Qualitative Interpretation:**  
  *Possible explanation:* Formal statutory terminology ("Tennessee law requires", "sales tax", "proceeds") carries positive weights toward `likely_reliable` in the fitted classifier, reflecting typical administrative and legislative verification language.

---

### Case 5: `likely_reliable` $\rightarrow$ Misclassified as `likely_fake`
- **ID:** `11972.json`
- **Statement:** *"Building a wall on the U.S.-Mexico border will take literally years."*
- **Actual Label:** `likely_reliable`
- **Predicted Label:** `likely_fake`
- **Model Probability Estimates:**
  - `prob_likely_fake`: **0.5229**
  - `prob_likely_reliable`: 0.1804
  - `prob_misleading`: 0.2967
- **Qualitative Interpretation:**  
  *Possible explanation:* Polarized political keywords ("wall", "U.S.-Mexico border") and informal hyperbole/intensifiers ("literally") frequently appeared in fabricated campaign talking points within the training corpus. The linear model assigned a 52.3% probability to `likely_fake`, failing to recognize that the factual assertion (infrastructure construction timelines) was verified as true by PolitiFact.

---

### Case 6: `likely_reliable` $\rightarrow$ Misclassified as `misleading`
- **ID:** `4888.json`
- **Statement:** *"Says the unemployment rate for college graduates is 4.4 percent and over 10 percent for noncollege-educated."*
- **Actual Label:** `likely_reliable`
- **Predicted Label:** `misleading`
- **Model Probability Estimates:**
  - `prob_likely_fake`: 0.2456
  - `prob_likely_reliable`: 0.3132
  - `prob_misleading`: **0.4411**
- **Qualitative Interpretation:**  
  *Possible explanation:* Comparative economic statistics ("unemployment rate", "percent") are commonly scrutinized in PolitiFact for selective framing, often landing in the `half-true` or `barely-true` categories (mapped to `misleading`). The model appears to associate comparative demographic rate claims with partial nuance rather than complete veracity.

---

### Case 7: `misleading` $\rightarrow$ Correctly Predicted
- **ID:** `5209.json`
- **Statement:** *"Suzanne Bonamici supports a plan that will cut choice for Medicare Advantage seniors."*
- **Actual Label:** `misleading`
- **Predicted Label:** `misleading`
- **Model Probability Estimates:**
  - `prob_likely_fake`: 0.3927
  - `prob_likely_reliable`: 0.1620
  - `prob_misleading`: **0.4453**
- **Qualitative Interpretation:**  
  *Possible explanation:* Standard political advertisement phrasing ("supports a plan that will cut choice for Medicare Advantage seniors") strongly aligns with typical political spin patterns that receive `half-true` ratings in the benchmark.

---

### Case 8: `misleading` $\rightarrow$ Misclassified as `likely_fake`
- **ID:** `2508.json`
- **Statement:** *"PolitiFact Texas says Congressman Edwards attacks on Bill Flores are false."*
- **Actual Label:** `misleading`
- **Predicted Label:** `likely_fake`
- **Model Probability Estimates:**
  - `prob_likely_fake`: **0.4730**
  - `prob_likely_reliable`: 0.1841
  - `prob_misleading`: 0.3429
- **Qualitative Interpretation:**  
  *Possible explanation:* The explicit presence of negative judgment terms ("attacks", "false") directly triggered large positive weights in the `likely_fake` linear classifier weights, despite the statement itself being an attack advertisement requiring context.

---

### Case 9: `misleading` $\rightarrow$ Misclassified as `likely_reliable`
- **ID:** `9198.json`
- **Statement:** *"A recent Gallup poll found that 72 percent of Americans and 56 percent of Democrats say the biggest threat to our nations security is big government."*
- **Actual Label:** `misleading`
- **Predicted Label:** `likely_reliable`
- **Model Probability Estimates:**
  - `prob_likely_fake`: 0.2549
  - `prob_likely_reliable`: **0.4807**
  - `prob_misleading`: 0.2644
- **Qualitative Interpretation:**  
  *Possible explanation:* Authority citations ("recent Gallup poll found", "percent of Americans") mimic verified empirical assertions. The model lacks the capability to check whether the polling question was accurately summarized or cherry-picked.

---

## 4. Fundamental Limitations of Classical Text-Only Models

The empirical results and failure modes highlight inherent limitations of classical surface-level ML baselines:

1. **Statistical Pattern Matching vs. Fact Verification:**
   The model is purely a surface-level statistical classifier. It evaluates lexical frequencies rather than claim truthfulness. A completely fabricated claim written with professional statistical phrasing ("According to the Census Bureau, 45%...") will consistently be predicted as reliable.
2. **Inability to Retrieve External Evidence:**
   Determining whether construction of a border wall takes years or whether 57% of federal spending goes to defense requires querying authoritative external databases, which bag-of-words models cannot do.
3. **Severe Nuance Confusion:**
   Claims with intermediate veracity (`misleading`) represent 37.5% of the test set and account for the largest proportion of classification errors (301 misclassifications). Nuanced statements share vocabulary equally with true statements and fabrications.

### Architectural Takeaway for Subsequent Steps
These findings directly motivate the hybrid architecture of the overall project:
- **Step 8:** Interpretable stylometric and red-flag signal extraction (e.g., emotional charge, clickbait syntax, hedging).
- **Later Steps:** Structured LLM claim extraction, automated external evidence retrieval, deterministic multi-signal fusion, and principled abstention (`unverifiable`).
