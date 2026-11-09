# Instamart Customer Intelligence & Support Copilot

## Overview

This repository implements a local, reproducible capstone pipeline over synthetic data. It connects data quality and SQLite analytics (Part A), RFM segmentation and lapse modelling (Part B), privacy-aware support triage (Part C), and policy retrieval plus a tool-using refund agent (Part D). All model-like calls use the supplied offline `mock_llm.py`; no API key or network service is required. The supplied data generator creates fictional records only; none of the generated data represents real Instamart customers or policy.

## Reproduce

Run these commands from the repository root with Python 3.11 or newer:

1. `python -m venv .venv`
2. `source .venv/bin/activate` (Windows: `.venv\\Scripts\\activate`)
3. `python -m pip install --upgrade pip`
4. `pip install -r requirements.txt`
5. `python generate_data.py`
6. `python run_all.py`

`run_all.py` runs Parts A–D in dependency order. Part A prints `ALL CHECKPOINTS PASSED` when its assertions pass. Fixed random seeds are used for data generation, K-means, splitting, and the model grids. The retry pipeline intentionally sleeps for a short jittered delay when the mock raises a transient error.

## Part A

### Cleaning rules

- Customers: trim and title-case city labels; map `male`/`m` and `female`/`f` variants to canonical values and blanks to `unknown`; convert invalid ages outside 10–100 to missing and fill missing ages using the city median; parse signup dates as dates.
- Orders: parse timestamps; remove exact duplicate rows; retain the latest timestamp per `order_id` to resolve earlier pending snapshots; convert amounts to numeric and reconstruct invalid values using `price_inr × quantity × (1 − discount_pct/100)`; replace negative delivery durations with missing; fill missing durations only for completed orders using the completed-order median for the product category. Cancelled orders retain missing delivery time.
- Products: retain the generated product catalogue and numeric INR prices.

### Validation

The pipeline asserts 1,200 customers, 40 products, 12,309 final orders, no pending statuses, no missing amounts, no missing completed-order delivery durations, missing delivery duration for all cancelled orders, six canonical cities, no missing ages, and completed-order revenue of INR 2,998,668.80. Successful validation prints `ALL CHECKPOINTS PASSED`.

Clean CSVs are written to `data/clean/`; the SQLite database is `data/instamart.db` with `customers`, `products`, and `orders` tables. Seven SQL examples covering `WHERE`/`HAVING`, joins, scalar subqueries, left joins, CTEs, ranking windows, and running totals are stored in `sql/queries.sql`.

### Query and distribution notes

Completed-order revenue is INR 2,998,668.80. The order amount distribution is right-skewed: reference mean INR 269.91, median INR 135.00, skewness about 4.61, Q1 INR 64.80, Q3 INR 297.00, upper 1.5×IQR fence INR 645.30, with 897 values above that fence and none below the lower fence. The largest absolute correlation in the reference data is between amount and discount percentage (about −0.095); this is a weak association and does not establish causation.

## Part B

The RFM snapshot is 2025-10-01. The pipeline writes `outputs/customer_segments.csv`, the K-means inertia/silhouette table to `outputs/kmeans_metrics.csv`, PCA variance to `outputs/pca_variance.csv`, the fitted lapse model to `outputs/lapse_model.joblib`, a top-Premium retention list to `outputs/retention_list.csv`, and run metrics to `outputs/lapse_report.json`.

The reference RFM population is 1,175 customers. The lapse cohort contains 1,156 customers, with 443 labelled lapsed; the stratified split is 924 train / 232 test. In the current reproducible run, the random-forest grid selected max depth 8 and minimum leaf size 10; test confusion matrix (actual rows, predicted columns) was `[[111, 32], [32, 57]]`, and lapsed-class precision, recall and F1 were each approximately 0.640. The model is saved and reloaded, and the code asserts that the reloaded model reproduces test predictions exactly.

No order dated on or after 2025-09-01 is used to construct lapse features. Segment labels are not model features. Encoders and scalers are in a scikit-learn pipeline fitted on training rows only. The Premium flag comes from whole-window RFM, including September; therefore the retention list is illustrative and is not a leakage-free evaluation.

## Part C

Tickets are processed in source-file order. Empty/whitespace tickets are skipped, inputs above 400 estimated tokens are flagged, and email, phone and order identifiers are redacted before any model call. The content of the user message sent to the mock is only the redacted ticket text. Model calls retry transient errors up to three times with exponential backoff and jitter. Replies are accepted only when they contain valid JSON with one of the four allowed categories, or when a non-JSON response contains exactly one allowed category as a lowercase word; other replies are sent to human review.

Outputs include `outputs/triage_zero_shot.csv`, `outputs/triage_few_shot.csv`, `outputs/triage_budget_demo.csv`, `outputs/triage_results.csv`, and `outputs/triage_report.json`. In the current run, zero-shot classified-ticket accuracy was 85.7% (126/147) using 7,060 tokens; few-shot accuracy was 100% using 11,116 tokens. Few-shot used about 75.6 tokens per classified ticket versus 48.0 for zero-shot. The added cost is justified when avoiding misrouting fraud/technical tickets is more important than token savings. The 6,000-token budget demonstration exhausted the budget for 23 tickets. Routing is deterministic: guardrail/invalid/budget statuses go to human review; fraud goes to the security team at high priority; otherwise Premium customers go to the premium desk at high priority; all others go to the support queue.

## Part D

Policy files are split into 40-word chunks with 10-word overlap and indexed with `TfidfVectorizer`; retrieval ranks chunks by cosine similarity. TF measures within-document term frequency; IDF downweights terms common across documents. TF-IDF is lexical, so equivalent wording with different terms can retrieve poorly. The evaluation output is `outputs/policy_eval_results.csv`; the current run retrieved an expected source for all 12 answerable questions, matched all 12 expected facts, and returned the required fallback for both unanswerable questions. The retriever creates 15 overlapping chunks.

The refund agent exposes `get_order`, `search_policy`, `calculate_refund`, `issue_refund`, and `escalate_to_human` through a tool map. Refunds apply only to completed orders delivered in more than 45 minutes. Amounts above INR 1,000 are escalated rather than automatically issued. Agent traces are saved in `outputs/agent_runs.jsonl`, and recorded actions are in `outputs/refund_ledger.jsonl`. The tool-loop mock is deterministic and is useful for exercising order lookup, policy retrieval, refund calculation, issuing, escalation, denial, and not-found cases.

### ReAct versus plan-and-execute

ReAct makes a model call after each observation, so it can react to an order-not-found response or a cancelled order before deciding whether to calculate or issue a refund. This typically costs more calls and tokens than asking for a complete plan once, but each tool result can change the next step. Plan-and-execute can be more efficient for predictable happy paths, but a stale plan is riskier when database observations contradict assumptions. For a refund workflow involving money, I would prefer ReAct with strict tool validation and human escalation because correctness and safe stopping matter more than minimizing a few local mock calls.

## Limitations and Next Steps

- The data is synthetic, and the offline mock is not a substitute for evaluation with a real language model or production support traffic.
- The lapse model is evaluated on one temporal split; monitoring, calibration, and time-based validation would be needed before operational use.
- The Premium retention list is illustrative and not a leakage-free evaluation because the Premium flag is based on whole-window RFM, including the lapse-label month.
- Refund actions are simulated in a local JSONL ledger; no payment system is connected. Production use would require authentication, authorization, idempotency, audit controls, and human approval for policy-sensitive cases.
- The repository is a working local implementation of the core pipeline. The full brief also asks for an explicit crash-and-resume demonstration and separate token-limit/timeout demonstrations; those operational demonstrations should be completed and recorded before treating this as a final graded submission.
