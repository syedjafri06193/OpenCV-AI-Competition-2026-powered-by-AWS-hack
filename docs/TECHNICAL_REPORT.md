# InspectAgent — Technical Report

OpenCV AI Competition 2026, powered by AWS · Focus path: **Agentic Vision with OpenCV 5**

## 1. Problem

Visual surface inspection (scratches, pits, cracks on machined or stamped metal) is a standard automation target, but deployed systems are brittle to capture conditions. Reflective parts produce specular glare that saturates the sensor; vibration and part-height variation defocus the frame; lighting drifts dim; long exposures add noise. A fixed single-pass pipeline makes two costly mistakes under these conditions: it **passes defective parts** whose defects the bad frame hid, and it **rejects good parts** when enhancement artifacts look like defects.

Human inspectors handle this by changing how they look: they tilt the part away from the light, step closer, look again. InspectAgent gives a vision system the same option.

## 2. Users

* **Line QA operators** — get a release / rework / hold decision per part, and a review queue for held parts with the evidence needed to decide quickly.
* **Quality engineers** — get a per-step trace for every decision (what the system saw, why it did what it did) for audits and root-cause analysis.
* **Small manufacturers** without budget for a dedicated machine-vision integrator — the system runs serverless and costs nothing when idle.

## 3. Architecture

![Architecture](architecture.png)

* **Compute:** one AWS Lambda container function, **arm64 (Graviton)**, 2 GB, Python 3.12, FastAPI behind Mangum, exposed through a Lambda Function URL. Reserved concurrency 10 bounds cost and abuse on the public demo URL.
* **Storage:** Amazon S3 (private, SSE, public access blocked, 90-day lifecycle) for input and annotated evidence images and full JSON traces; Amazon DynamoDB (on-demand, PITR) for inspection records and the human-review queue.
* **Reasoning (optional):** Amazon Bedrock Converse API with tool use as the LLM planner. Without a model ID configured, the rule planner is used.
* **Observability:** CloudWatch Embedded Metric Format (inspections, latency, agent steps, re-shoots, by disposition), structured log events for human reviews, X-Ray active tracing, and a Lambda-errors alarm.
* **Infrastructure as code:** `deploy/template.yaml` (AWS SAM), validated with `cfn-lint`; `deploy/deploy.sh` builds and deploys.

## 4. OpenCV 5 implementation

All perception runs on OpenCV 5.0.0 (`inspectagent/vision.py`); each function is a tool the agent may call.

| Tool | OpenCV operations | Notes |
|---|---|---|
| `assess_quality` | `Laplacian`, `GaussianBlur`, `medianBlur`, statistics | Sharpness is the median over 64-px tiles of Laplacian variance divided by high-pass variance — invariant to exposure and robust to glare and to the defects themselves. Noise is 1.4826·MAD of the median-filter residual. Glare is the fraction of pixels ≥ 250. |
| `fix_exposure` | `LUT` (adaptive gamma), `createCLAHE` | Gamma chosen to move the mean toward 150. |
| `denoise` | `fastNlMeansDenoising` | |
| `sharpen` | unsharp mask via `GaussianBlur` | |
| `suppress_glare` | illumination field by large `GaussianBlur`, division, `normalize`, `dilate` | Returns a mask of saturated pixels; detections inside it are discarded because the information there is lost. |
| `detect_defects` | `morphologyEx(MORPH_BLACKHAT)`, robust z-score threshold, `MORPH_OPEN/CLOSE`, `connectedComponentsWithStats`, `findContours`, `minAreaRect`, `convexHull` | Black-hat isolates dark structures narrower than a 31-px ellipse from the brushed texture. The threshold adapts to each image's texture (median/MAD). Shape features classify scratch (elongated, solid), pit (compact), crack (low solidity), unknown. |
| `zoom_verify` | crop + `resize(INTER_CUBIC)` ×2 + local `detect_defects` | Active perception: local statistics around a suspect replace whole-image statistics. |
| overlays | `rectangle`, `putText`, `imencode` | Evidence images. |

## 5. Agent design

![Agent workflow](agent_workflow.png)

The loop (`inspectagent/agent.py`) holds a state (current frame, quality before/after, candidates with status `detected | suspect | confirmed | rejected`, budgets, trace). Each iteration the planner receives a compact JSON observation and returns one tool call with a reason; the guard checks it; the tool runs; the observation is appended to the trace.

**OpenCV output changes later decisions in four places:**
1. Quality issues select the remedy. With a station attached, glare and blur trigger `request_recapture` because saturated or defocused pixels can't be restored in software; otherwise software remedies run.
2. The re-shot frame is re-assessed; if the new frame has a different problem (e.g. it came back dark), the agent applies that remedy instead.
3. Low-confidence detections and faint sweep hits become suspects that the agent zooms into.
4. The final disposition depends on confirmed defects, unresolved suspects, residual blur, and how much of the surface glare hid.

**Autonomy and human control.** The agent decides automatically when the evidence is sufficient, and holds the part for a person otherwise (`ESCALATE`): unverified suspects, glare that hid >2% of the surface, or a frame still blurry after sharpening. A held part is not released until a named reviewer chooses *Release part* or *Send to rework* in the UI; the decision, reviewer and timestamp are stored and logged. A part cannot be reviewed twice.

**Guard (applies to both planners).** Unknown tools → policy decision; re-applying a remedy → redirected; re-shoot or zoom beyond budget → refused; `finalize(PASS)` while defects or suspects exist → overridden by the policy; step budget (16) → policy decision. The guard means an LLM planner can make inspection slower or more cautious, but cannot make it less safe. This is covered by tests (`test_guard_overrides_unsafe_pass`, `test_bedrock_planner_contract_and_guard`).

**Failure handling.** Any planner exception (network, throttling, malformed response) falls back to the rule policy for that step; the trace labels the step `rules-fallback` and the result's planner field becomes `bedrock+fallback` (`test_planner_failure_falls_back_to_rules`).

Example traces with input and overlay images: `docs/traces/` — glare → re-shoot → FAIL; dark → exposure fix → PASS; upload with glare → ESCALATE; and a dark part where four faint suspects are zoom-verified (three rejected, one confirmed).

## 6. Data

To avoid licensing and consent issues, all evaluation images are generated (`inspectagent/synth.py`): brushed-metal texture (directional filtered noise, illumination gradient, sensor noise) with 0–2 planted defects (scratch, pit, crack) of random depth including faint ones, then one capture condition: good, blur, underexposed, glare, or noise. Each sample records exact defect boxes. `SimulatedStation` re-renders the same part (same seed, same defects) under a new capture condition — 80% good, 20% another degraded condition — so the agent must check the re-shot frame rather than trust it.

## 7. Evaluation

**Protocol.** Thresholds were tuned on seeds 5000–5149. Reported numbers are on held-out seeds 20000–20499 (500 parts, 290 defective), reproducible with `./eval/run_all.sh`. Baseline: the same `detect_defects` run once on the raw frame. The agent is evaluated with a station (can re-shoot) and in upload-only mode (cannot re-shoot). A defective part counts as *caught* if the system rejects or holds it. Defects match when a predicted box center lies within 12 px of the ground-truth box.

| Metric | Baseline | Agent + station | Agent, upload only |
|---|---|---|---|
| Defective parts escaped as PASS | 36 / 290 | **0 / 290** | **0 / 290** |
| Good parts falsely rejected | 0.0% | 0.0% | 1.9% |
| Held for human review | 0% | 0.6% | 12.2% |
| Accuracy of automatic decisions | 92.8% | 100% | 99.1% |
| Defect recall / precision | 85.5% / 100% | 99.7% / 99.3% | 99.2% / 97.3% |
| Mean latency, local x86 CPU | 19 ms | 67 ms | 66 ms |
| Mean agent steps | — | 4.3 | 4.5 |

Correct-and-safe outcome rate by capture condition (baseline → agent with station): noise 78.7% → 100%, underexposed 90.8% → 100%, glare 98% → 100%, blur 100% → 100%, good 99% → 100%.

**Reading the numbers.** Of the baseline's 36 escapes, 33 are noisy (23) or underexposed (10) frames where faint defects fall under the threshold; 2 are glare and 1 a good frame with a very faint defect. The agent removes all of them by conditioning the frame first and by sweeping and zoom-verifying faint candidates. Without a station, the cost of not being able to re-shoot shows up as review load (12.2%: 36 glare, 22 blur and 3 good-light frames out of 500) rather than as escaped defects — the intended trade-off for a safety-relevant decision.

## 8. Observability, security and responsible use

* Each inspection emits one EMF line; dashboards and alarms can be built on `InspectAgent/*` metrics without extra code. Human reviews are logged as structured events.
* The S3 bucket is private and encrypted; images are served only through the API. Records expire after 90 days.
* The IAM role grants only S3/DynamoDB on the stack's own resources, plus `bedrock:InvokeModel` only when a model is configured.
* Uploads are size-limited (8 MB), decoded with `imdecode` (invalid files return 400), and downscaled to ≤1024 px; image keys are validated.
* The demo URL is unauthenticated so judges can use it. A production deployment should put it behind IAM auth or Cognito, and should not take uploads of people — the system is designed for part surfaces and processes no personal data.
* The system is decision support for QA, not a certification: it is built to hold rather than guess, and the human decision is recorded separately from the agent's.

## 9. Limitations and failure cases

* **Synthetic data only.** The generator and detector were written by the same team, so the evaluation measures behavior on the failure modes we modeled, not on real factory images. Real surfaces (cast textures, oil, machining marks, colored coatings) will need threshold calibration and, likely, a learned detector as one more tool. We did not have a rights-cleared real defect dataset available in the build window.
* **Observed failure mode (upload mode, 1.9% false rejects).** All 4 false rejects in the held-out set (4 of 210 good parts) are blurry uploads: sharpening amplifies brushed-texture streaks into small blobs that zoom verification confirms as `unknown` defects (`eval/results/upload/failures/`). Candidate fixes — not applied, to keep the test set untouched: require a known defect class to confirm, or escalate instead of failing on `unknown` after sharpening.
* `zoom_verify` reads the conditioned frame; it does not yet request a physical close-up from the station.
* The Bedrock planner was tested against a stubbed Converse response (tool schema, parsing, guard, fallback). The reported evaluation uses the rule planner; LLM-planner effectiveness was not separately measured.
* Latency figures are from a local x86 CPU; Lambda arm64 latency should be measured from the deployed stack (CloudWatch `LatencyMs`).
* COOL (Cloud-Optimized OpenCV Library) is not used; the arm64 Lambda runs the standard OpenCV 5 wheel.

## 10. Reproducibility

* Pinned dependencies: `requirements.txt` (runtime), `requirements-dev.txt` (tests).
* `pytest -q` — 14 tests covering quality checks, detection, agent decisions, re-shoot behavior, the guard, planner fallback, the Bedrock contract and the HTTP API including review.
* `./eval/run_all.sh` — regenerates every number in section 7 deterministically.
* `./deploy/deploy.sh` — one-command deployment; prints the demo URL and a health check.
