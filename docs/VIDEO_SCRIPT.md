# Demo video script (≤ 5 minutes)

| Time | Show | Say |
|---|---|---|
| 0:00–0:25 | Team on camera | Who we are; the problem: inspection systems fail on bad captures — glare, blur, dim light — and either pass bad parts or reject good ones. |
| 0:25–0:55 | `docs/architecture.png` | OpenCV 5 tools inside an arm64 Graviton Lambda; S3 evidence, DynamoDB review queue, Bedrock planner option, CloudWatch metrics. |
| 0:55–1:20 | `docs/agent_workflow.png` | Perceive → fix the view → detect → sweep → zoom-verify → decide; the guard that blocks unsafe PASS. |
| 1:20–2:10 | Live demo URL: Capture condition **Glare**, Defects **1**, Shoot | Point at step 1 "Ask station to re-shoot" and its reason; toggle *First capture* vs *Findings*; expand OpenCV output; tag reads Reject, action divert to rework. |
| 2:10–2:40 | Capture **Underexposed**, Defects **0** | Exposure fix then PASS — no false reject. |
| 2:40–3:20 | **Upload a photo** (`docs/traces/upload_glare_escalate_input.jpg`) | No station → cannot re-shoot → part held. Enter a name, *Release part*; queue row updates with reviewer. |
| 3:20–4:15 | README results table | 0 of 290 defective parts escaped vs 36 for single-pass; review load 0.6% with a station, 12.2% without; the 4 false rejects and why; synthetic-data limitation stated plainly. |
| 4:15–4:45 | CloudWatch metrics / a trace JSON | Every decision is traceable; EMF metrics by disposition. |
| 4:45–5:00 | Repo | Pinned deps, tests, one-command deploy. |
