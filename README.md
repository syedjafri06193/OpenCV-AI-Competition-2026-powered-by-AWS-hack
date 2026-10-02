# OpenCV AI Competition 2026, powered by AWS

> Build vision systems that see, reason, and act with **OpenCV 5** and **Amazon Web Services (AWS)**. Win money and bragging rights!

- **Devpost page:** https://opencv26.devpost.com/
- **Official site:** https://opencv.org/opencv-ai-competition-2026/
- **Final submission deadline:** **Oct 26, 2026 @ 11:45pm PDT** (Devpost) — the official site says 11:59 p.m. PT; plan for the earlier time
- **Winners announced:** Nov 10, 2026, on the *OpenCV Live!* stream
- **Sponsor:** Amazon Web Services · **Administrator:** The Open Source Vision Foundation (OpenCV)
- **Devpost links:** [Rules](https://opencv26.devpost.com/rules) · [Resources](https://opencv26.devpost.com/resources) · [Updates](https://opencv26.devpost.com/updates) · [Discussions](https://opencv26.devpost.com/forum_topics) · [Project gallery](https://opencv26.devpost.com/project-gallery) · [My projects](https://devpost.com/submit-to/30984-opencv-ai-competition-2026-powered-by-aws/manage/submissions)

---

## Table of Contents
1. [At a Glance](#at-a-glance)
2. [Core Requirements](#core-requirements)
3. [Focus Paths (COOL & Agentic Vision)](#focus-paths)
4. [Suggested Project Areas](#suggested-project-areas)
5. [Schedule](#schedule)
6. [AWS Credits & Grant Proposals](#aws-credits--grant-proposals)
7. [Build Phase](#build-phase)
8. [Final Submission Package](#final-submission-package)
9. [Prizes](#prizes)
10. [Judging](#judging)
11. [Eligibility & Rules](#eligibility--rules)
12. [Resources](#resources)
13. [Contact](#contact)

---

## At a Glance

| | |
|---|---|
| Total value | **$20,250** — $12,000 cash prizes + $8,250 AWS credits |
| AWS grants | 55 selected participants × $150 AWS credits |
| Build window | ~2 months (Aug 26 – Oct 26, 2026) |
| Teams | Individual or team (official site says 1–5 members in one place and "no more than four" in the terms — check the Devpost Official Rules) |
| Languages / hardware | Any, as long as OpenCV 5 + AWS requirements are met |
| Emphasis | Physical AI and Generative AI systems where visual understanding drives decisions, actions, predictions, or interactions |

## Core Requirements

- Register on the official Devpost page.
- Submit the cloud compute grant proposal.
- Use **OpenCV 5** as a substantive runtime component of the application.
- Make **image or video analysis central** to the project, not incidental.
- Run a **meaningful project component on AWS**.
- Submit original work and hold all needed rights/licenses for third-party code, data, models, media, and trademarks.
- Provide judges a **working application** — a judge-accessible web endpoint is preferred; a scheduled live screen-share demo is acceptable.
- Comply with laws, AWS service terms, platform rules, and responsible-use requirements.

## Focus Paths

Optional — combine both, or neither. All entries remain eligible for the Overall Awards.

### Cloud-Optimized Vision with COOL
Use the [Cloud-Optimized OpenCV Library (COOL)](https://opencv.org/COOL/) — available via AWS Marketplace and optimized for **AWS Graviton** — to accelerate vision operations and move to scalable cloud deployment. Strong submissions:
- Show COOL executes the core image/video workload on the **Arm path**.
- Report reproducible measurements (latency, throughput, utilization, cost, or developer productivity) against a baseline.
- Explain any x86/Arm coexistence, container, server, or serverless architecture.
- **Requirement:** COOL must run the claimed core workload on AWS Graviton, or on the Arm component of a documented hybrid architecture.

### Agentic Vision with OpenCV 5
Build a workflow where an agent uses OpenCV 5 tools in a multi-step **perception → decision → action** loop. Image/video results must influence a later plan, tool call, action, or request for human approval.
- A chatbot that only explains a fixed vision result **does not qualify**.
- Examples: autonomous inspection, visual troubleshooting, embodied assistants, active perception, safety monitoring, human-in-the-loop operations.
- Any agent framework or model; OpenCV 5 (and COOL) may be invoked via APIs, tools, or **MCP**.
- Developer-agent workflows qualify only if they iteratively invoke, evaluate, and adjust an OpenCV 5 workload and end in a running image/video workload.
- Using an AI coding assistant to write the entry **does not** count as Agentic Vision.

## Suggested Project Areas

Not limited to these, but especially welcome:

1. **Active Perception** — autonomous inspection with agentic orchestration or MCP
2. **Video Prediction** — physics-informed prediction for process control
3. **Visual SLAM** — multi-agent SLAM with distributed context
4. **Spatial Digital Twins** — real-time twins of physical environments
5. **COOL Pipelines** — server, serverless, container, or hybrid x86/Arm
6. **Developer Agents** — integrate COOL with Codex, Claude Code, Kiro, or MCP systems

Other areas: Healthcare · Safety · Accessibility · Agriculture · Environmental Monitoring · Smart Cities · Education · Retail · Sports Analytics

## Schedule

All deadlines 11:59 p.m. Pacific Time unless stated otherwise.

| Date | Milestone |
|---|---|
| Aug 12, 2026 | Launch; registration and proposals open |
| Aug 17, 2026 | Registration closes (per official site) |
| Aug 18 onward | Rolling proposal review |
| Aug 25, 2026 | AWS credit recipient notices begin |
| Aug 26 – Oct 26, 2026 | Build phase |
| Sep 21 – Oct 2, 2026 | Required midpoint check-ins (grant recipients) |
| Oct 26, 2026 | Final submissions due (Devpost: 11:45pm PDT) |
| Oct 27 – Nov 9, 2026 | Final judging |
| Nov 10, 2026 | Winners announced |

## AWS Credits & Grant Proposals

**Cloud Compute Grants:** up to 55 participants receive **$150 in AWS credits** each ($8,250 total). [Submit proposal](https://www.jotform.com/form/262145877145059). Teams not selected can still compete and win prizes.

**Proposals must include:**
1. Team name
2. Problem statement and intended real-world impact
3. Planned OpenCV 5 image/video analysis
4. Planned AWS architecture and services
5. High-level architecture diagram or technical description
6. Target users / beneficiaries
7. Evaluation method and judge demonstration plan
8. Whether you're pursuing COOL, Agentic Vision, both, or neither
9. Short team bio incl. prior hackathons/competitions

**Grant selection criteria:**

| Criterion | Weight |
|---|---|
| OpenCV 5 & core image/video fit | 20% |
| Technical & AWS feasibility | 20% |
| Innovation | 20% |
| Real-world impact | 20% |
| Team strength & relevant work | 10% |
| Evaluation & demonstration plan | 10% |

Ties: OpenCV 5 fit → technical/AWS feasibility → team strength.

**Free Tier credits:** New AWS customers can get up to $100 in [AWS Free Tier credits](https://aws.amazon.com/free/) on a Free Plan account, plus up to $100 more via eligible activities — subject to [Free Tier terms](https://aws.amazon.com/free/terms/) and [AWS Promotional Credit Terms](https://aws.amazon.com/awscredits/). Credits aren't redeemable for cash.

## Build Phase

- Join the official Slack (`#opencvcomp26`) and share progress with **#OpenCVComp26**.
- Grant recipients must do one **30-minute Zoom check-in** (Sep 21 – Oct 2) to receive the **other 50%** of their grant.
- Maintain active development and give a concise progress update at check-in.

## Final Submission Package

- **Technical report:** problem, users, architecture, OpenCV 5 implementation, AWS deployment, evaluation, limitations, responsible-use considerations.
- **Code repository/archive** accessible to judges (public or private; need not be open source).
- **Pinned dependencies** + clear build, deploy, and test instructions.
- **Architecture diagram** showing OpenCV 5 and AWS components (and COOL/agent components if relevant).
- **Working web endpoint** or arranged live screen-share demo.
- **Video ≤ 5 minutes** (public or unlisted) showing the team, the app working, architecture, and principal results.
- **Evaluation evidence**, including failure cases and limitations.

**Extra for Best Use of COOL:** COOL version and AWS instance/deployment config; reproducible evaluation method, inputs, baselines, results; evidence COOL executes the claimed core workload.

**Extra for Agentic Vision:** agent workflow diagram (perception → decision/orchestration → action); a trace or demo showing OpenCV 5 output changing a later decision, tool call, or action; evaluation of task success, failure handling, observability, and human control.

## Prizes

Up to **$12,000** in cash.

| Award | Prize |
|---|---|
| 🥇 First Place | $5,000 |
| 🥈 Second Place | $3,000 |
| 🥉 Third Place | $2,000 |
| Best Use of COOL (special) | $1,000 |
| Agentic Vision Award (special) | $1,000 |

- Max **one Overall Award** per entry; can also win **one or both** Special Awards.
- Special Awards are scored independently and may go unawarded if no entry qualifies.
- Paid within 30 days of final judging after eligibility/tax docs; team representative receives and distributes payment.
- Prizes are non-transferable; sponsor may substitute a prize of equal or greater value.

## Judging

Judges come from the AI/CV community; AWS may nominate up to two. Each entry gets **≥ 2 independent, conflict-free scores**, averaged. Judges must disclose conflicts and recuse.

### Overall Rubric (100 points)

| Criterion | Weight | Covers |
|---|---|---|
| Technical execution | 30% | Correctness/depth of OpenCV 5 implementation, architecture, reliability, evaluation |
| Innovation | 20% | Originality, thoughtful use of CV/AI |
| Real-world impact | 20% | Problem importance, usefulness, evidence of benefit |
| User experience | 10% | Usability, accessibility, interaction quality |
| Documentation & presentation | 10% | Report, code, architecture, instructions, video |
| Cloud delivery, reproducibility & responsible operation | 10% | AWS deployment quality, repeatability, observability, security, responsible use |

COOL/agentic methods aren't required for Overall Awards; they're credited only insofar as they improve the project.

### Best Use of COOL Rubric

| Criterion | Weight |
|---|---|
| Verified COOL on AWS Graviton (or Arm part of documented hybrid) | 30% |
| Architecture & technical quality | 25% |
| Measured performance, cost, reliability, or productivity value | 20% |
| Innovation | 15% |
| Reproducibility & demonstration | 10% |

### Agentic Vision Award Rubric

| Criterion | Weight |
|---|---|
| Substantive OpenCV 5 + agent integration | 30% |
| Orchestration & appropriate autonomy | 25% |
| Task effectiveness & evaluation | 20% |
| Failure handling, observability, security, human control | 15% |
| UX, documentation & demonstration | 10% |

**Ties:** Overall → Technical Execution, then Real-World Impact, then an extra conflict-free judge. Special awards → first rubric criterion, then second, then an extra judge. Judges' decisions are final.

## Eligibility & Rules

**Eligibility & fair play**
- Must be **13+**; minors need parent/guardian permission.
- Every team member must be eligible; teams designate one representative.
- No manipulation of registration, voting, judging, or traffic via bots, fraud, or hacking. Disclosed agents that are integral to the project are allowed and encouraged.
- Ineligible: organizers' staff/contractors involved in administering or judging, and immediate household; Promotion Entities (Sponsor, Administrator) and their employees/families.
- Not open where U.S. sanctions, export controls, or local law prohibit participation/payment, or to those acting for sanctioned parties.
- Not open in areas prohibiting prize promotions (e.g. **Brazil**). Void where prohibited.

**Responsible & ethical use** — judges may reject entries that:
- Use data, models, or media without rights or consent
- Create safety, privacy, security, discrimination, or surveillance risks without safeguards
- Contain unlawful, defamatory, sexually explicit, gratuitously violent, hateful, or harassing content
- Misrepresent capabilities, results, benchmarks, or human review
- Violate OpenCV, AWS, Devpost, or third-party terms

**Ownership & privacy**
- You keep all IP (code, models, weights, datasets, methods, docs, etc.).
- Submitted Materials (proposal, report, presentation, video) are licensed to OpenCV and AWS — perpetual, worldwide, royalty-free, non-exclusive, sublicensable — for any lawful use. This does **not** cover private code, weights, datasets, or credentials merely referenced or demoed.
- Personal data is shared among administrator, OpenCV, and AWS only to run the competition; unrelated marketing needs separate opt-in.

**Administration**
- Organizers may disqualify ineligible, incomplete, misleading, unsafe, or rule-breaking entries, and may modify/suspend/cancel the competition if needed.
- Potential winners must complete eligibility, release, and tax docs on time or be replaced.
- Winners handle their own taxes; teams split prizes internally.
- Devpost Official Rules control if anything conflicts.
- **Liability:** organizers' aggregate liability is capped at USD $100; no indirect/consequential damages.

## Resources

- [OpenCV 5 documentation](https://docs.opencv.org/5.x/)
- [OpenCV 5 overview](https://opencv.org/opencv-5/)
- [Official competition website](https://opencv.org/opencv-ai-competition-2026/)
- [OpenCV COOL overview](https://opencv.org/COOL/)
- [COOL on AWS Marketplace](https://aws.amazon.com/marketplace/pp/prodview-fdvbfiewzuehs)
- [AWS account & Free Tier info](https://aws.amazon.com/free/)
- [AWS Promotional Credit Terms](https://aws.amazon.com/awscredits/)
- [OpenCV community Slack](https://bit.ly/opencv-slack) — channel `#opencvcomp26`
- [Grant proposal form](https://www.jotform.com/form/262145877145059)
- [OpenCV Forum](https://forum.opencv.org/)
- Free courses: [OpenCV Bootcamp](https://opencv.org/university/free-opencv-course/) · [PyTorch Bootcamp](https://opencv.org/university/free-pytorch-course/) · [VLM Bootcamp](https://opencv.org/university/vision-language-model-bootcamp/)

## Contact

Questions go to the contact email listed on the [official competition site](https://opencv.org/opencv-ai-competition-2026/#contact) (it's obfuscated on the page, so grab it there), or ask in the Slack / [Devpost discussions](https://opencv26.devpost.com/forum_topics).

---

*Sources: https://opencv26.devpost.com/rules, https://opencv26.devpost.com/resources, https://opencv.org/opencv-ai-competition-2026/ (retrieved Oct 2, 2026)*
