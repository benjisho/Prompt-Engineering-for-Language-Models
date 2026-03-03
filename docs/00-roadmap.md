# Roadmap: Prompt Engineering for Language Models

This roadmap is designed for learners who want to become **production-ready prompt engineers**, not just prompt hobbyists.

---

## Core principle

```text
Prompt quality = Good instructions + Good context + Good evaluation + Good guardrails
```

---

## Competency map

| Competency | Beginner | Intermediate | Advanced |
|---|---|---|---|
| Prompt design | clear single-task prompts | schema-first and few-shot patterns | orchestrated multi-step pipelines |
| Evaluation | manual spot checks | structured rubrics + pass/fail criteria | automated regression gates |
| Safety | basic refusals | policy-aware patterns + injection tests | layered guardrails + incident playbooks |
| System design | standalone prompts | prompt + retrieval integration | routing, caching, fallback, observability |
| Communication | write examples | report metrics and tradeoffs | lead design reviews and launch readiness |

---

## 30/60/90-day execution plan

## Day 0–30: Foundations + baseline systems

Deliverables:

- 3 baseline prompts (classification, extraction, grounded QA)
- first evaluation set (>=20 cases per task)
- failure taxonomy (top 10 failure patterns)

Success criteria:

- consistent output format
- explicit uncertainty handling
- documented known limitations

## Day 31–60: Reliability + safety hardening

Deliverables:

- structured output validators
- injection and policy safety test suite
- scorecard with weekly trend tracking

Success criteria:

- >=95% output format validity
- measurable quality improvement over baseline
- no critical policy escapes in standard adversarial set

## Day 61–90: Production-style architecture + portfolio

Deliverables:

- RAG integration for one knowledge-heavy use case
- fallback strategy for low-confidence cases
- polished case study and architecture diagram

Success criteria:

- stable regression test outcomes
- explainable design tradeoffs
- interview-ready project artifacts

---

## Iteration loop (ASCII)

```text
[Business Goal]
      |
      v
[Prompt + Context Design]
      |
      v
[Evaluation + Safety Tests]
      |
   pass? ------------------ no ------------------+
      |                                          |
     yes                                         v
      |                                  [Refine Prompt,
      v                                   Context, Routing,
[Release Candidate]                        Validators]
      |
      v
[Monitor Quality, Cost, Latency]
      |
      +------------------> [Next Iteration]
```

---

## Weekly rhythm (industry-friendly)

- **Monday:** analyze failures from previous week
- **Tuesday:** implement prompt/system changes
- **Wednesday:** run eval + safety suites
- **Thursday:** review metrics + approve rollout
- **Friday:** publish changelog and lessons learned

---

## Portfolio milestones

- Milestone 1: baseline + eval report
- Milestone 2: safety-hardened version
- Milestone 3: RAG-integrated version
- Milestone 4: production-readiness report

---

## Completion criteria

You are job-ready when you can repeatedly:

- define task success with measurable criteria,
- improve behavior through controlled iterations,
- harden against common safety failures,
- communicate architecture + tradeoffs to stakeholders.

---

## References in this repo

- [Documentation Index](README.md)
- [Restored original README code guide](07-restored-original-readme-code-guide.md)
- [Prompt Engineering Patterns](02-prompt-engineering-patterns.md)
- [Evaluation and Guardrails](03-evaluation-and-guardrails.md)
- [RAG and System Design](04-rag-and-system-design.md)
