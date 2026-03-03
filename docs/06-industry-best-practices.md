# Industry Best-Practice References

This repository aligns with recurring patterns from major LLM providers and experienced production teams.

---

## Official references

- OpenAI docs: <https://platform.openai.com/docs>
- Anthropic docs: <https://docs.anthropic.com>
- Google Vertex AI docs: <https://cloud.google.com/vertex-ai/generative-ai/docs>
- Microsoft Azure OpenAI docs: <https://learn.microsoft.com/azure/ai-services/openai>
- Prompting Guide: <https://www.promptingguide.ai>

---

## Synthesized best practices (cross-industry)

### 1) Start from task contracts, not model tricks

Define what “good” means using:

- objective outputs,
- explicit constraints,
- measurable acceptance criteria.

### 2) Use schema-first outputs for machine pipelines

If downstream code consumes model output, enforce strict JSON contracts and validation.

### 3) Treat evaluation as a release gate

Every prompt/model/retriever change should trigger regression tests.

### 4) Build defense in depth for safety

```text
Input filters -> policy enforcement -> model generation -> output validation -> escalation path
```

### 5) Monitor business outcomes, not only model metrics

Track:

- successful task completion,
- policy violation escape rate,
- cost per successful resolution,
- user satisfaction and deflection impact.

### 6) Version everything

Version prompt templates, model versions, retrieval index versions, and eval datasets together.

---

## Governance checklist

- documented acceptable-use policy,
- audit logging for critical interactions,
- PII redaction strategy,
- incident escalation ownership,
- periodic policy and dataset review cadence.

---

## Practical rollout checklist

1. Validate against dev + holdout eval suites.
2. Run adversarial safety set.
3. Deploy to canary traffic.
4. Watch quality/cost/safety dashboards.
5. Roll forward or rollback based on thresholds.

---

## Environment note

If this environment limits external HTTP access, verify provider docs in an unrestricted environment before final production implementation.

---

## References in this repo

- [Documentation Index](README.md)
- [Restored original README code guide](07-restored-original-readme-code-guide.md)
- [Prompt Engineering Patterns](02-prompt-engineering-patterns.md)
- [Evaluation and Guardrails](03-evaluation-and-guardrails.md)
- [RAG and System Design](04-rag-and-system-design.md)
