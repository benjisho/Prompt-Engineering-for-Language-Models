# Projects, Portfolio, and Career Readiness

This module helps convert knowledge into interview-ready evidence.

---

## Project 1: Prompt Quality Lab (Support Triage)

### Scenario

You own first-line support triage for an e-commerce company.

### Deliverables

- baseline prompt,
- eval dataset (20–100 cases),
- rubric and scoring report,
- improved prompt and measured gains.

### Example KPI summary

```text
Intent accuracy: 75% -> 89%
Schema validity: 82% -> 98%
p95 latency: 2.1s -> 1.8s
```

---

## Project 2: Structured Extraction (Contracts)

### Scenario

Extract legal metadata from contract text.

### Output schema

```json
{
  "party_a": "string",
  "party_b": "string",
  "effective_date": "YYYY-MM-DD",
  "renewal_term": "string|null",
  "governing_law": "string|null"
}
```

### Validator snippet

```python
required = {"party_a", "party_b", "effective_date", "renewal_term", "governing_law"}

def valid_schema(obj: dict) -> bool:
    return set(obj.keys()) == required
```

---

## Project 3: Grounded Policy Assistant (RAG)

### Scenario

Build an HR or IT policy assistant for internal users.

### Required controls

- citations for factual claims,
- fallback for missing evidence,
- policy-aware refusal behavior.

### Architecture sketch

```text
Question -> Router -> Retriever -> Generator -> Validator -> Response/Fallback
```

---

## Project 4: Safety Red-Team Harness

### Scenario

Stress-test a customer-facing assistant before launch.

### Attack categories

- instruction override,
- data exfiltration,
- role confusion,
- harmful-content elicitation.

### Outputs

- adversarial prompt suite,
- mitigation matrix,
- before/after safety scorecard.

---

## Case study template (recommended)

1. Problem and business context
2. Design approach
3. Evaluation methodology
4. Safety and risk controls
5. Results and impact
6. Limitations and next steps

---

## Interview readiness checklist

- [ ] I can explain a complete prompt iteration lifecycle.
- [ ] I can defend metric choices and thresholds.
- [ ] I can discuss hallucination and safety tradeoffs.
- [ ] I can describe schema validation and fallback behavior.
- [ ] I can show at least 2 projects with measurable improvements.

---

## Collaboration readiness checklist

- [ ] Versioning of prompt/model/retriever is documented.
- [ ] Eval steps are reproducible by teammates.
- [ ] Rollback process is explicit.
- [ ] Monitoring plan is defined for production.

---

## References in this repo

- [Documentation Index](README.md)
- [Restored original README code guide](07-restored-original-readme-code-guide.md)
- [Prompt Engineering Patterns](02-prompt-engineering-patterns.md)
- [Evaluation and Guardrails](03-evaluation-and-guardrails.md)
- [RAG and System Design](04-rag-and-system-design.md)
