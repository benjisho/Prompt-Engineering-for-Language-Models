# Prompt Engineering Patterns (Modern)

This module provides practical patterns used in production systems.

---

## Prompt anatomy (gold standard)

A robust prompt specifies:

1. Role
2. Task
3. Constraints
4. Context boundaries
5. Output schema
6. Quality criteria

### Visual mental model

```text
+--------------------- PROMPT CONTRACT ----------------------+
| ROLE       : who the model should act as                  |
| TASK       : what outcome is required                     |
| CONTEXT    : what evidence is allowed                     |
| CONSTRAINTS: what is forbidden/required                   |
| FORMAT     : strict machine-readable output               |
| QUALITY    : rubric for acceptance                        |
+------------------------------------------------------------+
```

---

## Pattern A: Baseline zero-shot prompt

```text
You are an e-commerce triage assistant.
Classify the user message into one label:
[shipping_issue, payment_issue, account_access, product_question, other]
Return strict JSON with keys: label, reason.
reason must be <= 20 words.
```

Use this to benchmark your first measurable baseline.

### Invocation snippet (Python)

```python
prompt = """
You are an e-commerce triage assistant.
Classify the user message into one label:
[shipping_issue, payment_issue, account_access, product_question, other]
Return strict JSON with keys: label, reason.
reason must be <= 20 words.
User: charged twice for one order
"""

# response = llm.generate(prompt)
# print(response)
```

---

## Pattern B: Few-shot prompt for ambiguity

```text
Example 1
Input: "delivered says yes but i got nothing"
Output: {"label":"shipping_issue","reason":"delivery discrepancy"}

Example 2
Input: "charged twice for one order"
Output: {"label":"payment_issue","reason":"duplicate charge"}

Now classify:
Input: "can't login after password reset"
```

Best practice:

- keep examples concise,
- include at least one edge case,
- avoid contradictory example logic.

---

## Pattern C: Schema-first generation

### Prompt requirement

```text
Return strict JSON with exactly:
- intent: enum[refund, exchange, troubleshooting, other]
- confidence: number 0..1
- next_action: string
No markdown. No extra keys.
```

### Validator snippet (Python/Pydantic)

```python
from pydantic import BaseModel, Field
from typing import Literal

class ResolutionPlan(BaseModel):
    intent: Literal["refund", "exchange", "troubleshooting", "other"]
    confidence: float = Field(ge=0.0, le=1.0)
    next_action: str
```

### Defensive parse wrapper

```python
import json

def parse_resolution(raw: str) -> ResolutionPlan | None:
    try:
        payload = json.loads(raw)
        return ResolutionPlan(**payload)
    except Exception:
        return None
```

---

## Pattern D: Decomposition and tool routing

```text
Step 1: classify intent.
Step 2: retrieve top-3 policy passages.
Step 3: draft answer grounded in retrieved context.
Step 4: validate citations and output schema.
```

### Rich orchestration diagram

```text
┌────────────┐      ┌────────────────────┐      ┌──────────────────┐
│ User Input │ ───► │ Intent Classifier  │ ───► │ Route Decision   │
└─────┬──────┘      └─────────┬──────────┘      └───────┬──────────┘
      │                       │                          │
      │                       │                          │
      │                       ▼                          ▼
      │                ┌──────────────┐          ┌───────────────┐
      │                │ Direct Prompt│          │ Retrieval Path│
      │                └──────┬───────┘          └───────┬───────┘
      │                       │                          │
      │                       │                   ┌──────▼───────┐
      │                       │                   │ Retriever     │
      │                       │                   └──────┬───────┘
      │                       │                          │
      │                       │                   ┌──────▼───────┐
      │                       │                   │ Context Pack  │
      │                       │                   └──────┬───────┘
      │                       └──────────────┬───────────┘
      │                                      ▼
      │                              ┌──────────────┐
      └────────────────────────────► │ LLM Generator│
                                     └──────┬───────┘
                                            ▼
                                     ┌──────────────┐
                                     │ Validator    │
                                     │ schema/policy│
                                     └───┬─────┬────┘
                                         │     │
                                         │     └────► Retry / Fallback
                                         ▼
                                       Final
```

---

## Pattern E: Critique-and-revise loop

```text
Draft answer.
Evaluate against criteria:
- factual support
- policy compliance
- format validity
Revise once if any criterion fails.
```

### Critique-revise code skeleton

```python
def generate_with_review(task_prompt: str) -> str:
    draft = llm_generate(task_prompt)
    review_prompt = f"""
Review this output for:
1) factual support
2) policy compliance
3) schema validity
Output PASS or FAIL with concise reasons.
Output to review:\n{draft}
"""
    verdict = llm_generate(review_prompt)
    if "FAIL" in verdict:
        repair_prompt = f"Revise output to satisfy all checks.\nOriginal:\n{draft}\nReview:\n{verdict}"
        return llm_generate(repair_prompt)
    return draft
```

---

## Anti-patterns (high risk)

- mega-prompt without modularity,
- no output schema for machine consumers,
- no uncertainty behavior,
- no adversarial testing,
- no distinction between facts and assumptions.

---

## Reusable enterprise template

```text
You are [role] for [domain].

Task:
[exact instruction]

Context:
[retrieved context or user input]

Hard constraints:
- [must do]
- [must not do]

Output contract:
[exact JSON keys/types]

Quality bar:
- factual grounding
- policy compliance
- concise answer

If insufficient evidence, state uncertainty and request missing information.
```

---

## Real-world example: policy-grounded HR assistant

### Bad prompt

```text
Answer employee policy questions.
```

### Better prompt

```text
You are an HR policy assistant.
Answer only using provided policy excerpts.
Cite each claim as [policy_id:section].
If policy support is missing, say: "Insufficient policy evidence in provided context."
Return JSON: {"answer":"...","citations":[...],"confidence":0..1}
```

### End-to-end mini pipeline

```python
def answer_hr_question(question: str, policy_chunks: list[str]) -> dict:
    context = "\n".join(policy_chunks[:5])
    prompt = f"""
You are an HR policy assistant.
Use only this context:\n{context}
Question: {question}
Return JSON with answer, citations, confidence.
"""
    raw = llm_generate(prompt)
    # validate JSON and fields in real implementation
    return {"raw": raw}
```

---

## References in this repo

- [Documentation Index](README.md)
- [Restored original README code guide](07-restored-original-readme-code-guide.md)
- [Prompt Engineering Patterns](02-prompt-engineering-patterns.md)
- [Evaluation and Guardrails](03-evaluation-and-guardrails.md)
- [RAG and System Design](04-rag-and-system-design.md)
