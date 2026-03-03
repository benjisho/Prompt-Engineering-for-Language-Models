# RAG and System Design Around Prompts

Prompt engineering becomes truly effective when integrated into robust systems.

---

## 1) Production reference architecture

```text
                           ┌─────────────────────────┐
User Request ────────────► │ API Gateway / Auth      │
                           └────────────┬────────────┘
                                        ▼
                           ┌─────────────────────────┐
                           │ Intent Router           │
                           └───────┬─────────┬───────┘
                                   │         │
                                   │         └─────────────────────────────┐
                                   ▼                                       ▼
                        ┌─────────────────────┐                   ┌─────────────────────┐
                        │ Direct Prompt Path  │                   │ Retrieval Path      │
                        └──────────┬──────────┘                   └──────────┬──────────┘
                                   │                                         ▼
                                   │                             ┌─────────────────────┐
                                   │                             │ Retriever (Hybrid)  │
                                   │                             └──────────┬──────────┘
                                   │                                         ▼
                                   │                             ┌─────────────────────┐
                                   │                             │ Context Builder     │
                                   │                             └──────────┬──────────┘
                                   └────────────────────┬────────────────────┘
                                                        ▼
                                           ┌─────────────────────┐
                                           │ LLM Generation      │
                                           └──────────┬──────────┘
                                                      ▼
                                           ┌─────────────────────┐
                                           │ Output Validator    │
                                           │ schema/citation/pol │
                                           └───────┬───────┬─────┘
                                                   │       │
                                                   │       └──► Fallback / Escalation
                                                   ▼
                                                Response
```

---

## 2) RAG implementation best practices

- chunk by semantic boundaries (sections/headings),
- use metadata (`source`, `version`, `timestamp`, `access_level`),
- evaluate retrieval quality separately from generation quality,
- enforce citation-backed answers.

### Grounded response contract

```text
Use only provided context.
If evidence is missing, explicitly say so.
Cite each factual claim as [doc_id:chunk_id].
```

### Chunking helper snippet

```python
def chunk_by_paragraph(text: str, max_chars: int = 800) -> list[str]:
    chunks, current = [], ""
    for para in text.split("\n\n"):
        if len(current) + len(para) + 2 <= max_chars:
            current += ("\n\n" if current else "") + para
        else:
            if current:
                chunks.append(current)
            current = para
    if current:
        chunks.append(current)
    return chunks
```

---

## 3) Orchestration pseudo-code

```python
def answer(query: str) -> dict:
    intent = route_intent(query)

    if intent in {"policy_qa", "technical_qa", "compliance_qa"}:
        chunks = retrieve(query, top_k=5)
        draft = generate_answer(query=query, context=chunks)
    else:
        draft = generate_answer(query=query)

    if not output_is_valid(draft):
        draft = retry_with_constraints(query)

    if not output_is_valid(draft):
        return {"status": "fallback", "message": "Unable to provide a reliable answer."}

    return {"status": "ok", "data": draft}
```

### Response validator skeleton

```python
import json

def output_is_valid(raw: str) -> bool:
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return False
    has_fields = all(k in obj for k in ["answer", "citations", "confidence"])
    has_citations = isinstance(obj.get("citations"), list)
    return has_fields and has_citations
```

---

## 4) Design decision matrix

| Need | Best first choice | Why |
|---|---|---|
| fast behavior shaping | prompting | minimal overhead, quick iteration |
| factual grounding | RAG | fresh, auditable source-backed responses |
| stable style/format at scale | fine-tuning | predictable behavior after convergence |
| policy-critical workflows | layered guardrails | safety cannot rely on prompt alone |

---

## 5) Reliability and observability checklist

- prompt/model/retriever versioning,
- canary rollout before full rollout,
- p50/p95 latency dashboards,
- cost per successful task,
- policy-violation and fallback rates,
- retriever freshness monitoring.

### Operational telemetry diagram

```text
[Requests] -> [Router] -> [LLM/RAG] -> [Validator] -> [Responses]
      |            |          |             |             |
      +------------+----------+-------------+-------------+
                                 |
                                 v
                         [Observability Bus]
                                 |
      +--------------------------+------------------------------+
      |                          |                              |
      v                          v                              v
 [Quality Metrics]         [Safety Metrics]               [Cost/Latency]
      |                          |                              |
      +--------------------------+---------------+--------------+
                                                 v
                                          [Release Decisions]
```

---

## 6) Cost optimization techniques

- route simple tasks to smaller models,
- cache retrieval and shared prompt prefixes,
- compress long history into bounded summaries,
- avoid unnecessary chain depth in low-risk tasks.

### Simple router snippet

```python
def choose_model(intent: str) -> str:
    if intent in {"classification", "simple_extraction"}:
        return "small-fast-model"
    return "large-reasoning-model"
```

---

## References in this repo

- [Documentation Index](README.md)
- [Restored original README code guide](07-restored-original-readme-code-guide.md)
- [Prompt Engineering Patterns](02-prompt-engineering-patterns.md)
- [Evaluation and Guardrails](03-evaluation-and-guardrails.md)
- [RAG and System Design](04-rag-and-system-design.md)
