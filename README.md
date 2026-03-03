# Prompt Engineering for Language Models

> A **state-of-the-art, portfolio-ready** guide for designing, evaluating, and shipping reliable LLM systems.

[![Documentation](https://img.shields.io/badge/docs-modular-blue)](docs/README.md)
[![Focus](https://img.shields.io/badge/focus-production%20prompt%20engineering-success)](docs/02-prompt-engineering-patterns.md)
[![Safety](https://img.shields.io/badge/safety-guardrails%20%2B%20eval-critical)](docs/03-evaluation-and-guardrails.md)

---

## ✨ What makes this repo different

Most prompt guides stop at writing prompts.
This repository teaches the **full engineering loop**:

- prompt design,
- structured outputs,
- evaluation and regression,
- safety and policy alignment,
- RAG/system architecture,
- portfolio-ready project execution.

---

## 🧭 Documentation Hub

Start with the index: **[`docs/README.md`](docs/README.md)**

- `00-roadmap.md` — roadmap, milestones, and delivery cadence
- `01-foundations-nlp-ml.md` — practical foundations for prompt engineers
- `02-prompt-engineering-patterns.md` — modern patterns, templates, and anti-patterns
- `03-evaluation-and-guardrails.md` — eval systems, safety layers, incident handling
- `04-rag-and-system-design.md` — retrieval, routing, orchestration, reliability
- `05-projects-and-career.md` — project blueprints, KPI framing, interview preparation
- `06-industry-best-practices.md` — reference map + best-practice synthesis
- `07-restored-original-readme-code-guide.md` — fully restored original README code/tutorial content

---

## 🚀 Recommended workflow (high signal)

1. Pick a real use case (support triage, contract extraction, policy Q&A).
2. Create baseline prompt + output schema.
3. Build an eval set and scoring rubric.
4. Add safety tests (injection, policy-sensitive, malformed input).
5. Iterate with metrics (quality, latency, cost).
6. Ship a documented case study to your portfolio.

---

## 🧩 Architecture mindset

```text
Business Task -> Prompt Design -> Eval + Safety -> System Integration -> Monitoring -> Iteration
```

Prompt engineering is not a one-shot writing task; it is an iterative product and reliability discipline.

---


## ♻️ Restored original README code examples

Per request, all previously removed code/tutorial content from the old monolithic README has been restored in:

- [`docs/07-restored-original-readme-code-guide.md`](docs/07-restored-original-readme-code-guide.md)

The modular docs now reference this restored guide as a code appendix.

---

## 🤝 Contributing

Contributions are welcome, especially:

- stronger eval datasets,
- better prompt templates and validators,
- richer real-world examples,
- clearer diagrams and walkthroughs.

---

## 📜 License

This repository is licensed under the [MIT License](LICENSE).
