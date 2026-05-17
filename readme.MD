# SHL Assessment Recommender

Conversational agent that helps hiring managers find the right SHL assessments through dialogue.

Built for the SHL Labs AI Intern take-home assignment.

---

## What it does

- Clarifies vague queries before recommending
- Recommends 1–10 assessments grounded in the SHL catalog
- Updates recommendations mid-conversation when user refines constraints
- Compares assessments using catalog data only
- Refuses off-topic questions and prompt injection attempts

---

## Stack

| Component | Choice |
|-----------|--------|
| API | FastAPI + Pydantic |
| LLM | llama-3.3-70b-versatile via Groq |
| Retrieval | BM25 (rank-bm25) over full 68-item catalog |
| Deployment | Railway |
| HTTP client | httpx (async) |

---

## Project structure

```
shl-recommender/
├── main.py          # FastAPI app, endpoints, prompt, LLM call
├── retriever.py     # BM25 catalog indexer and search
├── data/
│   └── catalog.json # Scraped SHL Individual Test Solutions (68 items)
├── requirements.txt
└── railway.toml
```

---

## API

### GET /health
Returns `{"status": "ok"}` with HTTP 200.

### POST /chat

Request:
```json
{
  "messages": [
    {"role": "user", "content": "I am hiring a Java developer"},
    {"role": "assistant", "content": "What seniority level?"},
    {"role": "user", "content": "Mid-level, 4 years experience"}
  ]
}
```

Response:
```json
{
  "reply": "Here are 4 assessments for a mid-level Java developer.",
  "recommendations": [
    {"name": "Java (New)", "url": "https://www.shl.com/...", "test_type": "K"},
    {"name": "OPQ32r", "url": "https://www.shl.com/...", "test_type": "P"}
  ],
  "end_of_conversation": false
}
```

`recommendations` is empty while the agent is still gathering context or refusing. `end_of_conversation` is true only when the task is complete.

---

## Running locally

```bash
# Install dependencies
pip install -r requirements.txt

# Set your Groq API key
export GROQ_API_KEY=your_key_here

# Start the server
python main.py
```

Server runs on `http://localhost:8080`.

---

## Deployment (Railway)

1. Push repo to GitHub
2. Create new Railway project from repo
3. Add `GROQ_API_KEY` as an environment variable in Railway dashboard
4. Railway uses `railway.toml` for config — no extra setup needed

---

## Key design decisions

**Full catalog in context** — The catalog has 68 items (~6,000 tokens). Passing all of them to the LLM on every call eliminates recall misses from BM25 keyword mismatch on semantic queries.

**Hallucination protection** — `validate_recommendations()` checks every LLM response against catalog URLs and names before returning. Any invented assessment is silently dropped.

**Stateless by design** — No session state is stored. Every `/chat` call carries the full conversation history, matching the evaluator's design.

**One question at a time** — The prompt explicitly limits the agent to one clarifying question per turn, preventing turn-cap failures.