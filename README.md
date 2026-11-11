# CRAG — Self-Corrective Retrieval-Augmented Generation

A mini project for **Prompt Engineering and Optimization for Generative AI (UE24AM343AB5)** that demonstrates how self-corrective retrieval can improve upon standard RAG pipelines.

## Architecture

```
                          ┌─────────────┐
                          │  User Query │
                          └──────┬──────┘
                                 │
                          ┌──────▼──────┐
                          │  Retriever  │ ── cosine similarity, top-k chunks
                          └──────┬──────┘
                                 │
                          ┌──────▼──────┐
                          │  Evaluator  │ ── LLM scores each chunk (0-1)
                          └──────┬──────┘
                                 │
                 ┌───────────────┼───────────────┐
                 │               │               │
          score >= 0.7    0.4 <= score < 0.7   score < 0.4
                 │               │               │
          ┌──────▼──────┐ ┌──────▼──────┐ ┌──────▼──────┐
          │   CORRECT   │ │  AMBIGUOUS  │ │  INCORRECT  │
          │  Refine     │ │ Refine +    │ │ Rewrite     │
          │  context    │ │ Web search  │ │ query +     │
          └──────┬──────┘ └──────┬──────┘ │ Web search  │
                 │               │        └──────┬──────┘
                 └───────────────┼───────────────┘
                                 │
                          ┌──────▼──────┐
                          │  Generator  │ ── Gemini LLM
                          └──────┬──────┘
                                 │
                          ┌──────▼──────┐
                          │   Answer    │
                          └─────────────┘
```

## Setup

```bash
# 1. Create virtual environment
python -m venv venv

# 2. Activate it
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your API key
# Copy .env.example to .env and add your Gemini API key
# Get a free key: https://aistudio.google.com/apikey
```

## How to Run

```bash
# Start the Streamlit app
streamlit run app.py

# Open http://localhost:8501 in your browser
```

## Run Evaluation

```bash
python -m src.evaluate
```

This generates `eval/results.csv` with baseline vs CRAG comparison across 12 test questions.

## File Structure

```
crag-project/
├── app.py                  # Streamlit UI
├── requirements.txt
├── .env / .env.example
├── README.md
├── data/
│   ├── docs/               # Knowledge base documents (.txt)
│   ├── index.npz           # Cached embeddings (generated)
│   └── chunks.json         # Cached chunks (generated)
├── src/
│   ├── __init__.py
│   ├── config.py            # Configuration & constants
│   ├── llm.py               # Gemini LLM wrapper
│   ├── ingest.py            # Document loading & chunking
│   ├── retriever.py         # Vector similarity retrieval
│   ├── web_search.py        # DuckDuckGo search wrapper
│   ├── evaluator.py         # Retrieval quality evaluator
│   ├── corrective.py        # Context refinement & query rewriting
│   ├── baseline_rag.py      # Plain RAG pipeline
│   ├── crag.py              # Full CRAG pipeline
│   └── evaluate.py          # Evaluation script
└── eval/
    ├── questions.json       # Test questions
    └── results.csv          # Evaluation results (generated)
```

## Sample Questions

| Category   | Question                                              |
|------------|-------------------------------------------------------|
| CORRECT    | What is RAG and who introduced it?                    |
| CORRECT    | How does CRAG evaluate retrieved documents?           |
| AMBIGUOUS  | How does Self-RAG differ from CRAG?                   |
| AMBIGUOUS  | Advanced prompt engineering for RAG?                  |
| INCORRECT  | Who won the most recent FIFA World Cup?               |
| INCORRECT  | How to make carbonara pasta?                          |

## Tech Stack

- Python 3.10+, Streamlit, sentence-transformers (all-MiniLM-L6-v2)
- numpy cosine similarity, DuckDuckGo search, Google Gemini (gemini-3.8-flash)
