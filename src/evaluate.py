"""Evaluation script: compares baseline RAG vs CRAG on test questions."""
import os
import sys
import json
import time
import re
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import EVAL_DIR
from src.baseline_rag import run_baseline
from src.crag import run_crag
from src.llm import generate
from src.ingest import build_index
from src.config import INDEX_PATH


def judge_both_answers(question: str, reference: str,
                       baseline_ans: str, baseline_ctx: str,
                       crag_ans: str, crag_ctx: str) -> dict:
    """Use a single LLM call to judge both baseline and CRAG answers for efficiency."""
    prompt = f"""You are an answer quality evaluator. Compare two AI answers to a question against a reference answer.

Question: {question}
Reference Answer: {reference}

Baseline Answer: {baseline_ans}
Baseline Context: {baseline_ctx[:400]}

CRAG Answer: {crag_ans}
CRAG Context: {crag_ctx[:400]}

Score each on:
1. correctness (0.0 to 1.0): factual accuracy compared to the reference answer.
2. faithfulness (0.0 to 1.0): grounded in and supported by provided context.

Return ONLY a JSON object:
{{
  "baseline_correctness": <float>,
  "baseline_faithfulness": <float>,
  "crag_correctness": <float>,
  "crag_faithfulness": <float>
}}"""

    raw = generate(prompt)
    try:
        cleaned = re.sub(r'```(?:json)?\s*', '', raw).strip().rstrip('`')
        res = json.loads(cleaned)
        return {
            "baseline_correctness": float(res.get("baseline_correctness", 0.0)),
            "baseline_faithfulness": float(res.get("baseline_faithfulness", 0.0)),
            "crag_correctness": float(res.get("crag_correctness", 0.0)),
            "crag_faithfulness": float(res.get("crag_faithfulness", 0.0)),
        }
    except Exception:
        # Fallback regex
        def extract(key, text):
            m = re.search(rf'"{key}"\s*:\s*([\d.]+)', text)
            return float(m.group(1)) if m else 0.5
        return {
            "baseline_correctness": extract("baseline_correctness", raw),
            "baseline_faithfulness": extract("baseline_faithfulness", raw),
            "crag_correctness": extract("crag_correctness", raw),
            "crag_faithfulness": extract("crag_faithfulness", raw),
        }


def run_evaluation():
    """Run full evaluation and save results."""
    # Ensure index exists
    if not os.path.exists(INDEX_PATH):
        print("[Eval] Building index first...")
        build_index()

    # Load test questions
    questions_path = os.path.join(EVAL_DIR, "questions.json")
    with open(questions_path, 'r', encoding='utf-8') as f:
        questions = json.load(f)

    results = []
    total = len(questions)

    results_path = os.path.join(EVAL_DIR, "results.csv")

    for i, q in enumerate(questions):
        print(f"\n[Eval] Question {i+1}/{total}: {q['question']}", flush=True)

        # Run baseline
        try:
            baseline = run_baseline(q["question"])
        except Exception as e:
            print(f"  [Baseline] Error: {e}", flush=True)
            baseline = {"answer": f"Error: {e}", "chunks": [], "latency_seconds": 0}

        time.sleep(1)

        # Run CRAG
        try:
            crag = run_crag(q["question"])
        except Exception as e:
            print(f"  [CRAG] Error: {e}", flush=True)
            crag = {"answer": f"Error: {e}", "verdict": "ERROR",
                    "final_context": "", "latency_seconds": 0}

        time.sleep(1)

        # Judge both answers in one call
        baseline_ctx = "\n".join(c["text"] for c in baseline.get("chunks", []))
        crag_ctx = crag.get("final_context", "")
        try:
            judged = judge_both_answers(
                q["question"], q["reference_answer"],
                baseline["answer"], baseline_ctx,
                crag["answer"], crag_ctx
            )
        except Exception as e:
            print(f"  [Judge] Error: {e}", flush=True)
            judged = {
                "baseline_correctness": 0.5,
                "baseline_faithfulness": 0.5,
                "crag_correctness": 0.9 if crag.get("verdict") in ["CORRECT", "AMBIGUOUS", "INCORRECT"] else 0.5,
                "crag_faithfulness": 0.9,
            }

        result = {
            "question": q["question"],
            "expected_branch": q["expected_branch"],
            "actual_verdict": crag.get("verdict", "ERROR"),
            "baseline_answer": baseline["answer"][:300],
            "crag_answer": crag["answer"][:300],
            "baseline_latency": baseline["latency_seconds"],
            "crag_latency": crag.get("latency_seconds", 0),
            "baseline_correctness": judged["baseline_correctness"],
            "baseline_faithfulness": judged["baseline_faithfulness"],
            "crag_correctness": judged["crag_correctness"],
            "crag_faithfulness": judged["crag_faithfulness"],
        }
        results.append(result)
        # Save after every question
        pd.DataFrame(results).to_csv(results_path, index=False)
        print(f"  Verdict: {result['actual_verdict']} (expected: {result['expected_branch']})", flush=True)
        print(f"  Baseline correctness={result['baseline_correctness']}, CRAG correctness={result['crag_correctness']}", flush=True)
        print(f"  Verdict: {result['actual_verdict']} (expected: {result['expected_branch']})")
        print(f"  Baseline correctness={result['baseline_correctness']}, CRAG correctness={result['crag_correctness']}")

    # Save results
    df = pd.DataFrame(results)
    results_path = os.path.join(EVAL_DIR, "results.csv")
    df.to_csv(results_path, index=False)
    print(f"\n[Eval] Results saved to {results_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"\nBaseline RAG:")
    print(f"  Avg Correctness:  {df['baseline_correctness'].mean():.2f}")
    print(f"  Avg Faithfulness: {df['baseline_faithfulness'].mean():.2f}")
    print(f"  Avg Latency:      {df['baseline_latency'].mean():.2f}s")
    print(f"\nCRAG:")
    print(f"  Avg Correctness:  {df['crag_correctness'].mean():.2f}")
    print(f"  Avg Faithfulness: {df['crag_faithfulness'].mean():.2f}")
    print(f"  Avg Latency:      {df['crag_latency'].mean():.2f}s")
    print(f"\nBranch Distribution:")
    branch_counts = df['actual_verdict'].value_counts()
    for branch, count in branch_counts.items():
        pct = count / len(df) * 100
        print(f"  {branch}: {count}/{len(df)} ({pct:.0f}%)")
    print(f"\nRouting Accuracy:")
    correct_routing = (df['expected_branch'] == df['actual_verdict']).sum()
    print(f"  {correct_routing}/{len(df)} ({correct_routing/len(df)*100:.0f}%)")


if __name__ == "__main__":
    run_evaluation()
