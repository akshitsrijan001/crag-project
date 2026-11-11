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


def judge_answer(question: str, reference: str, answer: str, context: str) -> dict:
    """Use LLM as judge to score correctness and faithfulness."""
    prompt = f"""You are an answer quality judge. Given a question, a reference answer, the actual answer, and the context used, score the answer on two criteria.

Question: {question}
Reference Answer: {reference}
Actual Answer: {answer}
Context Used (first 500 chars): {context[:500]}

Score on:
1. correctness: How factually correct is the answer compared to the reference? (0.0 to 1.0)
2. faithfulness: Is the answer grounded in / supported by the provided context? (0.0 to 1.0)

Return ONLY a JSON object: {{"correctness": <float>, "faithfulness": <float>}}"""

    raw = generate(prompt)
    try:
        cleaned = re.sub(r'```(?:json)?\s*', '', raw).strip().rstrip('`')
        result = json.loads(cleaned)
        return {
            "correctness": float(result.get("correctness", 0.0)),
            "faithfulness": float(result.get("faithfulness", 0.0)),
        }
    except (json.JSONDecodeError, KeyError, TypeError):
        # Fallback regex
        c_match = re.search(r'"correctness"\s*:\s*([\d.]+)', raw)
        f_match = re.search(r'"faithfulness"\s*:\s*([\d.]+)', raw)
        return {
            "correctness": float(c_match.group(1)) if c_match else 0.0,
            "faithfulness": float(f_match.group(1)) if f_match else 0.0,
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

    for i, q in enumerate(questions):
        print(f"\n[Eval] Question {i+1}/{total}: {q['question']}")

        # Run baseline
        try:
            baseline = run_baseline(q["question"])
        except Exception as e:
            print(f"  [Baseline] Error: {e}")
            baseline = {"answer": f"Error: {e}", "chunks": [], "latency_seconds": 0}

        time.sleep(2)  # Rate limit pause

        # Run CRAG
        try:
            crag = run_crag(q["question"])
        except Exception as e:
            print(f"  [CRAG] Error: {e}")
            crag = {"answer": f"Error: {e}", "verdict": "ERROR",
                    "final_context": "", "latency_seconds": 0}

        time.sleep(2)  # Rate limit pause

        # Judge baseline answer
        try:
            baseline_judge = judge_answer(
                q["question"], q["reference_answer"],
                baseline["answer"],
                "\n".join(c["text"] for c in baseline.get("chunks", []))
            )
        except Exception as e:
            print(f"  [Judge baseline] Error: {e}")
            baseline_judge = {"correctness": 0.0, "faithfulness": 0.0}

        time.sleep(2)

        # Judge CRAG answer
        try:
            crag_judge = judge_answer(
                q["question"], q["reference_answer"],
                crag["answer"], crag.get("final_context", "")
            )
        except Exception as e:
            print(f"  [Judge CRAG] Error: {e}")
            crag_judge = {"correctness": 0.0, "faithfulness": 0.0}

        time.sleep(2)

        result = {
            "question": q["question"],
            "expected_branch": q["expected_branch"],
            "actual_verdict": crag.get("verdict", "ERROR"),
            "baseline_answer": baseline["answer"][:300],
            "crag_answer": crag["answer"][:300],
            "baseline_latency": baseline["latency_seconds"],
            "crag_latency": crag.get("latency_seconds", 0),
            "baseline_correctness": baseline_judge["correctness"],
            "baseline_faithfulness": baseline_judge["faithfulness"],
            "crag_correctness": crag_judge["correctness"],
            "crag_faithfulness": crag_judge["faithfulness"],
        }
        results.append(result)
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
