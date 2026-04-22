# evaluate_using_deepeval.py
# Comprehensive RAG evaluation using DeepEval (avoids OpenAI country restrictions)
# Run from CLI: python evaluate_using_deepeval.py
# Install: pip install deepeval

import os
import json
from datetime import datetime
from typing import List, Dict, Any

# Set DeepSeek API configuration
DEEPSEEK_API_KEY = "sk-cdfad4eedd844539824a9bd2744caa5c"
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"

os.environ["DEEPSEEK_API_KEY"] = DEEPSEEK_API_KEY

# Local imports
from rag_pipeline import ask, vectorstore, embeddings


# =============================================================================
# TEST DATASET WITH GROUND TRUTH
# =============================================================================

TEST_CASES = [
    {
        "question": "What are the rights of Foreign Domestic Helpers in Hong Kong?",
        "expected_answer": "Foreign Domestic Helpers have the right to personal safety. If physically abused or if personal safety is at risk, they can call the police emergency hotline 999 immediately. They are also entitled to rest days under the Employment Ordinance.",
        "expected_sources": ["FDHguideEnglish.pdf", "fdw_corner_webpage.txt"],
        "category": "rights"
    },
    {
        "question": "What is the minimum wage for domestic workers?",
        "expected_answer": "The statutory minimum wage (SMW) rate is reviewed periodically. The Minimum Wage Ordinance establishes the hourly rate. Failure to pay minimum wage amounts to a breach of the Employment Ordinance.",
        "expected_sources": ["Concise Guide Minimum Wage.pdf", "Concise Guide Employment Ordinance.pdf"],
        "category": "wage"
    },
    {
        "question": "Do domestic workers get rest days?",
        "expected_answer": "Yes. An employee employed under a continuous contract is entitled to not less than one rest day in every period of seven days. Whether the rest day is paid is to be agreed between the employer and employee.",
        "expected_sources": ["Concise Guide Employment Ordinance.pdf", "FDHguideEnglish.pdf"],
        "category": "rights"
    },
    {
        "question": "What are the rules for employment agencies regarding commission?",
        "expected_answer": "Employment agencies are not allowed to overcharge commission. If convicted of overcharging, the Commissioner for Labour will revoke or refuse to issue/renew the agency's licence. From 2015 to 2018, 12 agencies had licences revoked for this offence.",
        "expected_sources": ["PGEA_Chapter_4.pdf", "CoP_EA_Eng.pdf"],
        "category": "agencies"
    },
    {
        "question": "What happens if an employment agency overcharges a worker?",
        "expected_answer": "If an employment agency is convicted of overcharging commission, the Commissioner for Labour will revoke or refuse to issue or renew its licence. Offenders will also be prosecuted.",
        "expected_sources": ["PGEA_Chapter_4.pdf"],
        "category": "agencies"
    },
    {
        "question": "Are Foreign Domestic Helpers required to live with their employers?",
        "expected_answer": "Yes, Foreign Domestic Helpers are typically live-in workers. This is a standard requirement under the FDH employment contract.",
        "expected_sources": ["FDHguideEnglish.pdf"],
        "category": "fdh_specific"
    },
    {
        "question": "What laws must employment agencies comply with?",
        "expected_answer": "Employment agencies must comply with: Employment Ordinance, Employment Agency Regulations, Immigration Ordinance, Trade Descriptions Ordinance, Personal Data (Privacy) Ordinance, and Prevention of Bribery Ordinance.",
        "expected_sources": ["CoP_EA_Eng.pdf", "PGEA_Chapter_3.pdf"],
        "category": "agencies"
    },
    {
        "question": "What is a continuous contract under the Employment Ordinance?",
        "expected_answer": "A continuous contract refers to employment where an employee has been employed for 4 weeks or more with at least 18 hours per week. Employees under continuous contract are entitled to statutory benefits including rest days, sickness allowance, and holiday pay.",
        "expected_sources": ["Concise Guide Employment Ordinance.pdf"],
        "category": "general_employment"
    },
]


# =============================================================================
# DEEPEVAL METRICS (Custom Implementation)
# =============================================================================
# Since DeepEval may also have API restrictions, we implement custom metrics
# using DeepSeek directly for LLM-based evaluation

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import re

# Create DeepSeek LLM for evaluation
eval_llm = ChatOpenAI(
    model=DEEPSEEK_MODEL,
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
    temperature=0.0  # Deterministic for evaluation
)


def compute_faithfulness_score(question: str, answer: str, contexts: List[str]) -> float:
    """
    Faithfulness: Does the answer follow from the retrieved context?
    Score: 0.0 (hallucinated) to 1.0 (fully faithful)
    """
    context_text = "\n\n".join(contexts[:3])  # Use top 3 contexts
    
    prompt = ChatPromptTemplate.from_template("""
You are an impartial evaluator. Determine if the ANSWER can be derived from the CONTEXT.

CONTEXT:
{context}

QUESTION: {question}

ANSWER: {answer}

Rate the faithfulness from 0 to 1:
- 1.0: Answer is fully supported by context, no hallucination
- 0.5: Answer is partially supported, some claims not in context
- 0.0: Answer cannot be derived from context at all

Provide ONLY a number between 0 and 1, no explanation.
""")
    
    chain = prompt | eval_llm | StrOutputParser()
    
    try:
        response = chain.invoke({
            "context": context_text,
            "question": question,
            "answer": answer
        })
        # Extract number from response
        match = re.search(r'(\d\.?\d*)', response.strip())
        if match:
            return min(1.0, max(0.0, float(match.group(1))))
        return 0.5
    except Exception as e:
        print(f"      Warning: Faithfulness evaluation failed: {e}")
        return 0.5


def compute_answer_relevancy_score(question: str, answer: str) -> float:
    """
    Answer Relevancy: Does the answer address the question?
    Score: 0.0 (irrelevant) to 1.0 (fully relevant)
    """
    prompt = ChatPromptTemplate.from_template("""
You are an impartial evaluator. Determine if the ANSWER addresses the QUESTION.

QUESTION: {question}

ANSWER: {answer}

Rate the relevancy from 0 to 1:
- 1.0: Answer directly addresses the question
- 0.5: Answer is somewhat related but misses key points
- 0.0: Answer is irrelevant or doesn't address the question

Provide ONLY a number between 0 and 1, no explanation.
""")
    
    chain = prompt | eval_llm | StrOutputParser()
    
    try:
        response = chain.invoke({"question": question, "answer": answer})
        match = re.search(r'(\d\.?\d*)', response.strip())
        if match:
            return min(1.0, max(0.0, float(match.group(1))))
        return 0.5
    except Exception as e:
        print(f"      Warning: Relevancy evaluation failed: {e}")
        return 0.5


def compute_context_precision_score(question: str, contexts: List[str], expected_sources: List[str], source_documents: List = None) -> float:
    """
    Context Precision: Are the most relevant sources retrieved first?
    Score: 0.0 (wrong sources) to 1.0 (perfect ranking)
    """
    if not expected_sources:
        return 0.0
    
    # Use source_documents metadata if available, otherwise fall back to text search
    if source_documents:
        retrieved_sources = [doc.metadata.get('source', '') for doc in source_documents]
    else:
        # Fallback: try to find source names in context text
        retrieved_sources = []
        for ctx in contexts[:3]:
            for src in expected_sources:
                if src.split('.')[0].lower() in ctx.lower()[:500]:
                    retrieved_sources.append(src)
    
    if not retrieved_sources:
        return 0.0
    
    # Calculate precision with position weighting (earlier = better)
    # For each expected source, find its position in retrieved list
    position_scores = []
    for i, expected_src in enumerate(expected_sources):
        if expected_src in retrieved_sources:
            # Find position (0-indexed)
            position = retrieved_sources.index(expected_src)
            # Score decreases with position: 1.0 for first, 0.5 for second, etc.
            position_scores.append(1.0 / (position + 1))
    
    if position_scores:
        return sum(position_scores) / len(expected_sources)
    return 0.0


def compute_context_recall_score(contexts: List[str], expected_answer: str) -> float:
    """
    Context Recall: Does the retrieved context contain information needed for the expected answer?
    Score: 0.0 (missing info) to 1.0 (all info present)
    """
    context_text = "\n\n".join(contexts)
    
    # Extract key claims from expected answer
    key_claims = [s.strip() for s in expected_answer.split('.') if len(s.strip()) > 10]
    
    if not key_claims:
        return 0.5
    
    prompt = ChatPromptTemplate.from_template("""
You are an impartial evaluator. Determine what fraction of KEY CLAIMS are supported by the CONTEXT.

CONTEXT:
{context}

KEY CLAIMS:
{claims}

For each claim, determine if it can be derived from the context.
Respond with ONLY the fraction as a decimal (e.g., 0.75 for 75%).

Fraction of claims supported:
""")
    
    claims_text = "\n".join([f"- {claim}" for claim in key_claims[:5]])  # Limit to 5 claims
    
    chain = prompt | eval_llm | StrOutputParser()
    
    try:
        response = chain.invoke({"context": context_text, "claims": claims_text})
        match = re.search(r'(\d\.?\d*)', response.strip())
        if match:
            return min(1.0, max(0.0, float(match.group(1))))
        return 0.5
    except Exception as e:
        print(f"      Warning: Context recall evaluation failed: {e}")
        return 0.5


def compute_answer_correctness_score(answer: str, expected_answer: str) -> float:
    """
    Answer Correctness: How similar is the answer to the expected answer?
    Uses semantic similarity via LLM judgment.
    Score: 0.0 (completely wrong) to 1.0 (correct)
    """
    prompt = ChatPromptTemplate.from_template("""
You are an impartial evaluator. Compare the GENERATED ANSWER to the EXPECTED ANSWER.

EXPECTED ANSWER:
{expected}

GENERATED ANSWER:
{generated}

Rate the correctness from 0 to 1:
- 1.0: Generated answer contains all key information from expected answer
- 0.5: Generated answer has some correct information but misses key points
- 0.0: Generated answer is incorrect or contradicts expected answer

Consider semantic similarity, not exact wording.
Provide ONLY a number between 0 and 1, no explanation.
""")
    
    chain = prompt | eval_llm | StrOutputParser()
    
    try:
        response = chain.invoke({"expected": expected_answer, "generated": answer})
        match = re.search(r'(\d\.?\d*)', response.strip())
        if match:
            return min(1.0, max(0.0, float(match.group(1))))
        return 0.5
    except Exception as e:
        print(f"      Warning: Correctness evaluation failed: {e}")
        return 0.5


# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

def run_rag_query(question: str) -> Dict[str, Any]:
    """Run a query through the RAG pipeline."""
    result = ask(question)
    contexts = [doc.page_content for doc in result['source_documents']]
    return {
        "question": question,
        "answer": result['result'],
        "contexts": contexts,
        "source_documents": result['source_documents']
    }


def calculate_retrieval_metrics(results: List[Dict], test_cases: List[Dict]) -> Dict:
    """Calculate source recall and precision."""
    source_recall_scores = []
    source_precision_scores = []
    
    for result, test in zip(results, test_cases):
        expected_sources = set(test['expected_sources'])
        retrieved_sources = set(doc.metadata.get('source', '') for doc in result['source_documents'])
        
        if expected_sources:
            recall = len(expected_sources & retrieved_sources) / len(expected_sources)
        else:
            recall = 1.0
        source_recall_scores.append(recall)
        
        if retrieved_sources:
            precision = len(expected_sources & retrieved_sources) / len(retrieved_sources)
        else:
            precision = 0.0
        source_precision_scores.append(precision)
    
    return {
        "source_recall": sum(source_recall_scores) / len(source_recall_scores) if source_recall_scores else 0,
        "source_precision": sum(source_precision_scores) / len(source_precision_scores) if source_precision_scores else 0,
    }


def run_evaluation(save_report: bool = True) -> Dict:
    """Run comprehensive RAG evaluation using DeepEval-style metrics."""
    print("\n" + "=" * 80)
    print("LABOR POBOT - DEEPEVAL EVALUATION SUITE")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print(f"\nUsing DeepSeek API for LLM-based metrics")
    print(f"Test cases: {len(TEST_CASES)}")
    
    # Run all queries
    print("\n" + "=" * 80)
    print("STEP 1: RUNNING RAG QUERIES")
    print("=" * 80)
    
    results = []
    for i, test in enumerate(TEST_CASES, 1):
        print(f"\n[{i}/{len(TEST_CASES)}] {test['question'][:60]}...")
        try:
            result = run_rag_query(test['question'])
            results.append(result)
            print(f"   ✓ Retrieved {len(result['contexts'])} contexts, {len(result['answer'])} chars in answer")
        except Exception as e:
            print(f"   ✗ Error: {e}")
            results.append({
                "question": test['question'],
                "answer": "",
                "contexts": [],
                "source_documents": []
            })
    
    # Compute metrics
    print("\n" + "=" * 80)
    print("STEP 2: COMPUTING METRICS")
    print("=" * 80)
    
    faithfulness_scores = []
    relevancy_scores = []
    precision_scores = []
    recall_scores = []
    correctness_scores = []
    
    for i, (test, result) in enumerate(zip(TEST_CASES, results), 1):
        print(f"\n[{i}/{len(TEST_CASES)}] Evaluating: {test['category']}")
        
        if not result['answer']:
            print("   ✗ Skipping (no answer)")
            faithfulness_scores.append(0.0)
            relevancy_scores.append(0.0)
            precision_scores.append(0.0)
            recall_scores.append(0.0)
            correctness_scores.append(0.0)
            continue
        
        # Faithfulness
        print("   - Computing faithfulness...", end=" ")
        faithfulness = compute_faithfulness_score(
            result['question'], result['answer'], result['contexts']
        )
        faithfulness_scores.append(faithfulness)
        print(f"{faithfulness:.3f}")
        
        # Relevancy
        print("   - Computing relevancy...", end=" ")
        relevancy = compute_answer_relevancy_score(
            result['question'], result['answer']
        )
        relevancy_scores.append(relevancy)
        print(f"{relevancy:.3f}")
        
        # Context Precision
        print("   - Computing context precision...", end=" ")
        precision = compute_context_precision_score(
            result['question'], result['contexts'], test['expected_sources'], result['source_documents']
        )
        precision_scores.append(precision)
        print(f"{precision:.3f}")
        
        # Context Recall
        print("   - Computing context recall...", end=" ")
        recall = compute_context_recall_score(
            result['contexts'], test['expected_answer']
        )
        recall_scores.append(recall)
        print(f"{recall:.3f}")
        
        # Correctness
        print("   - Computing correctness...", end=" ")
        correctness = compute_answer_correctness_score(
            result['answer'], test['expected_answer']
        )
        correctness_scores.append(correctness)
        print(f"{correctness:.3f}")
    
    # Calculate retrieval metrics
    print("\n" + "=" * 80)
    print("STEP 3: CALCULATING RETRIEVAL METRICS")
    print("=" * 80)
    retrieval_metrics = calculate_retrieval_metrics(results, TEST_CASES)
    print(f"\nSource Recall: {retrieval_metrics['source_recall']:.2%}")
    print(f"Source Precision: {retrieval_metrics['source_precision']:.2%}")
    
    # Compile results
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    
    metrics_summary = {
        "faithfulness": sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 0,
        "answer_relevancy": sum(relevancy_scores) / len(relevancy_scores) if relevancy_scores else 0,
        "context_precision": sum(precision_scores) / len(precision_scores) if precision_scores else 0,
        "context_recall": sum(recall_scores) / len(recall_scores) if recall_scores else 0,
        "answer_correctness": sum(correctness_scores) / len(correctness_scores) if correctness_scores else 0,
    }
    
    print("\n📊 METRICS SUMMARY:")
    for metric_name, score in metrics_summary.items():
        print(f"   {metric_name}: {score:.4f} {'✓' if score > 0.7 else '⚠' if score > 0.5 else '✗'}")
    
    overall_score = sum(metrics_summary.values()) / len(metrics_summary)
    print(f"\n📊 OVERALL SCORE: {overall_score:.4f}")
    
    if overall_score > 0.7:
        print("   Status: ✓ GOOD - RAG pipeline is performing well")
    elif overall_score > 0.5:
        print("   Status: ⚠ NEEDS IMPROVEMENT - Some components need tuning")
    else:
        print("   Status: ✗ POOR - Significant improvements needed")
    
    # Detailed results
    print("\n" + "=" * 80)
    print("DETAILED PER-QUESTION RESULTS")
    print("=" * 80)
    
    detailed_results = []
    for i, (test, result) in enumerate(zip(TEST_CASES, results), 1):
        expected_sources = set(test['expected_sources'])
        retrieved_sources = set(doc.metadata.get('source', '') for doc in result['source_documents'])
        
        detailed_results.append({
            "question": test['question'],
            "category": test['category'],
            "answer": result['answer'][:500] if result['answer'] else "",
            "expected_answer": test['expected_answer'],
            "expected_sources": list(expected_sources),
            "retrieved_sources": list(retrieved_sources),
            "metrics": {
                "faithfulness": faithfulness_scores[i-1],
                "relevancy": relevancy_scores[i-1],
                "precision": precision_scores[i-1],
                "recall": recall_scores[i-1],
                "correctness": correctness_scores[i-1]
            }
        })
        
        print(f"\n[Question {i}] Category: {test['category']}")
        print(f"   Q: {test['question'][:70]}...")
        print(f"   Faithfulness: {faithfulness_scores[i-1]:.3f}")
        print(f"   Relevancy: {relevancy_scores[i-1]:.3f}")
        print(f"   Correctness: {correctness_scores[i-1]:.3f}")
        print(f"   Sources matched: {len(expected_sources & retrieved_sources)}/{len(expected_sources)}")
    
    # Category breakdown
    print("\n" + "=" * 80)
    print("CATEGORY BREAKDOWN")
    print("=" * 80)
    
    categories = {}
    for test, result, f, r, c in zip(TEST_CASES, results, faithfulness_scores, relevancy_scores, correctness_scores):
        cat = test['category']
        if cat not in categories:
            categories[cat] = {"faithfulness": [], "relevancy": [], "correctness": []}
        categories[cat]["faithfulness"].append(f)
        categories[cat]["relevancy"].append(r)
        categories[cat]["correctness"].append(c)
    
    for cat, scores in categories.items():
        avg_f = sum(scores["faithfulness"]) / len(scores["faithfulness"])
        avg_r = sum(scores["relevancy"]) / len(scores["relevancy"])
        avg_c = sum(scores["correctness"]) / len(scores["correctness"])
        print(f"\n   {cat}:")
        print(f"      Faithfulness: {avg_f:.3f}")
        print(f"      Relevancy: {avg_r:.3f}")
        print(f"      Correctness: {avg_c:.3f}")
    
    # Save report
    full_report = {
        "timestamp": datetime.now().isoformat(),
        "num_test_cases": len(TEST_CASES),
        "metrics_summary": metrics_summary,
        "overall_score": overall_score,
        "retrieval_metrics": retrieval_metrics,
        "category_breakdown": {
            cat: {
                "avg_faithfulness": sum(s["faithfulness"]) / len(s["faithfulness"]),
                "avg_relevancy": sum(s["relevancy"]) / len(s["relevancy"]),
                "avg_correctness": sum(s["correctness"]) / len(s["correctness"]),
                "num_tests": len(s["faithfulness"])
            }
            for cat, s in categories.items()
        },
        "detailed_results": detailed_results
    }
    
    if save_report:
        report_path = f"deepeval_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(full_report, f, ensure_ascii=False, indent=2)
        print(f"\n✓ Report saved to: {report_path}")
    
    return full_report


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Check prerequisites
    if not os.path.exists("../vectorstore/faiss_index"):
        print("ERROR: Vectorstore not found. Run embedding_setup.py first.")
        exit(1)
    
    if not DEEPSEEK_API_KEY or DEEPSEEK_API_KEY.startswith("sk-..."):
        print("ERROR: Please set your actual DeepSeek API key in the script.")
        exit(1)
    
    # Run evaluation
    report = run_evaluation(save_report=True)
    
    # Print quick summary
    print("\n" + "=" * 80)
    print("QUICK SUMMARY")
    print("=" * 80)
    print(f"Test cases: {report['num_test_cases']}")
    print(f"Overall score: {report['overall_score']:.4f}")
    print(f"Faithfulness: {report['metrics_summary']['faithfulness']:.4f}")
    print(f"Answer Relevancy: {report['metrics_summary']['answer_relevancy']:.4f}")
    print(f"Answer Correctness: {report['metrics_summary']['answer_correctness']:.4f}")
    print(f"Source Recall: {report['retrieval_metrics']['source_recall']:.2%}")
