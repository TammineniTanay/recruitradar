"""
cli.py
Command-line interface for RecruitRadar AI.
Run analysis without the Streamlit UI — useful for batch processing.

Usage:
    python cli.py --resume resume.pdf --job "job description text"
    python cli.py --resume resume.pdf --job-file job.txt
    python cli.py --resume resume.pdf --job-file job.txt --no-eval
"""

import argparse
import sys
import os
from dotenv import load_dotenv

load_dotenv()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="RecruitRadar AI — CLI for candidate-job match analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py --resume resume.pdf --job "We are looking for a Python developer..."
  python cli.py --resume resume.pdf --job-file job_description.txt
  python cli.py --resume resume.pdf --job-file job.txt --top-k 3 --no-eval
        """
    )
    
    parser.add_argument(
        "--resume",
        type=str,
        default="resume.pdf",
        help="Path to resume PDF (default: resume.pdf)"
    )
    
    parser.add_argument(
        "--job",
        type=str,
        help="Job description as a string"
    )
    
    parser.add_argument(
        "--job-file",
        type=str,
        help="Path to text file containing job description"
    )
    
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of resume chunks to retrieve (default: 5)"
    )
    
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip pipeline quality evaluation (faster)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="Save results to JSON file"
    )
    
    return parser.parse_args()


def load_job_description(args) -> str:
    """Load job description from args or file."""
    if args.job:
        return args.job
    
    if args.job_file:
        if not os.path.exists(args.job_file):
            print(f"Error: Job file not found: {args.job_file}")
            sys.exit(1)
        with open(args.job_file, "r", encoding="utf-8") as f:
            return f.read()
    
    print("Error: Provide --job or --job-file")
    sys.exit(1)


def print_results(analysis: str, score: int, eval_report: dict = None):
    """Print analysis results in a clean format."""
    print("\n" + "=" * 60)
    print("RECRUITRADAR AI — ANALYSIS RESULTS")
    print("=" * 60)
    print(f"\nMATCH SCORE: {score}/100")
    
    if score >= 75:
        print("VERDICT: Strong Match ✅")
    elif score >= 50:
        print("VERDICT: Moderate Match ⚠️")
    else:
        print("VERDICT: Weak Match ❌")
    
    print("\n" + "-" * 60)
    print("DETAILED ANALYSIS:")
    print("-" * 60)
    print(analysis)
    
    if eval_report:
        print("\n" + "-" * 60)
        print("PIPELINE QUALITY METRICS:")
        print("-" * 60)
        print(f"Retrieval Quality : {eval_report['retrieval_quality']}/10")
        print(f"Coverage Score    : {eval_report['coverage_score']}/10")
        print(f"Overall Quality   : {eval_report['overall_quality']}/10")
    
    print("\n" + "=" * 60)


def main():
    """Main CLI entry point."""
    args = parse_args()
    
    # Validate resume exists
    if not os.path.exists(args.resume):
        print(f"Error: Resume not found: {args.resume}")
        sys.exit(1)
    
    # Validate API key
    if not os.getenv("GROQ_API_KEY"):
        print("Error: GROQ_API_KEY not set in .env file")
        sys.exit(1)
    
    # Load job description
    job_description = load_job_description(args)
    
    print("Initializing RAG pipeline...")
    from rag import initialize_rag, retrieve_relevant_chunks
    from agent import analyze_match, parse_score
    
    vector_store = initialize_rag()
    
    print("Retrieving relevant resume sections...")
    chunks = retrieve_relevant_chunks(vector_store, job_description, k=args.top_k)
    
    print("Analyzing match...")
    analysis = analyze_match(job_description, chunks)
    score = parse_score(analysis)
    
    eval_report = None
    if not args.no_eval:
        print("Running quality evaluation...")
        from evaluator import evaluate_pipeline
        eval_report = evaluate_pipeline(job_description, chunks, analysis)
    
    print_results(analysis, score, eval_report)
    
    if args.output:
        import json
        result = {
            "score": score,
            "analysis": analysis,
            "eval_report": eval_report
        }
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()