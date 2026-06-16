# evaluator.py
# This file measures HOW GOOD our RAG retrieval was.
# It answers: "Did we retrieve the RIGHT resume sections?"
# This is what RAGAS does professionally - we're building a simple version.

import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def calculate_relevance_score(job_description, chunks):
    """
    Measures how relevant the retrieved chunks are to the job description.
    
    For each chunk, we ask: on a scale of 0-10, how relevant is this
    resume section to what the job is asking for?
    
    Then we average all scores to get overall retrieval quality.
    This is similar to what RAGAS calls 'context precision'.
    """
    scores = []
    
    for i, chunk in enumerate(chunks):
        prompt = f"""Rate how relevant this resume section is to the job description.
        
Job Description: {job_description[:500]}

Resume Section: {chunk.page_content}

Reply with ONLY a number from 0-10.
0 = completely irrelevant
10 = perfectly relevant

Your rating:"""

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,      # Zero temperature = most consistent scoring
            max_tokens=5        # We only need a single number back
        )
        
        try:
            score = float(response.choices[0].message.content.strip())
            score = min(10, max(0, score))  # Keep between 0-10
            scores.append(score)
        except:
            scores.append(5.0)  # Default middle score if parsing fails
    
    avg_score = sum(scores) / len(scores) if scores else 0
    return round(avg_score, 2), scores

def calculate_coverage_score(job_description, analysis_text):
    """
    Measures how well the final analysis covers the job requirements.
    
    We ask: did the AI's analysis actually address what the job needs?
    This is similar to what RAGAS calls 'answer relevancy'.
    """
    prompt = f"""Rate how well this candidate analysis addresses the job requirements.

Job Description: {job_description[:500]}

Analysis: {analysis_text[:800]}

Reply with ONLY a number from 0-10.
0 = analysis completely ignores job requirements  
10 = analysis perfectly addresses all job requirements

Your rating:"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=5
    )
    
    try:
        score = float(response.choices[0].message.content.strip())
        return round(min(10, max(0, score)), 2)
    except:
        return 5.0

def evaluate_pipeline(job_description, chunks, analysis_text):
    """
    Master evaluation function - runs all metrics and returns a report.
    
    This is what we show on the dashboard:
    - Retrieval Quality: did we find the right resume sections?
    - Coverage Score: did the analysis address the job well?
    - Overall Quality: average of both
    
    Having these metrics is what separates a professional RAG system
    from a basic chatbot. This is exactly what RAGAS measures in production.
    """
    print("📊 Running pipeline evaluation...")
    
    # Metric 1: Retrieval Quality (Context Precision)
    retrieval_score, individual_scores = calculate_relevance_score(
        job_description, chunks
    )
    
    # Metric 2: Coverage Score (Answer Relevancy)  
    coverage_score = calculate_coverage_score(job_description, analysis_text)
    
    # Overall quality = average of both metrics
    overall_score = round((retrieval_score + coverage_score) / 2, 2)
    
    report = {
        "retrieval_quality": retrieval_score,      # 0-10: how good was retrieval?
        "coverage_score": coverage_score,           # 0-10: how well did we cover JD?
        "overall_quality": overall_score,           # 0-10: combined score
        "chunks_evaluated": len(chunks),            # how many chunks we retrieved
        "individual_chunk_scores": individual_scores # score for each chunk
    }
    
    print(f"✅ Retrieval Quality: {retrieval_score}/10")
    print(f"✅ Coverage Score: {coverage_score}/10")
    print(f"✅ Overall Quality: {overall_score}/10")
    
    return report