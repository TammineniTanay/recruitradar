# agent.py
# This file does 3 things:
# 1. Takes the retrieved resume chunks + job description
# 2. Uses Groq LLM to score the match (0-100)
# 3. Returns score + reasoning + missing skills

import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# Initialize Groq client
# Groq gives us access to Llama 3 - same model mentioned in your resume
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def format_chunks(chunks):
    """
    Takes the list of retrieved chunks and combines them into one readable text.
    Each chunk becomes a numbered section.
    """
    formatted = ""
    for i, chunk in enumerate(chunks):
        formatted += f"\n--- Resume Section {i+1} ---\n"
        formatted += chunk.page_content
    return formatted

def analyze_match(job_description, retrieved_chunks):
    """
    The core agent function.
    
    Sends the job description + relevant resume sections to Groq LLM.
    The LLM acts as an expert recruiter and returns:
    - A match score from 0-100
    - Why the candidate is a good fit
    - What skills are missing
    - A recommendation
    
    We use a structured prompt so the output is always consistent.
    """
    
    resume_context = format_chunks(retrieved_chunks)
    
    # This is the prompt - instructions we give the AI
    # We tell it exactly what role to play and what format to return
    prompt = f"""You are an expert technical recruiter with 10 years of experience evaluating AI/ML engineers.

Analyze how well this candidate matches the job description.

JOB DESCRIPTION:
{job_description}

RELEVANT RESUME SECTIONS:
{resume_context}

Provide your analysis in EXACTLY this format:

MATCH_SCORE: [number 0-100]

STRENGTHS:
- [strength 1]
- [strength 2]
- [strength 3]

GAPS:
- [missing skill or experience 1]
- [missing skill or experience 2]

RECOMMENDATION:
[2-3 sentences on whether to proceed with this candidate and why]

INTERVIEW_QUESTIONS:
- [suggested question 1 based on their background]
- [suggested question 2 based on their background]
- [suggested question 3 based on their background]
"""

    # Send to Groq API - using Llama 3 70B for best quality
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system", 
                "content": "You are an expert technical recruiter. Always respond in the exact format requested."
            },
            {
                "role": "user", 
                "content": prompt
            }
        ],
        temperature=0.3,  # Low temperature = more consistent, focused responses
        max_tokens=1000
    )
    
    return response.choices[0].message.content

def parse_score(analysis_text):
    """
    Extracts just the number from the MATCH_SCORE line.
    Example: "MATCH_SCORE: 87" → 87
    """
    for line in analysis_text.split('\n'):
        if 'MATCH_SCORE:' in line:
            try:
                score = int(line.split(':')[1].strip())
                return min(100, max(0, score))  # Ensure between 0-100
            except:
                return 0
    return 0