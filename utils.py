"""
utils.py
Shared utility functions for RecruitRadar AI pipeline.
"""

import re
import time
import functools
from typing import List, Dict, Any


def timer(func):
    """
    Decorator that measures and prints function execution time.
    Use on any pipeline function to track performance.
    
    Example:
        @timer
        def initialize_rag():
            ...
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = round(time.time() - start, 2)
        print(f"⏱️  {func.__name__} completed in {elapsed}s")
        return result
    return wrapper


def clean_text(text: str) -> str:
    """
    Clean extracted PDF text by removing noise and normalizing whitespace.
    
    Args:
        text: Raw text extracted from PDF
    
    Returns:
        Cleaned text ready for chunking
    """
    # Remove excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    
    # Remove page numbers (common in PDFs)
    text = re.sub(r'\n\d+\n', '\n', text)
    
    # Remove common PDF artifacts
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    
    return text.strip()


def extract_keywords(text: str) -> List[str]:
    """
    Extract meaningful keywords from text for ATS analysis.
    Filters out common stop words to keep only meaningful terms.
    
    Args:
        text: Job description or resume text
    
    Returns:
        List of unique meaningful keywords
    """
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
        'for', 'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were',
        'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
        'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can',
        'this', 'that', 'these', 'those', 'i', 'you', 'we', 'they', 'it'
    }
    
    # Extract words (letters and numbers only)
    words = re.findall(r'\b[a-zA-Z][a-zA-Z0-9+#.]*\b', text.lower())
    
    # Filter stop words and short words
    keywords = [w for w in words if w not in stop_words and len(w) > 2]
    
    # Return unique keywords preserving order
    seen = set()
    unique = []
    for kw in keywords:
        if kw not in seen:
            seen.add(kw)
            unique.append(kw)
    
    return unique


def find_keyword_gaps(resume_text: str, job_description: str) -> Dict[str, Any]:
    """
    Compare resume keywords against job description to find gaps.
    This is the core ATS gap analysis function.
    
    Args:
        resume_text: Full resume text
        job_description: Job description text
    
    Returns:
        Dict with matched keywords, missing keywords, and match percentage
    """
    resume_keywords = set(extract_keywords(resume_text))
    jd_keywords = set(extract_keywords(job_description))
    
    matched = resume_keywords.intersection(jd_keywords)
    missing = jd_keywords.difference(resume_keywords)
    
    match_pct = round(len(matched) / len(jd_keywords) * 100, 1) if jd_keywords else 0
    
    return {
        "matched_keywords": sorted(list(matched)),
        "missing_keywords": sorted(list(missing)),
        "match_percentage": match_pct,
        "total_jd_keywords": len(jd_keywords),
        "total_matched": len(matched)
    }


def format_score_label(score: int) -> str:
    """
    Convert numeric match score to human readable label with emoji.
    
    Args:
        score: Match score 0-100
    
    Returns:
        Formatted label string
    """
    if score >= 75:
        return f"Strong Match ✅ ({score}/100)"
    elif score >= 50:
        return f"Moderate Match ⚠️ ({score}/100)"
    else:
        return f"Weak Match ❌ ({score}/100)"


def truncate_text(text: str, max_chars: int = 500) -> str:
    """
    Truncate text to max_chars while preserving word boundaries.
    
    Args:
        text: Text to truncate
        max_chars: Maximum character count
    
    Returns:
        Truncated text with ellipsis if needed
    """
    if len(text) <= max_chars:
        return text
    
    truncated = text[:max_chars]
    last_space = truncated.rfind(' ')
    
    if last_space > 0:
        truncated = truncated[:last_space]
    
    return truncated + "..."