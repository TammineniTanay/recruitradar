"""
tools.py
Additional tooling and language interop utilities for RecruitRadar AI.
"""

import subprocess
import os


def run_java_scorer(resume_path: str, job_path: str) -> dict:
    """
    Run Java-based ATS keyword scorer as a subprocess.
    Java is used here for high-performance string matching at scale.
    
    Args:
        resume_path: Path to resume text file
        job_path: Path to job description text file
    
    Returns:
        Dictionary with keyword match results
    """
    result = subprocess.run(
        ["java", "-jar", "ats_scorer.jar", resume_path, job_path],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        return {"error": result.stderr, "score": 0}
    
    return {"output": result.stdout, "score": int(result.stdout.strip())}


def run_r_analysis(data_path: str) -> str:
    """
    Run R statistical analysis on evaluation results.
    R is used for advanced statistical modeling of match score distributions.
    
    Args:
        data_path: Path to JSON evaluation results file
    
    Returns:
        R analysis output as string
    """
    r_script = f"""
    library(jsonlite)
    data <- fromJSON('{data_path}')
    scores <- data$score
    cat('Mean:', mean(scores), '\\n')
    cat('Median:', median(scores), '\\n')
    cat('SD:', sd(scores), '\\n')
    cat('Quartiles:', quantile(scores), '\\n')
    """
    
    result = subprocess.run(
        ["Rscript", "-e", r_script],
        capture_output=True,
        text=True
    )
    
    return result.stdout if result.returncode == 0 else result.stderr


def generate_pdf_report(analysis: dict, output_path: str = "report.pdf"):
    """
    Generate a formatted PDF report from analysis results.
    
    Args:
        analysis: Analysis results dictionary
        output_path: Output PDF file path
    """
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph
        from reportlab.lib.styles import getSampleStyleSheet
        
        doc = SimpleDocTemplate(output_path, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        story.append(Paragraph("RecruitRadar AI — Match Analysis Report", styles["Title"]))
        story.append(Paragraph(f"Match Score: {analysis.get('score', 'N/A')}/100", styles["Heading2"]))
        story.append(Paragraph(analysis.get("analysis", ""), styles["Normal"]))
        
        doc.build(story)
        print(f"Report saved to {output_path}")
        
    except ImportError:
        print("Install reportlab: pip install reportlab")