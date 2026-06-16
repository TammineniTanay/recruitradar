# app.py
# This is the main UI file - built with Streamlit
# Run it with: streamlit run app.py
# It ties together rag.py + agent.py + evaluator.py into one clean interface

import streamlit as st
from rag import initialize_rag, retrieve_relevant_chunks
from agent import analyze_match, parse_score
from evaluator import evaluate_pipeline

# ─── PAGE CONFIGURATION ───────────────────────────────────────
# This must be the first Streamlit command - sets browser tab title and layout
st.set_page_config(
    page_title="RecruitRadar AI",
    page_icon="🎯",
    layout="wide"  # Uses full browser width - better for demo
)

# ─── CUSTOM STYLING ───────────────────────────────────────────
# Makes the app look professional - judges notice good UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1F4E79;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #595959;
        text-align: center;
        margin-bottom: 2rem;
    }
    .score-box {
        background: linear-gradient(135deg, #1F4E79, #2E75B6);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        font-size: 3rem;
        font-weight: bold;
    }
    .metric-box {
        background: #F0F7FF;
        border-left: 4px solid #2E75B6;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ─── HEADER ───────────────────────────────────────────────────
st.markdown('<div class="main-header">🎯 RecruitRadar AI</div>', 
            unsafe_allow_html=True)
st.markdown('<div class="sub-header">Intelligent Candidate-Job Match Analysis powered by RAG + LLM</div>', 
            unsafe_allow_html=True)

# ─── SESSION STATE ────────────────────────────────────────────
# Session state persists data between interactions
# Without this, every button click would reset everything
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

# ─── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ System Controls")
    
    # Initialize RAG button - loads resume into vector store
    if st.button("🚀 Initialize RAG Pipeline", type="primary", use_container_width=True):
        with st.spinner("Loading resume and building vector store..."):
            try:
                st.session_state.vector_store = initialize_rag()
                st.success("✅ RAG Pipeline Ready!")
                st.info("Resume loaded and indexed. Ready to analyze jobs.")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    # Show status
    if st.session_state.vector_store:
        st.markdown("**Pipeline Status:** 🟢 Active")
    else:
        st.markdown("**Pipeline Status:** 🔴 Not initialized")
    
    st.divider()
    
    # Settings
    st.subheader("🔧 Settings")
    num_chunks = st.slider(
        "Chunks to retrieve", 
        min_value=3, 
        max_value=10, 
        value=5,
        help="How many resume sections to retrieve per query"
    )
    run_evaluation = st.checkbox(
        "Run quality evaluation", 
        value=True,
        help="Measures retrieval and coverage quality (takes extra 10-15 seconds)"
    )

# ─── MAIN CONTENT ─────────────────────────────────────────────
# Two columns - left for input, right for results
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📋 Job Description Input")
    
    job_description = st.text_area(
        "Paste the job description here:",
        height=300,
        placeholder="""Example:
We are looking for a Senior AI/ML Engineer to join our team.

Requirements:
- 3+ years experience with Python and ML frameworks
- Experience with LLMs, RAG systems, or vector databases
- Knowledge of LangChain, LangGraph, or similar frameworks
- Experience deploying models to production (AWS, Docker)
- Strong understanding of transformer architectures

Nice to have:
- Experience with fine-tuning LLMs
- Publications or open source contributions
- FastAPI or similar backend frameworks"""
    )
    
    # Analyze button
    analyze_clicked = st.button(
        "🔍 Analyze Match", 
        type="primary", 
        use_container_width=True,
        disabled=not st.session_state.vector_store
    )
    
    if not st.session_state.vector_store:
        st.warning("⚠️ Initialize the RAG Pipeline first using the sidebar button")

with col2:
    st.subheader("📊 Analysis Results")
    
    # Run analysis when button clicked
    if analyze_clicked and job_description:
        
        with st.spinner("🔍 Retrieving relevant resume sections..."):
            chunks = retrieve_relevant_chunks(
                st.session_state.vector_store, 
                job_description, 
                k=num_chunks
            )
        
        with st.spinner("🤖 AI agent analyzing match..."):
            analysis = analyze_match(job_description, chunks)
            score = parse_score(analysis)
        
        # ── MATCH SCORE ──
        st.markdown("### 🎯 Match Score")
        
        # Color the score based on value
        if score >= 75:
            score_color = "#1E7145"  # Green for high scores
            verdict = "Strong Match ✅"
        elif score >= 50:
            score_color = "#C55A11"  # Orange for medium
            verdict = "Moderate Match ⚠️"
        else:
            score_color = "#C00000"  # Red for low
            verdict = "Weak Match ❌"
        
        # Display score prominently
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, {score_color}, #2E75B6);
                    color: white; padding: 1.5rem; border-radius: 10px;
                    text-align: center; margin-bottom: 1rem;">
            <div style="font-size: 4rem; font-weight: bold;">{score}</div>
            <div style="font-size: 1.2rem;">out of 100</div>
            <div style="font-size: 1rem; margin-top: 0.5rem;">{verdict}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # ── FULL ANALYSIS ──
        st.markdown("### 📝 Detailed Analysis")
        st.markdown(analysis)
        
        # ── RETRIEVED CHUNKS ──
        with st.expander("🔍 View Retrieved Resume Sections"):
            for i, chunk in enumerate(chunks):
                st.markdown(f"**Section {i+1}:**")
                st.text(chunk.page_content)
                st.divider()
        
        # ── EVALUATION METRICS ──
        if run_evaluation:
            with st.spinner("📊 Running quality evaluation..."):
                eval_report = evaluate_pipeline(
                    job_description, chunks, analysis
                )
            
            st.markdown("### 📈 Pipeline Quality Metrics")
            st.caption("These metrics measure how well the RAG system performed")
            
            # Display metrics in 3 columns
            m1, m2, m3 = st.columns(3)
            
            with m1:
                st.metric(
                    label="Retrieval Quality",
                    value=f"{eval_report['retrieval_quality']}/10",
                    help="How relevant were the retrieved resume sections?"
                )
            with m2:
                st.metric(
                    label="Coverage Score", 
                    value=f"{eval_report['coverage_score']}/10",
                    help="How well did the analysis cover job requirements?"
                )
            with m3:
                st.metric(
                    label="Overall Quality",
                    value=f"{eval_report['overall_quality']}/10",
                    help="Combined pipeline quality score"
                )
            
            # Individual chunk scores
            with st.expander("📊 Individual Chunk Relevance Scores"):
                for i, score_val in enumerate(eval_report['individual_chunk_scores']):
                    st.progress(
                        score_val/10, 
                        text=f"Section {i+1}: {score_val}/10"
                    )
        
        st.session_state.analysis_done = True

    elif analyze_clicked and not job_description:
        st.warning("Please paste a job description first")
    
    # Show placeholder when nothing has been analyzed yet
    if not st.session_state.analysis_done and not analyze_clicked:
        st.info("👈 Initialize the pipeline, paste a job description, then click Analyze Match")

# ─── FOOTER ───────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style="text-align: center; color: #595959; font-size: 0.85rem;">
    RecruitRadar AI — Built with LangChain + Qdrant + Groq (Llama 3) + Streamlit<br>
    RAG Pipeline | Agent Orchestration | Quality Evaluation
</div>
""", unsafe_allow_html=True)