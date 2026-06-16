```markdown
# 🎯 RecruitRadar AI

An AI system that reads your resume and tells you how well you match a job — with a score, detailed reasoning, and quality metrics.

Built at Buildathon Dallas 2026.

## The Problem

Applying to jobs is frustrating. You spend hours tailoring your resume and never hear back. Most of the time you don't even know why. RecruitRadar gives you an honest answer before you apply.

## What It Does

Paste any job description. The system reads your resume, finds the most relevant sections, and gives you:

- A match score from 0 to 100
- What makes you a strong fit
- What skills or experience you're missing
- Suggested interview questions based on your background
- A quality score measuring how well the analysis performed

## How It Works

I built this using a RAG pipeline — Retrieval Augmented Generation. Instead of asking an AI what it thinks about you from memory, the system first retrieves the most relevant parts of your resume, then generates the analysis based on that actual content.

The pipeline runs like this:

1. Your resume PDF gets split into small overlapping chunks
2. Each chunk gets converted into a vector using a sentence embedding model
3. Vectors get stored in Qdrant — an in-memory vector database
4. When you paste a job description, it gets vectorized too
5. Qdrant finds the top 5 most semantically similar resume sections
6. Those sections plus the job description go to Llama 3.3 70B via Groq
7. The LLM scores the match and gives structured reasoning
8. A custom evaluator measures retrieval quality and coverage

## Tech Stack

- **LangChain** — RAG pipeline orchestration
- **Qdrant** — vector storage and similarity search
- **sentence-transformers/all-MiniLM-L6-v2** — text embeddings
- **Llama 3.3 70B via Groq** — LLM scoring and reasoning
- **PyPDF** — resume text extraction
- **Streamlit** — UI
- **Custom evaluator** — RAGAS-inspired retrieval and coverage metrics

## Running It Locally

```bash
git clone https://github.com/TammineniTanay/recruitradar.git
cd recruitradar
pip install -r requirements.txt
```

Create a `.env` file:

```
GROQ_API_KEY=your_key_here
```

Add your resume as `resume.pdf` in the root folder, then:

```bash
streamlit run app.py
```

## Project Structure

```
recruitradar/
├── app.py          — Streamlit UI
├── rag.py          — PDF loading, chunking, embedding, retrieval
├── agent.py        — LLM match scoring and reasoning
├── evaluator.py    — Retrieval quality and coverage metrics
├── requirements.txt
└── README.md
```

## Evaluation

Most RAG demos ship with no way to measure if retrieval actually worked. I built a lightweight evaluator that scores two things independently:

**Retrieval Quality** — were the chunks we pulled actually relevant to the job description?

**Coverage Score** — did the final analysis actually address what the job was asking for?

Both scores run through the same Groq LLM at zero extra cost, inspired by how RAGAS measures RAG pipelines in production.

## Hackathon Tracks

- RAG & Retrieval
- Agent Orchestration
- Evals & Testing

## Author

Tanay Tammineni
- [GitHub](https://github.com/TammineniTanay)
- [LinkedIn](https://linkedin.com/in/tanay-tammineni)
- [Portfolio](https://tanaytammineni.vercel.app)
```