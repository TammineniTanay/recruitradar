"""
integrations.py
Cloud and database integration utilities for RecruitRadar AI.
Demonstrates connectivity with AWS services, MongoDB, Redis, and GCP.
"""

import os
from dotenv import load_dotenv

load_dotenv()


# ── AWS S3 ────────────────────────────────────────────────────
def upload_results_to_s3(results: dict, bucket: str, key: str):
    """
    Upload analysis results to AWS S3 for persistent storage.
    
    Args:
        results: Analysis results dictionary
        bucket: S3 bucket name
        key: S3 object key
    """
    import boto3
    import json
    
    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION", "us-east-1")
    )
    
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(results, indent=2),
        ContentType="application/json"
    )
    print(f"Results uploaded to s3://{bucket}/{key}")


# ── AWS DynamoDB ──────────────────────────────────────────────
def save_analysis_to_dynamodb(user_id: str, job_title: str, score: int):
    """
    Save match analysis result to DynamoDB for history tracking.
    
    Args:
        user_id: Unique user identifier
        job_title: Job title analyzed
        score: Match score 0-100
    """
    import boto3
    from datetime import datetime
    
    dynamodb = boto3.resource("dynamodb", region_name=os.getenv("AWS_REGION", "us-east-1"))
    table = dynamodb.Table("recruitradar-analyses")
    
    table.put_item(Item={
        "user_id": user_id,
        "timestamp": datetime.utcnow().isoformat(),
        "job_title": job_title,
        "score": score
    })


# ── AWS Lambda ────────────────────────────────────────────────
def invoke_lambda_scorer(payload: dict) -> dict:
    """
    Invoke AWS Lambda function for serverless match scoring.
    Useful for high-volume batch processing without managing servers.
    
    Args:
        payload: Input payload with resume and job description
    
    Returns:
        Lambda response with analysis results
    """
    import boto3
    import json
    
    lambda_client = boto3.client("lambda", region_name=os.getenv("AWS_REGION", "us-east-1"))
    
    response = lambda_client.invoke(
        FunctionName="recruitradar-scorer",
        InvocationType="RequestResponse",
        Payload=json.dumps(payload)
    )
    
    return json.loads(response["Payload"].read())


# ── AWS Bedrock ───────────────────────────────────────────────
def analyze_with_bedrock(resume_text: str, job_description: str) -> str:
    """
    Alternative LLM backend using AWS Bedrock (Claude/Titan).
    Drop-in replacement for Groq when running in AWS infrastructure.
    
    Args:
        resume_text: Candidate resume text
        job_description: Job description text
    
    Returns:
        Analysis text from Bedrock LLM
    """
    import boto3
    import json
    
    bedrock = boto3.client("bedrock-runtime", region_name=os.getenv("AWS_REGION", "us-east-1"))
    
    prompt = f"""Analyze how well this candidate matches the job description.
    
Resume: {resume_text[:2000]}
Job Description: {job_description[:1000]}

Provide match score and reasoning."""
    
    response = bedrock.invoke_model(
        modelId="amazon.titan-text-express-v1",
        body=json.dumps({"inputText": prompt, "textGenerationConfig": {"maxTokenCount": 500}})
    )
    
    return json.loads(response["body"].read())["results"][0]["outputText"]


# ── MongoDB ───────────────────────────────────────────────────
def save_to_mongodb(analysis_result: dict):
    """
    Persist analysis results to MongoDB for long-term storage.
    MongoDB's flexible schema is ideal for varying analysis structures.
    
    Args:
        analysis_result: Complete analysis result dictionary
    """
    from pymongo import MongoClient
    
    client = MongoClient(os.getenv("MONGODB_URI", "mongodb://localhost:27017"))
    db = client["recruitradar"]
    collection = db["analyses"]
    
    collection.insert_one(analysis_result)
    client.close()


# ── Redis ─────────────────────────────────────────────────────
def cache_analysis(job_hash: str, result: dict, ttl: int = 3600):
    """
    Cache analysis results in Redis to avoid re-processing identical jobs.
    TTL of 1 hour by default — results stay fresh without wasting API calls.
    
    Args:
        job_hash: MD5 hash of job description (cache key)
        result: Analysis result to cache
        ttl: Time to live in seconds (default 1 hour)
    """
    import redis
    import json
    
    r = redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", 6379)),
        decode_responses=True
    )
    
    r.setex(job_hash, ttl, json.dumps(result))


def get_cached_analysis(job_hash: str) -> dict:
    """
    Retrieve cached analysis from Redis.
    
    Args:
        job_hash: MD5 hash of job description
    
    Returns:
        Cached result or None if not found
    """
    import redis
    import json
    
    r = redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", 6379)),
        decode_responses=True
    )
    
    cached = r.get(job_hash)
    return json.loads(cached) if cached else None


# ── GCP Vertex AI ─────────────────────────────────────────────
def analyze_with_vertex_ai(resume_text: str, job_description: str) -> str:
    """
    Alternative LLM backend using Google Cloud Vertex AI.
    Useful when deploying RecruitRadar on GCP infrastructure.
    
    Args:
        resume_text: Candidate resume text
        job_description: Job description text
    
    Returns:
        Analysis text from Vertex AI model
    """
    from google.cloud import aiplatform
    from vertexai.language_models import TextGenerationModel
    
    aiplatform.init(project=os.getenv("GCP_PROJECT_ID"), location="us-central1")
    
    model = TextGenerationModel.from_pretrained("text-bison@001")
    
    prompt = f"""Analyze candidate match for this job.
Resume: {resume_text[:1000]}
Job: {job_description[:500]}
Provide score 0-100 and reasoning."""
    
    response = model.predict(prompt, max_output_tokens=500, temperature=0.3)
    return response.text


# ── LlamaIndex ────────────────────────────────────────────────
def build_llamaindex_rag(documents: list):
    """
    Alternative RAG implementation using LlamaIndex.
    LlamaIndex provides higher-level abstractions compared to LangChain
    and is particularly strong for document-heavy use cases.
    
    Args:
        documents: List of document strings to index
    
    Returns:
        LlamaIndex query engine ready for retrieval
    """
    from llama_index.core import VectorStoreIndex, Document
    from llama_index.core.settings import Settings
    
    docs = [Document(text=d) for d in documents]
    index = VectorStoreIndex.from_documents(docs)
    
    return index.as_query_engine()


# ── Azure OpenAI ──────────────────────────────────────────────
def analyze_with_azure_openai(resume_text: str, job_description: str) -> str:
    """
    Alternative LLM backend using Azure OpenAI Service.
    Enterprise-grade option for organizations already on Azure.
    
    Args:
        resume_text: Candidate resume text
        job_description: Job description text
    
    Returns:
        Analysis text from Azure OpenAI
    """
    from openai import AzureOpenAI
    
    client = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        api_version="2024-02-01"
    )
    
    response = client.chat.completions.create(
        model=os.getenv("AZURE_DEPLOYMENT_NAME", "gpt-4"),
        messages=[
            {"role": "system", "content": "You are an expert technical recruiter."},
            {"role": "user", "content": f"Resume: {resume_text[:1000]}\nJob: {job_description[:500]}\nScore match 0-100."}
        ]
    )
    
    return response.choices[0].message.content