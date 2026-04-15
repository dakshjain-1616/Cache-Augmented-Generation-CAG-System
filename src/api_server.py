#!/usr/bin/env python3
"""
Cache-Augmented Generation (CAG) - FastAPI Server

Provides REST API endpoints for:
- POST /ingest - Ingest documents into KV cache
- GET /status/{job_id} - Check ingestion status
- POST /query - Query a corpus
- GET /corpora - List available corpora
- GET /health - Health check

Usage:
    python api_server.py [--port 8000] [--host 0.0.0.0]
"""

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from contextlib import asynccontextmanager

import asyncio
import threading
import requests
import httpx
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Security
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
import uvicorn


# Global state
app_state = {
    "jobs": {},
    "project_dir": None,
    "env": {},
    "llama_server_url": None,
    "kv_slots_dir": None
}
jobs_lock = threading.Lock()   # protects app_state["jobs"] across background threads
slot_lock = None               # asyncio.Lock — initialized in lifespan (event-loop bound)

# API key auth — disabled if CAG_API_KEY env var is not set
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_api_key(api_key: str = Security(API_KEY_HEADER)):
    expected = os.environ.get("CAG_API_KEY")
    if expected and api_key != expected:
        raise HTTPException(status_code=403, detail="Invalid or missing API key")
    return api_key


# Pydantic models
class IngestRequest(BaseModel):
    documents: List[str] = Field(..., description="List of document contents to ingest")
    corpus_id: Optional[str] = Field(None, description="Unique identifier for this corpus")
    description: Optional[str] = Field(None, description="Description of the corpus")


class IngestResponse(BaseModel):
    job_id: str
    status: str
    corpus_id: str
    message: str


class JobStatus(BaseModel):
    job_id: str
    status: str  # pending, processing, completed, failed
    corpus_id: Optional[str] = None
    result: Optional[Dict] = None
    error: Optional[str] = None
    created_at: str
    updated_at: str


class QueryRequest(BaseModel):
    corpus_id: str = Field(..., description="Corpus ID to query")
    question: str = Field(..., description="Question to ask")
    max_tokens: int = Field(512, description="Maximum tokens to generate")
    temperature: float = Field(0.7, description="Sampling temperature")
    stream: bool = Field(True, description="Stream the response")


class QueryResponse(BaseModel):
    corpus_id: str
    question: str
    answer: str
    tokens_generated: int
    restore_time_ms: float
    decode_time_s: float
    tokens_per_sec: float
    ms_per_token: float
    timestamp: str


class CorpusInfo(BaseModel):
    corpus_id: str
    description: Optional[str]
    slot_id: str
    prompt_length: int
    prefill_ms: float
    total_time_s: float
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    llama_server: str
    timestamp: str


# Utility functions
def load_env(project_dir: Path) -> Dict[str, str]:
    """Load environment variables from .env file."""
    env = {}
    env_path = project_dir / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    env[key] = value
    return env


def load_manifest(project_dir: Path) -> Dict:
    """Load or create manifest.json."""
    manifest_path = project_dir / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            return json.load(f)
    return {"corpora": [], "version": "1.0"}


def save_manifest(project_dir: Path, manifest: Dict):
    """Save manifest.json."""
    manifest_path = project_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)


def load_jobs(project_dir: Path) -> Dict:
    """Load persisted jobs from disk."""
    jobs_path = project_dir / "jobs.json"
    if jobs_path.exists():
        with open(jobs_path) as f:
            return json.load(f)
    return {}


def save_jobs(project_dir: Path, jobs: Dict):
    """Persist jobs to disk so they survive restarts."""
    jobs_path = project_dir / "jobs.json"
    with open(jobs_path, "w") as f:
        json.dump(jobs, f, indent=2)


def create_ingestion_prompt(documents: list, corpus_id: str, description: Optional[str]) -> str:
    """Create a structured prompt for document ingestion."""
    prompt_parts = [
        "<|im_start|>system",
        "You are a helpful assistant with access to the following documents. "
        "These documents have been pre-loaded into your context and you should "
        "reference them when answering questions.",
        "",
        f"<|im_start|>user",
        f"Document Corpus ID: {corpus_id}",
    ]
    
    if description:
        prompt_parts.append(f"Description: {description}")
    
    prompt_parts.append("")
    prompt_parts.append("Documents:")
    prompt_parts.append("=" * 40)
    
    for i, doc in enumerate(documents, 1):
        prompt_parts.append(f"\n--- Document {i} ---")
        prompt_parts.append(doc)
    
    prompt_parts.append("=" * 40)
    prompt_parts.append("")
    prompt_parts.append("Please acknowledge that you have received these documents "
                       "and are ready to answer questions about them.")
    prompt_parts.append("")
    prompt_parts.append("<|im_start|>assistant")
    prompt_parts.append("I have received and processed the documents. "
                       "I am ready to answer questions about the content.")
    
    return "\n".join(prompt_parts)


def check_llama_server_health(url: str) -> bool:
    """Check if llama-server is healthy."""
    try:
        response = requests.get(f"{url}/health", timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False


# Background task for ingestion
def ingest_documents_task(
    job_id: str,
    server_url: str,
    kv_slots_dir: Path,
    project_dir: Path,
    documents: List[str],
    corpus_id: str,
    description: Optional[str]
):
    """Background task to ingest documents."""
    with jobs_lock:
        app_state["jobs"][job_id]["status"] = "processing"
        app_state["jobs"][job_id]["updated_at"] = datetime.now().isoformat()
        save_jobs(project_dir, app_state["jobs"])
    
    try:
        # Create the ingestion prompt
        prompt = create_ingestion_prompt(documents, corpus_id, description)
        slot_id = 0  # llama.cpp uses integer slot IDs; CAG uses slot 0

        # Send to llama-server
        payload = {
            "prompt": prompt,
            "id_slot": slot_id,
            "n_predict": 100,
            "temperature": 0.0,
            "stop": ["<|im_end|>"],
            "stream": False,
            "cache_prompt": True,
        }
        
        start_time = time.time()
        response = requests.post(
            f"{server_url}/completion",
            json=payload,
            timeout=300
        )
        response.raise_for_status()
        
        result = response.json()
        elapsed = time.time() - start_time
        timings = result.get("timings", {})
        prompt_ms = timings.get("prompt_ms", 0)
        
        # Save the KV slot
        slot_file = f"{corpus_id}_{slot_id}.slot"
        save_payload = {
            "filename": f"{corpus_id}.bin"
        }

        save_response = requests.post(
            f"{server_url}/slots/0?action=save",
            json=save_payload,
            timeout=60
        )
        
        save_success = save_response.status_code == 200
        
        # Update manifest
        manifest_entry = {
            "slot_id": slot_id,
            "corpus_id": corpus_id,
            "description": description or "",
            "prompt_length": len(prompt),
            "prefill_ms": prompt_ms,
            "total_time_s": elapsed,
            "timestamp": datetime.now().isoformat(),
            "slot_file": slot_file
        }
        
        manifest = load_manifest(project_dir)
        manifest["corpora"].append(manifest_entry)
        save_manifest(project_dir, manifest)
        
        # Update job status
        with jobs_lock:
            app_state["jobs"][job_id]["status"] = "completed"
            app_state["jobs"][job_id]["result"] = manifest_entry
            app_state["jobs"][job_id]["updated_at"] = datetime.now().isoformat()
            save_jobs(project_dir, app_state["jobs"])

    except Exception as e:
        with jobs_lock:
            app_state["jobs"][job_id]["status"] = "failed"
            app_state["jobs"][job_id]["error"] = str(e)
            app_state["jobs"][job_id]["updated_at"] = datetime.now().isoformat()
            save_jobs(project_dir, app_state["jobs"])


# FastAPI app
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    global slot_lock
    slot_lock = asyncio.Lock()   # must be created inside the running event loop

    script_dir = Path(__file__).parent.parent.resolve()
    app_state["project_dir"] = script_dir
    app_state["env"] = load_env(script_dir)
    llama_host = os.environ.get("LLAMA_SERVER_HOST", "localhost")
    app_state["llama_server_url"] = f"http://{llama_host}:{app_state['env'].get('LLAMA_CPP_PORT', '8080')}"
    app_state["kv_slots_dir"] = Path(app_state["env"].get("KV_SLOTS_DIR", script_dir / "kv_slots"))
    app_state["jobs"] = load_jobs(script_dir)   # restore jobs from disk on startup
    yield


app = FastAPI(
    title="CAG API",
    description="Cache-Augmented Generation API Server",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# API Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    llama_url = app_state["llama_server_url"]
    llama_status = "healthy" if check_llama_server_health(llama_url) else "unavailable"
    
    return HealthResponse(
        status="healthy",
        llama_server=llama_status,
        timestamp=datetime.now().isoformat()
    )


@app.post("/ingest", response_model=IngestResponse, dependencies=[Depends(verify_api_key)])
async def ingest_documents(
    request: IngestRequest,
    background_tasks: BackgroundTasks
):
    """Ingest documents into KV cache."""
    # Check llama-server
    if not check_llama_server_health(app_state["llama_server_url"]):
        raise HTTPException(status_code=503, detail="llama-server not available")
    
    # Generate job ID and corpus ID
    job_id = str(uuid.uuid4())
    corpus_id = request.corpus_id or f"corpus_{uuid.uuid4().hex[:8]}"
    
    # Create job entry
    now = datetime.now().isoformat()
    with jobs_lock:
        app_state["jobs"][job_id] = {
            "job_id": job_id,
            "status": "pending",
            "corpus_id": corpus_id,
            "created_at": now,
            "updated_at": now
        }
        save_jobs(app_state["project_dir"], app_state["jobs"])
    
    # Start background task
    background_tasks.add_task(
        ingest_documents_task,
        job_id=job_id,
        server_url=app_state["llama_server_url"],
        kv_slots_dir=app_state["kv_slots_dir"],
        project_dir=app_state["project_dir"],
        documents=request.documents,
        corpus_id=corpus_id,
        description=request.description
    )
    
    return IngestResponse(
        job_id=job_id,
        status="pending",
        corpus_id=corpus_id,
        message="Ingestion started"
    )


@app.get("/status/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get ingestion job status."""
    with jobs_lock:
        if job_id not in app_state["jobs"]:
            raise HTTPException(status_code=404, detail="Job not found")
        job = dict(app_state["jobs"][job_id])
    return JobStatus(**job)


@app.post("/query", dependencies=[Depends(verify_api_key)])
async def query_corpus(request: QueryRequest):
    """Query a corpus."""
    # Check llama-server
    if not check_llama_server_health(app_state["llama_server_url"]):
        raise HTTPException(status_code=503, detail="llama-server not available")
    
    # Find corpus
    manifest = load_manifest(app_state["project_dir"])
    corpus = None
    for c in manifest.get("corpora", []):
        if c.get("corpus_id") == request.corpus_id:
            corpus = c
            break
    
    if not corpus:
        raise HTTPException(status_code=404, detail=f"Corpus not found: {request.corpus_id}")
    
    slot_id = corpus.get("slot_id")
    slot_file = corpus.get("slot_file")
    
    if slot_id is None:
        raise HTTPException(status_code=500, detail="Corpus missing slot_id")
    
    # Restore slot
    restore_start = time.time()
    
    try:
        # Always restore slot from disk before querying
        if slot_file:
            slot_path = app_state["kv_slots_dir"] / Path(slot_file).name
            if slot_path.exists():
                # filename must be relative to the slot-save-path directory
                restore_payload = {
                    "filename": Path(slot_file).name
                }

                restore_response = requests.post(
                    f"{app_state['llama_server_url']}/slots/0?action=restore",
                    json=restore_payload,
                    timeout=120
                )

                if restore_response.status_code != 200:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Failed to restore slot: {restore_response.text}"
                    )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error restoring slot: {str(e)}")
    
    restore_time = time.time() - restore_start
    
    # Prepare query
    prompt = f"<|im_start|>user\n{request.question}<|im_end|>\n<|im_start|>assistant\n"
    
    payload = {
        "prompt": prompt,
        "id_slot": 0,
        "n_predict": request.max_tokens,
        "temperature": request.temperature,
        "stream": request.stream,
        "stop": ["<|im_end|>", "<|im_start|>"],
        "cache_prompt": True,
    }
    
    if request.stream:
        async def stream_generator():
            async with slot_lock:
                decode_start = time.time()
                tokens_generated = 0
                async with httpx.AsyncClient(timeout=300) as client:
                    async with client.stream("POST", f"{app_state['llama_server_url']}/completion", json=payload) as response:
                        async for line in response.aiter_lines():
                            if line.startswith("data: "):
                                try:
                                    data = json.loads(line[6:])
                                    if "content" in data:
                                        tokens_generated += 1
                                        yield f"data: {json.dumps({'content': data['content']})}\n\n"
                                    if data.get("stop"):
                                        decode_time = time.time() - decode_start
                                        tokens_per_sec = tokens_generated / decode_time if decode_time > 0 else 0
                                        ms_per_token = (decode_time * 1000) / tokens_generated if tokens_generated > 0 else 0
                                        yield f"data: {json.dumps({'done': True, 'tokens_generated': tokens_generated, 'restore_time_ms': restore_time * 1000, 'decode_time_s': decode_time, 'tokens_per_sec': tokens_per_sec, 'ms_per_token': ms_per_token})}\n\n"
                                        break
                                except json.JSONDecodeError:
                                    continue

        return StreamingResponse(stream_generator(), media_type="text/event-stream")
    else:
        async with slot_lock:
            decode_start = time.time()
            async with httpx.AsyncClient(timeout=300) as client:
                response = await client.post(f"{app_state['llama_server_url']}/completion", json=payload)
            response.raise_for_status()
            result = response.json()
            decode_time = time.time() - decode_start

        content = result.get("content", "")
        tokens_generated = result.get("timings", {}).get("predicted_n", len(content.split()))
        tokens_per_sec = tokens_generated / decode_time if decode_time > 0 else 0
        ms_per_token = (decode_time * 1000) / tokens_generated if tokens_generated > 0 else 0

        return QueryResponse(
            corpus_id=request.corpus_id,
            question=request.question,
            answer=content,
            tokens_generated=tokens_generated,
            restore_time_ms=restore_time * 1000,
            decode_time_s=decode_time,
            tokens_per_sec=tokens_per_sec,
            ms_per_token=ms_per_token,
            timestamp=datetime.now().isoformat()
        )


@app.get("/corpora", response_model=List[CorpusInfo])
async def list_corpora():
    """List all available corpora."""
    manifest = load_manifest(app_state["project_dir"])
    corpora = []
    
    for c in manifest.get("corpora", []):
        corpora.append(CorpusInfo(
            corpus_id=c.get("corpus_id", ""),
            description=c.get("description"),
            slot_id=str(c.get("slot_id", "")),
            prompt_length=c.get("prompt_length", 0),
            prefill_ms=c.get("prefill_ms", 0),
            total_time_s=c.get("total_time_s", 0),
            timestamp=c.get("timestamp", "")
        ))
    
    return corpora


@app.delete("/corpora/{corpus_id}", dependencies=[Depends(verify_api_key)])
async def delete_corpus(corpus_id: str):
    """Delete a corpus — removes manifest entry and KV slot file."""
    manifest = load_manifest(app_state["project_dir"])
    remaining, deleted = [], None
    for c in manifest.get("corpora", []):
        if c.get("corpus_id") == corpus_id:
            deleted = c
        else:
            remaining.append(c)

    if deleted is None:
        raise HTTPException(status_code=404, detail=f"Corpus not found: {corpus_id}")

    manifest["corpora"] = remaining
    save_manifest(app_state["project_dir"], manifest)

    slot_file = deleted.get("slot_file")
    if slot_file:
        slot_path = app_state["kv_slots_dir"] / slot_file
        if slot_path.exists():
            slot_path.unlink()

    return {"message": f"Corpus '{corpus_id}' deleted", "corpus_id": corpus_id}


def main():
    parser = argparse.ArgumentParser(description="CAG API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    
    args = parser.parse_args()
    
    print(f"[api_server] Starting CAG API Server on {args.host}:{args.port}")
    print(f"[api_server] Documentation: http://{args.host}:{args.port}/docs")
    
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
