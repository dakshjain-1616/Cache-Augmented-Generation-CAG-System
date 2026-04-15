#!/usr/bin/env python3
"""
Cache-Augmented Generation (CAG) - Document Ingestion Pipeline

This script ingests documents into the KV cache by:
1. Reading documents from file or stdin
2. Concatenating into a structured prompt
3. POSTing to llama-server to prefill and generate KV cache
4. Saving the KV slot via API
5. Recording metadata to manifest.json

Usage:
    python ingest.py <document_path> [--corpus-id <id>] [--description <desc>]
    cat document.txt | python ingest.py --corpus-id my_doc
"""

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

import requests


def read_pdf(path) -> str:
    """Extract text from a PDF file."""
    try:
        import pypdf
        reader = pypdf.PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except ImportError:
        print(f"[ingest] ⚠ pypdf not installed — reading {path.name} as raw bytes")
        with open(path, "rb") as f:
            return f.read().decode("utf-8", errors="ignore")


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~3.5 chars per token for English."""
    return int(len(text) / 3.5)


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


def create_ingestion_prompt(documents: list, corpus_id: str, description: str) -> str:
    """
    Create a structured prompt for document ingestion.
    This prompt format optimizes the KV cache for later retrieval.
    """
    prompt_parts = [
        "<|im_start|>system\nYou are a helpful assistant with access to the following documents. "
        "Answer questions based solely on this content.<|im_end|>",
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
                       "and are ready to answer questions about them.<|im_end|>")
    prompt_parts.append("")
    prompt_parts.append("<|im_start|>assistant")
    prompt_parts.append("I have received and processed the documents. "
                       "I am ready to answer questions about the content.")
    
    return "\n".join(prompt_parts)


def ingest_document(
    server_url: str,
    documents: list,
    corpus_id: str,
    description: str,
    kv_slots_dir: Path
) -> Optional[Dict]:
    """
    Ingest documents into KV cache via llama-server.
    
    Returns:
        Dict with slot_id and metadata on success, None on failure
    """
    # Create the ingestion prompt
    prompt = create_ingestion_prompt(documents, corpus_id, description)
    
    # Use integer slot 0 — llama.cpp slot IDs are integers, not strings
    slot_id = 0
    
    # Prepare the request for llama-server
    payload = {
        "prompt": prompt,
        "id_slot": slot_id,
        "n_predict": 0,
        "temperature": 0.0,
        "stop": ["<|im_end|>"],
        "stream": False,
        "cache_prompt": True,  # Enable KV cache
    }
    
    estimated_tokens = estimate_tokens(prompt)
    ctx_size = int(env.get("CTX_SIZE", "131072")) if 'env' in dir() else 131072
    usage_pct = estimated_tokens / ctx_size * 100 if ctx_size else 0
    print(f"[ingest] Sending documents to llama-server...")
    print(f"[ingest] Corpus ID: {corpus_id}")
    print(f"[ingest] Slot ID: {slot_id}")
    print(f"[ingest] Prompt length: {len(prompt):,} characters (~{estimated_tokens:,} tokens, {usage_pct:.0f}% of ctx)")
    if usage_pct > 95:
        print(f"[ingest] ⚠ WARNING: Exceeds 95% of context window — content will be truncated!")
    elif usage_pct > 80:
        print(f"[ingest] ⚠ Note: Using {usage_pct:.0f}% of context window")
    
    start_time = time.time()
    
    try:
        response = requests.post(
            f"{server_url}/completion",
            json=payload,
            timeout=7200  # 1M token prefill can take ~68 minutes
        )
        response.raise_for_status()
        
        result = response.json()
        elapsed = time.time() - start_time
        
        # Extract timing info
        timings = result.get("timings", {})
        prompt_ms = timings.get("prompt_ms", 0)
        
        print(f"[ingest] ✓ Documents ingested successfully")
        print(f"[ingest]   Prefill time: {prompt_ms:.1f}ms")
        print(f"[ingest]   Total time: {elapsed:.2f}s")
        
        # Save the KV slot
        save_payload = {
            "filename": f"{corpus_id}.bin"
        }

        save_response = requests.post(
            f"{server_url}/slots/{slot_id}?action=save",
            json=save_payload,
            timeout=60
        )
        
        if save_response.status_code == 200:
            print(f"[ingest] ✓ KV slot saved")
        else:
            print(f"[ingest] ⚠ Failed to save KV slot: {save_response.status_code}")
        
        return {
            "slot_id": slot_id,
            "corpus_id": corpus_id,
            "description": description,
            "prompt_length": len(prompt),
            "prefill_ms": prompt_ms,
            "total_time_s": elapsed,
            "timestamp": datetime.now().isoformat(),
            "slot_file": f"{corpus_id}.bin"
        }
        
    except requests.exceptions.RequestException as e:
        print(f"[ingest] ✗ Error: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Ingest documents into CAG KV cache"
    )
    parser.add_argument(
        "document",
        nargs="?",
        help="Path to document file (or use stdin)"
    )
    parser.add_argument(
        "--corpus-dir",
        default=None,
        help="Directory of .txt/.md/.pdf files to ingest as one corpus"
    )
    parser.add_argument(
        "--corpus-id",
        default=None,
        help="Unique identifier for this corpus (auto-generated if not provided)"
    )
    parser.add_argument(
        "--description",
        default="",
        help="Description of the corpus"
    )
    parser.add_argument(
        "--server-url",
        default=None,
        help="llama-server URL (default: http://localhost:8080)"
    )
    
    args = parser.parse_args()
    
    # Determine project directory
    script_dir = Path(__file__).parent.parent.resolve()
    
    # Load environment
    env = load_env(script_dir)
    
    # Set defaults from environment
    server_url = args.server_url or f"http://localhost:{env.get('LLAMA_CPP_PORT', '8080')}"
    kv_slots_dir = Path(env.get("KV_SLOTS_DIR", script_dir / "kv_slots"))
    
    # Generate corpus ID if not provided
    corpus_id = args.corpus_id or f"corpus_{uuid.uuid4().hex[:8]}"
    
    # Read documents
    documents = []
    
    if args.corpus_dir:
        # Read all supported files from directory
        corpus_dir = Path(args.corpus_dir)
        if not corpus_dir.is_dir():
            print(f"[ingest] ✗ Error: Not a directory: {args.corpus_dir}")
            sys.exit(1)
        supported = [".txt", ".md", ".pdf"]
        files = sorted([f for f in corpus_dir.iterdir() if f.suffix.lower() in supported])
        if not files:
            print(f"[ingest] ✗ Error: No .txt/.md/.pdf files in {args.corpus_dir}")
            sys.exit(1)
        for file_path in files:
            if file_path.suffix.lower() == ".pdf":
                content = read_pdf(file_path)
            else:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            documents.append(content)
            print(f"[ingest] Loaded: {file_path.name} ({len(content):,} chars)")
    elif args.document:
        # Read from single file
        doc_path = Path(args.document)
        if not doc_path.exists():
            print(f"[ingest] ✗ Error: File not found: {args.document}")
            sys.exit(1)
        if doc_path.suffix.lower() == ".pdf":
            documents.append(read_pdf(doc_path))
        else:
            with open(doc_path, "r", encoding="utf-8", errors="ignore") as f:
                documents.append(f.read())
        print(f"[ingest] Loaded document: {args.document}")
    else:
        # Read from stdin
        print("[ingest] Reading from stdin (Ctrl+D to finish)...")
        content = sys.stdin.read()
        if not content.strip():
            print("[ingest] ✗ Error: No content provided")
            sys.exit(1)
        documents.append(content)
        print(f"[ingest] Loaded {len(content):,} characters from stdin")
    
    # Check server health
    try:
        health = requests.get(f"{server_url}/health", timeout=5)
        if health.status_code != 200:
            print(f"[ingest] ✗ Error: llama-server not healthy (status: {health.status_code})")
            sys.exit(1)
    except requests.exceptions.RequestException:
        print(f"[ingest] ✗ Error: Cannot connect to llama-server at {server_url}")
        print(f"[ingest]    Make sure the server is running: ./start_server.sh")
        sys.exit(1)
    
    # Ingest documents
    result = ingest_document(
        server_url=server_url,
        documents=documents,
        corpus_id=corpus_id,
        description=args.description,
        kv_slots_dir=kv_slots_dir
    )
    
    if result is None:
        sys.exit(1)
    
    # Update manifest
    manifest = load_manifest(script_dir)
    manifest["corpora"].append(result)
    save_manifest(script_dir, manifest)
    
    print(f"\n[ingest] ✓ Ingestion complete!")
    print(f"[ingest]   Corpus ID: {corpus_id}")
    print(f"[ingest]   Slot ID: {result['slot_id']}")
    print(f"[ingest]   Manifest updated: {script_dir / 'manifest.json'}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
