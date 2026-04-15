#!/usr/bin/env python3
"""
Cache-Augmented Generation (CAG) - Query Tool

This script queries the CAG system by:
1. Restoring KV slot by corpus_id from manifest
2. Appending user question to the restored context
3. Streaming LLM response
4. Reporting restore time and decode speed

Usage:
    python query.py <corpus_id> '<question>'
    python query.py --list  # List available corpora
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

import requests


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
    """Load manifest.json."""
    manifest_path = project_dir / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            return json.load(f)
    return {"corpora": [], "version": "1.0"}


def list_corpora(project_dir: Path) -> List[Dict]:
    """List all available corpora from manifest."""
    manifest = load_manifest(project_dir)
    return manifest.get("corpora", [])


def find_corpus(project_dir: Path, corpus_id: str) -> Optional[Dict]:
    """Find corpus by ID in manifest."""
    corpora = list_corpora(project_dir)
    for corpus in corpora:
        if corpus.get("corpus_id") == corpus_id:
            return corpus
    return None


def query_corpus(
    server_url: str,
    corpus: Dict,
    question: str,
    kv_slots_dir: Path,
    max_tokens: int = 512,
    temperature: float = 0.7
) -> Optional[Dict]:
    """
    Query a corpus by restoring its KV slot and asking a question.
    
    Returns:
        Dict with response and timing info on success, None on failure
    """
    slot_id = 0  # llama.cpp slot IDs are integers; CAG uses slot 0
    slot_file = corpus.get("slot_file")

    print(f"[query] Restoring KV slot: {slot_id}")

    # Step 1: Restore the KV slot
    restore_start = time.time()

    try:
        # Always restore from disk — /slots returns a bare list, not {"slots":[...]}
        if slot_file:
            slot_path = kv_slots_dir / Path(slot_file).name
            if slot_path.exists():
                restore_payload = {
                    "filename": Path(slot_file).name
                }

                restore_response = requests.post(
                    f"{server_url}/slots/0?action=restore",
                    json=restore_payload,
                    timeout=120
                )

                if restore_response.status_code != 200:
                    print(f"[query] ✗ Error: Failed to restore slot: {restore_response.status_code}")
                    print(f"[query]    Response: {restore_response.text}")
                    return None

                print(f"[query] ✓ Slot restored from file")
            else:
                print(f"[query] ✗ Error: Slot file not found: {slot_path}")
                return None
        
        restore_time = time.time() - restore_start
        print(f"[query]   Restore time: {restore_time*1000:.1f}ms")
        
    except requests.exceptions.RequestException as e:
        print(f"[query] ✗ Error during restore: {e}")
        return None
    
    # Step 2: Query with the question
    print(f"[query] Asking question...")
    print(f"[query] Question: {question[:80]}{'...' if len(question) > 80 else ''}")
    
    # Format the question prompt
    prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    
    payload = {
        "prompt": prompt,
        "id_slot": slot_id,
        "n_predict": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stop": ["<|im_end|>", "<|im_start|>"],
        "cache_prompt": True,  # Use cached KV
    }
    
    decode_start = time.time()
    tokens_generated = 0
    response_text = []
    
    try:
        response = requests.post(
            f"{server_url}/completion",
            json=payload,
            stream=True,
            timeout=300
        )
        response.raise_for_status()
        
        print(f"\n{'='*60}")
        print("ANSWER:")
        print(f"{'='*60}")
        
        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                if line_str.startswith('data: '):
                    try:
                        data = json.loads(line_str[6:])
                        
                        if "content" in data:
                            content = data["content"]
                            print(content, end='', flush=True)
                            response_text.append(content)
                            tokens_generated += 1
                        
                        if data.get("stop", False):
                            break
                            
                    except json.JSONDecodeError:
                        continue
        
        decode_time = time.time() - decode_start
        
        print(f"\n{'='*60}")
        
        # Calculate metrics
        if tokens_generated > 0 and decode_time > 0:
            tokens_per_sec = tokens_generated / decode_time
            ms_per_token = (decode_time * 1000) / tokens_generated
        else:
            tokens_per_sec = 0
            ms_per_token = 0
        
        print(f"\n[query] ✓ Query complete")
        print(f"[query]   Tokens generated: {tokens_generated}")
        print(f"[query]   Decode time: {decode_time:.2f}s")
        print(f"[query]   Speed: {tokens_per_sec:.2f} tok/s ({ms_per_token:.1f}ms/tok)")
        
        return {
            "corpus_id": corpus.get("corpus_id"),
            "slot_id": slot_id,
            "question": question,
            "answer": "".join(response_text),
            "tokens_generated": tokens_generated,
            "restore_time_ms": restore_time * 1000,
            "decode_time_s": decode_time,
            "tokens_per_sec": tokens_per_sec,
            "ms_per_token": ms_per_token,
            "timestamp": datetime.now().isoformat()
        }
        
    except requests.exceptions.RequestException as e:
        print(f"[query] ✗ Error during query: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Query the CAG system"
    )
    parser.add_argument(
        "corpus_id",
        nargs="?",
        help="Corpus ID to query"
    )
    parser.add_argument(
        "question",
        nargs="?",
        help="Question to ask"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available corpora"
    )
    parser.add_argument(
        "--server-url",
        default=None,
        help="llama-server URL (default: http://localhost:8080)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate (default: 512)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)"
    )
    
    args = parser.parse_args()
    
    # Determine project directory
    script_dir = Path(__file__).parent.parent.resolve()
    
    # Load environment
    env = load_env(script_dir)
    
    # Set defaults from environment
    server_url = args.server_url or f"http://localhost:{env.get('LLAMA_CPP_PORT', '8080')}"
    kv_slots_dir = Path(env.get("KV_SLOTS_DIR", script_dir / "kv_slots"))
    
    # List mode
    if args.list:
        corpora = list_corpora(script_dir)
        if not corpora:
            print("[query] No corpora found. Ingest documents first.")
            return 0
        
        print(f"\nAvailable Corpora ({len(corpora)} total):")
        print(f"{'='*80}")
        for i, corpus in enumerate(corpora, 1):
            print(f"{i}. {corpus.get('corpus_id', 'unknown')}")
            if corpus.get('description'):
                print(f"   Description: {corpus['description']}")
            print(f"   Slot ID: {corpus.get('slot_id', 'unknown')}")
            print(f"   Ingested: {corpus.get('timestamp', 'unknown')}")
            print()
        return 0
    
    # Validate arguments
    if not args.corpus_id:
        print("[query] ✗ Error: corpus_id required (or use --list)")
        parser.print_help()
        sys.exit(1)
    
    if not args.question:
        print("[query] ✗ Error: question required")
        parser.print_help()
        sys.exit(1)
    
    # Find corpus
    corpus = find_corpus(script_dir, args.corpus_id)
    if not corpus:
        print(f"[query] ✗ Error: Corpus not found: {args.corpus_id}")
        print(f"[query] Run with --list to see available corpora")
        sys.exit(1)
    
    # Check server health
    try:
        health = requests.get(f"{server_url}/health", timeout=5)
        if health.status_code != 200:
            print(f"[query] ✗ Error: llama-server not healthy (status: {health.status_code})")
            sys.exit(1)
    except requests.exceptions.RequestException:
        print(f"[query] ✗ Error: Cannot connect to llama-server at {server_url}")
        print(f"[query]    Make sure the server is running: ./start_server.sh")
        sys.exit(1)
    
    # Execute query
    result = query_corpus(
        server_url=server_url,
        corpus=corpus,
        question=args.question,
        kv_slots_dir=kv_slots_dir,
        max_tokens=args.max_tokens,
        temperature=args.temperature
    )
    
    if result is None:
        sys.exit(1)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
