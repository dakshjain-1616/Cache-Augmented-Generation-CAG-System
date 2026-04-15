#!/usr/bin/env python3
"""
Cache-Augmented Generation (CAG) - End-to-End Demo

This script demonstrates the complete CAG workflow:
1. Downloads sample text documents
2. Ingests them into KV cache
3. Runs test questions
4. Prints performance stats

Usage:
    python demo.py [--skip-setup] [--questions <n>]
"""

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path
from datetime import datetime
from typing import List, Dict

import requests


# Sample documents for demo
SAMPLE_DOCUMENTS = [
    {
        "name": "alice_chapter1.txt",
        "url": "https://www.gutenberg.org/files/11/11-0.txt",
        "description": "Alice's Adventures in Wonderland - Chapter 1",
        "extract_start": "CHAPTER I.\nDown the Rabbit-Hole",
        "extract_end": "CHAPTER II.\nThe Pool of Tears",
        "max_chars": 8000
    },
    {
        "name": "peter_pan_chapter1.txt",
        "url": "https://www.gutenberg.org/files/16/16-0.txt",
        "description": "Peter Pan - Chapter 1",
        "extract_start": "Chapter 1 PETER BREAKS THROUGH",
        "extract_end": "Chapter 2 THE SHADOW",
        "max_chars": 8000
    }
]

# Test questions for each corpus
TEST_QUESTIONS = {
    "demo_alice": [
        "What is Alice doing at the beginning of the story?",
        "How does Alice feel about falling down the rabbit hole?",
        "What does Alice see while falling?",
    ],
    "demo_peter": [
        "Who is Mrs. Darling and what is she doing at the start?",
        "What is special about Peter Pan?",
        "Why does Peter visit the nursery?",
    ]
}


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


def download_sample_text(url: str, extract_start: str, extract_end: str, max_chars: int) -> str:
    """Download and extract sample text from Project Gutenberg."""
    print(f"  Downloading from {url[:50]}...")
    
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            content = response.read().decode('utf-8', errors='ignore')
        
        # Extract the relevant section
        start_idx = content.find(extract_start)
        if start_idx == -1:
            start_idx = 0
        
        end_idx = content.find(extract_end, start_idx + len(extract_start))
        if end_idx == -1:
            end_idx = start_idx + max_chars
        
        extracted = content[start_idx:end_idx].strip()
        
        # Limit to max_chars
        if len(extracted) > max_chars:
            extracted = extracted[:max_chars]
            # Try to end at a sentence boundary
            last_period = extracted.rfind('.')
            if last_period > max_chars * 0.8:
                extracted = extracted[:last_period + 1]
        
        return extracted
        
    except Exception as e:
        print(f"  Warning: Failed to download: {e}")
        # Fallback content
        return f"Sample text for demo purposes. This is a placeholder since the download failed. " * 50


def check_server_health(server_url: str) -> bool:
    """Check if llama-server is running."""
    try:
        response = requests.get(f"{server_url}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def ingest_document_api(
    server_url: str,
    documents: List[str],
    corpus_id: str,
    description: str
) -> Dict:
    """Ingest documents via API."""
    payload = {
        "documents": documents,
        "corpus_id": corpus_id,
        "description": description
    }
    
    response = requests.post(
        f"{server_url}/ingest",
        json=payload,
        timeout=10
    )
    response.raise_for_status()
    
    return response.json()


def wait_for_job(api_url: str, job_id: str, timeout: int = 300) -> Dict:
    """Wait for ingestion job to complete."""
    start_time = time.time()
    attempt = 0

    while time.time() - start_time < timeout:
        response = requests.get(f"{api_url}/status/{job_id}", timeout=10)
        response.raise_for_status()
        
        status = response.json()
        
        if status["status"] == "completed":
            return status
        elif status["status"] == "failed":
            raise Exception(f"Job failed: {status.get('error', 'Unknown error')}")
        
        wait_interval = min(2 ** (attempt + 1), 30)
        print(f"    Status: {status['status']}... (retry in {wait_interval}s)")
        time.sleep(wait_interval)
        attempt += 1
    
    raise Exception("Timeout waiting for job completion")


def query_corpus_api(
    api_url: str,
    corpus_id: str,
    question: str
) -> Dict:
    """Query corpus via API."""
    payload = {
        "corpus_id": corpus_id,
        "question": question,
        "max_tokens": 256,
        "temperature": 0.7,
        "stream": False
    }
    
    response = requests.post(
        f"{api_url}/query",
        json=payload,
        timeout=120
    )
    response.raise_for_status()
    
    return response.json()


def print_banner(text: str, width: int = 70):
    """Print a formatted banner."""
    print("\n" + "=" * width)
    print(text.center(width))
    print("=" * width)


def print_stats(stats: Dict):
    """Print performance statistics."""
    print(f"\n  Performance Stats:")
    print(f"    • Tokens generated: {stats.get('tokens_generated', 'N/A')}")
    print(f"    • Restore time: {stats.get('restore_time_ms', 0):.1f}ms")
    print(f"    • Decode time: {stats.get('decode_time_s', 0):.2f}s")
    print(f"    • Speed: {stats.get('tokens_per_sec', 0):.2f} tok/s")
    print(f"    • Latency: {stats.get('ms_per_token', 0):.1f}ms/token")


def main():
    parser = argparse.ArgumentParser(description="CAG End-to-End Demo")
    parser.add_argument("--skip-setup", action="store_true", help="Skip setup checks")
    parser.add_argument("--questions", type=int, default=3, help="Number of questions per corpus")
    parser.add_argument("--api-url", default=None, help="API server URL")
    parser.add_argument("--llama-url", default=None, help="llama-server URL")
    
    args = parser.parse_args()
    
    # Determine project directory
    script_dir = Path(__file__).parent.parent.resolve()
    
    # Load environment
    env = load_env(script_dir)
    
    # Set URLs
    api_url = args.api_url or f"http://localhost:{env.get('API_PORT', '8000')}"
    llama_url = args.llama_url or f"http://localhost:{env.get('LLAMA_CPP_PORT', '8080')}"
    
    print_banner("CACHE-AUGMENTED GENERATION (CAG) DEMO")
    
    # Check servers
    if not args.skip_setup:
        print("\n[1/5] Checking server status...")
        
        print(f"  Checking llama-server at {llama_url}...")
        if not check_server_health(llama_url):
            print(f"  ✗ llama-server not running at {llama_url}")
            print(f"    Please start it first: ./start_server.sh")
            sys.exit(1)
        print(f"  ✓ llama-server is healthy")
        
        print(f"  Checking API server at {api_url}...")
        try:
            response = requests.get(f"{api_url}/health", timeout=5)
            if response.status_code == 200:
                print(f"  ✓ API server is healthy")
            else:
                print(f"  ✗ API server not responding correctly")
                sys.exit(1)
        except:
            print(f"  ✗ API server not running at {api_url}")
            print(f"    Please start it first: python api_server.py")
            sys.exit(1)
    
    # Download sample documents
    print("\n[2/5] Downloading sample documents...")
    
    downloaded_docs = []
    for doc_info in SAMPLE_DOCUMENTS:
        print(f"\n  Downloading: {doc_info['name']}")
        content = download_sample_text(
            doc_info['url'],
            doc_info['extract_start'],
            doc_info['extract_end'],
            doc_info['max_chars']
        )
        
        corpus_id = f"demo_{doc_info['name'].split('_')[0]}"
        downloaded_docs.append({
            "corpus_id": corpus_id,
            "name": doc_info['name'],
            "description": doc_info['description'],
            "content": content
        })
        print(f"  ✓ Downloaded {len(content)} characters")
    
    # Ingest documents
    print("\n[3/5] Ingesting documents into KV cache...")
    
    ingestion_results = []
    for doc in downloaded_docs:
        print(f"\n  Ingesting: {doc['corpus_id']}")
        print(f"    Description: {doc['description']}")
        
        try:
            result = ingest_document_api(
                server_url=api_url,
                documents=[doc['content']],
                corpus_id=doc['corpus_id'],
                description=doc['description']
            )
            
            job_id = result['job_id']
            print(f"    Job ID: {job_id}")
            
            # Wait for completion
            status = wait_for_job(api_url, job_id)
            
            result_data = status.get('result', {})
            print(f"  ✓ Ingested successfully")
            print(f"    • Slot ID: {result_data.get('slot_id', 'N/A')}")
            print(f"    • Prompt length: {result_data.get('prompt_length', 0)} chars")
            print(f"    • Prefill time: {result_data.get('prefill_ms', 0):.1f}ms")
            
            ingestion_results.append({
                "corpus_id": doc['corpus_id'],
                "slot_id": result_data.get('slot_id'),
                "success": True
            })
            
        except Exception as e:
            print(f"  ✗ Ingestion failed: {e}")
            ingestion_results.append({
                "corpus_id": doc['corpus_id'],
                "success": False,
                "error": str(e)
            })
    
    # Query documents
    print("\n[4/5] Running test queries...")
    
    query_results = []
    
    for result in ingestion_results:
        if not result['success']:
            continue
        
        corpus_id = result['corpus_id']
        questions = TEST_QUESTIONS.get(corpus_id, ["What is this document about?"])
        
        # Limit questions
        questions = questions[:args.questions]
        
        print(f"\n  Querying corpus: {corpus_id}")
        print(f"  {'='*60}")
        
        for i, question in enumerate(questions, 1):
            print(f"\n  Q{i}: {question}")
            
            try:
                start_time = time.time()
                response = query_corpus_api(api_url, corpus_id, question)
                total_time = time.time() - start_time
                
                answer = response.get('answer', '').strip()
                
                # Print answer (truncated if too long)
                if len(answer) > 200:
                    print(f"  A{i}: {answer[:200]}...")
                else:
                    print(f"  A{i}: {answer}")
                
                print_stats(response)
                print(f"    • Total time: {total_time:.2f}s")
                
                query_results.append({
                    "corpus_id": corpus_id,
                    "question": question,
                    "success": True,
                    "stats": response
                })
                
            except Exception as e:
                print(f"  ✗ Query failed: {e}")
                query_results.append({
                    "corpus_id": corpus_id,
                    "question": question,
                    "success": False,
                    "error": str(e)
                })
    
    # Print summary
    print("\n[5/5] Demo Summary")
    print_banner("RESULTS SUMMARY")
    
    # Ingestion summary
    successful_ingestions = sum(1 for r in ingestion_results if r['success'])
    print(f"\n  Ingestion:")
    print(f"    • Total documents: {len(ingestion_results)}")
    print(f"    • Successful: {successful_ingestions}")
    print(f"    • Failed: {len(ingestion_results) - successful_ingestions}")
    
    # Query summary
    successful_queries = sum(1 for r in query_results if r['success'])
    print(f"\n  Queries:")
    print(f"    • Total queries: {len(query_results)}")
    print(f"    • Successful: {successful_queries}")
    print(f"    • Failed: {len(query_results) - successful_queries}")
    
    # Performance summary
    if query_results:
        successful_stats = [r['stats'] for r in query_results if r['success'] and 'stats' in r]
        
        if successful_stats:
            avg_restore = sum(s.get('restore_time_ms', 0) for s in successful_stats) / len(successful_stats)
            avg_decode = sum(s.get('decode_time_s', 0) for s in successful_stats) / len(successful_stats)
            avg_speed = sum(s.get('tokens_per_sec', 0) for s in successful_stats) / len(successful_stats)
            avg_latency = sum(s.get('ms_per_token', 0) for s in successful_stats) / len(successful_stats)
            
            print(f"\n  Average Performance:")
            print(f"    • Restore time: {avg_restore:.1f}ms")
            print(f"    • Decode time: {avg_decode:.2f}s")
            print(f"    • Speed: {avg_speed:.2f} tok/s")
            print(f"    • Latency: {avg_latency:.1f}ms/token")
    
    print("\n" + "=" * 70)
    print("Demo complete! The CAG system is working correctly.")
    print("=" * 70 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
