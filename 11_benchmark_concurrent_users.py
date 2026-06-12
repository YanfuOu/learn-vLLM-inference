#!/usr/bin/env python3
"""
Concurrent User Throughput Benchmark

This benchmark emulates real users by having each simulated user send a small
sequence of requests back-to-back, optionally with a short think-time between
requests. It measures total token throughput, request throughput, latency, and
success rate under different concurrency levels.
"""

import argparse
import asyncio
import json
import os
import time

import aiohttp
import requests


async def send_completion_request(session, url, model, prompt, max_tokens):
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }
    start = time.time()
    try:
        async with session.post(
            f"{url}/v1/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
        ) as resp:
            data = await resp.json()
            end = time.time()
            return {
                "latency": end - start,
                "tokens": data.get("usage", {}).get("completion_tokens", 0),
                "success": resp.status == 200,
                "status": resp.status,
                "error": None if resp.status == 200 else data,
            }
    except Exception as exc:
        return {
            "latency": time.time() - start,
            "tokens": 0,
            "success": False,
            "status": None,
            "error": str(exc),
        }


async def run_user_session(session, url, model, prompts, requests_per_user, max_tokens, think_time):
    results = []
    for index in range(requests_per_user):
        prompt = prompts[index % len(prompts)]
        result = await send_completion_request(session, url, model, prompt, max_tokens)
        results.append(result)
        if think_time and index < requests_per_user - 1:
            await asyncio.sleep(think_time)
    return results


async def run_benchmark(url, model, prompts, concurrent_users, requests_per_user, max_tokens, think_time):
    async with aiohttp.ClientSession() as session:
        tasks = [
            run_user_session(session, url, model, prompts, requests_per_user, max_tokens, think_time)
            for _ in range(concurrent_users)
        ]
        start = time.time()
        responses = await asyncio.gather(*tasks)
        elapsed = time.time() - start

    all_results = [result for user_results in responses for result in user_results]
    successful = [r for r in all_results if r["success"]]
    total_tokens = sum(r["tokens"] for r in successful)
    avg_latency = sum(r["latency"] for r in successful) / len(successful) if successful else 0
    success_rate = len(successful) / len(all_results) * 100 if all_results else 0
    throughput = total_tokens / elapsed if elapsed > 0 else 0
    requests_per_second = len(all_results) / elapsed if elapsed > 0 else 0

    return {
        "users": concurrent_users,
        "requests_per_user": requests_per_user,
        "total_requests": len(all_results),
        "successful_requests": len(successful),
        "success_rate": success_rate,
        "total_tokens": total_tokens,
        "total_time": elapsed,
        "throughput": throughput,
        "requests_per_second": requests_per_second,
        "avg_latency": avg_latency,
    }


def verify_server(server_url):
    try:
        health = requests.get(f"{server_url}/health", timeout=5)
        return health.status_code == 200
    except Exception:
        return False


def build_prompts():
    return [
        "What is machine learning?",
        "Explain neural networks briefly.",
        "How does a transformer model work?",
        "What is natural language processing?",
        "Describe deep learning in one paragraph.",
        "What are tokens in the context of LLMs?",
        "How is AI used in healthcare?",
        "What is the difference between AI and ML?",
        "Explain what fine-tuning means.",
        "What is transfer learning?",
    ]


def parse_args():
    parser = argparse.ArgumentParser(description="Concurrent user throughput benchmark for vLLM")
    parser.add_argument("--server-url", default="http://localhost:8000", help="Base URL for the vLLM server")
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM-135M", help="Model name used by the server")
    parser.add_argument("--user-counts", nargs="+", type=int, default=[1, 5, 10, 20], help="Concurrent user counts to benchmark")
    parser.add_argument("--requests-per-user", type=int, default=1, help="How many sequential requests each user sends")
    parser.add_argument("--think-time", type=float, default=0.0, help="Think time between requests from the same user (seconds)")
    parser.add_argument("--max-tokens", type=int, default=50, help="Max tokens for each completion request")
    parser.add_argument("--output", default="markers/11-concurrent_TRT-LLM_user_benchmark.json", help="Path to save benchmark results")
    return parser.parse_args()


def main():
    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    markers_dir = os.path.join(script_dir, "markers")
    os.makedirs(markers_dir, exist_ok=True)

    if not verify_server(args.server_url):
        print("ERROR: vLLM server is not running or healthy at", args.server_url)
        print("Start the server first, for example: python project-files/5_api_server.py")
        return

    prompts = build_prompts()
    results = []

    print(f"Benchmark server={args.server_url}, model={args.model}")
    print(f"Simulating {args.requests_per_user} sequential requests per user with {args.think_time}s think time")
    print("Using prompts:")
    for prompt in prompts:
        print(f"  - {prompt}")
    print()

    for users in args.user_counts:
        print(f"Running benchmark with {users} concurrent users...")
        result = asyncio.run(
            run_benchmark(
                args.server_url,
                args.model,
                prompts,
                users,
                args.requests_per_user,
                args.max_tokens,
                args.think_time,
            )
        )
        results.append(result)
        print(
            f"  Users={users}  Requests={result['total_requests']}  "
            f"Throughput={result['throughput']:.1f} tok/s  "
            f"Req/s={result['requests_per_second']:.1f}  "
            f"AvgLatency={result['avg_latency']:.2f}s  "
            f"Success={result['success_rate']:.0f}%"
        )
        print()

    print("\n--- BENCHMARK RESULTS ---")
    print(f"{'Users':>5} {'Reqs':>5} {'Tok':>7} {'Time':>7} {'Tok/s':>8} {'Req/s':>8} {'Lat(s)':>8} {'Success':>8}")
    print("" + "-" * 64)
    for r in results:
        print(
            f"{r['users']:>5} {r['total_requests']:>5} {r['total_tokens']:>7} "
            f"{r['total_time']:>7.2f} {r['throughput']:>8.1f} {r['requests_per_second']:>8.1f} "
            f"{r['avg_latency']:>8.2f} {r['success_rate']:>8.0f}%"
        )

    with open(os.path.join(script_dir, args.output), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved benchmark results to {args.output}")


if __name__ == "__main__":
    main()
