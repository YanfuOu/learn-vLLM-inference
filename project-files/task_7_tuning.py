#!/usr/bin/env python3
"""
Task 7: Tuning vLLM Parameters for Production
Experiment with key vLLM configuration options.
"""

import os
import sys
import time
import json
import signal
import subprocess
import asyncio

# Configure vLLM for CPU-only execution
os.environ["VLLM_TARGET_DEVICE"] = "cpu"
os.environ.setdefault("VLLM_CPU_KVCACHE_SPACE", "1")
os.environ["TORCHDYNAMO_DISABLE"] = "1"


async def send_request(session, url, model, prompt, max_tokens=50):
    """Send a single completion request."""
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
            tokens = data.get("usage", {}).get("completion_tokens", 0)
            return {"latency": end - start, "tokens": tokens, "success": True}
    except Exception:
        return {"latency": time.time() - start, "tokens": 0, "success": False}


async def run_quick_benchmark(url, model, num_requests=10):
    """Run a quick benchmark with concurrent requests."""
    import aiohttp

    prompts = [
        "What is machine learning?",
        "Explain neural networks.",
        "How does AI work?",
        "What are transformers?",
        "Describe deep learning.",
    ]

    async with aiohttp.ClientSession() as session:
        tasks = [
            send_request(session, url, model, prompts[i % len(prompts)])
            for i in range(num_requests)
        ]
        start = time.time()
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start

    successful = [r for r in results if r["success"]]
    total_tokens = sum(r["tokens"] for r in successful)
    avg_latency = sum(r["latency"] for r in successful) / len(successful) if successful else 0
    throughput = total_tokens / total_time if total_time > 0 else 0

    return {
        "throughput": throughput,
        "avg_latency": avg_latency,
        "total_tokens": total_tokens,
        "total_time": total_time,
        "success_count": len(successful),
    }


def stop_server():
    """Stop any running vLLM server."""
    pid_file = "/root/markers/vllm_server_pid.txt"
    if os.path.exists(pid_file):
        with open(pid_file, "r") as f:
            pid = int(f.read().strip())
        try:
            os.kill(pid, signal.SIGTERM)
            time.sleep(2)
            print("  Previous server stopped.")
        except ProcessLookupError:
            pass

    # Also try killing by port
    try:
        result = subprocess.run(
            ["fuser", "-k", "8000/tcp"],
            capture_output=True, timeout=5
        )
    except Exception:
        pass
    time.sleep(1)


def start_server(model, max_model_len, max_num_seqs, swap_space=2):
    """Start vLLM server with given parameters."""
    import requests

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--port", "8000",
        "--max-model-len", str(max_model_len),
        "--max-num-seqs", str(max_num_seqs),
        # swap-space removed in vLLM v0.18+ (shown in output only)
        "--enforce-eager",
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Save PID
    os.makedirs("/root/markers", exist_ok=True)
    with open("/root/markers/vllm_server_pid.txt", "w") as f:
        f.write(str(proc.pid))

    # Wait for ready
    timeout = 120
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get("http://localhost:8000/health")
            if resp.status_code == 200:
                return proc
        except Exception:
            pass
        time.sleep(2)

    return None


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    markers_dir = os.path.join(script_dir, "markers")
    print("=" * 65)
    print("Task 7: Tuning vLLM Parameters for Production")
    print("=" * 65)

    model_name = "HuggingFaceTB/SmolLM-135M"
    server_url = "http://localhost:8000"
    num_test_requests = 10

    print(f"\nModel: {model_name}")
    print(f"Benchmark: {num_test_requests} concurrent requests per config")
    print("-" * 65)

    # Define configurations to test
    configs = [
        {
            "name": "A: Default",
            "max_model_len": 128,
            "max_num_seqs": 256,
            "swap_space": 1,
        },
        {
            "name": "B: Shorter Context",
            # TODO 1: Set a shorter context length
            # Hint: Shorter context = less memory per request
            "max_model_len": 64,  # TODO: Set to 64
            "max_num_seqs": 256,
            "swap_space": 1,
        },
        {
            "name": "C: Limited Concurrency",
            "max_model_len": 64,
            # TODO 2: Limit concurrent sequences
            # Hint: Fewer concurrent sequences = less memory pressure
            "max_num_seqs": 8,  # TODO: Set to 8
            "swap_space": 1,
        },
    ]

    results = []

    for i, config in enumerate(configs):
        print(f"\n--- CONFIG {config['name']} ---")
        print(f"  max_model_len={config['max_model_len']}, "
              f"max_num_seqs={config['max_num_seqs']}, "
              f"swap_space={config['swap_space']}GB")

        # Stop existing server
        print("  Stopping previous server...")
        stop_server()

        # Start with new config
        print(f"  Starting server with config {config['name']}...")
        proc = start_server(
            model_name,
            config["max_model_len"],
            config["max_num_seqs"],
            config["swap_space"],
        )

        if proc is None:
            print("  ERROR: Server failed to start with this config.")
            results.append({"config": config["name"], "throughput": 0, "avg_latency": 0})
            continue

        print("  Server ready! Running benchmark...")
        benchmark = asyncio.run(run_quick_benchmark(server_url, model_name, num_test_requests))

        results.append({
            "config": config["name"],
            "max_model_len": config["max_model_len"],
            "max_num_seqs": config["max_num_seqs"],
            "throughput": benchmark["throughput"],
            "avg_latency": benchmark["avg_latency"],
            "total_tokens": benchmark["total_tokens"],
        })

        print(f"  Result: {benchmark['throughput']:.1f} tok/s, "
              f"{benchmark['avg_latency']:.2f}s avg latency")

    # --- COMPARISON TABLE ---
    print(f"\n--- CONFIGURATION COMPARISON ---")
    print(f"{'Config':<22} {'max_model_len':>14} {'max_num_seqs':>13} {'Throughput':>11} {'Latency':>9}")
    print("-" * 72)
    for r in results:
        print(
            f"{r['config']:<22} "
            f"{r.get('max_model_len', 'N/A'):>14} "
            f"{r.get('max_num_seqs', 'N/A'):>13} "
            f"{r['throughput']:>8.1f} tok/s "
            f"{r['avg_latency']:>7.2f}s"
        )

    # --- KEY PARAMETERS ---
    print(f"\n--- KEY PARAMETERS EXPLAINED ---")
    print(f"  max_model_len:  Maximum context length per request.")
    print(f"                  Lower = less memory per request.")
    print(f"  max_num_seqs:   Maximum concurrent sequences in a batch.")
    print(f"                  Controls concurrency vs per-request resources.")
    print(f"  swap_space:     CPU swap space (GB) for KV cache overflow.")
    print(f"                  Extends capacity beyond available RAM.")

    # Save results
    os.makedirs(markers_dir, exist_ok=True)
    with open(os.path.join(markers_dir, "tuning_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # --- KEY INSIGHT ---
    print("\n" + "=" * 65)
    print("KEY INSIGHT:")
    print("- Lower max_model_len saves memory per request")
    print("- max_num_seqs controls concurrency vs per-request resources")
    print("- swap_space extends KV cache to CPU RAM when memory is tight")
    print("- Always tune based on YOUR workload pattern")
    print("- Next: Build a monitoring dashboard to track these metrics (Task 8)")
    print("=" * 65)

    # Create marker
    with open(os.path.join(markers_dir, "task7_complete.txt"), "w") as f:
        f.write("TASK_7_COMPLETE\n")

    print("\nTask 7 Complete!")
    print("Next: python " + os.path.join(script_dir, "task_8_dashboard.py"))


if __name__ == "__main__":
    main()
