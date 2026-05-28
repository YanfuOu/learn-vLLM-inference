#!/usr/bin/env python3
"""
Task 5: Launch vLLM as an OpenAI-Compatible API Server
Serve SmolLM via HTTP and interact using the OpenAI Python client.
"""

import os
import sys
import time
import subprocess
import signal

# Configure vLLM for CPU-only execution
os.environ["VLLM_TARGET_DEVICE"] = "cpu"
os.environ.setdefault("VLLM_CPU_KVCACHE_SPACE", "1")
os.environ["TORCHDYNAMO_DISABLE"] = "1"


def wait_for_server(url, timeout=120):
    """Wait for the vLLM server to be ready."""
    import requests

    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(f"{url}/health")
            if resp.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(2)
        elapsed = int(time.time() - start)
        print(f"  Waiting for server... ({elapsed}s)", end="\r")
    return False


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    markers_dir = os.path.join(script_dir, "markers")
    print("=" * 65)
    print("Task 5: vLLM OpenAI-Compatible API Server")
    print("=" * 65)

    model_name = "HuggingFaceTB/SmolLM-135M"
    server_url = "http://localhost:8000"
    prompt = "What is inference in machine learning?"

    print(f"\nModel: {model_name}")
    print(f"Server URL: {server_url}")
    print(f"Prompt: \"{prompt}\"")
    print("-" * 65)

    # --- START vLLM SERVER ---
    print("\nStarting vLLM server (this may take a moment)...")
    print("Command: vllm serve HuggingFaceTB/SmolLM-135M --port 8000")

    # Check if server is already running
    import requests
    try:
        resp = requests.get(f"{server_url}/health")
        if resp.status_code == 200:
            print("  Server is already running!")
            server_process = None
    except Exception:
        # Start the server in the background
        server_process = subprocess.Popen(
            [
                sys.executable, "-m", "vllm.entrypoints.openai.api_server",
                "--model", model_name,
                "--port", "8000",
                "--max-model-len", "128",
                "--enforce-eager",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        print(f"  Server process started (PID: {server_process.pid})")

        # Save PID for later tasks
        os.makedirs("/root/markers", exist_ok=True)
        with open("/root/markers/vllm_server_pid.txt", "w") as f:
            f.write(str(server_process.pid))

    # Wait for server to be ready
    print("\n  Waiting for server to be ready...")
    if wait_for_server(server_url):
        print("  Server is ready!")
    else:
        print("  ERROR: Server did not start within timeout.")
        print("  Try running manually: vllm serve HuggingFaceTB/SmolLM-135M --port 8000")
        return

    # --- SEND REQUEST ---
    print(f"\n--- SENDING REQUEST ---")
    print(f"Endpoint: {server_url}/v1/completions")

    from openai import OpenAI

    # TODO 1: Configure the OpenAI client to point to the local vLLM server
    # Hint: Point the client to the local vLLM server URL
    client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")  # TODO: Set to "http://localhost:8000/v1" and "not-needed"

    # TODO 2: Send a completion request
    # Hint: Use the model_name variable ("HuggingFaceTB/SmolLM-135M")
    start_time = time.time()
    response = client.completions.create(
        model="HuggingFaceTB/SmolLM-135M",  # TODO: Set to "HuggingFaceTB/SmolLM-135M"
        prompt=prompt,
        max_tokens=50,
        temperature=0.7,
    )
    end_time = time.time()

    # Extract response
    response_text = response.choices[0].text
    latency = end_time - start_time

    # --- RESPONSE ---
    print(f"\n--- RESPONSE ---")
    print(f"Model: {response.model}")
    print(f"Response: {response_text[:200]}")
    print(f"Latency: {latency:.2f}s")

    if response.usage:
        print(f"Prompt tokens: {response.usage.prompt_tokens}")
        print(f"Completion tokens: {response.usage.completion_tokens}")

    # --- API DETAILS ---
    print(f"\n--- API DETAILS ---")
    print(f"Endpoint: {server_url}/v1/completions")
    print(f"Format: OpenAI-compatible (drop-in replacement)")
    print(f"Auth: No API key needed (local server)")

    # --- KEY INSIGHT ---
    print("\n" + "=" * 65)
    print("KEY INSIGHT:")
    print("- vLLM serves an OpenAI-compatible API out of the box")
    print("- Any app using the OpenAI SDK works with vLLM - zero code changes")
    print("- This is how you self-host LLMs in production")
    print("- The server stays running for Tasks 6-8")
    print("=" * 65)

    # Create marker
    with open(os.path.join(markers_dir, "task5_complete.txt"), "w") as f:
        f.write("TASK_5_COMPLETE\n")

    print("\nTask 5 Complete!")
    print("Next: python " + os.path.join(script_dir, "task_6_multi_user_load.py"))


if __name__ == "__main__":
    main()
