Task 8: Production Monitoring Dashboard (Capstone)

TASK OVERVIEW

InferenceIO is going live. In this capstone task, you will build a Gradio monitoring dashboard that shows real-time inference metrics and a complete summary of your lab journey.

CONCEPT: Production Monitoring

In production, you need to monitor:

Tokens per second - Is the model performing well?
Latency - How long are users waiting?
Throughput under load - Can the system handle peak traffic?
Before/After comparison - Did optimizations actually help?
Monitoring is how you know when to scale, tune, or upgrade your inference setup.

REAL-WORLD USE CASE:
Production LLM systems use tools like Prometheus and Grafana to track inference metrics. This dashboard gives you a simplified version of that monitoring stack.

YOUR TASK

Step 1: Open /root/code/task_8_dashboard.py

Step 2: Complete 3 TODOs:

TODO 1 (line 85): Set to requests.post()
TODO 2 (line 114): Set to hf_tps, vllm_tps
TODO 3 (line 119): Set to vllm_tps / hf_tps
