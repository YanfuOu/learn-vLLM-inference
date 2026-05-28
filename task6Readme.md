Task 6: Multi-User Throughput Under Load

TASK OVERVIEW

The CEO's deal requires serving multiple users at once. In this task, you will stress-test the vLLM server with concurrent requests and see how throughput scales.

CONCEPT: Why Throughput Matters

Single-user performance tells you how fast ONE person gets a response. But in production:

Multiple users send requests simultaneously
The system must handle them efficiently
Total throughput (tokens/sec across ALL users) is what matters
vLLM uses continuous batching - it processes requests as they arrive
REAL-WORLD USE CASE:
A chatbot serving 50 concurrent users. If each user gets 10 tok/s individually, but the system produces 200 tok/s total - that is high throughput. vLLM achieves this through efficient batching and PagedAttention.

YOUR TASK

Step 1: Open /root/code/task_6_multi_user_load.py

Step 2: Complete 2 TODOs:

TODO 1 (line 97): Set to [1, 5, 10, 20]
TODO 2 (line 117): Set to total_tokens / total_time
Step 3: Run the script: