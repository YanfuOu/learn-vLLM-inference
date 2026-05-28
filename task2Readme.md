Task 2: vLLM Offline Inference - See the Difference

TASK OVERVIEW

In this task, you will run the SAME model and SAME prompt using vLLM instead of HuggingFace. You will see a side-by-side comparison of tokens per second.

CONCEPT: What is an Inference Engine?

vLLM is an inference engine - a system designed specifically to serve LLMs efficiently. There are many inference engines:

vLLM - High throughput, great for serving multiple users
llama.cpp - Optimized for CPU/RAM, great for local use
TensorRT-LLM - Optimized for NVIDIA GPUs
SGLang - Fast structured generation
Hugging Face TGI - Simple deployment
The same model can run at very different speeds depending on which engine you use.

REAL-WORLD USE CASE:
A company switching from basic HuggingFace inference to vLLM can serve the same model to more users with the same hardware.

