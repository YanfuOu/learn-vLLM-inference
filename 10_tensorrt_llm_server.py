#!/usr/bin/env python3
"""
TensorRT-LLM API Server

This script launches a FastAPI server that exposes a simple OpenAI-compatible
completion endpoint backed by TensorRT-LLM.

Run it with:
    python3 project-files/10_tensorrt_llm_server.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0

Then send requests to:
    http://localhost:8000/v1/completions

Supported body fields: model, prompt, max_tokens, temperature, top_p.
"""

import argparse
import json
import os
import sys
import time
import uuid
import asyncio
from typing import List, Optional, Union

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse

try:
    from tensorrt_llm import LLM, SamplingParams
except ImportError as exc:
    print("ERROR: tensorrt_llm is not installed or failed to import:", exc, file=sys.stderr)
    sys.exit(1)

app = FastAPI(title="TensorRT-LLM Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"] ,
    allow_headers=["*"] ,
)

llm = None
inference_lock = None
server_model_name = None


class CompletionRequest(BaseModel):
    model: Optional[str] = None
    prompt: Union[str, List[str]]
    max_tokens: Optional[int] = 50
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.95
    n: Optional[int] = 1
    stop: Optional[Union[str, List[str]]] = None


class CompletionChoice(BaseModel):
    text: str
    index: int
    logprobs: Optional[dict] = None
    finish_reason: Optional[str] = None


@app.on_event("startup")
async def startup_event():
    if llm is None:
        raise RuntimeError("LLM backend is not initialized")


@app.get("/health")
async def health():
    return {"status": "ok"}


def _run_generate(prompts: List[str], sampling_params: SamplingParams):
    outputs = []
    for output in llm.generate(prompts, sampling_params):
        outputs.append(output)
    return outputs


@app.post("/v1/completions")
async def completions(request: Request, body: CompletionRequest):
    prompts = [body.prompt] if isinstance(body.prompt, str) else body.prompt
    if not prompts:
        raise HTTPException(status_code=400, detail="prompt must not be empty")

    if body.model and body.model != server_model_name:
        raise HTTPException(
            status_code=400,
            detail=f"Server is configured for model '{server_model_name}', not '{body.model}'",
        )

    sampling_params = SamplingParams(
        temperature=body.temperature,
        top_p=body.top_p,
    )

    try:
        async with inference_lock:
            outputs = await run_in_threadpool(_run_generate, prompts, sampling_params)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    choices = []
    total_completion_tokens = 0
    for index, output in enumerate(outputs):
        text = output.outputs[0].text if output.outputs else ""
        token_count = len(text.split())
        total_completion_tokens += token_count
        choices.append(
            {
                "text": text,
                "index": index,
                "logprobs": None,
                "finish_reason": "length",
            }
        )

    response = {
        "id": str(uuid.uuid4()),
        "object": "text_completion",
        "created": int(time.time()),
        "model": server_model_name,
        "choices": choices,
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_completion_tokens,
        },
    }

    return JSONResponse(content=response)


def parse_args():
    parser = argparse.ArgumentParser(description="Start a TensorRT-LLM HTTP server")
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", help="Model name or path for TensorRT-LLM")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--max-model-len", type=int, default=2048, help="Model context length")
    parser.add_argument("--num-beams", type=int, default=1, help="Beam size")
    parser.add_argument("--device", default="cuda", help="Target device for TensorRT-LLM")
    parser.add_argument("--disable-progress", action="store_true", help="Disable model load progress output")
    return parser.parse_args()


def main():
    global llm, inference_lock, server_model_name

    args = parse_args()
    server_model_name = args.model
    inference_lock = asyncio.Lock()

    print(f"Starting TensorRT-LLM server with model: {args.model}")
    print(f"Target device: {args.device}")
    print("Loading model... this may take a while.")

    os.environ.setdefault("VLLM_TARGET_DEVICE", args.device)

    llm = LLM(model=args.model)

    print("Model loaded successfully.")
    print(f"OpenAI-compatible completion endpoint available at http://{args.host}:{args.port}/v1/completions")

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
