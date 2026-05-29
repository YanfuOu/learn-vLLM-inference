#!/usr/bin/env python3
"""
Naive HuggingFace Inference - The Baseline
Measure baseline inference speed using raw HuggingFace transformers.
"""

import os
import time


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "HuggingFaceTB/SmolLM-135M"
    prompt = "Explain what a large language model is in simple terms."
    # expect chall = 28010
    print(f"\nModel: {model_name}")
    print(f"Prompt: \"{prompt}\"")
    print("-" * 65)

    # --- LOAD MODEL ---
    print("\nLoading model with HuggingFace transformers...")
    model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-135M")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("-------> vocab list")
    vocab = tokenizer.get_vocab()
    print(list(vocab.items())[0:20])
    print("-------> tokenizer.vocab_size", tokenizer.vocab_size)

    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Model loaded successfully.")

    # --- GENERATE ---
    print("\nGenerating with HuggingFace transformers...")
    inputs = tokenizer(prompt, return_tensors="pt")
    print("---> inputs: ", inputs)

    start_time = time.time()
    outputs = model.generate(
        **inputs,
        max_new_tokens=50, 
        do_sample=True,
        temperature=0.7,
    )
    end_time = time.time()

    # visualize the output tokens and their decoded text
    print("---> the output is ", outputs, " below is each tet and its encoding")
    for text in outputs[0]:
        decoded_text = tokenizer.decode(text, skip_special_tokens=True)
        print(f"(encoded: {text}, decoded: {decoded_text})")

    # Calculate metrics
    input_tokens = inputs["input_ids"].shape[1]
    total_tokens = outputs.shape[1]
    generated_tokens = total_tokens - input_tokens
    total_time = end_time - start_time
    tokens_per_second = generated_tokens / total_time

    # Decode output
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # --- RESULTS ---
    print("\n--- RESULTS ---")
    print(f"Generated text: {output_text[:1000]}...")
    print(f"\nGenerated tokens: {generated_tokens}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Tokens per second: {tokens_per_second:.1f} tok/s")

    # Save baseline for later comparison 
    script_dir = os.path.dirname(os.path.abspath(__file__))
    markers_dir = os.path.join(script_dir, "markers")
    baseline_file = os.path.join(markers_dir, "hf_baseline.txt")
    os.makedirs(markers_dir, exist_ok=True)
    with open(baseline_file, "w") as f:
        f.write(f"tokens_per_second={tokens_per_second:.2f}\n")
        f.write(f"total_time={total_time:.4f}\n")
        f.write(f"generated_tokens={generated_tokens}\n")

    # Clean up
    del model
    del tokenizer


if __name__ == "__main__":
    main()
