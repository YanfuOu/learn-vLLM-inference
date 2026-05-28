from vllm import LLM, SamplingParams

print("Initializing...")

llm = LLM(
    model="HuggingFaceTB/SmolLM-135M",
    enforce_eager=True,
    max_model_len=128,
)

print("Loaded.")

params = SamplingParams(
    temperature=0.7,
    max_tokens=50,
)

outputs = llm.generate(
    ["What is an LLM?"],
    params,
)

print(outputs[0].outputs[0].text)