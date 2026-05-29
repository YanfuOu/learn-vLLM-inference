# 1 HuggingFace Inference Baseline
Goal: Get the `HuggingFaceTB/SmolLM-135M` model running locally using HuggingFace transformers. </br> 
This represents out baseline single-request performance. Under load with multiple users, requests would queue up.</br>
Why `SmolLM-135M`: A tiny 135M parameter model was chosen because my 3 year old Ryzen 7 7840HS laptop doesn't even have a GPU. The following was done as a learning experiment, so feel free for folllowing. Thanks for reading. 

1. `docker pull vllm/vllm-openai-cpu:latest-x86_64` from vLLM's website for Intel/AMD x86 CPU only pre-built images. Run `docker run -it -v ./project-files:/project-files  --entrypoint bash   vllm/vllm-openai-cpu:latest-x86_64` to use the container.  
2. Ensure that `pip show torch` shows `Version: 2.11.0+cpu` and `pip show vllm` shows `Version: 0.21.0+cpu` 
3. `from transformers import AutoModelForCausalLM, AutoTokenizer` to import the needed libraries to run the model and to tokenize prompts.
4. `model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-135M")` loads the pretrained language model, while `tokenizer = AutoTokenizer.from_pretrained(model_name)` loads the tokenizer associated with that model.

Since language models cannot directly understand raw text, the tokenizer converts human-readable text into numerical tokens that the model can process. These tokens represent pieces of text (words, subwords, or characters) according to the model’s vocabulary. I have printed out a couple of vocabs from the model as well as the vocab size. 

5.  `inputs = tokenizer(prompt, return_tensors="pt")`  tokenizes the input
6. `outputs = model.generate(**inputs, max_new_tokens=100, do_sample=True, temperature=0.7,)` sends the model the tokenized input prompt and generates the output. 
7. Times and calculates the # of generated tokens, total time, and tokens/sec. 
Baseline Metrics:
```
tokens_per_second=49.62
total_time=2.0155
generated_tokens=100
```
Key insights:
- This is SINGLE-REQUEST performance, so no batching - one request at a time
- Under load with multiple users, requests would queue up

# 2 vLLM Inference Setup and Comparison
Goal: Run the exact same model `HuggingFaceTB/SmolLM-135M` but using vLLM inference engine and compare the results. 
1. `from vllm import LLM, SamplingParams` import the right libraries 
2. `llm = LLM(model=model_name, max_model_len=128, enforce_eager=True)` initializes the vLLM engine with our model. `max_model_len` represents the max total number of tokens the model is allowed to keep in its context window during inference. The total prompt tokens are calculated:  input `prompt tokens` + `generated tokens` = total tokens
3. `sampling_params = SamplingParams(temperature=0.7, max_tokens=100)`  This line creates a configuration object that controls how the language model generates text.
4. `outputs = llm.generate([prompt], sampling_params)` actual answer generation by the LLM 
5. `generated_text = outputs[0].outputs[0].text` gets the generated text 

Metrics
```
Generated tokens: 50
Total time: 0.82 seconds
Tokens per second: 61.3 tok/s

--- COMPARISON: HuggingFace vs vLLM ---
Metric                HuggingFace         vLLM
----------------------------------------------
Tokens/sec                   49.8         61.3
Total time                  1.00s        0.82s
vLLM is 1.2x faster in tokens/sec
```

Insights: 
- vLLM optimizes inference even for single requests because it was designed as a high-performance LLM serving engine, where as Transformers is a general-purpose model library
- vLLM has better KV cache optimizations 
- vLLM has more efficient memory management through PagedAttention(explained below) 

