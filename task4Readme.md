Task 4: PagedAttention - vLLM's Solution

TASK OVERVIEW

Now let's see how vLLM solves the KV cache problem. vLLM introduced PagedAttention, which was directly inspired by how operating systems manage virtual memory using paging.

CONCEPT: PagedAttention (OS Paging Analogy)

Instead of pre-allocating a large contiguous block per request:

vLLM uses small, fixed-size pages (like OS memory pages)
Pages are allocated ON DEMAND as the model generates tokens
No worst-case padding needed - memory grows dynamically
Utilization jumps from ~20% to ~95%
This is the exact same concept as OS virtual memory:

OS divides RAM into 4KB pages, allocated on demand
vLLM divides KV cache into token-sized pages, allocated on demand
REAL-WORLD USE CASE:
Instead of reserving 50 theater seats per group, you give seats one at a time as people arrive. Now the same theater fits 4-5x more groups.