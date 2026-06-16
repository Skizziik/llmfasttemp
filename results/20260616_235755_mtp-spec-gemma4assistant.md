# Eval run — mtp-spec-gemma4assistant

- **UTC:** 2026-06-16 23:57:55
- **Backend:** llama.cpp server (build bankenichi-mtp)
- **Hardware:** NVIDIA GeForce RTX 4060, 8188 MiB
- **Model:** gemma-4-E4B-it-Q4_K_M.gguf
- **Settings:** ngl=99, ctx=4096, temperature=0.0 (greedy)
- **VRAM (model resident):** ~3702 MiB (idle 1000 → loaded 4702 MiB)
- **Avg generation speed:** **80.28 tok/s** across 10 prompts

## Summary

| # | id | category | entropy | tokens | tok/s | accept | latency (ms) | finish | peak VRAM (MiB) |
|---|----|----------|---------|-------:|------:|-------:|-------------:|--------|----------------:|
| 1 | code_fib | code | low | 184 | 97.0 | 0.736 | 1896.9 | stop | 4731 |
| 2 | json_book | structured | low | 71 | 92.64 | 0.667 | 766.4 | stop | 4731 |
| 3 | list_primes | list | very-low | 57 | 108.04 | 0.9 | 527.6 | stop | 4712 |
| 4 | reason_batball | reasoning | medium | 193 | 107.68 | 0.871 | 1792.3 | stop | 4711 |
| 5 | fact_capital | factual | low | 5 | 88.18 | 0.5 | 56.7 | stop | 4706 |
| 6 | prose_lighthouse | creative | high | 82 | 55.77 | 0.294 | 1470.3 | stop | 4708 |
| 7 | explain_hashmap | explanation | medium | 70 | 63.08 | 0.321 | 1109.7 | stop | 4708 |
| 8 | summarize_photosynthesis | summarization | medium | 55 | 99.39 | 0.727 | 553.4 | stop | 4690 |
| 9 | ru_blackhole | multilingual-ru | medium | 61 | 65.44 | 0.382 | 932.1 | stop | 4700 |
| 10 | longform_watercycle | long-form | medium | 132 | 59.18 | 0.351 | 2230.5 | stop | 4711 |

## Full answers (for quality comparison across versions)

### 1. `code_fib` — code (97.0 tok/s, 184 tok)

**Prompt:** Write a Python function `fib(n)` that returns the nth Fibonacci number using memoization. Return only the code in a single code block.

**Answer:**

```
```python
def fib(n):
    """
    Calculates the nth Fibonacci number using memoization.

    Args:
        n (int): The index of the Fibonacci number to calculate (n >= 0).

    Returns:
        int: The nth Fibonacci number.
    """
    memo = {}

    def fib_recursive(k):
        if k in memo:
            return memo[k]
        if k <= 1:
            return k
        
        result = fib_recursive(k - 1) + fib_recursive(k - 2)
        memo[k] = result
        return result

    if not isinstance(n, int) or n < 0:
        raise ValueError("Input must be a non-negative integer.")
        
    return fib_recursive(n)
```
```

### 2. `json_book` — structured (92.64 tok/s, 71 tok)

**Prompt:** Return a single JSON object describing a book with fields: title (string), author (string), year (integer), genres (array of strings). Output only valid JSON, no commentary.

**Answer:**

```
```json
{
  "title": "The Hitchhiker's Guide to the Galaxy",
  "author": "Douglas Adams",
  "year": 1979,
  "genres": [
    "Science Fiction",
    "Comedy",
    "Adventure"
  ]
}
```
```

### 3. `list_primes` — list (108.04 tok/s, 57 tok)

**Prompt:** List the first 20 prime numbers as a single comma-separated line, nothing else.

**Answer:**

```
2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71
```

### 4. `reason_batball` — reasoning (107.68 tok/s, 193 tok)

**Prompt:** A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost? Show your working briefly, then give the final answer.

**Answer:**

```
Let $B$ be the cost of the bat and $L$ be the cost of the ball.

We are given two pieces of information:
1. The total cost is $\$1.10$:
   $$B + L = 1.10$$
2. The bat costs $\$1.00$ more than the ball:
   $$B = L + 1.00$$

Substitute the second equation into the first equation:
$$(L + 1.00) + L = 1.10$$
$$2L + 1.00 = 1.10$$
$$2L = 1.10 - 1.00$$
$$2L = 0.10$$
$$L = 0.05$$

The ball costs $\$0.05$.

**Final Answer:** $\$0.05$
```

### 5. `fact_capital` — factual (88.18 tok/s, 5 tok)

**Prompt:** What is the capital of Australia? Answer with just the city name.

**Answer:**

```
Canberra
```

### 6. `prose_lighthouse` — creative (55.77 tok/s, 82 tok)

**Prompt:** Write a four-sentence story about a lighthouse keeper who befriends a whale. Exactly four sentences.

**Answer:**

```
Old Silas, the lighthouse keeper, found an enormous, gentle whale stranded near the rocks one foggy morning. Over the next few weeks, the massive creature began visiting the tower, breaching near the beam of light. They developed a quiet understanding, communicating through shared silences and the rhythmic sweep of the lamp. Soon, Silas would wave from the window as the whale swam out to meet the rising sun.
```

### 7. `explain_hashmap` — explanation (63.08 tok/s, 70 tok)

**Prompt:** Explain how a hash map works and why average lookup is O(1), in exactly three sentences.

**Answer:**

```
A hash map uses a hash function to convert keys into indices within an underlying array, which determines where the corresponding value is stored. This direct calculation allows the map to locate the correct storage slot in near-constant time, assuming minimal collisions. Therefore, the average time complexity for insertion, deletion, and lookup remains $O(1)$.
```

### 8. `summarize_photosynthesis` — summarization (99.39 tok/s, 55 tok)

**Prompt:** Summarize the following in two sentences: Photosynthesis is the process by which green plants, algae, and some bacteria convert light energy, usually from the sun, into chemical energy stored in glucose. It takes place mainly in the chloroplasts using the pigment chlorophyll, consumes carbon dioxide and water, and releases oxygen as a byproduct.

**Answer:**

```
Photosynthesis is the vital process where green plants, algae, and certain bacteria convert light energy, typically from the sun, into chemical energy stored as glucose. This process occurs primarily in chloroplasts using chlorophyll, consuming carbon dioxide and water while releasing oxygen as a byproduct.
```

### 9. `ru_blackhole` — multilingual-ru (65.44 tok/s, 61 tok)

**Prompt:** Объясни простыми словами в двух предложениях, что такое чёрная дыра.

**Answer:**

```
Чёрная дыра — это место в космосе, где гравитация настолько сильна, что даже свет не может оттуда убежать. Из-за этой невероятной силы она "засасывает" всё, что попадает в её гравитационное поле.
```

### 10. `longform_watercycle` — long-form (59.18 tok/s, 132 tok)

**Prompt:** Describe the water cycle in about 120 words, covering evaporation, condensation, precipitation, and collection.

**Answer:**

```
The water cycle is Earth's continuous movement of water through different states. It begins with **evaporation**, where solar energy heats liquid water from oceans, rivers, and lakes, turning it into water vapor that rises into the atmosphere. As this vapor ascends, it cools and undergoes **condensation**, changing back into tiny liquid droplets that form clouds. When these droplets become too heavy, they fall back to Earth as **precipitation**—rain, snow, sleet, or hail. Finally, this water is **collected** in bodies of water, or it soaks into the ground (infiltration), ready to restart the cycle, ensuring life on Earth.
```
