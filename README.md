# Shake n Break Sort (Breakpoint-Guided Window Shuffle Sort)

An **experimental, stochastic sorting algorithm** that treats sorting like an optimization problem.

Instead of comparing/splitting/merging like traditional sorts, Shake n Break Sort repeatedly **targets local disorder** (adjacent inversions) and tries to reduce it by **rearranging a sliding window** around a breakpoint. It mixes:
- **Deterministic “smart moves”** (sort the window, swap the breakpoint pair, rotate triples)
- **Random shuffles** (stochastic search inside the window)
- **Simulated annealing** (occasionally accept worse moves to escape local minima)
- **A “shake” mechanic** (random swaps when stagnating)
- **An insertion-sort finisher** (fast when nearly sorted)

This repo also includes a full **benchmark suite** to compare Shake n Break against Timsort / NumPy / reference Python implementations across many input distributions.

---

## What “breakpoints” are

A **breakpoint** is an adjacent inversion:

> breakpoint at index *i* if `a[i] > a[i+1]`

If an array has **0 breakpoints**, it is already sorted (nondecreasing).

Shake n Break Sort uses the **breakpoint count** as a lightweight “unsortedness score” to minimize.

---

## How the algorithm works

At a high level, each step does:

1. **Pick a breakpoint** to target  
   - Uses random sampling for speed, falls back to a full scan if sampling misses.

2. **Adapt parameters dynamically**  
   Based on:
   - `score_ratio = breakpoints / (n-1)` (how unsorted we are)
   - `stuck_ratio` (how long we’ve gone without improving)

   This adjusts:
   - window size (`batch_size`)
   - shuffle tries per step
   - annealing temperature `T`
   - “shake after” threshold

3. **Build a window** around the chosen breakpoint

4. **Try deterministic improvements**
   - **A:** sort the window
   - **B:** swap the breakpoint pair
   - **C:** rotate triples around the breakpoint

5. **Try random shuffles** inside the same window

6. **Accept / reject the best candidate**
   - Always accept improvements
   - Sometimes accept equal or worse moves (annealing)

7. **Shake if stuck**
   - If no progress for long enough, do random swaps inside the window

8. **Finish with insertion sort**
   - Once nearly sorted (< ~1% breakpoint ratio), insertion sort is very fast

### Key optimization
Only the window’s internal pairs **plus two boundary pairs** affect the global breakpoint score.  
So the algorithm can update the score using a **local delta** instead of rescanning the full array every step.

---

## Files in this repo

- **`Shake_n_Break_Sort.py`**  
  The algorithm implementation.
  - `dynamic_batch_bogo_sort(...)` — main entry point
  - `total_breakpoints(...)`, `breakpoint_indices(...)` and helpers

- **`benchmark_core.py`**  
  Benchmark engine + statistics, input generators, algorithm registry, report types, HTML output.

- **`benchmark_sorts.py`**  
  CLI runner for quick runs, full suites, scaling analysis, and convergence profiling.

---

## Quick start: use the sorter

```python
from Shake_n_Break_Sort import dynamic_batch_bogo_sort

arr = [5, 2, 9, 1, 5, 6]
sorted_arr = dynamic_batch_bogo_sort(arr, max_steps=300_000)

print(sorted_arr)  # should be sorted
```

### Notes
- The function **does not mutate the input**; it sorts a copy and returns it.
- It is stochastic; results can vary slightly by RNG state / seed.
- If `max_steps` is too low for a tough input, it returns the **current best-effort state at cutoff** (not guaranteed to be the best-ever state).

---

## API

### `dynamic_batch_bogo_sort(arr, max_steps=300_000, on_step=None, viz_every=1)`

- **`arr`**: any iterable of comparable elements
- **`max_steps`**: optimization budget (more steps → higher success rate)
- **`on_step(a, score, start, end, step, accepted, shook)`** *(optional)*:
  callback for visualization / logging  
  - `a`: current array state (copy reference to internal working list)
  - `score`: current total breakpoint score
  - `start, end`: window bounds used that step
  - `accepted`: whether a candidate move was accepted
  - `shook`: whether a shake happened that step
- **`viz_every`**: call `on_step` every N steps

---

## Benchmark suite

The benchmark runner supports:
- **Quick sanity check**
- **Full distribution suite** (random, reversed, nearly sorted, duplicates, etc.)
- **Scaling analysis** (estimate time complexity via curve fitting)
- **Convergence profiling** (how breakpoint score drops over steps)
- **Deep characteristic analysis**

### Useful flags
- `--slow` include slow pure-Python reference algorithms (heapsort/mergesort/quicksort/insertion)
- `--no-numpy` exclude NumPy algorithms even if NumPy is installed
- `--runs` control benchmark repetitions (default 10)
- `--quiet` reduce console output

### Run a quick benchmark
```bash
python benchmark_sorts.py --quick
```

### Full suite + HTML report
```bash
python benchmark_sorts.py --full --n 10000 --html report.html
```

### Scaling analysis
```bash
python benchmark_sorts.py --scaling --max-n 50000
```

### Convergence profile (stochastic behavior)
```bash
python benchmark_sorts.py --profile --n 5000 --max-steps 300000
```

### Deep analysis (behavior + sensitivity)
```bash
python benchmark_sorts.py --analyze --n 3000 --max-steps 300000
```

### Export JSON results
```bash
python benchmark_sorts.py --full --n 10000 --output report.json
```

---

## Expected performance and limitations

This is an **experimental** algorithm intended for exploration, not production:

- **Worst-case runtime is not bounded** in the same way as deterministic sorts.
- Expected behavior (empirically) tends to look **superlinear**; the benchmark suite labels it
  as **“O(n²) expected”** in `benchmark_core.py`.
- Performance depends heavily on:
  - `max_steps`
  - input distribution
  - duplicates / repeated values
  - randomness (stochastic search)

In contrast, **Timsort** (Python’s built-in) is extremely optimized and will dominate for real workloads.

---

## Design goals

Shake n Break Sort is useful if you care about:
- experimenting with **stochastic optimization** applied to classic problems
- building intuition for **local vs global disorder metrics**
- exploring “metaheuristic” hybrids (greedy + random + annealing + shake + finisher)
- producing interesting convergence and scaling plots via the benchmark suite

---

## Contributions

AI was used to create the benchmark suite. AI was also used to create comments in all files and write some of the README other than that everything else was created by me. 

---

## License

MIT License
