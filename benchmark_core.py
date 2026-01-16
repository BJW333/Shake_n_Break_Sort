"""
Sorting Algorithm Benchmark Suite - Core Module
===============================================

Contains: statistics, input generators, algorithm definitions,
benchmark engine, and report generation.
"""

from __future__ import annotations
import gc, json, math, platform, random, statistics, sys, time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple
from contextlib import contextmanager

from Shake_n_Break_Sort import dynamic_batch_bogo_sort, total_breakpoints

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False

# =============================================================================
# Configuration
# =============================================================================

@dataclass(frozen=True)
class BenchmarkConfig:
    seed: int = 42
    warmup_runs: int = 2
    min_runs: int = 5
    max_runs: int = 50
    target_uncertainty: float = 0.05
    confidence_level: float = 0.95
    outlier_threshold: float = 2.5
    gc_between_runs: bool = True
    timeout_seconds: float = 60.0
    max_steps_stochastic: int = 300_000

SCALING_SIZES = [100, 250, 500, 1000, 2500, 5000, 10000, 25000, 50000]

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'
    
    @classmethod
    def disable(cls):
        for a in ['HEADER','BLUE','CYAN','GREEN','YELLOW','RED','BOLD','UNDERLINE','END']:
            setattr(cls, a, '')

if not sys.stdout.isatty():
    Colors.disable()

# =============================================================================
# Statistics
# =============================================================================

@dataclass
class Statistics:
    n: int
    mean: float
    median: float
    std_dev: float
    std_err: float
    ci_lower: float
    ci_upper: float
    min_val: float
    max_val: float
    p5: float
    p25: float
    p75: float
    p95: float
    iqr: float
    outliers_removed: int
    raw_values: List[float] = field(default_factory=list, repr=False)

    @classmethod
    def from_samples(cls, samples: List[float], confidence: float = 0.95,
                     remove_outliers: bool = True, outlier_iqr_mult: float = 2.5) -> "Statistics":
        if not samples:
            return cls(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,[])
        
        def pct(data, p):
            k = (len(data)-1) * p / 100
            f, c = int(k), min(int(k)+1, len(data)-1)
            return data[f] if f == c else data[f]*(c-k) + data[c]*(k-f)
        
        s = sorted(samples)
        n = len(s)
        p25, p75 = pct(s, 25), pct(s, 75)
        iqr = p75 - p25
        outliers = 0
        
        if remove_outliers and n > 4:
            lo, hi = p25 - outlier_iqr_mult*iqr, p75 + outlier_iqr_mult*iqr
            filtered = [x for x in s if lo <= x <= hi]
            if len(filtered) >= 3:
                outliers = n - len(filtered)
                s = filtered
                n = len(s)
        
        mean = statistics.mean(s)
        med = statistics.median(s)
        std = statistics.stdev(s) if n > 1 else 0.0
        se = std / math.sqrt(n) if n > 0 else 0.0
        
        # t-distribution critical values for 95% CI
        t_vals = {3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447,
                  8: 2.365, 9: 2.306, 10: 2.262, 15: 2.145, 20: 2.093, 30: 2.045}
        
        if n <= 2:
            t = 12.706
        elif n >= 100:
            t = 1.96
        else:
            # Find smallest key >= n, or use 2.0 as fallback
            candidates = [k for k in t_vals if k >= n]
            t = t_vals[min(candidates)] if candidates else 2.0
        
        margin = t * se
        
        return cls(n=n, mean=mean, median=med, std_dev=std, std_err=se,
                   ci_lower=mean-margin, ci_upper=mean+margin,
                   min_val=s[0], max_val=s[-1],
                   p5=pct(s,5), p25=pct(s,25), p75=pct(s,75), p95=pct(s,95),
                   iqr=iqr, outliers_removed=outliers, raw_values=samples)

    @property
    def relative_ci(self) -> float:
        return (self.ci_upper - self.ci_lower) / (2 * self.mean) if self.mean else float('inf')


def estimate_complexity(sizes: List[int], times: List[float]) -> Tuple[str, float]:
    """Estimate Big-O complexity via curve fitting."""
    if len(sizes) < 3 or len(times) < 3:
        return ("unknown", 0.0)
    
    def r2(pred, act):
        m = sum(act) / len(act)
        ss_tot = sum((a - m) ** 2 for a in act)
        if ss_tot == 0:
            return 0.0
        return 1 - sum((a - p) ** 2 for a, p in zip(act, pred)) / ss_tot
    
    def fit(x, y):
        n = len(x)
        sx, sy = sum(x), sum(y)
        sxy = sum(a * b for a, b in zip(x, y))
        sxx = sum(a * a for a in x)
        d = n * sxx - sx * sx
        if abs(d) < 1e-10:
            return (0, sy / n if n else 0)
        return ((n * sxy - sx * sy) / d, (sy * sxx - sx * sxy) / d)
    
    nln = [s * math.log(s) for s in sizes]
    n2 = [s * s for s in sizes]
    
    cands = []
    
    # O(n)
    m, b = fit(sizes, times)
    cands.append(("O(n)", r2([m*s + b for s in sizes], times)))
    
    # O(n log n)
    m, b = fit(nln, times)
    cands.append(("O(n log n)", r2([m*s + b for s in nln], times)))
    
    # O(n^2)
    m, b = fit(n2, times)
    cands.append(("O(n^2)", r2([m*s + b for s in n2], times)))
    
    return max(cands, key=lambda x: x[1])


# =============================================================================
# Input Generators
# =============================================================================

class InputGenerator(ABC):
    @property
    @abstractmethod
    def name(self) -> str: pass
    
    @property
    @abstractmethod
    def description(self) -> str: pass
    
    @abstractmethod
    def generate(self, n: int, rng: random.Random) -> List[int]: pass


class RandomPermutation(InputGenerator):
    name = "random"
    description = "Uniformly random permutation"
    def generate(self, n, rng):
        a = list(range(n))
        rng.shuffle(a)
        return a

class ReverseSorted(InputGenerator):
    name = "reversed"
    description = "Descending order"
    def generate(self, n, rng):
        return list(range(n-1, -1, -1))

class AlreadySorted(InputGenerator):
    name = "sorted"
    description = "Already sorted"
    def generate(self, n, rng):
        return list(range(n))

class NearlySorted(InputGenerator):
    name = "nearly_sorted"
    description = "Sorted with ~1% swaps"
    def generate(self, n, rng):
        a = list(range(n))
        for _ in range(max(1, n // 100)):
            i, j = rng.randrange(n), rng.randrange(n)
            a[i], a[j] = a[j], a[i]
        return a

class FewUnique(InputGenerator):
    name = "few_unique"
    description = "Only 10 unique values"
    def generate(self, n, rng):
        return [rng.randrange(10) for _ in range(n)]

class ManyDuplicates(InputGenerator):
    name = "duplicates"
    description = "~sqrt(n) unique values"
    def generate(self, n, rng):
        return [rng.randrange(max(2, int(math.sqrt(n)))) for _ in range(n)]

class Sawtooth(InputGenerator):
    name = "sawtooth"
    description = "Repeating ascending runs"
    def generate(self, n, rng):
        t = max(1, n // 10)
        return [i % t for i in range(n)]

class PipeOrgan(InputGenerator):
    name = "pipe_organ"
    description = "Ascending then descending"
    def generate(self, n, rng):
        m = n // 2
        return list(range(m)) + list(range(n - m - 1, -1, -1))

class PushFront(InputGenerator):
    name = "push_front"
    description = "Sorted except smallest at end"
    def generate(self, n, rng):
        return list(range(1, n)) + [0]

class RandomRange(InputGenerator):
    name = "random_range"
    description = "Random ints in wide range"
    def generate(self, n, rng):
        return [rng.randint(-10**9, 10**9) for _ in range(n)]

class AllEqual(InputGenerator):
    name = "all_equal"
    description = "All elements identical"
    def generate(self, n, rng):
        return [42] * n

class BitonicSequence(InputGenerator):
    name = "bitonic"
    description = "Ascending then descending"
    def generate(self, n, rng):
        m = n // 2
        return list(range(m)) + list(range(m, 0, -1))


INPUT_GENERATORS: Dict[str, InputGenerator] = {g.name: g for g in [
    RandomPermutation(), ReverseSorted(), AlreadySorted(), NearlySorted(),
    FewUnique(), ManyDuplicates(), Sawtooth(), PipeOrgan(), PushFront(),
    RandomRange(), AllEqual(), BitonicSequence()
]}


# =============================================================================
# Algorithms
# =============================================================================

@dataclass
class AlgorithmInfo:
    name: str
    function: Callable
    category: str
    expected_complexity: str
    stable: bool
    in_place: bool
    description: str
    
    def __call__(self, arr):
        return self.function(arr)


def _timsort(a):
    return sorted(a)

def _timsort_ip(a):
    a.sort()
    return a

def _heapsort(a):
    import heapq
    h = a[:]
    heapq.heapify(h)
    return [heapq.heappop(h) for _ in range(len(h))]

def _mergesort(a):
    if len(a) <= 1:
        return a[:]
    m = len(a) // 2
    L, R = _mergesort(a[:m]), _mergesort(a[m:])
    res, i, j = [], 0, 0
    while i < len(L) and j < len(R):
        if L[i] <= R[j]:
            res.append(L[i])
            i += 1
        else:
            res.append(R[j])
            j += 1
    return res + L[i:] + R[j:]

def _quicksort(a):
    if len(a) <= 1:
        return a[:]
    p = sorted([a[0], a[len(a)//2], a[-1]])[1]
    return (_quicksort([x for x in a if x < p]) +
            [x for x in a if x == p] +
            _quicksort([x for x in a if x > p]))

def _insertion(a):
    r = a[:]
    for i in range(1, len(r)):
        k, j = r[i], i - 1
        while j >= 0 and r[j] > k:
            r[j + 1] = r[j]
            j -= 1
        r[j + 1] = k
    return r

def _numpy_sort(a):
    return np.sort(a).tolist()

def _numpy_stable(a):
    x = np.array(a)
    x.sort(kind='stable')
    return x.tolist()

def _numpy_qs(a):
    x = np.array(a)
    x.sort(kind='quicksort')
    return x.tolist()

def _numpy_hs(a):
    x = np.array(a)
    x.sort(kind='heapsort')
    return x.tolist()

def _make_bp_sort(max_steps):
    def f(a):
        return dynamic_batch_bogo_sort(a, max_steps=max_steps)
    return f


def get_algorithms(config: BenchmarkConfig, include_slow=False, include_numpy=True) -> Dict[str, AlgorithmInfo]:
    algos = {
        "Timsort (sorted)": AlgorithmInfo(
            "Timsort (sorted)", _timsort, "production",
            "O(n log n)", True, False, "Python built-in sorted()"),
        "Timsort (inplace)": AlgorithmInfo(
            "Timsort (inplace)", _timsort_ip, "production",
            "O(n log n)", True, True, "Python list.sort()"),
        "Shake n Break Sort": AlgorithmInfo(
            "Shake n Break Sort", _make_bp_sort(config.max_steps_stochastic),
            "experimental", "O(n^2) expected", False, False,
            "Stochastic breakpoint minimization via window shuffling"),
    }
    
    if HAS_NUMPY and include_numpy:
        algos.update({
            "NumPy sort": AlgorithmInfo(
                "NumPy sort", _numpy_sort, "production",
                "O(n log n)", False, False, "NumPy default sort"),
            "NumPy stable": AlgorithmInfo(
                "NumPy stable", _numpy_stable, "production",
                "O(n log n)", True, True, "NumPy stable mergesort"),
            "NumPy quicksort": AlgorithmInfo(
                "NumPy quicksort", _numpy_qs, "production",
                "O(n log n)", False, True, "NumPy introsort"),
            "NumPy heapsort": AlgorithmInfo(
                "NumPy heapsort", _numpy_hs, "production",
                "O(n log n)", False, True, "NumPy heapsort"),
        })
    
    if include_slow:
        algos.update({
            "Heapsort (Python)": AlgorithmInfo(
                "Heapsort (Python)", _heapsort, "reference",
                "O(n log n)", False, False, "Pure Python heapsort"),
            "Mergesort (Python)": AlgorithmInfo(
                "Mergesort (Python)", _mergesort, "reference",
                "O(n log n)", True, False, "Pure Python mergesort"),
            "Quicksort (Python)": AlgorithmInfo(
                "Quicksort (Python)", _quicksort, "reference",
                "O(n log n)", False, False, "Pure Python quicksort"),
            "Insertion (Python)": AlgorithmInfo(
                "Insertion (Python)", _insertion, "reference",
                "O(n^2)", True, True, "Pure Python insertion sort"),
        })
    
    return algos


# =============================================================================
# Result Types
# =============================================================================

@dataclass
class AlgorithmResult:
    algorithm: str
    input_type: str
    n: int
    stats: Statistics
    correct: bool
    stable: Optional[bool]
    error: Optional[str] = None
    
    def to_dict(self):
        return {
            "algorithm": self.algorithm,
            "input_type": self.input_type,
            "n": self.n,
            "correct": self.correct,
            "stable": self.stable,
            "error": self.error,
            "stats": {
                "n_samples": self.stats.n,
                "mean_seconds": self.stats.mean,
                "median_seconds": self.stats.median,
                "std_dev": self.stats.std_dev,
                "ci_lower": self.stats.ci_lower,
                "ci_upper": self.stats.ci_upper,
                "min": self.stats.min_val,
                "max": self.stats.max_val,
                "p5": self.stats.p5,
                "p25": self.stats.p25,
                "p75": self.stats.p75,
                "p95": self.stats.p95,
            }
        }


@dataclass
class ScalingResult:
    algorithm: str
    sizes: List[int]
    times: List[float]
    estimated_complexity: str
    r_squared: float
    
    def to_dict(self):
        return {
            "algorithm": self.algorithm,
            "sizes": self.sizes,
            "median_times": self.times,
            "estimated_complexity": self.estimated_complexity,
            "r_squared": self.r_squared,
        }


@dataclass
class BenchmarkReport:
    metadata: Dict[str, Any]
    algorithm_info: Dict[str, Dict[str, Any]]
    results: List[AlgorithmResult]
    scaling: List[ScalingResult]
    convergence_profile: Optional[Dict[str, Any]] = None
    
    def to_dict(self):
        return {
            "metadata": self.metadata,
            "algorithm_info": self.algorithm_info,
            "results": [r.to_dict() for r in self.results],
            "scaling": [s.to_dict() for s in self.scaling],
            "convergence_profile": self.convergence_profile,
        }


# =============================================================================
# Benchmark Engine
# =============================================================================

class BenchmarkEngine:
    def __init__(self, config: BenchmarkConfig = BenchmarkConfig()):
        self.config = config
    
    @contextmanager
    def _gc_pause(self):
        if self.config.gc_between_runs:
            gc.collect()
            gc.disable()
        try:
            yield
        finally:
            if self.config.gc_between_runs:
                gc.enable()
    
    def time_once(self, fn, arr):
        a = arr[:]
        with self._gc_pause():
            t0 = time.perf_counter()
            result = fn(a)
            t1 = time.perf_counter()
        return (t1 - t0, result if result is not None else a)
    
    def verify(self, fn, arr):
        try:
            result = fn(arr[:])
            return (list(result if result else arr) == sorted(arr), None)
        except Exception as e:
            return (False, str(e))
    
    def benchmark_algorithm(self, algo: AlgorithmInfo, gen: InputGenerator, n: int) -> AlgorithmResult:
        inputs = [gen.generate(n, random.Random(self.config.seed + i))
                  for i in range(self.config.max_runs + self.config.warmup_runs)]
        
        correct, err = self.verify(algo.function, inputs[0])
        if not correct:
            return AlgorithmResult(algo.name, gen.name, n, Statistics.from_samples([]),
                                   False, None, err)
        
        # Warmup
        for i in range(self.config.warmup_runs):
            try:
                self.time_once(algo.function, inputs[i])
            except:
                pass
        
        # Timed runs
        times = []
        for i in range(self.config.warmup_runs, len(inputs)):
            try:
                t, _ = self.time_once(algo.function, inputs[i])
                times.append(t)
                if len(times) >= self.config.min_runs:
                    s = Statistics.from_samples(times)
                    if s.relative_ci <= self.config.target_uncertainty:
                        break
            except Exception as e:
                return AlgorithmResult(algo.name, gen.name, n, Statistics.from_samples(times),
                                       True, None, str(e))
        
        return AlgorithmResult(algo.name, gen.name, n, Statistics.from_samples(times),
                               correct, None, None)
    
    def run_scaling_analysis(self, algo: AlgorithmInfo, gen: InputGenerator,
                             sizes: List[int], runs_per_size: int = 5) -> ScalingResult:
        medians = []
        for n in sizes:
            inputs = [gen.generate(n, random.Random(self.config.seed + i))
                      for i in range(runs_per_size)]
            times = []
            for arr in inputs:
                try:
                    t, _ = self.time_once(algo.function, arr)
                    times.append(t)
                except:
                    times.append(float('inf'))
            medians.append(statistics.median(times))
        
        valid = [(s, t) for s, t in zip(sizes, medians) if t < float('inf')]
        if len(valid) >= 3:
            valid_sizes, valid_times = zip(*valid)
            comp, r2 = estimate_complexity(list(valid_sizes), list(valid_times))
        else:
            comp, r2 = ("unknown", 0.0)
        
        return ScalingResult(algo.name, sizes, medians, comp, r2)


# =============================================================================
# Profiler
# =============================================================================

def profile_convergence(n=5000, seed=42, max_steps=300_000, sample_interval=100):
    """Profile the breakpoint sort algorithm's convergence behavior."""
    rng = random.Random(seed)
    arr = list(range(n))
    rng.shuffle(arr)
    
    initial = total_breakpoints(arr)
    samples = []
    window_sizes = []
    accepted_count = 0
    total_count = 0
    shake_count = 0
    
    def tracker(a, score, start, end, step, acc, shook):
        nonlocal accepted_count, total_count, shake_count
        total_count += 1
        if acc:
            accepted_count += 1
        if shook:
            shake_count += 1
        if step % sample_interval == 0:
            samples.append({
                "step": step,
                "breakpoints": score,
                "reduction_pct": 100 * (1 - score / initial) if initial else 100,
                "window_size": end - start,
            })
            window_sizes.append(end - start)
    
    t0 = time.perf_counter()
    result = dynamic_batch_bogo_sort(arr, max_steps=max_steps, on_step=tracker, viz_every=1)
    elapsed = time.perf_counter() - t0
    
    return {
        "n": n,
        "initial_breakpoints": initial,
        "final_breakpoints": total_breakpoints(result),
        "correctly_sorted": result == sorted(arr),
        "total_time_seconds": elapsed,
        "steps_taken": samples[-1]["step"] if samples else 0,
        "steps_per_second": (samples[-1]["step"] / elapsed) if samples and elapsed else 0,
        "convergence_samples": samples,
        "avg_acceptance_rate": accepted_count / total_count if total_count else 0,
        "total_shakes": shake_count,
        "avg_window_size": statistics.mean(window_sizes) if window_sizes else 0,
    }


def analyze_algorithm_characteristics(n=5000, seed=42, max_steps=300_000):
    """Deep analysis of breakpoint sort behavior."""
    rng = random.Random(seed)
    results = {"n": n, "analyses": {}}
    
    # 1. Input sensitivity analysis
    sensitivity = {}
    for name, gen in [("random", RandomPermutation()), ("nearly_sorted", NearlySorted()),
                      ("reversed", ReverseSorted()), ("few_unique", FewUnique())]:
        arr = gen.generate(n, random.Random(seed))
        initial_bp = total_breakpoints(arr)
        t0 = time.perf_counter()
        result = dynamic_batch_bogo_sort(arr, max_steps=max_steps)
        elapsed = time.perf_counter() - t0
        sensitivity[name] = {
            "initial_breakpoints": initial_bp,
            "time_seconds": elapsed,
            "success": result == sorted(arr),
        }
    results["analyses"]["input_sensitivity"] = sensitivity
    
    # 2. Empirical complexity
    sizes = [500, 1000, 2000, 3000, 5000]
    times = []
    for sz in sizes:
        arr = list(range(sz))
        rng.shuffle(arr)
        t0 = time.perf_counter()
        dynamic_batch_bogo_sort(arr, max_steps=max_steps)
        times.append(time.perf_counter() - t0)
    
    complexity, r2 = estimate_complexity(sizes, times)
    results["analyses"]["complexity"] = {
        "sizes": sizes,
        "times": times,
        "estimated": complexity,
        "r_squared": r2,
    }
    
    # 3. Comparison with insertion sort
    arr = list(range(n))
    rng.shuffle(arr)
    
    t0 = time.perf_counter()
    _insertion(arr[:])
    insertion_time = time.perf_counter() - t0
    
    t0 = time.perf_counter()
    dynamic_batch_bogo_sort(arr[:], max_steps=max_steps)
    bp_time = time.perf_counter() - t0
    
    results["analyses"]["vs_insertion"] = {
        "insertion_sort_seconds": insertion_time,
        "breakpoint_sort_seconds": bp_time,
        "ratio": bp_time / insertion_time if insertion_time > 0 else float('inf'),
    }
    
    return results


# =============================================================================
# Formatting & Output
# =============================================================================

def fmt_time(t):
    """Format time with appropriate units."""
    if t == float('inf'):
        return "timeout"
    if t < 1e-6:
        return f"{t*1e9:.1f}ns"
    if t < 1e-3:
        return f"{t*1e6:.1f}us"
    if t < 1:
        return f"{t*1e3:.2f}ms"
    return f"{t:.3f}s"


def fmt_ci(s):
    """Format confidence interval."""
    return f"+/-{s.relative_ci*100:.1f}%"


def get_system_info():
    """Gather system information for reproducibility."""
    info = {
        "timestamp": datetime.now().isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "machine": platform.machine(),
        "numpy_available": HAS_NUMPY,
    }
    if HAS_NUMPY:
        info["numpy_version"] = np.__version__
    return info


def print_header(text):
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.HEADER}{text.center(70)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'='*70}{Colors.END}\n")


def print_subheader(text):
    print(f"\n{Colors.BOLD}{Colors.CYAN}{text}{Colors.END}")
    print(f"{Colors.CYAN}{'-'*len(text)}{Colors.END}")


def print_results_table(results, baseline="Timsort (sorted)"):
    """Print formatted benchmark results table."""
    base = next((r.stats.median for r in results if r.algorithm == baseline and r.stats.n > 0), None)
    results = sorted(results, key=lambda r: r.stats.median if r.stats.n > 0 else float('inf'))
    
    hdr = f"{'Algorithm':<24} {'Median':>10} {'95% CI':>10} {'Min':>10} {'Max':>10} {'vs Base':>10} {'Status':>8}"
    print(f"{Colors.BOLD}{hdr}{Colors.END}")
    print("-" * len(hdr))
    
    for r in results:
        if r.stats.n == 0 or r.error:
            print(f"{r.algorithm:<24} {'-':>10} {'-':>10} {'-':>10} {'-':>10} {'-':>10} {Colors.RED}FAIL{Colors.END}")
            continue
        
        s = r.stats
        ratio = f"{s.median/base:.2f}x" if base else "-"
        
        if base and s.median <= base * 1.1:
            clr = Colors.GREEN
        elif base and s.median <= base * 2:
            clr = Colors.YELLOW
        else:
            clr = Colors.RED
        
        status = f"{Colors.GREEN}OK{Colors.END}" if r.correct else f"{Colors.RED}FAIL{Colors.END}"
        print(f"{r.algorithm:<24} {fmt_time(s.median):>10} {fmt_ci(s):>10} "
              f"{fmt_time(s.min_val):>10} {fmt_time(s.max_val):>10} {clr}{ratio}{Colors.END:>10} {status}")


def print_scaling_table(results):
    """Print scaling analysis results."""
    if not results:
        return
    
    sizes = results[0].sizes
    hdr = f"{'Algorithm':<24} " + " ".join(f"{'n='+str(s):>10}" for s in sizes) + f" {'Complexity':>14}"
    print(f"{Colors.BOLD}{hdr}{Colors.END}")
    print("-" * len(hdr))
    
    for sr in results:
        row = f"{sr.algorithm:<24} " + " ".join(f"{fmt_time(t):>10}" for t in sr.times)
        print(row + f" {sr.estimated_complexity:>14}")


def print_convergence_profile(p):
    """Print convergence profile for the stochastic algorithm."""
    print_subheader("Convergence Profile: Shake n Break Sort")
    
    print(f"  Input size (n):        {p['n']:,}")
    print(f"  Initial breakpoints:   {p['initial_breakpoints']:,}")
    print(f"  Final breakpoints:     {p['final_breakpoints']}")
    print(f"  Correctly sorted:      {p['correctly_sorted']}")
    print(f"  Total time:            {fmt_time(p['total_time_seconds'])}")
    print(f"  Steps taken:           {p['steps_taken']:,}")
    print(f"  Throughput:            {p['steps_per_second']:,.0f} steps/sec")
    print(f"  Acceptance rate:       {p.get('avg_acceptance_rate', 0)*100:.1f}%")
    print(f"  Avg window size:       {p.get('avg_window_size', 0):.1f}")
    print(f"  Total shakes:          {p.get('total_shakes', 0):,}")
    
    samples = p.get('convergence_samples', [])
    if samples:
        print(f"\n  {Colors.CYAN}Convergence Milestones:{Colors.END}")
        milestones_hit = set()
        for m in [50, 75, 90, 95, 99, 100]:
            for s in samples:
                if s['reduction_pct'] >= m and m not in milestones_hit:
                    print(f"    {m:>3}% reduction at step {s['step']:>6,} (breakpoints: {s['breakpoints']})")
                    milestones_hit.add(m)
                    break


def print_analysis_report(analysis: dict):
    """Print deep analysis results."""
    print_subheader("Algorithm Characteristics Analysis")
    
    # Input sensitivity
    print(f"\n  {Colors.CYAN}Input Sensitivity:{Colors.END}")
    sens = analysis["analyses"]["input_sensitivity"]
    for name, data in sens.items():
        status = f"{Colors.GREEN}OK{Colors.END}" if data["success"] else f"{Colors.RED}FAIL{Colors.END}"
        print(f"    {name:<15} {fmt_time(data['time_seconds']):>10}  "
              f"(initial bp: {data['initial_breakpoints']:>5}) {status}")
    
    # Complexity
    comp = analysis["analyses"]["complexity"]
    print(f"\n  {Colors.CYAN}Empirical Complexity:{Colors.END}")
    print(f"    Estimated: {comp['estimated']} (R^2 = {comp['r_squared']:.4f})")
    print(f"    Data points: {list(zip(comp['sizes'], [f'{t:.3f}s' for t in comp['times']]))}")
    
    # vs Insertion sort
    vs = analysis["analyses"]["vs_insertion"]
    print(f"\n  {Colors.CYAN}Comparison with Insertion Sort (both ~O(n^2)):{Colors.END}")
    print(f"    Insertion sort:  {fmt_time(vs['insertion_sort_seconds'])}")
    print(f"    Breakpoint sort: {fmt_time(vs['breakpoint_sort_seconds'])}")
    print(f"    Ratio: {vs['ratio']:.1f}x slower")


def generate_html_report(report: BenchmarkReport, path: str):
    """Generate HTML report with Chart.js visualizations."""
    css = """<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:system-ui,-apple-system,sans-serif;background:#0d1117;color:#c9d1d9;padding:2rem}
.container{max-width:1200px;margin:0 auto}
h1{font-size:2rem;background:linear-gradient(90deg,#58a6ff,#a371f7);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:1rem}
h2{color:#58a6ff;margin:2rem 0 1rem;border-bottom:1px solid #21262d;padding-bottom:.5rem}
.card{background:#161b22;border-radius:8px;padding:1.5rem;margin-bottom:1.5rem;border:1px solid #30363d}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:1rem}
.stat{background:#21262d;padding:1rem;border-radius:6px}
.stat-label{font-size:.75rem;color:#8b949e;text-transform:uppercase}
.stat-value{font-size:1.25rem;color:#fff;margin-top:.25rem}
table{width:100%;border-collapse:collapse}
th,td{padding:.75rem;text-align:left;border-bottom:1px solid #21262d}
th{background:#21262d;color:#58a6ff}
.ok{color:#3fb950}.fail{color:#f85149}
.chart{height:350px;margin:1rem 0}
footer{text-align:center;padding:2rem;color:#6e7681}
</style>"""
    
    # Build results table
    results_html = ""
    for r in sorted(report.results, key=lambda x: (x.input_type, x.stats.median if x.stats.n else float('inf'))):
        if r.stats.n == 0:
            results_html += f"<tr><td>{r.algorithm}</td><td>{r.input_type}</td><td>{r.n:,}</td><td>-</td><td>-</td><td class='fail'>FAIL</td></tr>"
        else:
            s = r.stats
            status_cls = "ok" if r.correct else "fail"
            status_txt = "OK" if r.correct else "FAIL"
            results_html += f"<tr><td>{r.algorithm}</td><td>{r.input_type}</td><td>{r.n:,}</td><td>{fmt_time(s.median)}</td><td>+/-{s.relative_ci*100:.1f}%</td><td class='{status_cls}'>{status_txt}</td></tr>"
    
    # Scaling chart JS
    scaling_js = ""
    if report.scaling:
        datasets = []
        colors = ['#58a6ff', '#a371f7', '#3fb950', '#d29922', '#f85149', '#db61a2']
        for i, sr in enumerate(report.scaling):
            c = colors[i % len(colors)]
            data = [t * 1000 for t in sr.times]  # Convert to ms
            datasets.append(f"{{label:'{sr.algorithm}',data:{data},borderColor:'{c}',tension:0.3,fill:false}}")
        
        scaling_js = f"""
const ctx = document.getElementById('scaling').getContext('2d');
new Chart(ctx, {{
    type: 'line',
    data: {{
        labels: {report.scaling[0].sizes},
        datasets: [{','.join(datasets)}]
    }},
    options: {{
        responsive: true,
        maintainAspectRatio: false,
        scales: {{
            y: {{type: 'logarithmic', title: {{display: true, text: 'Time (ms)'}}}},
            x: {{title: {{display: true, text: 'n'}}}}
        }}
    }}
}});"""
    
    # Convergence chart JS
    conv_js = ""
    if report.convergence_profile and report.convergence_profile.get('convergence_samples'):
        samples = report.convergence_profile['convergence_samples']
        steps = [s['step'] for s in samples]
        bps = [s['breakpoints'] for s in samples]
        conv_js = f"""
const ctx2 = document.getElementById('conv').getContext('2d');
new Chart(ctx2, {{
    type: 'line',
    data: {{
        labels: {steps},
        datasets: [{{
            label: 'Breakpoints',
            data: {bps},
            borderColor: '#a371f7',
            fill: true,
            backgroundColor: 'rgba(163,113,247,0.1)',
            pointRadius: 0
        }}]
    }},
    options: {{
        responsive: true,
        maintainAspectRatio: false,
        scales: {{y: {{beginAtZero: true}}}}
    }}
}});"""
    
    m = report.metadata
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sorting Benchmark Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    {css}
</head>
<body>
<div class="container">
    <h1>Sorting Algorithm Benchmark Report</h1>
    <p style="color:#8b949e">{m.get('timestamp', '')}</p>
    
    <div class="card">
        <h2>System Information</h2>
        <div class="grid">
            <div class="stat">
                <div class="stat-label">Python</div>
                <div class="stat-value">{m.get('python_version', '').split()[0]}</div>
            </div>
            <div class="stat">
                <div class="stat-label">Platform</div>
                <div class="stat-value">{m.get('platform', '')[:30]}</div>
            </div>
            <div class="stat">
                <div class="stat-label">NumPy</div>
                <div class="stat-value">{'Yes' if m.get('numpy_available') else 'No'}</div>
            </div>
        </div>
    </div>
    
    <div class="card">
        <h2>Performance Results</h2>
        <table>
            <thead>
                <tr><th>Algorithm</th><th>Input</th><th>n</th><th>Median</th><th>95% CI</th><th>Status</th></tr>
            </thead>
            <tbody>{results_html}</tbody>
        </table>
    </div>
    
    {"<div class='card'><h2>Scaling Analysis</h2><div class='chart'><canvas id='scaling'></canvas></div></div>" if report.scaling else ""}
    {"<div class='card'><h2>Convergence Profile</h2><div class='chart'><canvas id='conv'></canvas></div></div>" if report.convergence_profile and report.convergence_profile.get('convergence_samples') else ""}
    
    <footer>Sorting Algorithm Benchmark Suite v2.0</footer>
</div>
<script>
{scaling_js}
{conv_js}
</script>
</body>
</html>"""
    
    with open(path, 'w', encoding='utf-8') as f:
        f.write(html)