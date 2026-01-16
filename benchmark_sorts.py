#!/usr/bin/env python3
"""
Sorting Algorithm Benchmark Suite - Main Runner
================================================

A comprehensive benchmarking framework for comparing sorting algorithms.

Usage:
    python benchmark_sorts.py --quick
    python benchmark_sorts.py --full --html report.html
    python benchmark_sorts.py --scaling --max-n 50000
    python benchmark_sorts.py --profile --n 5000
    python benchmark_sorts.py --analyze --n 3000
"""

import argparse
import json
import sys
from datetime import datetime

from benchmark_core import (
    BenchmarkConfig, BenchmarkEngine, BenchmarkReport,
    INPUT_GENERATORS, SCALING_SIZES,
    get_algorithms, get_system_info,
    profile_convergence, estimate_complexity,
    analyze_algorithm_characteristics,
    print_header, print_subheader, print_results_table,
    print_scaling_table, print_convergence_profile,
    print_analysis_report,
    generate_html_report, Colors, HAS_NUMPY,
    AlgorithmResult, ScalingResult,
)


def run_quick_benchmark(config: BenchmarkConfig, algos: dict, quiet: bool = False):
    """Quick sanity check benchmark."""
    engine = BenchmarkEngine(config)
    generator = INPUT_GENERATORS["random"]
    n = 5000
    
    if not quiet:
        print_header("Quick Benchmark")
        print(f"  n = {n}, input = random permutation\n")
    
    results = []
    for name, algo in algos.items():
        if not quiet:
            print(f"  Testing {name}...", end=" ", flush=True)
        result = engine.benchmark_algorithm(algo, generator, n)
        results.append(result)
        if not quiet:
            if result.error:
                print(f"{Colors.RED}ERROR{Colors.END}")
            elif result.correct:
                print(f"{Colors.GREEN}OK{Colors.END} ({result.stats.median*1000:.2f}ms)")
            else:
                print(f"{Colors.RED}INCORRECT{Colors.END}")
    
    if not quiet:
        print()
        print_results_table(results)
    
    return results


def run_full_benchmark(config: BenchmarkConfig, algos: dict, n: int, quiet: bool = False):
    """Full benchmark across all input distributions."""
    engine = BenchmarkEngine(config)
    all_results = []
    
    if not quiet:
        print_header("Full Benchmark Suite")
        print(f"  n = {n}, testing {len(INPUT_GENERATORS)} input distributions\n")
    
    for gen_name, generator in INPUT_GENERATORS.items():
        if not quiet:
            print_subheader(f"Input: {gen_name}")
            print(f"  {generator.description}\n")
        
        results = []
        for algo_name, algo in algos.items():
            if not quiet:
                print(f"    {algo_name}...", end=" ", flush=True)
            result = engine.benchmark_algorithm(algo, generator, n)
            results.append(result)
            all_results.append(result)
            if not quiet:
                if result.stats.n > 0 and result.correct:
                    print(f"{result.stats.median*1000:.2f}ms")
                else:
                    print(f"{Colors.RED}FAILED{Colors.END}")
        
        if not quiet:
            print()
            print_results_table(results)
    
    return all_results


def run_scaling_analysis(config: BenchmarkConfig, algos: dict, max_n: int, quiet: bool = False):
    """Analyze scaling behavior across input sizes."""
    engine = BenchmarkEngine(config)
    generator = INPUT_GENERATORS["random"]
    
    # Filter sizes based on max_n
    sizes = [s for s in SCALING_SIZES if s <= max_n]
    if max_n not in sizes:
        sizes.append(max_n)
    sizes = sorted(sizes)
    
    if not quiet:
        print_header("Scaling Analysis")
        print(f"  Sizes: {sizes}\n")
    
    scaling_results = []
    for algo_name, algo in algos.items():
        if not quiet:
            print(f"  Analyzing {algo_name}...", flush=True)
        result = engine.run_scaling_analysis(algo, generator, sizes, runs_per_size=3)
        scaling_results.append(result)
        if not quiet:
            print(f"    Estimated complexity: {result.estimated_complexity} (R^2={result.r_squared:.3f})")
    
    if not quiet:
        print()
        print_scaling_table(scaling_results)
    
    return scaling_results


def run_profile(n: int, seed: int, max_steps: int, quiet: bool = False):
    """Profile the stochastic algorithm's convergence."""
    if not quiet:
        print_header("Convergence Profile")
        print(f"  Profiling Shake n Break Sort with n={n}\n")
    
    profile = profile_convergence(n=n, seed=seed, max_steps=max_steps)
    
    if not quiet:
        print_convergence_profile(profile)
    
    return profile


def build_report(config: BenchmarkConfig, algos: dict, results: list,
                 scaling: list, profile: dict = None) -> BenchmarkReport:
    """Build complete benchmark report."""
    metadata = get_system_info()
    metadata.update({
        "benchmark_version": "2.0.0",
        "config": {
            "seed": config.seed,
            "warmup_runs": config.warmup_runs,
            "min_runs": config.min_runs,
            "max_runs": config.max_runs,
            "confidence_level": config.confidence_level,
        }
    })
    
    algo_info = {
        name: {
            "category": algo.category,
            "expected_complexity": algo.expected_complexity,
            "stable": algo.stable,
            "in_place": algo.in_place,
            "description": algo.description,
        }
        for name, algo in algos.items()
    }
    
    return BenchmarkReport(
        metadata=metadata,
        algorithm_info=algo_info,
        results=results,
        scaling=scaling,
        convergence_profile=profile,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Sorting Algorithm Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --quick                           Quick sanity check
  %(prog)s --full --html report.html         Full suite with HTML report
  %(prog)s --scaling --max-n 100000          Scaling analysis
  %(prog)s --profile --n 5000                Profile stochastic algorithm
  %(prog)s --analyze --n 3000                Deep algorithm analysis
        """
    )
    
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true", help="Quick benchmark")
    mode.add_argument("--full", action="store_true", help="Full benchmark suite")
    mode.add_argument("--scaling", action="store_true", help="Scaling analysis")
    mode.add_argument("--profile", action="store_true", help="Profile convergence")
    mode.add_argument("--analyze", action="store_true", help="Deep algorithm analysis")
    
    parser.add_argument("--n", type=int, default=10000, help="Array size (default: 10000)")
    parser.add_argument("--max-n", type=int, default=50000, help="Max size for scaling (default: 50000)")
    parser.add_argument("--runs", type=int, default=10, help="Benchmark runs (default: 10)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--max-steps", type=int, default=300000, help="Max steps for stochastic algo (default: 300000)")
    parser.add_argument("--case", choices=list(INPUT_GENERATORS.keys()), default="random",
                        help="Input distribution (default: random)")
    parser.add_argument("--slow", action="store_true", help="Include slow pure-Python algorithms")
    parser.add_argument("--no-numpy", action="store_true", help="Exclude NumPy algorithms")
    parser.add_argument("--output", "-o", type=str, help="JSON output path")
    parser.add_argument("--html", type=str, help="HTML report output path")
    parser.add_argument("--quiet", "-q", action="store_true", help="Minimal output")
    
    args = parser.parse_args()
    
    # Build config
    config = BenchmarkConfig(
        seed=args.seed,
        max_runs=args.runs,
        max_steps_stochastic=args.max_steps,
    )
    
    # Get algorithms
    algos = get_algorithms(
        config,
        include_slow=args.slow,
        include_numpy=(not args.no_numpy) and HAS_NUMPY,
    )
    
    if not args.quiet:
        print(f"\n{Colors.BOLD}Sorting Algorithm Benchmark Suite v2.0{Colors.END}")
        print(f"Python {sys.version.split()[0]} | NumPy: {'Yes' if HAS_NUMPY else 'No'}\n")
    
    results = []
    scaling = []
    profile = None
    
    # Run selected mode
    if args.profile:
        profile = run_profile(args.n, args.seed, args.max_steps, args.quiet)
    
    elif args.analyze:
        if not args.quiet:
            print_header("Deep Algorithm Analysis")
        analysis = analyze_algorithm_characteristics(args.n, args.seed, args.max_steps)
        if not args.quiet:
            print_analysis_report(analysis)
        profile = analysis  # Include in report
    
    elif args.scaling:
        scaling = run_scaling_analysis(config, algos, args.max_n, args.quiet)
    
    elif args.full:
        results = run_full_benchmark(config, algos, args.n, args.quiet)
        scaling = run_scaling_analysis(config, algos, min(args.n, 25000), args.quiet)
        profile = run_profile(min(args.n, 5000), args.seed, args.max_steps, args.quiet)
    
    else:  # Default to quick
        results = run_quick_benchmark(config, algos, args.quiet)
    
    # Build and export report
    report = build_report(config, algos, results, scaling, profile)
    
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(report.to_dict(), f, indent=2, default=str)
        if not args.quiet:
            print(f"\n{Colors.GREEN}JSON report saved to {args.output}{Colors.END}")
    
    if args.html:
        generate_html_report(report, args.html)
        if not args.quiet:
            print(f"{Colors.GREEN}HTML report saved to {args.html}{Colors.END}")
    
    if not args.quiet:
        print(f"\n{Colors.CYAN}Benchmark complete.{Colors.END}\n")


if __name__ == "__main__":
    main()