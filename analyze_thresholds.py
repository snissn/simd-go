#!/usr/bin/env python3
"""Analyze threshold benchmarks to find SIMD vs scalar crossover points."""

import re
import sys
from collections import defaultdict

def parse_benchmark(line):
    """Parse a benchmark line and return (func, impl, n, ns_per_op)."""
    # BenchmarkThreshold/fn=SumFloat64/impl=Scalar/n=1-4   	1000000000	         0.3125 ns/op
    m = re.match(r'BenchmarkThreshold/fn=([^/]+)/impl=([^/]+)/n=(\d+)-\d+\s+\d+\s+([\d.]+)\s+ns/op', line)
    if m:
        return m.group(1), m.group(2), int(m.group(3)), float(m.group(4))
    return None

def main():
    if len(sys.argv) < 2:
        print("Usage: analyze_thresholds.py <benchmark_output>")
        sys.exit(1)
    
    # Parse results: {func: {n: {impl: ns_per_op}}}
    results = defaultdict(lambda: defaultdict(dict))
    
    with open(sys.argv[1]) as f:
        for line in f:
            parsed = parse_benchmark(line.strip())
            if parsed:
                func, impl, n, ns = parsed
                results[func][n][impl] = ns
    
    print("=" * 80)
    print("SIMD vs Scalar Threshold Analysis")
    print("=" * 80)
    
    for func in sorted(results.keys()):
        print(f"\n### {func}")
        print(f"{'n':>6} | {'Scalar':>10} | {'NEON':>10} | {'SVE':>10} | {'NEON win':>10} | {'SVE win':>10}")
        print("-" * 70)
        
        neon_threshold = None
        sve_threshold = None
        
        for n in sorted(results[func].keys()):
            impls = results[func][n]
            scalar = impls.get('Scalar', 0)
            neon = impls.get('NEON', 0)
            sve = impls.get('SVE', 0)
            
            neon_win = ""
            sve_win = ""
            
            if neon and scalar:
                if neon < scalar:
                    neon_win = f"{scalar/neon:.2f}x"
                    if neon_threshold is None:
                        neon_threshold = n
                else:
                    neon_win = f"{neon/scalar:.2f}x slower"
            
            if sve and scalar:
                if sve < scalar:
                    sve_win = f"{scalar/sve:.2f}x"
                    if sve_threshold is None:
                        sve_threshold = n
                else:
                    sve_win = f"{sve/scalar:.2f}x slower"
            
            print(f"{n:>6} | {scalar:>10.2f} | {neon:>10.2f} | {sve:>10.2f} | {neon_win:>10} | {sve_win:>10}")
        
        print()
        if neon_threshold:
            print(f"  NEON threshold: n >= {neon_threshold}")
        else:
            print(f"  NEON threshold: NEVER wins (or not available)")
        if sve_threshold:
            print(f"  SVE threshold:  n >= {sve_threshold}")
        else:
            print(f"  SVE threshold:  NEVER wins (or not available)")

if __name__ == '__main__':
    main()
