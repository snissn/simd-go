# simd-go

[![Go Reference](https://pkg.go.dev/badge/github.com/axiomhq/simd-go.svg)](https://pkg.go.dev/github.com/axiomhq/simd-go)
[![CI](https://github.com/axiomhq/simd-go/actions/workflows/ci.yml/badge.svg)](https://github.com/axiomhq/simd-go/actions/workflows/ci.yml)
[![Go Report Card](https://goreportcard.com/badge/github.com/axiomhq/simd-go)](https://goreportcard.com/report/github.com/axiomhq/simd-go)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

SIMD-accelerated numeric operations for Go, optimized for ARM64 (NEON and SVE).

## Features

- **Hand-tuned ARM64 assembly** for NEON and SVE instruction sets
- **Automatic CPU detection** - detects Graviton3/4, uses optimal thresholds
- **Threshold-based dispatch** - uses scalar for small arrays, SIMD for large
- **Scalar fallbacks** for non-ARM64 platforms
- **Zero allocations** in hot paths
- **Fuzz tested** for correctness

## Installation

```bash
go get github.com/axiomhq/simd-go
```

## Usage

```go
package main

import (
    "fmt"

    simd "github.com/axiomhq/simd-go"
)

func main() {
    vals := []float64{1.0, 2.0, 3.0, 4.0, 5.0}

    sum := simd.SumFloat64(vals)
    min := simd.MinFloat64(vals)
    max := simd.MaxFloat64(vals)

    fmt.Printf("Sum: %v, Min: %v, Max: %v\n", sum, min, max)
    // Output: Sum: 15, Min: 1, Max: 5
}
```

## Supported Operations

### Float64

| Function | Description |
|----------|-------------|
| `SumFloat64(vals []float64) float64` | Sum of all values |
| `MinFloat64(vals []float64) float64` | Minimum value |
| `MaxFloat64(vals []float64) float64` | Maximum value |
| `DotProductFloat64(a, b []float64) float64` | Dot product of two vectors |

### Int64

| Function | Description |
|----------|-------------|
| `SumInt64(vals []int64) int64` | Sum of all values |
| `MinInt64(vals []int64) int64` | Minimum value |
| `MaxInt64(vals []int64) int64` | Maximum value |
| `DotProductInt64(a, b []int64) int64` | Dot product of two vectors |
| `SumSqInt64(vals []int64) int64` | Sum of squares (Σv²) |
| `AnyAbsGreaterThan(vals []int64, threshold int64) bool` | Check if any \|v\| > threshold |

### CPU Feature Detection

| Function | Description |
|----------|-------------|
| `HasSVE() bool` | Returns true if CPU supports SVE |
| `HasNEON() bool` | Returns true if CPU supports NEON |
| `IsARM64() bool` | Returns true if running on ARM64 |
| `CPUName() string` | Returns detected CPU name (e.g., "AWS Graviton4 (Neoverse-V2)") |

## Performance

Benchmarks on AWS Graviton3 (SVE-capable ARM with 256-bit vectors):

### Speedup vs Scalar

| Operation | NEON | SVE |
|-----------|------|-----|
| SumFloat64 | 2.3x faster | 2.9x faster |
| MinFloat64 | 7.5x faster | 7.2x faster |
| MaxFloat64 | 2.3x faster | 2.3x faster |
| DotProductFloat64 | 1.8x faster | 2.0x faster |
| SumInt64 | 2.6x faster | 2.9x faster |
| MinInt64 | 1.8x faster | 2.4x faster |
| MaxInt64 | 1.8x faster | 2.4x faster |
| DotProductInt64 | — | 1.5x faster |
| SumSqInt64 | 1.2x faster | 1.6x faster |
| **Geometric mean** | **1.9x faster** | **2.4x faster** |

### Detailed Latency Results (n=10,000)

```
                          │   Scalar   │         NEON          │          SVE          │
                          │   sec/op   │  sec/op    vs base    │  sec/op    vs base    │
SumFloat64                   3.868µ      1.401µ    -63.79%       1.243µ    -67.86%
MinFloat64                  15.417µ      1.564µ    -89.86%       1.612µ    -89.54%
MaxFloat64                   4.101µ      1.564µ    -61.86%       1.612µ    -60.69%
DotProductFloat64            3.868µ      1.942µ    -49.79%       1.767µ    -54.32%
SumInt64                     3.869µ      1.404µ    -63.71%       1.246µ    -67.81%
MinInt64                     3.868µ      1.997µ    -48.37%       1.613µ    -58.30%
MaxInt64                     3.862µ      1.994µ    -48.37%       1.613µ    -58.23%
DotProductInt64              3.866µ      3.917µ     +1.32%       2.347µ    -39.28%
SumSqInt64                   3.869µ      3.179µ    -17.82%       2.185µ    -43.52%
```

### Throughput (n=10,000)

```
                          │   Scalar   │          NEON          │          SVE           │
                          │    B/s     │    B/s      vs base    │    B/s      vs base    │
SumFloat64                  19.26Gi      53.19Gi    +176.16%      59.94Gi    +211.22%
MinFloat64                   4.83Gi      47.65Gi    +885.89%      46.21Gi    +856.15%
MaxFloat64                  18.17Gi      47.63Gi    +162.13%      46.21Gi    +154.34%
DotProductFloat64           38.52Gi      76.73Gi     +99.19%      84.33Gi    +118.91%
SumInt64                    19.26Gi      53.07Gi    +175.59%      59.82Gi    +210.63%
MinInt64                    19.26Gi      37.31Gi     +93.70%      46.18Gi    +139.73%
MaxInt64                    19.29Gi      37.37Gi     +93.70%      46.18Gi    +139.37%
DotProductInt64             38.55Gi      38.04Gi      -1.31%      63.49Gi     +64.70%
SumSqInt64                  19.26Gi      23.44Gi     +21.69%      34.09Gi     +77.01%
```

Note: DotProductInt64 shows no NEON speedup because NEON lacks 64-bit integer multiply; it falls back to scalar on NEON-only platforms.

### Run Your Own Benchmarks

```bash
go test -bench=. -benchmem -count=5 ./... | tee bench.txt
benchstat bench.txt
```

Use benchstat for comparison:
```bash
benchstat -row /fn,/n -col /impl bench.txt
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Public API (simd.go)                  │
│  SumFloat64, MinFloat64, MaxFloat64, DotProductFloat64  │
│  SumInt64, MinInt64, MaxInt64, DotProductInt64, ...     │
└─────────────────────────────┬───────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              │                               │
    ┌─────────▼─────────┐         ┌──────────▼──────────┐
    │  impl_arm64.go    │         │   impl_stub.go      │
    │  (ARM64 dispatch) │         │   (other platforms) │
    │                   │         │                     │
    │  1. Detect CPU    │         │  → scalar.go        │
    │     (G3/G4/other) │         │                     │
    │  2. Check size    │         └─────────────────────┘
    │     vs threshold  │
    │  3. Dispatch to:  │
    │     scalar/NEON/  │
    │     SVE           │
    └─────────┬─────────┘
              │
    ┌─────────┼─────────┐
    │         │         │
┌───▼───┐ ┌───▼───┐ ┌───▼───┐
│Scalar │ │ NEON  │ │  SVE  │
│ (.go) │ │ (.s)  │ │ (.s)  │
└───────┘ └───────┘ └───────┘
```

## Safety Notes

### SumSqInt64 Overflow

`SumSqInt64` can overflow if input values are too large. Use `AnyAbsGreaterThan` to check:

```go
const sqrtMaxInt64 = 3037000499

if simd.AnyAbsGreaterThan(vals, sqrtMaxInt64) {
    // Handle potential overflow - use float64 or arbitrary precision
}
sum := simd.SumSqInt64(vals)
```

## License

[MIT](LICENSE)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
