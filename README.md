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

### Float32

| Function | Description |
|----------|-------------|
| `SumFloat32(vals []float32) float32` | Sum of all values |
| `MinFloat32(vals []float32) float32` | Minimum value |
| `MaxFloat32(vals []float32) float32` | Maximum value |
| `DotProductFloat32(a, b []float32) float32` | Dot product of two vectors |

### Int64

| Function | Description |
|----------|-------------|
| `SumInt64(vals []int64) int64` | Sum of all values |
| `MinInt64(vals []int64) int64` | Minimum value |
| `MaxInt64(vals []int64) int64` | Maximum value |
| `DotProductInt64(a, b []int64) int64` | Dot product of two vectors |
| `SumSqInt64(vals []int64) int64` | Sum of squares (Σv²) |
| `AnyAbsGreaterThan(vals []int64, threshold int64) bool` | Check if any \|v\| > threshold |

### Int32

| Function | Description |
|----------|-------------|
| `SumInt32(vals []int32) int64` | Sum of all values (returns int64 to avoid overflow) |
| `MinInt32(vals []int32) int32` | Minimum value |
| `MaxInt32(vals []int32) int32` | Maximum value |
| `DotProductInt32(a, b []int32) int64` | Dot product of two vectors (returns int64 to avoid overflow) |
| `SumSqInt32(vals []int32) int64` | Sum of squares (returns int64 to avoid overflow) |
| `AnyAbsGreaterThanInt32(vals []int32, threshold int32) bool` | Check if any \|v\| > threshold |

### Int16

| Function | Description |
|----------|-------------|
| `SumInt16(vals []int16) int64` | Sum of all values (returns int64 to avoid overflow) |
| `MinInt16(vals []int16) int16` | Minimum value |
| `MaxInt16(vals []int16) int16` | Maximum value |
| `DotProductInt16(a, b []int16) int64` | Dot product of two vectors (returns int64 to avoid overflow) |
| `SumSqInt16(vals []int16) int64` | Sum of squares (returns int64 to avoid overflow) |
| `AnyAbsGreaterThanInt16(vals []int16, threshold int16) bool` | Check if any \|v\| > threshold |

### CPU Feature Detection

| Function | Description |
|----------|-------------|
| `HasSVE() bool` | Returns true if CPU supports SVE |
| `HasNEON() bool` | Returns true if CPU supports NEON |
| `IsARM64() bool` | Returns true if running on ARM64 |
| `CPUName() string` | Returns detected CPU name (e.g., "AWS Graviton4 (Neoverse-V2)") |

## Performance

All benchmarks run with n=10,000 elements.

### Apple M3 (NEON only)

| Type | Operation | Scalar | NEON | Speedup |
|------|-----------|--------|------|---------|
| Float64 | Sum | 2.94µs (27.2 GB/s) | 560ns (142.8 GB/s) | **5.3x** |
| Float64 | Min | 11.8µs (6.8 GB/s) | 560ns (142.9 GB/s) | **21.1x** |
| Float64 | Max | 11.8µs (6.8 GB/s) | 560ns (142.8 GB/s) | **21.1x** |
| Float64 | DotProduct | 2.96µs (54.0 GB/s) | 1.02µs (157.0 GB/s) | **2.9x** |
| Float32 | Sum | 2.96µs (13.5 GB/s) | 283ns (141.1 GB/s) | **10.5x** |
| Float32 | Min | 11.7µs (3.4 GB/s) | 282ns (141.8 GB/s) | **41.5x** |
| Float32 | Max | 11.8µs (3.4 GB/s) | 286ns (139.8 GB/s) | **41.3x** |
| Float32 | DotProduct | 2.96µs (27.0 GB/s) | 510ns (156.8 GB/s) | **5.8x** |
| Int64 | Sum | 2.97µs (26.9 GB/s) | 560ns (142.8 GB/s) | **5.3x** |
| Int64 | Min | 2.96µs (27.1 GB/s) | 799ns (100.1 GB/s) | **3.7x** |
| Int64 | Max | 2.99µs (26.8 GB/s) | 816ns (98.0 GB/s) | **3.7x** |
| Int64 | SumSq | 3.03µs (26.4 GB/s) | 1.60µs (50.1 GB/s) | **1.9x** |
| Int32 | Sum | 2.96µs (13.5 GB/s) | 375ns (106.8 GB/s) | **7.9x** |
| Int32 | Min | 2.92µs (13.7 GB/s) | 280ns (142.7 GB/s) | **10.4x** |
| Int32 | Max | 2.96µs (13.5 GB/s) | 285ns (140.3 GB/s) | **10.4x** |
| Int32 | DotProduct | 2.97µs (26.9 GB/s) | 747ns (107.1 GB/s) | **4.0x** |
| Int32 | SumSq | 2.92µs (13.7 GB/s) | 746ns (53.6 GB/s) | **3.9x** |
| Int16 | Sum | 2.97µs (6.7 GB/s) | 284ns (70.5 GB/s) | **10.5x** |
| Int16 | Min | 2.96µs (6.8 GB/s) | 149ns (134.5 GB/s) | **19.9x** |
| Int16 | Max | 2.98µs (6.7 GB/s) | 149ns (134.3 GB/s) | **20.0x** |
| Int16 | DotProduct | 2.95µs (13.5 GB/s) | 561ns (71.3 GB/s) | **5.3x** |
| Int16 | SumSq | 2.94µs (6.8 GB/s) | 559ns (35.8 GB/s) | **5.3x** |

### AWS Graviton3 (Neoverse-V1, SVE 256-bit)

| Type | Operation | Scalar | NEON | SVE | Best |
|------|-----------|--------|------|-----|------|
| Float64 | Sum | 3.87µs (20.7 GB/s) | 1.38µs (58.2 GB/s) | 1.24µs (64.3 GB/s) | **3.1x** SVE |
| Float64 | Min | 15.4µs (5.2 GB/s) | 1.56µs (51.3 GB/s) | 1.61µs (49.7 GB/s) | **9.9x** NEON |
| Float64 | Max | 15.3µs (5.2 GB/s) | 1.56µs (51.3 GB/s) | 1.61µs (49.7 GB/s) | **9.8x** NEON |
| Float64 | DotProduct | 3.87µs (41.3 GB/s) | 1.97µs (81.4 GB/s) | 1.76µs (90.9 GB/s) | **2.2x** SVE |
| Float32 | Sum | 3.87µs (10.3 GB/s) | 552ns (72.5 GB/s) | 391ns (102.4 GB/s) | **9.9x** SVE |
| Float32 | Min | 15.4µs (2.6 GB/s) | 570ns (70.2 GB/s) | 573ns (69.8 GB/s) | **27.0x** NEON |
| Float32 | Max | 15.3µs (2.6 GB/s) | 570ns (70.1 GB/s) | 575ns (69.6 GB/s) | **26.8x** NEON |
| Float32 | DotProduct | 3.87µs (20.7 GB/s) | 814ns (98.3 GB/s) | 702ns (114.0 GB/s) | **5.5x** SVE |
| Int64 | Sum | 3.87µs (20.7 GB/s) | 1.38µs (58.1 GB/s) | 1.25µs (64.1 GB/s) | **3.1x** SVE |
| Int64 | Min | 3.87µs (20.7 GB/s) | 2.00µs (40.1 GB/s) | 1.61µs (49.6 GB/s) | **2.4x** SVE |
| Int64 | Max | 3.87µs (20.7 GB/s) | 2.00µs (40.0 GB/s) | 1.61µs (49.6 GB/s) | **2.4x** SVE |
| Int64 | DotProduct | 3.87µs (41.3 GB/s) | — | 2.35µs (68.2 GB/s) | **1.6x** SVE |
| Int64 | SumSq | 3.86µs (20.7 GB/s) | 3.19µs (25.1 GB/s) | 2.19µs (36.6 GB/s) | **1.8x** SVE |
| Int32 | Sum | 3.86µs (10.4 GB/s) | 764ns (52.3 GB/s) | 1.27µs (31.5 GB/s) | **5.1x** NEON |
| Int32 | Min | 3.87µs (10.3 GB/s) | 563ns (71.1 GB/s) | 392ns (102.0 GB/s) | **9.9x** SVE |
| Int32 | Max | 3.86µs (10.4 GB/s) | 563ns (71.0 GB/s) | 393ns (101.8 GB/s) | **9.8x** SVE |
| Int32 | DotProduct | 3.87µs (20.7 GB/s) | 1.77µs (45.3 GB/s) | 2.76µs (29.0 GB/s) | **2.2x** NEON |
| Int32 | SumSq | 3.86µs (10.4 GB/s) | 1.43µs (28.0 GB/s) | 2.11µs (19.0 GB/s) | **2.7x** NEON |
| Int16 | Sum | 3.87µs (5.2 GB/s) | 542ns (36.9 GB/s) | 1.51µs (13.2 GB/s) | **7.1x** NEON |
| Int16 | Min | 3.87µs (5.2 GB/s) | 283ns (70.8 GB/s) | 201ns (99.3 GB/s) | **19.2x** SVE |
| Int16 | Max | 3.86µs (5.2 GB/s) | 283ns (70.7 GB/s) | 202ns (99.1 GB/s) | **19.1x** SVE |
| Int16 | DotProduct | 3.88µs (10.3 GB/s) | 1.25µs (31.9 GB/s) | 2.12µs (18.9 GB/s) | **3.1x** NEON |
| Int16 | SumSq | 3.87µs (5.2 GB/s) | 1.12µs (17.8 GB/s) | 1.87µs (10.7 GB/s) | **3.4x** NEON |

### AWS Graviton4 (Neoverse-V2, SVE2 128-bit)

| Type | Operation | Scalar | NEON | SVE | SVE2 | Best |
|------|-----------|--------|------|-----|------|------|
| Float64 | Sum | 3.59µs (22.3 GB/s) | 1.00µs (79.7 GB/s) | 1.00µs (79.6 GB/s) | — | **3.6x** NEON |
| Float64 | Min | 14.3µs (5.6 GB/s) | 1.22µs (65.7 GB/s) | 1.23µs (65.1 GB/s) | — | **11.7x** NEON |
| Float64 | Max | 14.3µs (5.6 GB/s) | 1.22µs (65.6 GB/s) | 1.23µs (65.0 GB/s) | — | **11.7x** NEON |
| Float64 | DotProduct | 4.06µs (40.0 GB/s) | 1.71µs (93.6 GB/s) | 1.66µs (96.6 GB/s) | — | **2.5x** SVE |
| Float32 | Sum | 3.90µs (10.4 GB/s) | 461ns (86.8 GB/s) | 444ns (90.1 GB/s) | — | **8.8x** SVE |
| Float32 | Min | 14.3µs (2.8 GB/s) | 563ns (71.1 GB/s) | 565ns (70.8 GB/s) | — | **25.4x** NEON |
| Float32 | Max | 14.2µs (2.8 GB/s) | 563ns (71.1 GB/s) | 565ns (70.8 GB/s) | — | **25.3x** NEON |
| Float32 | DotProduct | 3.59µs (22.3 GB/s) | 759ns (105.4 GB/s) | 770ns (103.9 GB/s) | — | **4.7x** NEON |
| Int64 | Sum | 4.06µs (20.0 GB/s) | 1.00µs (79.9 GB/s) | 1.00µs (79.8 GB/s) | — | **4.1x** NEON |
| Int64 | Min | 3.90µs (20.8 GB/s) | 1.58µs (50.7 GB/s) | 1.23µs (65.2 GB/s) | — | **3.2x** SVE |
| Int64 | Max | 3.58µs (22.3 GB/s) | 1.51µs (53.2 GB/s) | 1.23µs (65.1 GB/s) | — | **2.9x** SVE |
| Int64 | DotProduct | 3.59µs (44.6 GB/s) | — | 1.96µs (81.6 GB/s) | — | **1.8x** SVE |
| Int64 | SumSq | 3.59µs (22.3 GB/s) | 1.97µs (40.6 GB/s) | 1.80µs (44.4 GB/s) | — | **2.0x** SVE |
| Int32 | Sum | 3.59µs (11.2 GB/s) | 612ns (65.4 GB/s) | 1.02µs (39.2 GB/s) | 657ns (60.9 GB/s) | **5.9x** NEON |
| Int32 | Min | 4.19µs (9.7 GB/s) | 558ns (71.7 GB/s) | 444ns (90.2 GB/s) | — | **9.5x** SVE |
| Int32 | Max | 3.65µs (11.0 GB/s) | 558ns (71.7 GB/s) | 445ns (89.8 GB/s) | — | **8.2x** SVE |
| Int32 | DotProduct | 3.59µs (22.3 GB/s) | 1.38µs (57.9 GB/s) | 2.17µs (36.9 GB/s) | 1.35µs (59.5 GB/s) | **2.7x** SVE2 |
| Int32 | SumSq | 3.59µs (11.2 GB/s) | 1.12µs (35.6 GB/s) | 1.90µs (21.1 GB/s) | 1.12µs (35.6 GB/s) | **3.2x** NEON |
| Int16 | Sum | 3.78µs (5.3 GB/s) | 408ns (49.1 GB/s) | 1.26µs (15.9 GB/s) | 339ns (59.0 GB/s) | **11.1x** SVE2 |
| Int16 | Min | 3.90µs (5.2 GB/s) | 344ns (58.2 GB/s) | 224ns (89.2 GB/s) | — | **17.4x** SVE |
| Int16 | Max | 3.90µs (5.2 GB/s) | 342ns (58.5 GB/s) | 224ns (89.2 GB/s) | — | **17.4x** SVE |
| Int16 | DotProduct | 3.90µs (10.4 GB/s) | 965ns (41.5 GB/s) | 2.01µs (19.9 GB/s) | 746ns (53.7 GB/s) | **5.2x** SVE2 |
| Int16 | SumSq | 3.59µs (5.6 GB/s) | 896ns (22.3 GB/s) | 1.75µs (11.4 GB/s) | 587ns (34.1 GB/s) | **6.1x** SVE2 |

### Notes

- DotProductInt64 has no NEON implementation (NEON lacks 64-bit integer multiply)
- Graviton4 uses 128-bit SVE vectors; SVE2 provides additional instructions for better Int16/Int32 performance
- Apple M3 has no SVE support

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
