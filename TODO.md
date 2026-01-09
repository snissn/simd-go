# simd-go Roadmap

This document outlines the development roadmap for simd-go, including new data type support and optimization opportunities.

## Current State

### Supported Types
| Type | Sum | Min | Max | DotProduct | SumSq | AnyAbsGreaterThan |
|------|-----|-----|-----|------------|-------|-------------------|
| float64 | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
| int64 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

### Performance (Graviton4)
- **+47% for Int64 Min/Max** - SVE's native SMIN/SMAX vs NEON's CMGT+BIT emulation
- **+25% for SumSqInt64** - SVE's native 64-bit MUL
- **+4-5% for Float64 ops** - marginal gain (same 4 FP ALUs, predicate overhead)
- **-10% for AnyAbsGreaterThan** ⚠️ SVE is SLOWER than scalar

### Architecture Context

| Feature | Graviton3 (Neoverse V1) | Graviton4 (Neoverse V2) |
|---------|-------------------------|-------------------------|
| Vector width | 256-bit SVE | 256-bit SVE/SVE2 |
| FP execution units | 4 | 4 |
| L2 cache | 1 MB/core | 2 MB/core |
| SVE2 | ❌ | ✅ |

---

## Priority 0: Fix Regressions

### P0.1: Fix `anyAbsGreaterThan` SVE Implementation
**Effort:** Medium | **Impact:** Fix 10% regression

```
Scalar: 46.25 ns (17.3 GB/s)
NEON:   42.88 ns (18.7 GB/s)  
SVE:    51.10 ns (15.6 GB/s) ← REGRESSION
```

**Root cause:** Expensive predicate operations (16 CMPGTs, 8 ORRs, tree reduction).

**Solution:** Use `ABS` instruction first, then single unsigned comparison.

**Files:** `int64_arm64_sve.s`

---

## Priority 1: New Data Type Support

### Vector Throughput Comparison

| Type | Bits | NEON lanes | SVE-256 lanes | Elements/iteration (16 accum) |
|------|------|------------|---------------|-------------------------------|
| float64 | 64 | 2 | 4 | 32 (current) |
| int64 | 64 | 2 | 4 | 32 (current) |
| **float32** | 32 | 4 | 8 | 64 |
| **int32** | 32 | 4 | 8 | 64 |
| **int16** | 16 | 8 | 16 | 128 |

New types process **2-4x more elements per vector** than 64-bit types.

### P1.1: Add int32 Support
**Effort:** Medium | **Impact:** High (common data type)

#### Operations
| Operation | NEON | SVE | Notes |
|-----------|------|-----|-------|
| SumInt32 | ✅ | ✅ | Accumulate to int64 to avoid overflow |
| MinInt32 | ✅ | ✅ | Native SMIN.4S |
| MaxInt32 | ✅ | ✅ | Native SMAX.4S |
| DotProductInt32 | ✅ | ✅ | Native MUL.4S (unlike int64!) |
| SumSqInt32 | ✅ | ✅ | Native MUL.4S |

#### Key Instructions
```asm
// NEON (4 elements per vector)
ADD V0.4S, V0.4S, V1.4S      // integer add
MUL V0.4S, V0.4S, V1.4S      // 32-bit multiply (NATIVE - unlike int64!)
SMIN V0.4S, V0.4S, V1.4S     // signed min (NATIVE - unlike int64!)
SMAX V0.4S, V0.4S, V1.4S     // signed max (NATIVE - unlike int64!)
SADDLV D0, V0.4S             // widening horizontal sum to 64-bit

// SVE
LD1W {Z0.S}, P0/Z, [X0]      // load 32-bit elements
ADD Z0.S, P0/M, Z0.S, Z1.S   // add
MUL Z0.S, P0/M, Z0.S, Z1.S   // multiply
SMIN Z0.S, P0/M, Z0.S, Z1.S  // signed min
UADDV D0, P0, Z0.S           // horizontal sum
```

#### Files to Create/Modify
- `int32_arm64_neon.s` - NEON assembly
- `int32_arm64_sve.s` - SVE assembly
- `simd.go` - Add public API
- `scalar.go` - Add scalar fallbacks
- `impl_arm64.go` - Add dispatch + declarations
- `impl_stub.go` - Add stubs for non-arm64
- `caps_arm64.go` - Add thresholds

#### NEON Advantage over Int64
Unlike int64, NEON has **native** 32-bit MUL, SMIN, SMAX. This means:
- DotProduct will be fast on NEON (no scalar fallback needed)
- Min/Max won't need CMGT+BIT emulation
- SVE advantage will be smaller than for int64

---

### P1.2: Add float32 Support
**Effort:** Medium | **Impact:** High (ML, graphics)

#### Operations
| Operation | NEON | SVE | Notes |
|-----------|------|-----|-------|
| SumFloat32 | ✅ | ✅ | FADD.4S |
| MinFloat32 | ✅ | ✅ | FMIN.4S |
| MaxFloat32 | ✅ | ✅ | FMAX.4S |
| DotProductFloat32 | ✅ | ✅ | FMLA.4S |

#### Key Instructions
```asm
// NEON (4 elements per vector)
FADD V0.4S, V0.4S, V1.4S     // fp add
FMUL V0.4S, V0.4S, V1.4S     // fp multiply
FMIN V0.4S, V0.4S, V1.4S     // fp min
FMAX V0.4S, V0.4S, V1.4S     // fp max
FMLA V0.4S, V1.4S, V2.4S     // fused multiply-add
FADDP V0.4S, V0.4S, V1.4S    // pairwise add for reduction

// SVE
LD1W {Z0.S}, P0/Z, [X0]      // load 32-bit floats
FADD Z0.S, P0/M, Z0.S, Z1.S  // fp add
FMLA Z0.S, P0/M, Z1.S, Z2.S  // fused multiply-add
FADDV S0, P0, Z0.S           // horizontal sum
```

#### Files to Create/Modify
- `float32_arm64_neon.s`
- `float32_arm64_sve.s`
- `simd.go`, `scalar.go`, `impl_arm64.go`, `impl_stub.go`, `caps_arm64.go`

#### SVE2 Opportunity: FMMLA (Matrix Multiply)
**Not available for float64!**

```asm
FMMLA Z0.S, Z1.S, Z2.S  // 2x4 × 4x2 matrix multiply-accumulate
```

If adding matrix operations, float32 FMMLA provides massive speedups for ML workloads.

---

### P1.3: Add int16 Support
**Effort:** Medium-High | **Impact:** Medium (audio, signal processing)

#### Operations
| Operation | NEON | SVE | Notes |
|-----------|------|-----|-------|
| SumInt16 | ✅ | ✅ | Must widen to int32/int64 to avoid overflow |
| MinInt16 | ✅ | ✅ | Native SMIN.8H |
| MaxInt16 | ✅ | ✅ | Native SMAX.8H |
| DotProductInt16 | ✅ | ✅ | Widen to int32, use SMLAL |

#### Key Instructions
```asm
// NEON (8 elements per vector)
ADD V0.8H, V0.8H, V1.8H      // 16-bit add
MUL V0.8H, V0.8H, V1.8H      // 16-bit multiply
SMIN V0.8H, V0.8H, V1.8H     // signed min
SMAX V0.8H, V0.8H, V1.8H     // signed max
SADDLP V0.4S, V1.8H          // add pairs, widen to 32-bit
SADDLV S0, V0.8H             // widening horizontal sum

// SVE
LD1H {Z0.H}, P0/Z, [X0]      // load 16-bit elements
ADD Z0.H, P0/M, Z0.H, Z1.H   // add
SMIN Z0.H, P0/M, Z0.H, Z1.H  // signed min
SADDV D0, P0, Z0.H           // horizontal sum (widening)

// SVE2 widening
SADDLB Z0.S, Z1.H, Z2.H      // widen bottom halves, add
SADDLT Z0.S, Z1.H, Z2.H      // widen top halves, add
```

#### Overflow Handling Strategy
int16 sum can overflow quickly (32K elements of max value). Options:
1. **Widen immediately**: Accumulate into int32/int64 accumulators
2. **Periodic widening**: Sum in int16, periodically widen to int64
3. **Document limitation**: Require caller to handle overflow

**Recommendation:** Widen to int64 accumulators (safest, still fast).

#### Files to Create/Modify
- `int16_arm64_neon.s`
- `int16_arm64_sve.s`
- `simd.go`, `scalar.go`, `impl_arm64.go`, `impl_stub.go`, `caps_arm64.go`

#### SVE2 Opportunity: MATCH Instruction
**Only works on 8-bit and 16-bit - not available for 64-bit!**

```asm
MATCH P0.H, P1/Z, Z0.H, Z1.H  // Check if any element in Z0 matches any in Z1
```

This enables new operations:
```go
// New API (int16 only)
func ContainsAnyInt16(data []int16, needles []int16) bool
func IndexAnyInt16(data []int16, needles []int16) int
```

**Benchmark expectations (from ARM):**
| Hit Rate | Scalar | SVE2 MATCH | Speedup |
|----------|--------|------------|---------|
| 0% | 145 µs | 1.5 µs | 95x |
| 0.01% | 22 µs | 0.3 µs | 70x |

---

### Implementation Order

**Recommended:** int32 → float32 → int16

| Phase | Type | Rationale |
|-------|------|-----------|
| 1 | int32 | Most straightforward; similar to int64 but simpler (native MUL/MIN/MAX) |
| 2 | float32 | Similar to float64; just change suffixes |
| 3 | int16 | Requires widening logic; enables MATCH |

---

## Priority 2: Infrastructure

### P2.1: Add SVE2 Runtime Detection
**Effort:** Low | **Impact:** Enables all SVE2 optimizations

```go
// caps_arm64.go
var hasSVE2 = cpu.ARM64.HasSVE2

func HasSVE2() bool { return hasSVE2 }
```

### P2.2: Create SVE2 Assembly Files
**Effort:** Medium | **Impact:** Clean separation for SVE2-only instructions

Create separate files for SVE2-specific implementations:
- `*_arm64_sve2.s` files for MATCH, FMMLA, pairwise reductions

---

## Priority 3: SVE2-Specific Optimizations

### P3.1: Pairwise Reductions (SVE2)
**Effort:** Medium | **Impact:** 10-15% faster final reduction

```asm
// SVE2 pairwise reduction (all types)
FADDP Z0.S, P0/M, Z0.S, Z1.S  // float32 pairwise add
ADDP Z0.S, P0/M, Z0.S, Z1.S   // int32 pairwise add
FMAXP Z0.S, P0/M, Z0.S, Z1.S  // float32 pairwise max
SMAXP Z0.S, P0/M, Z0.S, Z1.S  // int32 pairwise max
```

### P3.2: MATCH-based Search (SVE2, int16/int8 only)
**Effort:** High | **Impact:** 66-95x speedup

See P1.3 for details. This is a unique capability for narrow types.

### P3.3: Matrix Multiply (SVE2, float32 only)
**Effort:** High | **Impact:** Massive for ML workloads

```go
// Potential new API
func MatMul4x4Float32(a, b, c []float32)  // C += A × B
```

Uses `FMMLA` instruction for 2×2 tile operations.

---

## Priority 4: General SVE Improvements

### P4.1: Software Prefetching for Large Arrays
**Effort:** Low | **Impact:** 5-10% for n ≥ 10,000

```asm
PRFB PLDL1STRM, P0, [X0, #8, MUL VL]  // prefetch bytes
PRFH PLDL1STRM, P0, [X0, #8, MUL VL]  // prefetch halfwords
PRFW PLDL1STRM, P0, [X0, #8, MUL VL]  // prefetch words
```

### P4.2: Optimize Tail Handling
**Effort:** Low | **Impact:** 2-5% for small arrays

For small remainders, unrolled scalar may beat WHILELO loops.

---

## Priority 5: Additional Operations

### P5.1: Byte/uint8 Operations with MATCH
**Effort:** High | **Impact:** Very high for parsing

```go
func ContainsAnyByte(data []byte, chars []byte) bool
func IndexAnyByte(data []byte, chars []byte) int
```

Use cases: CSV/JSON parsing, delimiter search, forbidden character checks.

### P5.2: HISTCNT for Histograms (SVE2)
**Effort:** High | **Impact:** Domain-specific

```go
func HistogramUint8(data []byte, counts []uint32)
```

---

## File Structure After Implementation

```
simd-go/
├── simd.go                    # Public API (all types)
├── scalar.go                  # Scalar fallbacks (all types)
├── impl_arm64.go              # Dispatch functions + declarations
├── impl_stub.go               # Non-arm64 stubs
├── caps_arm64.go              # CPU detection + thresholds
├── caps_stub.go               # Non-arm64 caps
│
├── float64_arm64_neon.s       # Existing
├── float64_arm64_sve.s        # Existing
├── int64_arm64_neon.s         # Existing
├── int64_arm64_sve.s          # Existing
│
├── float32_arm64_neon.s       # NEW
├── float32_arm64_sve.s        # NEW
├── int32_arm64_neon.s         # NEW
├── int32_arm64_sve.s          # NEW
├── int16_arm64_neon.s         # NEW
├── int16_arm64_sve.s          # NEW
│
├── simd_test.go               # Correctness tests
├── simd_bench_test.go         # Benchmarks
├── simd_fuzz_test.go          # Fuzz tests
└── threshold_bench_test.go    # Threshold calibration
```

---

## API After Implementation

```go
// Float64 (existing)
func SumFloat64(vals []float64) float64
func MinFloat64(vals []float64) float64
func MaxFloat64(vals []float64) float64
func DotProductFloat64(a, b []float64) float64

// Int64 (existing)
func SumInt64(vals []int64) int64
func MinInt64(vals []int64) int64
func MaxInt64(vals []int64) int64
func DotProductInt64(a, b []int64) int64
func SumSqInt64(vals []int64) int64
func AnyAbsGreaterThan(vals []int64, threshold int64) bool

// Float32 (NEW)
func SumFloat32(vals []float32) float32
func MinFloat32(vals []float32) float32
func MaxFloat32(vals []float32) float32
func DotProductFloat32(a, b []float32) float32

// Int32 (NEW)
func SumInt32(vals []int32) int64        // Returns int64 to avoid overflow
func MinInt32(vals []int32) int32
func MaxInt32(vals []int32) int32
func DotProductInt32(a, b []int32) int64 // Returns int64 to avoid overflow
func SumSqInt32(vals []int32) int64      // Returns int64 to avoid overflow

// Int16 (NEW)
func SumInt16(vals []int16) int64        // Returns int64 to avoid overflow
func MinInt16(vals []int16) int16
func MaxInt16(vals []int16) int16
func DotProductInt16(a, b []int16) int64 // Returns int64 to avoid overflow

// Search operations (NEW, SVE2 only, scalar fallback)
func ContainsAnyInt16(data []int16, needles []int16) bool
func IndexAnyInt16(data []int16, needles []int16) int
func ContainsAnyByte(data []byte, chars []byte) bool
func IndexAnyByte(data []byte, chars []byte) int
```

---

## Benchmarking Checklist

After implementing new types, run:

```bash
# Full benchmark suite
go test -bench=. -count=6 -benchmem | tee benchmark_results.txt

# Compare implementations for new types
go test -bench="Int32|Float32|Int16" -count=6

# Verify MATCH speedups (SVE2 only)
go test -bench="ContainsAny|IndexAny" -count=6
```

Key metrics:
- [ ] int32 NEON should match or beat int64 NEON (2x elements/vector)
- [ ] float32 NEON should match or beat float64 NEON (2x elements/vector)
- [ ] int16 should show ~4x throughput vs int64 for large arrays
- [ ] MATCH operations should show 50-90x speedup on SVE2

---

## References

- [AWS Graviton Getting Started](https://github.com/aws/aws-graviton-getting-started)
- [Neoverse V2 Software Optimization Guide](https://developer.arm.com/documentation/PJDOC-466751330-593177/latest/)
- [ARM SVE2 MATCH Learning Path](https://learn.arm.com/learning-paths/servers-and-cloud-computing/sve2-match/)
- [Introduction to SVE2](https://developer.arm.com/documentation/102340/latest/)
- [ARM Intrinsics Reference](https://developer.arm.com/architectures/instruction-sets/intrinsics/)
