# Dynamic Threshold Calibration Design

This document describes the design for auto-calibrating SIMD dispatch thresholds at runtime.

## Executive Summary

**Recommendation: Implement as opt-in feature, but consider low priority.**

Given that we already have well-tuned static thresholds for Apple M3, Graviton3, and Graviton4—the primary target platforms—dynamic calibration provides marginal benefit for the added complexity. However, it's valuable for:
- Unknown ARM CPUs where we don't have hardware access
- Power users who want optimal thresholds for their specific hardware
- Future-proofing for new CPU models

## Current State

### Static Thresholds
- **36 operations** across 5 data types (Float64, Float32, Int64, Int32, Int16)
- **3 known CPU families**: Apple M3, Graviton3, Graviton4
- **Conservative defaults** for unknown CPUs
- Thresholds defined in `caps_arm64.go` as `var neonThresholds` and `var sveThresholds`

### Threshold Ranges
| Implementation | Typical Threshold Range |
|----------------|------------------------|
| NEON | 4-64 elements |
| SVE | 8-256 elements |
| "Disabled" | 256+ (force NEON/scalar) |

## Design

### API

```go
// CalibrateThresholds measures actual Scalar vs NEON vs SVE performance
// and updates dispatch thresholds for optimal performance on the current CPU.
//
// This function should be called early, before concurrent use of simd functions.
// It runs ~5-10ms of microbenchmarks and is typically called via:
//
//   SIMD_CALIBRATE=1 ./myapp
//
// Or explicitly:
//
//   func main() {
//       simd.CalibrateThresholds()
//       // ... use simd functions
//   }
func CalibrateThresholds()
```

### Activation

1. **Environment variable** (checked in `init()`):
   ```go
   func init() {
       if os.Getenv("SIMD_CALIBRATE") == "1" {
           CalibrateThresholds()
       }
   }
   ```

2. **Explicit call** for programmatic control:
   ```go
   simd.CalibrateThresholds()
   ```

### Policy

| CPU State | Default Behavior | With SIMD_CALIBRATE=1 |
|-----------|------------------|----------------------|
| Known (M3, G3, G4) | Use static thresholds | Run calibration, override static |
| Unknown ARM64 | Use conservative defaults | Run calibration, override defaults |

### Calibration Algorithm

#### 1. Candidate Sizes
```go
var calibrationSizes = []int{4, 8, 16, 32, 48, 64, 96, 128}
```

These cover all meaningful threshold points. Larger sizes (192, 256) are only needed for "disable SVE" thresholds.

#### 2. Measurement Parameters
```go
const (
    calibrationIters = 256  // iterations per measurement
    calibrationReps  = 3    // repetitions, take minimum
    marginFactor     = 0.95 // SIMD must be 5% faster to win
)
```

#### 3. Per-Operation Calibration

For each operation (e.g., `SumFloat64`):

```go
func calibrateNEONThreshold[T any](
    scalarFn func([]T) T,
    neonFn func([]T) T,
    buf []T,
) int {
    for _, n := range calibrationSizes {
        tScalar := stableMeasure(func() time.Duration {
            return benchFn(scalarFn, buf[:n], calibrationIters)
        })
        tNEON := stableMeasure(func() time.Duration {
            return benchFn(neonFn, buf[:n], calibrationIters)
        })
        
        // NEON wins if it's at least 5% faster
        if float64(tNEON)*marginFactor < float64(tScalar) {
            return n
        }
    }
    return 1 << 30 // Disable SIMD - never faster
}
```

#### 4. SVE vs NEON/Scalar

SVE threshold calibration considers the already-computed NEON threshold:

```go
func calibrateSVEThreshold[T any](
    baseFn func([]T) T,  // NEON or scalar depending on size
    sveFn func([]T) T,
    buf []T,
    neonThreshold int,
) int {
    for _, n := range calibrationSizes {
        // Baseline is NEON if n >= neonThreshold, else scalar
        tBase := stableMeasure(func() time.Duration {
            return benchFn(baseFn, buf[:n], calibrationIters)
        })
        tSVE := stableMeasure(func() time.Duration {
            return benchFn(sveFn, buf[:n], calibrationIters)
        })
        
        if float64(tSVE)*marginFactor < float64(tBase) {
            // SVE wins, but threshold must be >= NEON threshold
            // to maintain dispatch ordering
            return max(n, neonThreshold)
        }
    }
    return 1 << 30 // Disable SVE
}
```

#### 5. Threshold Snapping

To reduce noise and maintain consistency:

```go
func snapToGrid(n int) int {
    grid := []int{4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256}
    for _, g := range grid {
        if n <= g {
            return g
        }
    }
    return 1 << 30
}
```

### Startup Cost Analysis

| Configuration | Calls per Op | Time per Op | Total (36 ops) |
|---------------|--------------|-------------|----------------|
| 8 sizes × 2 impl × 256 iters × 3 reps | 12,288 | ~0.25ms | ~9ms |
| 6 sizes × 2 impl × 256 iters × 2 reps | 6,144 | ~0.12ms | ~4.5ms |

**Realistic estimate: 5-10ms** including SVE where available.

This is acceptable for an opt-in feature triggered by environment variable or explicit call.

### Storage

Thresholds are stored by overwriting the global structs:

```go
var calibrateOnce sync.Once

func CalibrateThresholds() {
    calibrateOnce.Do(calibrateAll)
}

func calibrateAll() {
    buf64 := make([]float64, 256)
    buf32 := make([]float32, 256)
    // ... fill with test data
    
    nt := neonThresholds  // copy current
    st := sveThresholds
    
    // Float64
    nt.SumFloat64 = calibrateNEONThreshold(sumFloat64Scalar, sumFloat64NEON, buf64)
    if hasSVE {
        st.SumFloat64 = calibrateSVEThreshold(...)
    }
    // ... repeat for all 36 operations
    
    // Atomic replacement
    neonThresholds = nt
    sveThresholds = st
}
```

### Concurrency Safety

- `sync.Once` ensures calibration runs at most once
- Calibration must complete before concurrent simd use (documented requirement)
- No locks on hot-path reads of thresholds (they're immutable after init)

### Disk Caching

**Not recommended for initial implementation.**

Reasons:
- Calibration cost (5-10ms) is acceptable for opt-in use
- Cache key complexity (CPU ID, SVE length, OS version, simd version)
- Cache invalidation on upgrades
- Disk I/O latency in containerized environments

If needed later, cache key could be:
```json
{
  "cpu_id": "0x410FD4F0",
  "sve_len": 256,
  "simd_version": "v1.3.0"
}
```

## File Structure

```
simd-go/
├── calibrate_arm64.go      # Calibration implementation
├── calibrate_arm64_test.go # Calibration tests
└── caps_arm64.go           # Existing (no changes needed)
```

## Testing Strategy

1. **Unit tests**: Verify calibration produces reasonable thresholds
2. **Integration tests**: Compare calibrated vs static thresholds
3. **Benchmark validation**: Ensure calibrated thresholds don't regress performance

```go
func TestCalibration(t *testing.T) {
    // Save original thresholds
    origNEON := neonThresholds
    origSVE := sveThresholds
    defer func() {
        neonThresholds = origNEON
        sveThresholds = origSVE
    }()
    
    CalibrateThresholds()
    
    // Thresholds should be reasonable
    if neonThresholds.SumFloat64 < 2 || neonThresholds.SumFloat64 > 256 {
        t.Errorf("SumFloat64 threshold %d outside expected range", neonThresholds.SumFloat64)
    }
    // ... more assertions
}
```

## Verbose Mode

For debugging and validation:

```bash
SIMD_CALIBRATE=1 SIMD_CALIBRATE_VERBOSE=1 ./myapp
```

Output:
```
simd: calibrating thresholds...
simd: SumFloat64: NEON threshold=32 (scalar=12.3ns, neon=5.1ns at n=32)
simd: SumFloat64: SVE threshold=16 (neon=5.1ns, sve=4.8ns at n=16)
...
simd: calibration complete in 6.2ms
```

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Noisy measurements | Multiple reps, take minimum, 5% margin |
| CPU frequency scaling | Document "run on idle system", verbose mode for validation |
| Concurrent access during calibration | sync.Once, document early-call requirement |
| Bad thresholds on unusual systems | Conservative margin, snap to grid, verbose logging |

## Alternatives Considered

### 1. Expand Static Tables Only
**Pros**: Zero runtime cost, deterministic, easy to test
**Cons**: Requires hardware access for each new CPU

**Verdict**: This remains the primary approach. Calibration is supplementary.

### 2. Compile-Time Calibration
**Pros**: No runtime cost
**Cons**: Doesn't help users on unknown hardware, complex build process

**Verdict**: Not worth the complexity.

### 3. Background Adaptive Calibration
**Pros**: Can refine during runtime
**Cons**: Complex, unpredictable behavior, hard to debug

**Verdict**: Over-engineered for this use case.

## Implementation Priority

**Low-Medium Priority**

Given the existing static thresholds cover the main platforms (M3, G3, G4), this is primarily a "nice to have" for:
- Unknown ARM CPUs
- Power users
- Future-proofing

Recommended timeline:
1. Finish any remaining P0/P1 items first
2. Implement calibration when there's demand from users on unsupported CPUs
3. Consider implementing if expanding to more ARM variants (Ampere Altra, etc.)

## Effort Estimate

- Initial implementation: 1-2 days
- Testing and validation: 0.5-1 day
- Documentation: 0.5 day
- **Total: 2-4 days**
