// Package simd provides SIMD-accelerated numeric operations.
//
// All functions have scalar fallbacks for unsupported platforms.
// SIMD implementations are provided for arm64 (NEON, with SVE on supported CPUs).
//
// The implementations use multiple accumulators to hide memory latency
// and maximize instruction-level parallelism.
package simd

// SumFloat64 computes the sum of float64 values.
func SumFloat64(vals []float64) float64 {
	if len(vals) == 0 {
		return 0
	}
	return sumFloat64Impl(vals)
}

// MinFloat64 finds the minimum float64 value.
// Returns 0 for empty slices.
func MinFloat64(vals []float64) float64 {
	if len(vals) == 0 {
		return 0
	}
	return minFloat64Impl(vals)
}

// MaxFloat64 finds the maximum float64 value.
// Returns 0 for empty slices.
func MaxFloat64(vals []float64) float64 {
	if len(vals) == 0 {
		return 0
	}
	return maxFloat64Impl(vals)
}

// DotProductFloat64 computes the dot product of two float64 slices.
// Uses the minimum length of the two slices.
func DotProductFloat64(a, b []float64) float64 {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	if n == 0 {
		return 0
	}
	return dotProductFloat64Impl(a[:n], b[:n])
}

// SumInt64 computes the sum of int64 values.
func SumInt64(vals []int64) int64 {
	if len(vals) == 0 {
		return 0
	}
	return sumInt64Impl(vals)
}

// MinInt64 finds the minimum int64 value.
// Returns 0 for empty slices.
func MinInt64(vals []int64) int64 {
	if len(vals) == 0 {
		return 0
	}
	return minInt64Impl(vals)
}

// MaxInt64 finds the maximum int64 value.
// Returns 0 for empty slices.
func MaxInt64(vals []int64) int64 {
	if len(vals) == 0 {
		return 0
	}
	return maxInt64Impl(vals)
}

// DotProductInt64 computes the dot product of two int64 slices.
// Uses the minimum length of the two slices.
func DotProductInt64(a, b []int64) int64 {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	if n == 0 {
		return 0
	}
	return dotProductInt64Impl(a[:n], b[:n])
}

// SumSqInt64 computes the sum of squares of int64 values.
// The caller MUST ensure no value has |v| > 3037000499 (sqrt(MaxInt64))
// to avoid overflow. Use AnyAbsGreaterThan to check first.
func SumSqInt64(vals []int64) int64 {
	if len(vals) == 0 {
		return 0
	}
	return sumSqInt64Impl(vals)
}

// AnyAbsGreaterThan checks if any |v| > threshold.
// Useful for overflow detection before SumSqInt64.
func AnyAbsGreaterThan(vals []int64, threshold int64) bool {
	if len(vals) == 0 {
		return false
	}
	return anyAbsGreaterThanImpl(vals, threshold)
}
