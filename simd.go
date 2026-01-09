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

// SumInt32 computes the sum of int32 values.
// Returns int64 to avoid overflow.
func SumInt32(vals []int32) int64 {
	if len(vals) == 0 {
		return 0
	}
	return sumInt32Impl(vals)
}

// MinInt32 finds the minimum int32 value.
// Returns 0 for empty slices.
func MinInt32(vals []int32) int32 {
	if len(vals) == 0 {
		return 0
	}
	return minInt32Impl(vals)
}

// MaxInt32 finds the maximum int32 value.
// Returns 0 for empty slices.
func MaxInt32(vals []int32) int32 {
	if len(vals) == 0 {
		return 0
	}
	return maxInt32Impl(vals)
}

// DotProductInt32 computes the dot product of two int32 slices.
// Returns int64 to avoid overflow.
// Uses the minimum length of the two slices.
func DotProductInt32(a, b []int32) int64 {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	if n == 0 {
		return 0
	}
	return dotProductInt32Impl(a[:n], b[:n])
}

// SumSqInt32 computes the sum of squares of int32 values.
// Returns int64 to avoid overflow.
func SumSqInt32(vals []int32) int64 {
	if len(vals) == 0 {
		return 0
	}
	return sumSqInt32Impl(vals)
}

// AnyAbsGreaterThanInt32 checks if any |v| > threshold.
// Useful for overflow detection before SumSqInt32.
func AnyAbsGreaterThanInt32(vals []int32, threshold int32) bool {
	if len(vals) == 0 {
		return false
	}
	return anyAbsGreaterThanInt32Impl(vals, threshold)
}

// SumInt16 computes the sum of int16 values.
// Returns int64 to avoid overflow.
func SumInt16(vals []int16) int64 {
	if len(vals) == 0 {
		return 0
	}
	return sumInt16Impl(vals)
}

// MinInt16 finds the minimum int16 value.
// Returns 0 for empty slices.
func MinInt16(vals []int16) int16 {
	if len(vals) == 0 {
		return 0
	}
	return minInt16Impl(vals)
}

// MaxInt16 finds the maximum int16 value.
// Returns 0 for empty slices.
func MaxInt16(vals []int16) int16 {
	if len(vals) == 0 {
		return 0
	}
	return maxInt16Impl(vals)
}

// DotProductInt16 computes the dot product of two int16 slices.
// Returns int64 to avoid overflow.
// Uses the minimum length of the two slices.
func DotProductInt16(a, b []int16) int64 {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	if n == 0 {
		return 0
	}
	return dotProductInt16Impl(a[:n], b[:n])
}

// SumSqInt16 computes the sum of squares of int16 values.
// Returns int64 to avoid overflow.
func SumSqInt16(vals []int16) int64 {
	if len(vals) == 0 {
		return 0
	}
	return sumSqInt16Impl(vals)
}

// AnyAbsGreaterThanInt16 checks if any |v| > threshold.
// Useful for overflow detection before SumSqInt16.
func AnyAbsGreaterThanInt16(vals []int16, threshold int16) bool {
	if len(vals) == 0 {
		return false
	}
	return anyAbsGreaterThanInt16Impl(vals, threshold)
}

// SumFloat32 computes the sum of float32 values.
func SumFloat32(vals []float32) float32 {
	if len(vals) == 0 {
		return 0
	}
	return sumFloat32Impl(vals)
}

// MinFloat32 finds the minimum float32 value.
// Returns 0 for empty slices.
func MinFloat32(vals []float32) float32 {
	if len(vals) == 0 {
		return 0
	}
	return minFloat32Impl(vals)
}

// MaxFloat32 finds the maximum float32 value.
// Returns 0 for empty slices.
func MaxFloat32(vals []float32) float32 {
	if len(vals) == 0 {
		return 0
	}
	return maxFloat32Impl(vals)
}

// DotProductFloat32 computes the dot product of two float32 slices.
// Uses the minimum length of the two slices.
func DotProductFloat32(a, b []float32) float32 {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	if n == 0 {
		return 0
	}
	return dotProductFloat32Impl(a[:n], b[:n])
}
