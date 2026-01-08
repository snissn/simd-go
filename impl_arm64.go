//go:build arm64

package simd

// Dispatch functions select scalar, SVE, or NEON based on CPU capabilities and input size.
// Small arrays use scalar to avoid SIMD call overhead.

func sumFloat64Impl(vals []float64) float64 {
	n := len(vals)
	if hasSVE && n >= sveThresholds.SumFloat64 {
		return sumFloat64SVE(vals)
	}
	if n >= neonThresholds.SumFloat64 {
		return sumFloat64NEON(vals)
	}
	return sumFloat64Scalar(vals)
}

func minFloat64Impl(vals []float64) float64 {
	n := len(vals)
	if hasSVE && n >= sveThresholds.MinFloat64 {
		return minFloat64SVE(vals)
	}
	if n >= neonThresholds.MinFloat64 {
		return minFloat64NEON(vals)
	}
	return minFloat64Scalar(vals)
}

func maxFloat64Impl(vals []float64) float64 {
	n := len(vals)
	if hasSVE && n >= sveThresholds.MaxFloat64 {
		return maxFloat64SVE(vals)
	}
	if n >= neonThresholds.MaxFloat64 {
		return maxFloat64NEON(vals)
	}
	return maxFloat64Scalar(vals)
}

func dotProductFloat64Impl(a, b []float64) float64 {
	n := len(a)
	if hasSVE && n >= sveThresholds.DotProductFloat64 {
		return dotProductFloat64SVE(a, b)
	}
	if n >= neonThresholds.DotProductFloat64 {
		return dotProductFloat64NEON(a, b)
	}
	return dotProductFloat64Scalar(a, b)
}

func sumInt64Impl(vals []int64) int64 {
	n := len(vals)
	if hasSVE && n >= sveThresholds.SumInt64 {
		return sumInt64SVE(vals)
	}
	if n >= neonThresholds.SumInt64 {
		return sumInt64NEON(vals)
	}
	return sumInt64Scalar(vals)
}

func minInt64Impl(vals []int64) int64 {
	n := len(vals)
	if hasSVE && n >= sveThresholds.MinInt64 {
		return minInt64SVE(vals)
	}
	if n >= neonThresholds.MinInt64 {
		return minInt64NEON(vals)
	}
	return minInt64Scalar(vals)
}

func maxInt64Impl(vals []int64) int64 {
	n := len(vals)
	if hasSVE && n >= sveThresholds.MaxInt64 {
		return maxInt64SVE(vals)
	}
	if n >= neonThresholds.MaxInt64 {
		return maxInt64NEON(vals)
	}
	return maxInt64Scalar(vals)
}

func dotProductInt64Impl(a, b []int64) int64 {
	n := len(a)
	// No NEON impl for int64 dot product (lacks 64-bit MUL)
	if hasSVE && n >= sveThresholds.DotProductInt64 {
		return dotProductInt64SVE(a, b)
	}
	return dotProductInt64Scalar(a, b)
}

func sumSqInt64Impl(vals []int64) int64 {
	n := len(vals)
	if hasSVE && n >= sveThresholds.SumSqInt64 {
		return sumSqInt64SVE(vals)
	}
	if n >= neonThresholds.SumSqInt64 {
		return sumSqInt64NEON(vals)
	}
	return sumSqInt64Scalar(vals)
}

func anyAbsGreaterThanImpl(vals []int64, threshold int64) bool {
	n := len(vals)
	if hasSVE && n >= sveThresholds.AnyAbsGreaterThan {
		return anyAbsGreaterThanSVE(vals, threshold)
	}
	if n >= neonThresholds.AnyAbsGreaterThan {
		return anyAbsGreaterThanNEON(vals, threshold)
	}
	return anyAbsGreaterThanScalar(vals, threshold)
}

// NEON assembly implementations (in float64_arm64_neon.s and int64_arm64_neon.s)
func sumFloat64NEON(vals []float64) float64
func minFloat64NEON(vals []float64) float64
func maxFloat64NEON(vals []float64) float64
func dotProductFloat64NEON(a, b []float64) float64
func sumInt64NEON(vals []int64) int64
func minInt64NEON(vals []int64) int64
func maxInt64NEON(vals []int64) int64
func sumSqInt64NEON(vals []int64) int64
func anyAbsGreaterThanNEON(vals []int64, threshold int64) bool

// SVE assembly implementations (in float64_arm64_sve.s and int64_arm64_sve.s)
func sumFloat64SVE(vals []float64) float64
func minFloat64SVE(vals []float64) float64
func maxFloat64SVE(vals []float64) float64
func dotProductFloat64SVE(a, b []float64) float64
func sumInt64SVE(vals []int64) int64
func minInt64SVE(vals []int64) int64
func maxInt64SVE(vals []int64) int64
func dotProductInt64SVE(a, b []int64) int64
func sumSqInt64SVE(vals []int64) int64
func anyAbsGreaterThanSVE(vals []int64, threshold int64) bool
