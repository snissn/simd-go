//go:build arm64

package simd

// Dispatch functions select SVE or NEON implementation based on CPU capabilities.

func sumFloat64Impl(vals []float64) float64 {
	if hasSVE {
		return sumFloat64SVE(vals)
	}
	return sumFloat64NEON(vals)
}

func minFloat64Impl(vals []float64) float64 {
	if hasSVE {
		return minFloat64SVE(vals)
	}
	return minFloat64NEON(vals)
}

func maxFloat64Impl(vals []float64) float64 {
	if hasSVE {
		return maxFloat64SVE(vals)
	}
	return maxFloat64NEON(vals)
}

func dotProductFloat64Impl(a, b []float64) float64 {
	if hasSVE {
		return dotProductFloat64SVE(a, b)
	}
	return dotProductFloat64NEON(a, b)
}

func sumInt64Impl(vals []int64) int64 {
	if hasSVE {
		return sumInt64SVE(vals)
	}
	return sumInt64NEON(vals)
}

func minInt64Impl(vals []int64) int64 {
	if hasSVE {
		return minInt64SVE(vals)
	}
	return minInt64NEON(vals)
}

func maxInt64Impl(vals []int64) int64 {
	if hasSVE {
		return maxInt64SVE(vals)
	}
	return maxInt64NEON(vals)
}

func dotProductInt64Impl(a, b []int64) int64 {
	if hasSVE {
		return dotProductInt64SVE(a, b)
	}
	// NEON lacks 64-bit integer MUL, so its "NEON" version uses scalar MUL
	// with extra overhead. Benchmarks show scalar is faster on NEON-only platforms.
	return dotProductInt64Scalar(a, b)
}

func sumSqInt64Impl(vals []int64) int64 {
	if hasSVE {
		return sumSqInt64SVE(vals)
	}
	return sumSqInt64NEON(vals)
}

func anyAbsGreaterThanImpl(vals []int64, threshold int64) bool {
	if hasSVE {
		return anyAbsGreaterThanSVE(vals, threshold)
	}
	return anyAbsGreaterThanNEON(vals, threshold)
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
