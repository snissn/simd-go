//go:build !arm64

package simd

// Scalar implementations for non-ARM64 platforms.
// Delegates to scalar implementations in scalar.go.

func sumFloat64Impl(vals []float64) float64 {
	return sumFloat64Scalar(vals)
}

func minFloat64Impl(vals []float64) float64 {
	return minFloat64Scalar(vals)
}

func maxFloat64Impl(vals []float64) float64 {
	return maxFloat64Scalar(vals)
}

func dotProductFloat64Impl(a, b []float64) float64 {
	return dotProductFloat64Scalar(a, b)
}

func sumInt64Impl(vals []int64) int64 {
	return sumInt64Scalar(vals)
}

func minInt64Impl(vals []int64) int64 {
	return minInt64Scalar(vals)
}

func maxInt64Impl(vals []int64) int64 {
	return maxInt64Scalar(vals)
}

func dotProductInt64Impl(a, b []int64) int64 {
	return dotProductInt64Scalar(a, b)
}

func sumSqInt64Impl(vals []int64) int64 {
	return sumSqInt64Scalar(vals)
}

func anyAbsGreaterThanImpl(vals []int64, threshold int64) bool {
	return anyAbsGreaterThanScalar(vals, threshold)
}
