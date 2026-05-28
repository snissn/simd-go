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

func sumInt32Impl(vals []int32) int64 {
	return sumInt32Scalar(vals)
}

func minInt32Impl(vals []int32) int32 {
	return minInt32Scalar(vals)
}

func maxInt32Impl(vals []int32) int32 {
	return maxInt32Scalar(vals)
}

func dotProductInt32Impl(a, b []int32) int64 {
	return dotProductInt32Scalar(a, b)
}

func sumSqInt32Impl(vals []int32) int64 {
	return sumSqInt32Scalar(vals)
}

func anyAbsGreaterThanInt32Impl(vals []int32, threshold int32) bool {
	return anyAbsGreaterThanInt32Scalar(vals, threshold)
}

func sumInt16Impl(vals []int16) int64 {
	return sumInt16Scalar(vals)
}

func minInt16Impl(vals []int16) int16 {
	return minInt16Scalar(vals)
}

func maxInt16Impl(vals []int16) int16 {
	return maxInt16Scalar(vals)
}

func dotProductInt16Impl(a, b []int16) int64 {
	return dotProductInt16Scalar(a, b)
}

func sumSqInt16Impl(vals []int16) int64 {
	return sumSqInt16Scalar(vals)
}

func anyAbsGreaterThanInt16Impl(vals []int16, threshold int16) bool {
	return anyAbsGreaterThanInt16Scalar(vals, threshold)
}

func sumFloat32Impl(vals []float32) float32 {
	return sumFloat32Scalar(vals)
}

func minFloat32Impl(vals []float32) float32 {
	return minFloat32Scalar(vals)
}

func maxFloat32Impl(vals []float32) float32 {
	return maxFloat32Scalar(vals)
}

func dotProductFloat32Impl(a, b []float32) float32 {
	return dotProductFloat32Scalar(a, b)
}

func dotProductFloat32IndexedImpl(dst []float32, base []float32, query []float32, rowIDs []uint32, dims int) bool {
	if len(dst) == 0 || !dotProductFloat32IndexedShapeOK(base, query, rowIDs, dims) {
		return false
	}
	dotProductFloat32IndexedDotLoop(dst, base, query, rowIDs, dims)
	return false
}

func dotProductFloat32StridedImpl(dst []float32, base []float32, query []float32, rowCount, dims, stride int) bool {
	if rowCount == 0 || !dotProductFloat32StridedShapeOK(base, query, rowCount, dims, stride) {
		return false
	}
	dotProductFloat32StridedDotLoop(dst, base, query, rowCount, dims, stride)
	return false
}
