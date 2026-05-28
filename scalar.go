package simd

// Scalar implementations for testing and non-SIMD platforms.
// These are simple scalar loops used to verify SIMD correctness.

func sumFloat64Scalar(vals []float64) float64 {
	var sum float64
	for _, v := range vals {
		sum += v
	}
	return sum
}

func minFloat64Scalar(vals []float64) float64 {
	if len(vals) == 0 {
		return 0
	}
	min := vals[0]
	for _, v := range vals[1:] {
		if v < min {
			min = v
		}
	}
	return min
}

func maxFloat64Scalar(vals []float64) float64 {
	if len(vals) == 0 {
		return 0
	}
	max := vals[0]
	for _, v := range vals[1:] {
		if v > max {
			max = v
		}
	}
	return max
}

func dotProductFloat64Scalar(a, b []float64) float64 {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	var sum float64
	for i := 0; i < n; i++ {
		sum += a[i] * b[i]
	}
	return sum
}

func sumInt64Scalar(vals []int64) int64 {
	var sum int64
	for _, v := range vals {
		sum += v
	}
	return sum
}

func minInt64Scalar(vals []int64) int64 {
	if len(vals) == 0 {
		return 0
	}
	min := vals[0]
	for _, v := range vals[1:] {
		if v < min {
			min = v
		}
	}
	return min
}

func maxInt64Scalar(vals []int64) int64 {
	if len(vals) == 0 {
		return 0
	}
	max := vals[0]
	for _, v := range vals[1:] {
		if v > max {
			max = v
		}
	}
	return max
}

func dotProductInt64Scalar(a, b []int64) int64 {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	var sum int64
	for i := 0; i < n; i++ {
		sum += a[i] * b[i]
	}
	return sum
}

func sumSqInt64Scalar(vals []int64) int64 {
	var sum int64
	for _, v := range vals {
		sum += v * v
	}
	return sum
}

func anyAbsGreaterThanScalar(vals []int64, threshold int64) bool {
	for _, v := range vals {
		if v > threshold || v < -threshold {
			return true
		}
	}
	return false
}

// Int32 scalar implementations

func sumInt32Scalar(vals []int32) int64 {
	var sum int64
	for _, v := range vals {
		sum += int64(v)
	}
	return sum
}

func minInt32Scalar(vals []int32) int32 {
	if len(vals) == 0 {
		return 0
	}
	min := vals[0]
	for _, v := range vals[1:] {
		if v < min {
			min = v
		}
	}
	return min
}

func maxInt32Scalar(vals []int32) int32 {
	if len(vals) == 0 {
		return 0
	}
	max := vals[0]
	for _, v := range vals[1:] {
		if v > max {
			max = v
		}
	}
	return max
}

func dotProductInt32Scalar(a, b []int32) int64 {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	var sum int64
	for i := 0; i < n; i++ {
		sum += int64(a[i]) * int64(b[i])
	}
	return sum
}

func sumSqInt32Scalar(vals []int32) int64 {
	var sum int64
	for _, v := range vals {
		sum += int64(v) * int64(v)
	}
	return sum
}

func anyAbsGreaterThanInt32Scalar(vals []int32, threshold int32) bool {
	for _, v := range vals {
		if v > threshold || v < -threshold {
			return true
		}
	}
	return false
}

// Int16 scalar implementations

func sumInt16Scalar(vals []int16) int64 {
	var sum int64
	for _, v := range vals {
		sum += int64(v)
	}
	return sum
}

func minInt16Scalar(vals []int16) int16 {
	if len(vals) == 0 {
		return 0
	}
	min := vals[0]
	for _, v := range vals[1:] {
		if v < min {
			min = v
		}
	}
	return min
}

func maxInt16Scalar(vals []int16) int16 {
	if len(vals) == 0 {
		return 0
	}
	max := vals[0]
	for _, v := range vals[1:] {
		if v > max {
			max = v
		}
	}
	return max
}

func dotProductInt16Scalar(a, b []int16) int64 {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	var sum int64
	for i := 0; i < n; i++ {
		sum += int64(a[i]) * int64(b[i])
	}
	return sum
}

func sumSqInt16Scalar(vals []int16) int64 {
	var sum int64
	for _, v := range vals {
		sum += int64(v) * int64(v)
	}
	return sum
}

func anyAbsGreaterThanInt16Scalar(vals []int16, threshold int16) bool {
	for _, v := range vals {
		if v > threshold || v < -threshold {
			return true
		}
	}
	return false
}

// Float32 scalar implementations

func sumFloat32Scalar(vals []float32) float32 {
	var sum float32
	for _, v := range vals {
		sum += v
	}
	return sum
}

func minFloat32Scalar(vals []float32) float32 {
	if len(vals) == 0 {
		return 0
	}
	min := vals[0]
	for _, v := range vals[1:] {
		if v < min {
			min = v
		}
	}
	return min
}

func maxFloat32Scalar(vals []float32) float32 {
	if len(vals) == 0 {
		return 0
	}
	max := vals[0]
	for _, v := range vals[1:] {
		if v > max {
			max = v
		}
	}
	return max
}

func dotProductFloat32Scalar(a, b []float32) float32 {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	var sum float32
	for i := 0; i < n; i++ {
		sum += a[i] * b[i]
	}
	return sum
}

const (
	dotProductFloat32BatchSize    = 4
	dotProductFloat32BatchMinDims = 64
)

func dotProductFloat32IndexedShapeOK(base []float32, query []float32, rowIDs []uint32, dims int) bool {
	if dims <= 0 || len(query) < dims {
		return false
	}
	if len(rowIDs) == 0 {
		return true
	}
	maxStart := len(base) - dims
	if maxStart < 0 {
		return false
	}
	maxRowID := uint64(maxStart) / uint64(dims)
	for _, rowID := range rowIDs {
		if uint64(rowID) > maxRowID {
			return false
		}
	}
	return true
}

func dotProductFloat32StridedShapeOK(base []float32, query []float32, rowCount, dims, stride int) bool {
	if rowCount < 0 || dims <= 0 || stride < dims || len(query) < dims {
		return false
	}
	if rowCount == 0 {
		return true
	}
	maxStart := len(base) - dims
	if maxStart < 0 {
		return false
	}
	return uint64(rowCount-1) <= uint64(maxStart)/uint64(stride)
}

func dotProductFloat32IndexedUseNEON(rowIDs []uint32, dims int) bool {
	rowCount := len(rowIDs)
	if rowCount < dotProductFloat32BatchSize || dims < dotProductFloat32BatchMinDims {
		return false
	}
	if dims <= 64 {
		return true
	}
	if dims <= 128 {
		return rowCount%dotProductFloat32BatchSize == 0
	}
	sequential := dotProductFloat32RowIDsSequential(rowIDs)
	if dims >= 2048 {
		if sequential {
			return rowCount <= 8
		}
		if rowCount > dotProductFloat32BatchSize && rowCount < 16 {
			return false
		}
		return true
	}
	if rowCount < 256 {
		return true
	}
	return !sequential
}

func dotProductFloat32StridedUseNEON(rowCount, dims int) bool {
	if rowCount < dotProductFloat32BatchSize || dims < dotProductFloat32BatchMinDims {
		return false
	}
	if dims <= 64 {
		return true
	}
	if dims <= 128 {
		return rowCount%dotProductFloat32BatchSize == 0
	}
	return false
}

func dotProductFloat32RowIDsSequential(rowIDs []uint32) bool {
	if len(rowIDs) < 2 {
		return true
	}
	prev := rowIDs[0]
	for _, rowID := range rowIDs[1:] {
		if rowID != prev+1 {
			return false
		}
		prev = rowID
	}
	return true
}

func dotProductFloat32IndexedDotLoop(dst []float32, base []float32, query []float32, rowIDs []uint32, dims int) {
	query = query[:dims]
	for i := 0; i < len(dst); i++ {
		start := int(rowIDs[i]) * dims
		dst[i] = DotProductFloat32(base[start:start+dims], query)
	}
}

func dotProductFloat32StridedDotLoop(dst []float32, base []float32, query []float32, rowCount, dims, stride int) {
	query = query[:dims]
	for i := 0; i < rowCount; i++ {
		start := i * stride
		dst[i] = DotProductFloat32(base[start:start+dims], query)
	}
}

func dotProductFloat32IndexedScalar(dst []float32, base []float32, query []float32, rowIDs []uint32, dims int) {
	for i := 0; i < len(dst); i++ {
		start := int(rowIDs[i]) * dims
		var sum float32
		for j := 0; j < dims; j++ {
			sum += base[start+j] * query[j]
		}
		dst[i] = sum
	}
}

func dotProductFloat32StridedScalar(dst []float32, base []float32, query []float32, rowCount, dims, stride int) {
	for i := 0; i < rowCount; i++ {
		start := i * stride
		var sum float32
		for j := 0; j < dims; j++ {
			sum += base[start+j] * query[j]
		}
		dst[i] = sum
	}
}
