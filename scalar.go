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
