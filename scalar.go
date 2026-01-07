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
