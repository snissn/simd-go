package simd

import (
	"math"
	"testing"
)

func FuzzSumFloat64(f *testing.F) {
	f.Add([]byte{})
	f.Add([]byte{0, 0, 0, 0, 0, 0, 0, 0})
	f.Add([]byte{0, 0, 0, 0, 0, 0, 240, 63}) // 1.0

	f.Fuzz(func(t *testing.T, data []byte) {
		vals := bytesToFloat64s(data)
		if containsNaN(vals) || hasExtremeMagnitudeSpread(vals) {
			t.Skip("NaN/Inf values or extreme magnitude spread")
		}
		got := SumFloat64(vals)
		want := sumFloat64Scalar(vals)
		if !floatEqual(got, want) {
			t.Errorf("SumFloat64(%v) = %v, want %v", vals, got, want)
		}
	})
}

func FuzzMinFloat64(f *testing.F) {
	f.Add([]byte{})
	f.Add([]byte{0, 0, 0, 0, 0, 0, 240, 63})

	f.Fuzz(func(t *testing.T, data []byte) {
		vals := bytesToFloat64s(data)
		if containsNaN(vals) {
			t.Skip("NaN values")
		}
		got := MinFloat64(vals)
		want := minFloat64Scalar(vals)
		if got != want {
			t.Errorf("MinFloat64(%v) = %v, want %v", vals, got, want)
		}
	})
}

func FuzzMaxFloat64(f *testing.F) {
	f.Add([]byte{})
	f.Add([]byte{0, 0, 0, 0, 0, 0, 240, 63})

	f.Fuzz(func(t *testing.T, data []byte) {
		vals := bytesToFloat64s(data)
		if containsNaN(vals) {
			t.Skip("NaN values")
		}
		got := MaxFloat64(vals)
		want := maxFloat64Scalar(vals)
		if got != want {
			t.Errorf("MaxFloat64(%v) = %v, want %v", vals, got, want)
		}
	})
}

func FuzzDotProductFloat64(f *testing.F) {
	f.Add([]byte{}, []byte{})
	f.Add([]byte{0, 0, 0, 0, 0, 0, 240, 63}, []byte{0, 0, 0, 0, 0, 0, 0, 64})

	f.Fuzz(func(t *testing.T, dataA, dataB []byte) {
		a := bytesToFloat64s(dataA)
		b := bytesToFloat64s(dataB)
		if containsNaN(a) || containsNaN(b) || hasExtremeMagnitudeSpread(a) || hasExtremeMagnitudeSpread(b) {
			t.Skip("NaN/Inf values or extreme magnitude spread")
		}
		got := DotProductFloat64(a, b)
		want := dotProductFloat64Scalar(a, b)
		if math.IsNaN(got) || math.IsNaN(want) {
			t.Skip("Result is NaN")
		}
		if !floatEqual(got, want) {
			t.Errorf("DotProductFloat64 = %v, want %v", got, want)
		}
	})
}

func FuzzSumInt64(f *testing.F) {
	f.Add([]byte{})
	f.Add([]byte{1, 0, 0, 0, 0, 0, 0, 0})
	f.Add([]byte{255, 255, 255, 255, 255, 255, 255, 127}) // max int64

	f.Fuzz(func(t *testing.T, data []byte) {
		vals := bytesToInt64s(data)
		got := SumInt64(vals)
		want := sumInt64Scalar(vals)
		if got != want {
			t.Errorf("SumInt64(%v) = %v, want %v", vals, got, want)
		}
	})
}

func FuzzMinInt64(f *testing.F) {
	f.Add([]byte{})
	f.Add([]byte{1, 0, 0, 0, 0, 0, 0, 0})

	f.Fuzz(func(t *testing.T, data []byte) {
		vals := bytesToInt64s(data)
		got := MinInt64(vals)
		want := minInt64Scalar(vals)
		if got != want {
			t.Errorf("MinInt64(%v) = %v, want %v", vals, got, want)
		}
	})
}

func FuzzMaxInt64(f *testing.F) {
	f.Add([]byte{})
	f.Add([]byte{1, 0, 0, 0, 0, 0, 0, 0})

	f.Fuzz(func(t *testing.T, data []byte) {
		vals := bytesToInt64s(data)
		got := MaxInt64(vals)
		want := maxInt64Scalar(vals)
		if got != want {
			t.Errorf("MaxInt64(%v) = %v, want %v", vals, got, want)
		}
	})
}

func FuzzDotProductInt64(f *testing.F) {
	f.Add([]byte{}, []byte{})
	f.Add([]byte{1, 0, 0, 0, 0, 0, 0, 0}, []byte{2, 0, 0, 0, 0, 0, 0, 0})
	f.Add([]byte{255, 255, 255, 255, 255, 255, 255, 127}, []byte{1, 0, 0, 0, 0, 0, 0, 0})

	f.Fuzz(func(t *testing.T, dataA, dataB []byte) {
		a := bytesToInt64s(dataA)
		b := bytesToInt64s(dataB)
		if int64sWouldOverflow(a, b) {
			t.Skip("overflow")
		}
		got := DotProductInt64(a, b)
		want := dotProductInt64Scalar(a, b)
		if got != want {
			t.Errorf("DotProductInt64 = %v, want %v", got, want)
		}
	})
}

func FuzzSumSqInt64(f *testing.F) {
	f.Add([]byte{})
	f.Add([]byte{1, 0, 0, 0, 0, 0, 0, 0})
	f.Add([]byte{10, 0, 0, 0, 0, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0})

	f.Fuzz(func(t *testing.T, data []byte) {
		vals := bytesToInt64s(data)
		if int64sWouldOverflow(vals, vals) {
			t.Skip("overflow")
		}
		got := SumSqInt64(vals)
		want := sumSqInt64Scalar(vals)
		if got != want {
			t.Errorf("SumSqInt64(%v) = %v, want %v", vals, got, want)
		}
	})
}

func FuzzAnyAbsGreaterThan(f *testing.F) {
	f.Add([]byte{}, int64(100))
	f.Add([]byte{1, 0, 0, 0, 0, 0, 0, 0}, int64(10))
	f.Add([]byte{255, 255, 255, 255, 255, 255, 255, 127}, int64(1000))

	f.Fuzz(func(t *testing.T, data []byte, threshold int64) {
		vals := bytesToInt64s(data)
		if threshold < 0 {
			threshold = -threshold
		}
		got := AnyAbsGreaterThan(vals, threshold)
		want := anyAbsGreaterThanScalar(vals, threshold)
		if got != want {
			t.Errorf("AnyAbsGreaterThan(%v, %d) = %v, want %v", vals, threshold, got, want)
		}
	})
}

// bytesToFloat64s converts bytes to float64 slice (8 bytes per element).
func bytesToFloat64s(data []byte) []float64 {
	n := len(data) / 8
	if n == 0 {
		return nil
	}
	vals := make([]float64, n)
	for i := 0; i < n; i++ {
		bits := uint64(data[i*8]) |
			uint64(data[i*8+1])<<8 |
			uint64(data[i*8+2])<<16 |
			uint64(data[i*8+3])<<24 |
			uint64(data[i*8+4])<<32 |
			uint64(data[i*8+5])<<40 |
			uint64(data[i*8+6])<<48 |
			uint64(data[i*8+7])<<56
		vals[i] = math.Float64frombits(bits)
	}
	return vals
}

// bytesToInt64s converts bytes to int64 slice (8 bytes per element).
func bytesToInt64s(data []byte) []int64 {
	n := len(data) / 8
	if n == 0 {
		return nil
	}
	vals := make([]int64, n)
	for i := 0; i < n; i++ {
		vals[i] = int64(data[i*8]) |
			int64(data[i*8+1])<<8 |
			int64(data[i*8+2])<<16 |
			int64(data[i*8+3])<<24 |
			int64(data[i*8+4])<<32 |
			int64(data[i*8+5])<<40 |
			int64(data[i*8+6])<<48 |
			int64(data[i*8+7])<<56
	}
	return vals
}

// containsNaN returns true if any value is NaN or Inf.
func containsNaN(vals []float64) bool {
	for _, v := range vals {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return true
		}
	}
	return false
}

// hasExtremeMagnitudeSpread returns true if values span extreme magnitudes.
func hasExtremeMagnitudeSpread(vals []float64) bool {
	if len(vals) < 2 {
		return false
	}
	var minAbs, maxAbs float64 = math.MaxFloat64, 0
	for _, v := range vals {
		abs := math.Abs(v)
		if abs > 0 && abs < minAbs {
			minAbs = abs
		}
		if abs > maxAbs {
			maxAbs = abs
		}
	}
	if minAbs == 0 || minAbs == math.MaxFloat64 {
		return false
	}
	return maxAbs/minAbs > 1e30
}

// floatEqual compares floats with tolerance for accumulated FP error.
func floatEqual(a, b float64) bool {
	if a == b {
		return true
	}
	diff := math.Abs(a - b)
	max := math.Max(math.Abs(a), math.Abs(b))
	if max == 0 {
		return diff < 1e-15
	}
	return diff/max < 1e-10
}

// Int32 fuzz tests

func FuzzSumInt32(f *testing.F) {
	f.Add([]byte{})
	f.Add([]byte{1, 0, 0, 0})
	f.Add([]byte{255, 255, 255, 127}) // max int32

	f.Fuzz(func(t *testing.T, data []byte) {
		vals := bytesToInt32s(data)
		got := SumInt32(vals)
		want := sumInt32Scalar(vals)
		if got != want {
			t.Errorf("SumInt32(%v) = %v, want %v", vals, got, want)
		}
	})
}

func FuzzMinInt32(f *testing.F) {
	f.Add([]byte{})
	f.Add([]byte{1, 0, 0, 0})

	f.Fuzz(func(t *testing.T, data []byte) {
		vals := bytesToInt32s(data)
		got := MinInt32(vals)
		want := minInt32Scalar(vals)
		if got != want {
			t.Errorf("MinInt32(%v) = %v, want %v", vals, got, want)
		}
	})
}

func FuzzMaxInt32(f *testing.F) {
	f.Add([]byte{})
	f.Add([]byte{1, 0, 0, 0})

	f.Fuzz(func(t *testing.T, data []byte) {
		vals := bytesToInt32s(data)
		got := MaxInt32(vals)
		want := maxInt32Scalar(vals)
		if got != want {
			t.Errorf("MaxInt32(%v) = %v, want %v", vals, got, want)
		}
	})
}

func FuzzDotProductInt32(f *testing.F) {
	f.Add([]byte{}, []byte{})
	f.Add([]byte{1, 0, 0, 0}, []byte{2, 0, 0, 0})

	f.Fuzz(func(t *testing.T, dataA, dataB []byte) {
		a := bytesToInt32s(dataA)
		b := bytesToInt32s(dataB)
		if int32sWouldOverflow(a, b) {
			t.Skip("overflow")
		}
		got := DotProductInt32(a, b)
		want := dotProductInt32Scalar(a, b)
		if got != want {
			t.Errorf("DotProductInt32 = %v, want %v", got, want)
		}
	})
}

func FuzzSumSqInt32(f *testing.F) {
	f.Add([]byte{})
	f.Add([]byte{1, 0, 0, 0})
	f.Add([]byte{10, 0, 0, 0, 20, 0, 0, 0})

	f.Fuzz(func(t *testing.T, data []byte) {
		vals := bytesToInt32s(data)
		if int32sWouldOverflow(vals, vals) {
			t.Skip("overflow")
		}
		got := SumSqInt32(vals)
		want := sumSqInt32Scalar(vals)
		if got != want {
			t.Errorf("SumSqInt32(%v) = %v, want %v", vals, got, want)
		}
	})
}

func FuzzAnyAbsGreaterThanInt32(f *testing.F) {
	f.Add([]byte{}, int32(100))
	f.Add([]byte{1, 0, 0, 0}, int32(10))
	f.Add([]byte{255, 255, 255, 127}, int32(1000))

	f.Fuzz(func(t *testing.T, data []byte, threshold int32) {
		vals := bytesToInt32s(data)
		if threshold < 0 {
			threshold = -threshold
		}
		got := AnyAbsGreaterThanInt32(vals, threshold)
		want := anyAbsGreaterThanInt32Scalar(vals, threshold)
		if got != want {
			t.Errorf("AnyAbsGreaterThanInt32(%v, %d) = %v, want %v", vals, threshold, got, want)
		}
	})
}

// Int16 fuzz tests

func FuzzSumInt16(f *testing.F) {
	f.Add([]byte{})
	f.Add([]byte{1, 0})
	f.Add([]byte{255, 127}) // max int16

	f.Fuzz(func(t *testing.T, data []byte) {
		vals := bytesToInt16s(data)
		got := SumInt16(vals)
		want := sumInt16Scalar(vals)
		if got != want {
			t.Errorf("SumInt16(%v) = %v, want %v", vals, got, want)
		}
	})
}

func FuzzMinInt16(f *testing.F) {
	f.Add([]byte{})
	f.Add([]byte{1, 0})

	f.Fuzz(func(t *testing.T, data []byte) {
		vals := bytesToInt16s(data)
		got := MinInt16(vals)
		want := minInt16Scalar(vals)
		if got != want {
			t.Errorf("MinInt16(%v) = %v, want %v", vals, got, want)
		}
	})
}

func FuzzMaxInt16(f *testing.F) {
	f.Add([]byte{})
	f.Add([]byte{1, 0})

	f.Fuzz(func(t *testing.T, data []byte) {
		vals := bytesToInt16s(data)
		got := MaxInt16(vals)
		want := maxInt16Scalar(vals)
		if got != want {
			t.Errorf("MaxInt16(%v) = %v, want %v", vals, got, want)
		}
	})
}

func FuzzDotProductInt16(f *testing.F) {
	f.Add([]byte{}, []byte{})
	f.Add([]byte{1, 0}, []byte{2, 0})

	f.Fuzz(func(t *testing.T, dataA, dataB []byte) {
		a := bytesToInt16s(dataA)
		b := bytesToInt16s(dataB)
		got := DotProductInt16(a, b)
		want := dotProductInt16Scalar(a, b)
		if got != want {
			t.Errorf("DotProductInt16 = %v, want %v", got, want)
		}
	})
}

func FuzzSumSqInt16(f *testing.F) {
	f.Add([]byte{})
	f.Add([]byte{1, 0})
	f.Add([]byte{10, 0, 20, 0})

	f.Fuzz(func(t *testing.T, data []byte) {
		vals := bytesToInt16s(data)
		got := SumSqInt16(vals)
		want := sumSqInt16Scalar(vals)
		if got != want {
			t.Errorf("SumSqInt16(%v) = %v, want %v", vals, got, want)
		}
	})
}

func FuzzAnyAbsGreaterThanInt16(f *testing.F) {
	f.Add([]byte{}, int16(100))
	f.Add([]byte{1, 0}, int16(10))
	f.Add([]byte{255, 127}, int16(1000))

	f.Fuzz(func(t *testing.T, data []byte, threshold int16) {
		vals := bytesToInt16s(data)
		if threshold < 0 {
			threshold = -threshold
		}
		got := AnyAbsGreaterThanInt16(vals, threshold)
		want := anyAbsGreaterThanInt16Scalar(vals, threshold)
		if got != want {
			t.Errorf("AnyAbsGreaterThanInt16(%v, %d) = %v, want %v", vals, threshold, got, want)
		}
	})
}

// Float32 fuzz tests

func FuzzSumFloat32(f *testing.F) {
	f.Add([]byte{})
	f.Add([]byte{0, 0, 128, 63}) // 1.0

	f.Fuzz(func(t *testing.T, data []byte) {
		vals := bytesToFloat32s(data)
		if containsNaNFloat32(vals) || hasExtremeMagnitudeSpreadFloat32(vals) {
			t.Skip("NaN/Inf values or extreme magnitude spread")
		}
		got := SumFloat32(vals)
		want := sumFloat32Scalar(vals)
		if !float32Equal(got, want) {
			t.Errorf("SumFloat32(%v) = %v, want %v", vals, got, want)
		}
	})
}

func FuzzMinFloat32(f *testing.F) {
	f.Add([]byte{})
	f.Add([]byte{0, 0, 128, 63})

	f.Fuzz(func(t *testing.T, data []byte) {
		vals := bytesToFloat32s(data)
		if containsNaNFloat32(vals) {
			t.Skip("NaN values")
		}
		got := MinFloat32(vals)
		want := minFloat32Scalar(vals)
		if got != want {
			t.Errorf("MinFloat32(%v) = %v, want %v", vals, got, want)
		}
	})
}

func FuzzMaxFloat32(f *testing.F) {
	f.Add([]byte{})
	f.Add([]byte{0, 0, 128, 63})

	f.Fuzz(func(t *testing.T, data []byte) {
		vals := bytesToFloat32s(data)
		if containsNaNFloat32(vals) {
			t.Skip("NaN values")
		}
		got := MaxFloat32(vals)
		want := maxFloat32Scalar(vals)
		if got != want {
			t.Errorf("MaxFloat32(%v) = %v, want %v", vals, got, want)
		}
	})
}

func FuzzDotProductFloat32(f *testing.F) {
	f.Add([]byte{}, []byte{})
	f.Add([]byte{0, 0, 128, 63}, []byte{0, 0, 0, 64})

	f.Fuzz(func(t *testing.T, dataA, dataB []byte) {
		a := bytesToFloat32s(dataA)
		b := bytesToFloat32s(dataB)
		if containsNaNFloat32(a) || containsNaNFloat32(b) || hasExtremeMagnitudeSpreadFloat32(a) || hasExtremeMagnitudeSpreadFloat32(b) {
			t.Skip("NaN/Inf values or extreme magnitude spread")
		}
		got := DotProductFloat32(a, b)
		want := dotProductFloat32Scalar(a, b)
		if math.IsNaN(float64(got)) || math.IsNaN(float64(want)) {
			t.Skip("Result is NaN")
		}
		if !float32Equal(got, want) {
			t.Errorf("DotProductFloat32 = %v, want %v", got, want)
		}
	})
}

// Helper functions for int32/int16/float32

func bytesToInt32s(data []byte) []int32 {
	n := len(data) / 4
	if n == 0 {
		return nil
	}
	vals := make([]int32, n)
	for i := 0; i < n; i++ {
		vals[i] = int32(data[i*4]) |
			int32(data[i*4+1])<<8 |
			int32(data[i*4+2])<<16 |
			int32(data[i*4+3])<<24
	}
	return vals
}

func bytesToInt16s(data []byte) []int16 {
	n := len(data) / 2
	if n == 0 {
		return nil
	}
	vals := make([]int16, n)
	for i := 0; i < n; i++ {
		vals[i] = int16(data[i*2]) | int16(data[i*2+1])<<8
	}
	return vals
}

func bytesToFloat32s(data []byte) []float32 {
	n := len(data) / 4
	if n == 0 {
		return nil
	}
	vals := make([]float32, n)
	for i := 0; i < n; i++ {
		bits := uint32(data[i*4]) |
			uint32(data[i*4+1])<<8 |
			uint32(data[i*4+2])<<16 |
			uint32(data[i*4+3])<<24
		vals[i] = math.Float32frombits(bits)
	}
	return vals
}

func containsNaNFloat32(vals []float32) bool {
	for _, v := range vals {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return true
		}
	}
	return false
}

func hasExtremeMagnitudeSpreadFloat32(vals []float32) bool {
	if len(vals) < 2 {
		return false
	}
	var minAbs, maxAbs float32 = math.MaxFloat32, 0
	for _, v := range vals {
		abs := float32(math.Abs(float64(v)))
		if abs > 0 && abs < minAbs {
			minAbs = abs
		}
		if abs > maxAbs {
			maxAbs = abs
		}
	}
	if minAbs == 0 || minAbs == math.MaxFloat32 {
		return false
	}
	return maxAbs/minAbs > 1e15
}

func float32Equal(a, b float32) bool {
	if a == b {
		return true
	}
	diff := math.Abs(float64(a - b))
	max := math.Max(math.Abs(float64(a)), math.Abs(float64(b)))
	if max == 0 {
		return diff < 1e-7
	}
	return diff/max < 1e-5
}

// int32sWouldOverflow checks if dot product or sum of squares would overflow int64.
func int32sWouldOverflow(a, b []int32) bool {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	var sum int64
	for i := 0; i < n; i++ {
		prod := int64(a[i]) * int64(b[i])
		// Check if adding prod would overflow
		if prod > 0 && sum > math.MaxInt64-prod {
			return true
		}
		if prod < 0 && sum < math.MinInt64-prod {
			return true
		}
		sum += prod
	}
	return false
}

// int64sWouldOverflow checks if dot product or sum of squares would overflow int64.
// For int64, the multiplication itself can overflow, so we check that too.
func int64sWouldOverflow(a, b []int64) bool {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	var sum int64
	for i := 0; i < n; i++ {
		// Check if multiplication would overflow
		if a[i] != 0 && b[i] != 0 {
			if a[i] > 0 && b[i] > 0 && a[i] > math.MaxInt64/b[i] {
				return true
			}
			if a[i] < 0 && b[i] < 0 && a[i] < math.MaxInt64/b[i] {
				return true
			}
			if a[i] > 0 && b[i] < 0 && b[i] < math.MinInt64/a[i] {
				return true
			}
			if a[i] < 0 && b[i] > 0 && a[i] < math.MinInt64/b[i] {
				return true
			}
		}
		prod := a[i] * b[i]
		// Check if adding prod would overflow
		if prod > 0 && sum > math.MaxInt64-prod {
			return true
		}
		if prod < 0 && sum < math.MinInt64-prod {
			return true
		}
		sum += prod
	}
	return false
}
