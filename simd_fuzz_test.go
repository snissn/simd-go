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
		// Clamp values to avoid overflow
		for i := range a {
			if a[i] > 1000000 {
				a[i] = a[i] % 1000000
			} else if a[i] < -1000000 {
				a[i] = -((-a[i]) % 1000000)
			}
		}
		for i := range b {
			if b[i] > 1000000 {
				b[i] = b[i] % 1000000
			} else if b[i] < -1000000 {
				b[i] = -((-b[i]) % 1000000)
			}
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
		// Clamp values to avoid overflow (|v| <= 1000)
		for i := range vals {
			if vals[i] > 1000 {
				vals[i] = vals[i] % 1000
			} else if vals[i] < -1000 {
				vals[i] = -((-vals[i]) % 1000)
			}
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
