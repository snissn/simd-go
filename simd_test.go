package simd

import (
	"fmt"
	"math"
	"testing"
)

func TestSumFloat64(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name   string
		vals   []float64
		expect float64
	}{
		{"empty", nil, 0},
		{"single", []float64{42.5}, 42.5},
		{"multiple", []float64{1, 2, 3, 4, 5}, 15},
		{"negative", []float64{-1, -2, 3}, 0},
		{"large", make100Float64(), 4950},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := SumFloat64(tc.vals); got != tc.expect {
				t.Errorf("SumFloat64() = %v, want %v", got, tc.expect)
			}
		})
	}
}

func TestMinFloat64(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name   string
		vals   []float64
		expect float64
	}{
		{"empty", nil, 0},
		{"single", []float64{42.5}, 42.5},
		{"multiple", []float64{5, 1, 3, 2}, 1},
		{"negative", []float64{-1, -5, 3}, -5},
		{"first", []float64{1, 2, 3}, 1},
		{"last", []float64{3, 2, 1}, 1},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := MinFloat64(tc.vals); got != tc.expect {
				t.Errorf("MinFloat64() = %v, want %v", got, tc.expect)
			}
		})
	}
}

func TestMaxFloat64(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name   string
		vals   []float64
		expect float64
	}{
		{"empty", nil, 0},
		{"single", []float64{42.5}, 42.5},
		{"multiple", []float64{5, 1, 3, 2}, 5},
		{"negative", []float64{-1, -5, 3}, 3},
		{"first", []float64{3, 2, 1}, 3},
		{"last", []float64{1, 2, 3}, 3},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := MaxFloat64(tc.vals); got != tc.expect {
				t.Errorf("MaxFloat64() = %v, want %v", got, tc.expect)
			}
		})
	}
}

func TestDotProductFloat64(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name   string
		a, b   []float64
		expect float64
	}{
		{"empty", nil, nil, 0},
		{"single", []float64{2}, []float64{3}, 6},
		{"multiple", []float64{1, 2, 3}, []float64{4, 5, 6}, 32},
		{"unequal_len", []float64{1, 2, 3, 4}, []float64{1, 1}, 3},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := DotProductFloat64(tc.a, tc.b); got != tc.expect {
				t.Errorf("DotProductFloat64() = %v, want %v", got, tc.expect)
			}
		})
	}
}

func TestSumInt64(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name   string
		vals   []int64
		expect int64
	}{
		{"empty", nil, 0},
		{"single", []int64{42}, 42},
		{"multiple", []int64{1, 2, 3, 4, 5}, 15},
		{"negative", []int64{-1, -2, 3}, 0},
		{"large", make100Int64(), 4950},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := SumInt64(tc.vals); got != tc.expect {
				t.Errorf("SumInt64() = %v, want %v", got, tc.expect)
			}
		})
	}
}

func TestMinInt64(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name   string
		vals   []int64
		expect int64
	}{
		{"empty", nil, 0},
		{"single", []int64{42}, 42},
		{"multiple", []int64{5, 1, 3, 2}, 1},
		{"negative", []int64{-1, -5, 3}, -5},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := MinInt64(tc.vals); got != tc.expect {
				t.Errorf("MinInt64() = %v, want %v", got, tc.expect)
			}
		})
	}
}

func TestMaxInt64(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name   string
		vals   []int64
		expect int64
	}{
		{"empty", nil, 0},
		{"single", []int64{42}, 42},
		{"multiple", []int64{5, 1, 3, 2}, 5},
		{"negative", []int64{-1, -5, 3}, 3},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := MaxInt64(tc.vals); got != tc.expect {
				t.Errorf("MaxInt64() = %v, want %v", got, tc.expect)
			}
		})
	}
}

func TestDotProductInt64(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name   string
		a, b   []int64
		expect int64
	}{
		{"empty", nil, nil, 0},
		{"single", []int64{2}, []int64{3}, 6},
		{"multiple", []int64{1, 2, 3}, []int64{4, 5, 6}, 32},
		{"unequal_len", []int64{1, 2, 3, 4}, []int64{1, 1}, 3},
		{"negative", []int64{-1, 2, -3}, []int64{4, -5, 6}, -32},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := DotProductInt64(tc.a, tc.b); got != tc.expect {
				t.Errorf("DotProductInt64() = %v, want %v", got, tc.expect)
			}
		})
	}
}

func TestSumSqInt64(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name   string
		vals   []int64
		expect int64
	}{
		{"empty", nil, 0},
		{"single", []int64{3}, 9},
		{"multiple", []int64{1, 2, 3}, 14},
		{"negative", []int64{-2, 3}, 13},
		{"zeros", []int64{0, 0, 0}, 0},
		{"large_safe", []int64{1000, 2000, 3000}, 14000000},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := SumSqInt64(tc.vals); got != tc.expect {
				t.Errorf("SumSqInt64() = %v, want %v", got, tc.expect)
			}
		})
	}
}

func TestAnyAbsGreaterThan(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name      string
		vals      []int64
		threshold int64
		expect    bool
	}{
		{"empty", nil, 100, false},
		{"none_above", []int64{1, 2, 3}, 10, false},
		{"one_above", []int64{1, 20, 3}, 10, true},
		{"negative_above", []int64{1, -20, 3}, 10, true},
		{"at_threshold", []int64{10, 5, 3}, 10, false},
		{"just_above", []int64{10, 11, 3}, 10, true},
		{"all_above", []int64{100, 200, 300}, 10, true},
		{"large_threshold", []int64{1, 2, 3}, 3037000499, false},
		{"at_sqrt_max", []int64{3037000499}, 3037000499, false},
		{"above_sqrt_max", []int64{3037000500}, 3037000499, true},
		{"min_int64", []int64{-9223372036854775808}, 9223372036854775807, true},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := AnyAbsGreaterThan(tc.vals, tc.threshold); got != tc.expect {
				t.Errorf("AnyAbsGreaterThan() = %v, want %v", got, tc.expect)
			}
		})
	}
}

// TestMatchesScalar verifies SIMD matches scalar for various sizes (loop tail edge cases).
func TestMatchesScalar(t *testing.T) {
	t.Parallel()
	sizes := []int{0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 100, 1000}

	for _, size := range sizes {
		floats := makeFloatSlice(size)
		ints := makeInt64Slice(size)
		smallInts := makeSmallInt64Slice(size)
		int32s := makeInt32Slice(size)
		smallInt32s := makeSmallInt32Slice(size)
		int16s := makeInt16Slice(size)

		t.Run(fmt.Sprintf("SumFloat64/%d", size), func(t *testing.T) {
			if got, want := SumFloat64(floats), sumFloat64Scalar(floats); got != want {
				t.Errorf("got %v, want %v", got, want)
			}
		})
		t.Run(fmt.Sprintf("MinFloat64/%d", size), func(t *testing.T) {
			if got, want := MinFloat64(floats), minFloat64Scalar(floats); got != want {
				t.Errorf("got %v, want %v", got, want)
			}
		})
		t.Run(fmt.Sprintf("MaxFloat64/%d", size), func(t *testing.T) {
			if got, want := MaxFloat64(floats), maxFloat64Scalar(floats); got != want {
				t.Errorf("got %v, want %v", got, want)
			}
		})
		t.Run(fmt.Sprintf("DotProductFloat64/%d", size), func(t *testing.T) {
			got := DotProductFloat64(floats, floats)
			want := dotProductFloat64Scalar(floats, floats)
			if diff := math.Abs(got - want); diff > math.Abs(want)*1e-10 {
				t.Errorf("got %v, want %v (diff %v)", got, want, diff)
			}
		})
		t.Run(fmt.Sprintf("SumInt64/%d", size), func(t *testing.T) {
			if got, want := SumInt64(ints), sumInt64Scalar(ints); got != want {
				t.Errorf("got %v, want %v", got, want)
			}
		})
		t.Run(fmt.Sprintf("MinInt64/%d", size), func(t *testing.T) {
			if got, want := MinInt64(ints), minInt64Scalar(ints); got != want {
				t.Errorf("got %v, want %v", got, want)
			}
		})
		t.Run(fmt.Sprintf("MaxInt64/%d", size), func(t *testing.T) {
			if got, want := MaxInt64(ints), maxInt64Scalar(ints); got != want {
				t.Errorf("got %v, want %v", got, want)
			}
		})
		t.Run(fmt.Sprintf("DotProductInt64/%d", size), func(t *testing.T) {
			if got, want := DotProductInt64(ints, ints), dotProductInt64Scalar(ints, ints); got != want {
				t.Errorf("got %v, want %v", got, want)
			}
		})
		t.Run(fmt.Sprintf("SumSqInt64/%d", size), func(t *testing.T) {
			if got, want := SumSqInt64(smallInts), sumSqInt64Scalar(smallInts); got != want {
				t.Errorf("got %v, want %v", got, want)
			}
		})
		t.Run(fmt.Sprintf("AnyAbsGreaterThan/%d", size), func(t *testing.T) {
			if got, want := AnyAbsGreaterThan(ints, 1000), anyAbsGreaterThanScalar(ints, 1000); got != want {
				t.Errorf("got %v, want %v", got, want)
			}
		})
		// Int32 tests
		t.Run(fmt.Sprintf("SumInt32/%d", size), func(t *testing.T) {
			if got, want := SumInt32(int32s), sumInt32Scalar(int32s); got != want {
				t.Errorf("got %v, want %v", got, want)
			}
		})
		t.Run(fmt.Sprintf("MinInt32/%d", size), func(t *testing.T) {
			if got, want := MinInt32(int32s), minInt32Scalar(int32s); got != want {
				t.Errorf("got %v, want %v", got, want)
			}
		})
		t.Run(fmt.Sprintf("MaxInt32/%d", size), func(t *testing.T) {
			if got, want := MaxInt32(int32s), maxInt32Scalar(int32s); got != want {
				t.Errorf("got %v, want %v", got, want)
			}
		})
		t.Run(fmt.Sprintf("DotProductInt32/%d", size), func(t *testing.T) {
			if got, want := DotProductInt32(int32s, int32s), dotProductInt32Scalar(int32s, int32s); got != want {
				t.Errorf("got %v, want %v", got, want)
			}
		})
		t.Run(fmt.Sprintf("SumSqInt32/%d", size), func(t *testing.T) {
			if got, want := SumSqInt32(smallInt32s), sumSqInt32Scalar(smallInt32s); got != want {
				t.Errorf("got %v, want %v", got, want)
			}
		})
		// Float32 tests
		float32s := makeFloat32Slice(size)
		t.Run(fmt.Sprintf("SumFloat32/%d", size), func(t *testing.T) {
			if got, want := SumFloat32(float32s), sumFloat32Scalar(float32s); got != want {
				t.Errorf("got %v, want %v", got, want)
			}
		})
		t.Run(fmt.Sprintf("MinFloat32/%d", size), func(t *testing.T) {
			if got, want := MinFloat32(float32s), minFloat32Scalar(float32s); got != want {
				t.Errorf("got %v, want %v", got, want)
			}
		})
		t.Run(fmt.Sprintf("MaxFloat32/%d", size), func(t *testing.T) {
			if got, want := MaxFloat32(float32s), maxFloat32Scalar(float32s); got != want {
				t.Errorf("got %v, want %v", got, want)
			}
		})
		t.Run(fmt.Sprintf("DotProductFloat32/%d", size), func(t *testing.T) {
			got := DotProductFloat32(float32s, float32s)
			want := dotProductFloat32Scalar(float32s, float32s)
			if diff := float64(got) - float64(want); diff > float64(want)*1e-5 || diff < -float64(want)*1e-5 {
				t.Errorf("got %v, want %v (diff %v)", got, want, diff)
			}
		})
		// Int16 tests
		t.Run(fmt.Sprintf("SumInt16/%d", size), func(t *testing.T) {
			if got, want := SumInt16(int16s), sumInt16Scalar(int16s); got != want {
				t.Errorf("got %v, want %v", got, want)
			}
		})
		t.Run(fmt.Sprintf("MinInt16/%d", size), func(t *testing.T) {
			if got, want := MinInt16(int16s), minInt16Scalar(int16s); got != want {
				t.Errorf("got %v, want %v", got, want)
			}
		})
		t.Run(fmt.Sprintf("MaxInt16/%d", size), func(t *testing.T) {
			if got, want := MaxInt16(int16s), maxInt16Scalar(int16s); got != want {
				t.Errorf("got %v, want %v", got, want)
			}
		})
		t.Run(fmt.Sprintf("DotProductInt16/%d", size), func(t *testing.T) {
			if got, want := DotProductInt16(int16s, int16s), dotProductInt16Scalar(int16s, int16s); got != want {
				t.Errorf("got %v, want %v", got, want)
			}
		})
	}
}

// Helper functions for test data generation.

func make100Float64() []float64 {
	vals := make([]float64, 100)
	for i := range vals {
		vals[i] = float64(i)
	}
	return vals
}

func make100Int64() []int64 {
	vals := make([]int64, 100)
	for i := range vals {
		vals[i] = int64(i)
	}
	return vals
}

func makeFloatSlice(n int) []float64 {
	vals := make([]float64, n)
	for i := range vals {
		vals[i] = float64(i) - float64(n)/2
	}
	// Shuffle to avoid branch prediction artifacts in scalar min/max benchmarks
	shuffleFloat64(vals)
	return vals
}

func shuffleFloat64(vals []float64) {
	// Use deterministic seed for reproducible benchmarks
	for i := len(vals) - 1; i > 0; i-- {
		j := int(uint64(i) * 2654435761 % uint64(i+1)) // Knuth multiplicative hash
		vals[i], vals[j] = vals[j], vals[i]
	}
}

func makeInt64Slice(n int) []int64 {
	vals := make([]int64, n)
	for i := range vals {
		vals[i] = int64(i) - int64(n)/2
	}
	shuffleInt64(vals)
	return vals
}

func shuffleInt64(vals []int64) {
	for i := len(vals) - 1; i > 0; i-- {
		j := int(uint64(i) * 2654435761 % uint64(i+1))
		vals[i], vals[j] = vals[j], vals[i]
	}
}

func makeSmallInt64Slice(n int) []int64 {
	vals := make([]int64, n)
	for i := range vals {
		vals[i] = int64(i%100) - 50
	}
	shuffleInt64(vals)
	return vals
}

// Int32 tests

func TestSumInt32(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name   string
		vals   []int32
		expect int64
	}{
		{"empty", nil, 0},
		{"single", []int32{42}, 42},
		{"multiple", []int32{1, 2, 3, 4, 5}, 15},
		{"negative", []int32{-1, -2, 3}, 0},
		{"large", make100Int32(), 4950},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := SumInt32(tc.vals); got != tc.expect {
				t.Errorf("SumInt32() = %v, want %v", got, tc.expect)
			}
		})
	}
}

func TestMinInt32(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name   string
		vals   []int32
		expect int32
	}{
		{"empty", nil, 0},
		{"single", []int32{42}, 42},
		{"multiple", []int32{5, 1, 3, 2}, 1},
		{"negative", []int32{-1, -5, 3}, -5},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := MinInt32(tc.vals); got != tc.expect {
				t.Errorf("MinInt32() = %v, want %v", got, tc.expect)
			}
		})
	}
}

func TestMaxInt32(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name   string
		vals   []int32
		expect int32
	}{
		{"empty", nil, 0},
		{"single", []int32{42}, 42},
		{"multiple", []int32{5, 1, 3, 2}, 5},
		{"negative", []int32{-1, -5, 3}, 3},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := MaxInt32(tc.vals); got != tc.expect {
				t.Errorf("MaxInt32() = %v, want %v", got, tc.expect)
			}
		})
	}
}

func TestDotProductInt32(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name   string
		a, b   []int32
		expect int64
	}{
		{"empty", nil, nil, 0},
		{"single", []int32{2}, []int32{3}, 6},
		{"multiple", []int32{1, 2, 3}, []int32{4, 5, 6}, 32},
		{"unequal_len", []int32{1, 2, 3, 4}, []int32{1, 1}, 3},
		{"negative", []int32{-1, 2, -3}, []int32{4, -5, 6}, -32},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := DotProductInt32(tc.a, tc.b); got != tc.expect {
				t.Errorf("DotProductInt32() = %v, want %v", got, tc.expect)
			}
		})
	}
}

func TestSumSqInt32(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name   string
		vals   []int32
		expect int64
	}{
		{"empty", nil, 0},
		{"single", []int32{3}, 9},
		{"multiple", []int32{1, 2, 3}, 14},
		{"negative", []int32{-2, 3}, 13},
		{"zeros", []int32{0, 0, 0}, 0},
		{"large_safe", []int32{1000, 2000, 3000}, 14000000},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := SumSqInt32(tc.vals); got != tc.expect {
				t.Errorf("SumSqInt32() = %v, want %v", got, tc.expect)
			}
		})
	}
}

func make100Int32() []int32 {
	vals := make([]int32, 100)
	for i := range vals {
		vals[i] = int32(i)
	}
	return vals
}

func makeInt32Slice(n int) []int32 {
	vals := make([]int32, n)
	for i := range vals {
		vals[i] = int32(i) - int32(n)/2
	}
	shuffleInt32(vals)
	return vals
}

func makeSmallInt32Slice(n int) []int32 {
	vals := make([]int32, n)
	for i := range vals {
		vals[i] = int32(i%100) - 50
	}
	shuffleInt32(vals)
	return vals
}

func shuffleInt32(vals []int32) {
	for i := len(vals) - 1; i > 0; i-- {
		j := int(uint64(i) * 2654435761 % uint64(i+1))
		vals[i], vals[j] = vals[j], vals[i]
	}
}

// Int16 tests

func TestSumInt16(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name   string
		vals   []int16
		expect int64
	}{
		{"empty", nil, 0},
		{"single", []int16{42}, 42},
		{"multiple", []int16{1, 2, 3, 4, 5}, 15},
		{"negative", []int16{-1, -2, 3}, 0},
		{"large", make100Int16(), 4950},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := SumInt16(tc.vals); got != tc.expect {
				t.Errorf("SumInt16() = %v, want %v", got, tc.expect)
			}
		})
	}
}

func TestMinInt16(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name   string
		vals   []int16
		expect int16
	}{
		{"empty", nil, 0},
		{"single", []int16{42}, 42},
		{"multiple", []int16{5, 1, 3, 2}, 1},
		{"negative", []int16{-1, -5, 3}, -5},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := MinInt16(tc.vals); got != tc.expect {
				t.Errorf("MinInt16() = %v, want %v", got, tc.expect)
			}
		})
	}
}

func TestMaxInt16(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name   string
		vals   []int16
		expect int16
	}{
		{"empty", nil, 0},
		{"single", []int16{42}, 42},
		{"multiple", []int16{5, 1, 3, 2}, 5},
		{"negative", []int16{-1, -5, 3}, 3},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := MaxInt16(tc.vals); got != tc.expect {
				t.Errorf("MaxInt16() = %v, want %v", got, tc.expect)
			}
		})
	}
}

func TestDotProductInt16(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name   string
		a, b   []int16
		expect int64
	}{
		{"empty", nil, nil, 0},
		{"single", []int16{2}, []int16{3}, 6},
		{"multiple", []int16{1, 2, 3}, []int16{4, 5, 6}, 32},
		{"unequal_len", []int16{1, 2, 3, 4}, []int16{1, 1}, 3},
		{"negative", []int16{-1, 2, -3}, []int16{4, -5, 6}, -32},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := DotProductInt16(tc.a, tc.b); got != tc.expect {
				t.Errorf("DotProductInt16() = %v, want %v", got, tc.expect)
			}
		})
	}
}

func make100Int16() []int16 {
	vals := make([]int16, 100)
	for i := range vals {
		vals[i] = int16(i)
	}
	return vals
}

func makeInt16Slice(n int) []int16 {
	vals := make([]int16, n)
	for i := range vals {
		vals[i] = int16(i) - int16(n)/2
	}
	shuffleInt16(vals)
	return vals
}

func shuffleInt16(vals []int16) {
	for i := len(vals) - 1; i > 0; i-- {
		j := int(uint64(i) * 2654435761 % uint64(i+1))
		vals[i], vals[j] = vals[j], vals[i]
	}
}

// Float32 tests

func TestSumFloat32(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name   string
		vals   []float32
		expect float32
	}{
		{"empty", nil, 0},
		{"single", []float32{42.5}, 42.5},
		{"multiple", []float32{1, 2, 3, 4, 5}, 15},
		{"negative", []float32{-1, -2, 3}, 0},
		{"large", make100Float32(), 4950},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := SumFloat32(tc.vals); got != tc.expect {
				t.Errorf("SumFloat32() = %v, want %v", got, tc.expect)
			}
		})
	}
}

func TestMinFloat32(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name   string
		vals   []float32
		expect float32
	}{
		{"empty", nil, 0},
		{"single", []float32{42.5}, 42.5},
		{"multiple", []float32{5, 1, 3, 2}, 1},
		{"negative", []float32{-1, -5, 3}, -5},
		{"first", []float32{1, 2, 3}, 1},
		{"last", []float32{3, 2, 1}, 1},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := MinFloat32(tc.vals); got != tc.expect {
				t.Errorf("MinFloat32() = %v, want %v", got, tc.expect)
			}
		})
	}
}

func TestMaxFloat32(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name   string
		vals   []float32
		expect float32
	}{
		{"empty", nil, 0},
		{"single", []float32{42.5}, 42.5},
		{"multiple", []float32{5, 1, 3, 2}, 5},
		{"negative", []float32{-1, -5, 3}, 3},
		{"first", []float32{3, 2, 1}, 3},
		{"last", []float32{1, 2, 3}, 3},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := MaxFloat32(tc.vals); got != tc.expect {
				t.Errorf("MaxFloat32() = %v, want %v", got, tc.expect)
			}
		})
	}
}

func TestDotProductFloat32(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name   string
		a, b   []float32
		expect float32
	}{
		{"empty", nil, nil, 0},
		{"single", []float32{2}, []float32{3}, 6},
		{"multiple", []float32{1, 2, 3}, []float32{4, 5, 6}, 32},
		{"unequal_len", []float32{1, 2, 3, 4}, []float32{1, 1}, 3},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := DotProductFloat32(tc.a, tc.b); got != tc.expect {
				t.Errorf("DotProductFloat32() = %v, want %v", got, tc.expect)
			}
		})
	}
}

func make100Float32() []float32 {
	vals := make([]float32, 100)
	for i := range vals {
		vals[i] = float32(i)
	}
	return vals
}

func makeFloat32Slice(n int) []float32 {
	vals := make([]float32, n)
	for i := range vals {
		vals[i] = float32(i) - float32(n)/2
	}
	shuffleFloat32(vals)
	return vals
}

func shuffleFloat32(vals []float32) {
	for i := len(vals) - 1; i > 0; i-- {
		j := int(uint64(i) * 2654435761 % uint64(i+1))
		vals[i], vals[j] = vals[j], vals[i]
	}
}
