//go:build arm64

package simd

import (
	"fmt"
	"testing"
)

// TestSVEMatchesNEON verifies that SVE and NEON implementations produce identical results.
func TestSVEMatchesNEON(t *testing.T) {
	if !HasSVE() {
		t.Skip("SVE not supported on this CPU")
	}

	sizes := []int{0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 100, 1000}

	for _, size := range sizes {
		floats := makeFloatSlice(size)
		ints := makeInt64Slice(size)
		smallInts := makeSmallInt64Slice(size)

		t.Run(fmt.Sprintf("SumFloat64/%d", size), func(t *testing.T) {
			if got, want := sumFloat64SVE(floats), sumFloat64NEON(floats); got != want {
				t.Errorf("SVE=%v, NEON=%v", got, want)
			}
		})
		t.Run(fmt.Sprintf("MinFloat64/%d", size), func(t *testing.T) {
			if got, want := minFloat64SVE(floats), minFloat64NEON(floats); got != want {
				t.Errorf("SVE=%v, NEON=%v", got, want)
			}
		})
		t.Run(fmt.Sprintf("MaxFloat64/%d", size), func(t *testing.T) {
			if got, want := maxFloat64SVE(floats), maxFloat64NEON(floats); got != want {
				t.Errorf("SVE=%v, NEON=%v", got, want)
			}
		})
		t.Run(fmt.Sprintf("DotProductFloat64/%d", size), func(t *testing.T) {
			if got, want := dotProductFloat64SVE(floats, floats), dotProductFloat64NEON(floats, floats); got != want {
				t.Errorf("SVE=%v, NEON=%v", got, want)
			}
		})
		t.Run(fmt.Sprintf("SumInt64/%d", size), func(t *testing.T) {
			if got, want := sumInt64SVE(ints), sumInt64NEON(ints); got != want {
				t.Errorf("SVE=%v, NEON=%v", got, want)
			}
		})
		t.Run(fmt.Sprintf("MinInt64/%d", size), func(t *testing.T) {
			if got, want := minInt64SVE(ints), minInt64NEON(ints); got != want {
				t.Errorf("SVE=%v, NEON=%v", got, want)
			}
		})
		t.Run(fmt.Sprintf("MaxInt64/%d", size), func(t *testing.T) {
			if got, want := maxInt64SVE(ints), maxInt64NEON(ints); got != want {
				t.Errorf("SVE=%v, NEON=%v", got, want)
			}
		})
		t.Run(fmt.Sprintf("DotProductInt64/%d", size), func(t *testing.T) {
			if got, want := dotProductInt64SVE(ints, ints), dotProductInt64Scalar(ints, ints); got != want {
				t.Errorf("SVE=%v, Scalar=%v", got, want)
			}
		})
		t.Run(fmt.Sprintf("SumSqInt64/%d", size), func(t *testing.T) {
			if got, want := sumSqInt64SVE(smallInts), sumSqInt64NEON(smallInts); got != want {
				t.Errorf("SVE=%v, NEON=%v", got, want)
			}
		})
		t.Run(fmt.Sprintf("AnyAbsGreaterThan/%d", size), func(t *testing.T) {
			if got, want := anyAbsGreaterThanSVE(ints, 1000), anyAbsGreaterThanNEON(ints, 1000); got != want {
				t.Errorf("SVE=%v, NEON=%v", got, want)
			}
		})
	}
}
