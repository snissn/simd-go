//go:build arm64

package simd

import (
	"fmt"
	"testing"
)

var benchSizes = []int{100, 1000, 10000, 100000, 1000000}

type benchCase struct {
	name      string
	scalar    func()
	neon      func()
	sve       func()
	bytesPerN int // bytes processed per element (0 = skip throughput reporting)
}

func BenchmarkSIMD(b *testing.B) {
	for _, size := range benchSizes {
		floats := makeFloatSlice(size)
		ints := makeInt64Slice(size)
		smallInts := makeSmallInt64Slice(size)

		// For AnyAbsGreaterThan edge cases:
		// - earlyExit: first element exceeds threshold (best case - immediate return)
		// - noMatch: no element exceeds threshold (worst case - full scan)
		earlyExitInts := make([]int64, size)
		earlyExitInts[0] = 1e18 // First element triggers early exit
		for i := 1; i < size; i++ {
			earlyExitInts[i] = 0
		}

		noMatchInts := make([]int64, size)
		for i := range noMatchInts {
			noMatchInts[i] = int64(i % 100) // All values < 100, threshold will be higher
		}

		cases := []benchCase{
			// Float64 operations
			{
				name:      "SumFloat64",
				scalar:    func() { sumFloat64Scalar(floats) },
				neon:      func() { sumFloat64NEON(floats) },
				sve:       func() { sumFloat64SVE(floats) },
				bytesPerN: 8,
			},
			{
				name:      "MinFloat64",
				scalar:    func() { minFloat64Scalar(floats) },
				neon:      func() { minFloat64NEON(floats) },
				sve:       func() { minFloat64SVE(floats) },
				bytesPerN: 8,
			},
			{
				name:      "MaxFloat64",
				scalar:    func() { maxFloat64Scalar(floats) },
				neon:      func() { maxFloat64NEON(floats) },
				sve:       func() { maxFloat64SVE(floats) },
				bytesPerN: 8,
			},
			{
				name:      "DotProductFloat64",
				scalar:    func() { dotProductFloat64Scalar(floats, floats) },
				neon:      func() { dotProductFloat64NEON(floats, floats) },
				sve:       func() { dotProductFloat64SVE(floats, floats) },
				bytesPerN: 16, // reads two arrays
			},

			// Int64 operations
			{
				name:      "SumInt64",
				scalar:    func() { sumInt64Scalar(ints) },
				neon:      func() { sumInt64NEON(ints) },
				sve:       func() { sumInt64SVE(ints) },
				bytesPerN: 8,
			},
			{
				name:      "MinInt64",
				scalar:    func() { minInt64Scalar(ints) },
				neon:      func() { minInt64NEON(ints) },
				sve:       func() { minInt64SVE(ints) },
				bytesPerN: 8,
			},
			{
				name:      "MaxInt64",
				scalar:    func() { maxInt64Scalar(ints) },
				neon:      func() { maxInt64NEON(ints) },
				sve:       func() { maxInt64SVE(ints) },
				bytesPerN: 8,
			},
			{
				name:      "DotProductInt64",
				scalar:    func() { dotProductInt64Scalar(ints, ints) },
				neon:      nil, // NEON lacks 64-bit MUL, dispatches to scalar
				sve:       func() { dotProductInt64SVE(ints, ints) },
				bytesPerN: 16, // reads two arrays
			},
			{
				name:      "SumSqInt64",
				scalar:    func() { sumSqInt64Scalar(smallInts) },
				neon:      func() { sumSqInt64NEON(smallInts) },
				sve:       func() { sumSqInt64SVE(smallInts) },
				bytesPerN: 8,
			},

			// AnyAbsGreaterThan: worst case (no match, full scan)
			{
				name:      "AnyAbsGreaterThan/worst",
				scalar:    func() { anyAbsGreaterThanScalar(noMatchInts, 1000) },
				neon:      func() { anyAbsGreaterThanNEON(noMatchInts, 1000) },
				sve:       func() { anyAbsGreaterThanSVE(noMatchInts, 1000) },
				bytesPerN: 8,
			},
			// AnyAbsGreaterThan: best case (early exit on first element)
			{
				name:      "AnyAbsGreaterThan/best",
				scalar:    func() { anyAbsGreaterThanScalar(earlyExitInts, 1000) },
				neon:      func() { anyAbsGreaterThanNEON(earlyExitInts, 1000) },
				sve:       func() { anyAbsGreaterThanSVE(earlyExitInts, 1000) },
				bytesPerN: 0, // throughput is meaningless for early-exit
			},
		}

		for _, bc := range cases {
			bytes := int64(size * bc.bytesPerN)

			b.Run(fmt.Sprintf("fn=%s/impl=Scalar/n=%d", bc.name, size), func(b *testing.B) {
				if bytes > 0 {
					b.SetBytes(bytes)
				}
				for b.Loop() {
					bc.scalar()
				}
			})

			if bc.neon != nil {
				b.Run(fmt.Sprintf("fn=%s/impl=NEON/n=%d", bc.name, size), func(b *testing.B) {
					if bytes > 0 {
						b.SetBytes(bytes)
					}
					for b.Loop() {
						bc.neon()
					}
				})
			}

			if HasSVE() {
				b.Run(fmt.Sprintf("fn=%s/impl=SVE/n=%d", bc.name, size), func(b *testing.B) {
					if bytes > 0 {
						b.SetBytes(bytes)
					}
					for b.Loop() {
						bc.sve()
					}
				})
			}
		}
	}
}
