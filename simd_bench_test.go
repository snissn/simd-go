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
	sve2      func()
	bytesPerN int // bytes processed per element (0 = skip throughput reporting)
}

func BenchmarkSIMD(b *testing.B) {
	for _, size := range benchSizes {
		floats64 := makeFloatSlice(size)
		floats32 := makeFloat32Slice(size)
		ints64 := makeInt64Slice(size)
		ints32 := makeInt32Slice(size)
		ints16 := makeInt16Slice(size)
		smallInts64 := makeSmallInt64Slice(size)
		smallInts32 := makeSmallInt32Slice(size)

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

		// For int32 AnyAbsGreaterThan
		noMatchInts32 := make([]int32, size)
		for i := range noMatchInts32 {
			noMatchInts32[i] = int32(i % 100)
		}

		// For int16 AnyAbsGreaterThan
		noMatchInts16 := make([]int16, size)
		for i := range noMatchInts16 {
			noMatchInts16[i] = int16(i % 100)
		}

		cases := []benchCase{
			// Float64 operations
			{
				name:      "SumFloat64",
				scalar:    func() { sumFloat64Scalar(floats64) },
				neon:      func() { sumFloat64NEON(floats64) },
				sve:       func() { sumFloat64SVE(floats64) },
				bytesPerN: 8,
			},
			{
				name:      "MinFloat64",
				scalar:    func() { minFloat64Scalar(floats64) },
				neon:      func() { minFloat64NEON(floats64) },
				sve:       func() { minFloat64SVE(floats64) },
				bytesPerN: 8,
			},
			{
				name:      "MaxFloat64",
				scalar:    func() { maxFloat64Scalar(floats64) },
				neon:      func() { maxFloat64NEON(floats64) },
				sve:       func() { maxFloat64SVE(floats64) },
				bytesPerN: 8,
			},
			{
				name:      "DotProductFloat64",
				scalar:    func() { dotProductFloat64Scalar(floats64, floats64) },
				neon:      func() { dotProductFloat64NEON(floats64, floats64) },
				sve:       func() { dotProductFloat64SVE(floats64, floats64) },
				bytesPerN: 16, // reads two arrays
			},

			// Float32 operations
			{
				name:      "SumFloat32",
				scalar:    func() { sumFloat32Scalar(floats32) },
				neon:      func() { sumFloat32NEON(floats32) },
				sve:       func() { sumFloat32SVE(floats32) },
				bytesPerN: 4,
			},
			{
				name:      "MinFloat32",
				scalar:    func() { minFloat32Scalar(floats32) },
				neon:      func() { minFloat32NEON(floats32) },
				sve:       func() { minFloat32SVE(floats32) },
				bytesPerN: 4,
			},
			{
				name:      "MaxFloat32",
				scalar:    func() { maxFloat32Scalar(floats32) },
				neon:      func() { maxFloat32NEON(floats32) },
				sve:       func() { maxFloat32SVE(floats32) },
				bytesPerN: 4,
			},
			{
				name:      "DotProductFloat32",
				scalar:    func() { dotProductFloat32Scalar(floats32, floats32) },
				neon:      func() { dotProductFloat32NEON(floats32, floats32) },
				sve:       func() { dotProductFloat32SVE(floats32, floats32) },
				bytesPerN: 8, // reads two arrays
			},

			// Int64 operations
			{
				name:      "SumInt64",
				scalar:    func() { sumInt64Scalar(ints64) },
				neon:      func() { sumInt64NEON(ints64) },
				sve:       func() { sumInt64SVE(ints64) },
				bytesPerN: 8,
			},
			{
				name:      "MinInt64",
				scalar:    func() { minInt64Scalar(ints64) },
				neon:      func() { minInt64NEON(ints64) },
				sve:       func() { minInt64SVE(ints64) },
				bytesPerN: 8,
			},
			{
				name:      "MaxInt64",
				scalar:    func() { maxInt64Scalar(ints64) },
				neon:      func() { maxInt64NEON(ints64) },
				sve:       func() { maxInt64SVE(ints64) },
				bytesPerN: 8,
			},
			{
				name:      "DotProductInt64",
				scalar:    func() { dotProductInt64Scalar(ints64, ints64) },
				neon:      nil, // NEON lacks 64-bit MUL, dispatches to scalar
				sve:       func() { dotProductInt64SVE(ints64, ints64) },
				bytesPerN: 16, // reads two arrays
			},
			{
				name:      "SumSqInt64",
				scalar:    func() { sumSqInt64Scalar(smallInts64) },
				neon:      func() { sumSqInt64NEON(smallInts64) },
				sve:       func() { sumSqInt64SVE(smallInts64) },
				bytesPerN: 8,
			},

			// Int32 operations
			{
				name:      "SumInt32",
				scalar:    func() { sumInt32Scalar(ints32) },
				neon:      func() { sumInt32NEON(ints32) },
				sve:       func() { sumInt32SVE(ints32) },
				sve2:      func() { sumInt32SVE2(ints32) },
				bytesPerN: 4,
			},
			{
				name:      "MinInt32",
				scalar:    func() { minInt32Scalar(ints32) },
				neon:      func() { minInt32NEON(ints32) },
				sve:       func() { minInt32SVE(ints32) },
				bytesPerN: 4,
			},
			{
				name:      "MaxInt32",
				scalar:    func() { maxInt32Scalar(ints32) },
				neon:      func() { maxInt32NEON(ints32) },
				sve:       func() { maxInt32SVE(ints32) },
				bytesPerN: 4,
			},
			{
				name:      "DotProductInt32",
				scalar:    func() { dotProductInt32Scalar(ints32, ints32) },
				neon:      func() { dotProductInt32NEON(ints32, ints32) },
				sve:       func() { dotProductInt32SVE(ints32, ints32) },
				sve2:      func() { dotProductInt32SVE2(ints32, ints32) },
				bytesPerN: 8, // reads two arrays
			},
			{
				name:      "SumSqInt32",
				scalar:    func() { sumSqInt32Scalar(smallInts32) },
				neon:      func() { sumSqInt32NEON(smallInts32) },
				sve:       func() { sumSqInt32SVE(smallInts32) },
				sve2:      func() { sumSqInt32SVE2(smallInts32) },
				bytesPerN: 4,
			},
			{
				name:      "AnyAbsGreaterThanInt32/worst",
				scalar:    func() { anyAbsGreaterThanInt32Scalar(noMatchInts32, 1000) },
				neon:      func() { anyAbsGreaterThanInt32NEON(noMatchInts32, 1000) },
				sve:       func() { anyAbsGreaterThanInt32SVE(noMatchInts32, 1000) },
				bytesPerN: 4,
			},

			// Int16 operations
			{
				name:      "SumInt16",
				scalar:    func() { sumInt16Scalar(ints16) },
				neon:      func() { sumInt16NEON(ints16) },
				sve:       func() { sumInt16SVE(ints16) },
				sve2:      func() { sumInt16SVE2(ints16) },
				bytesPerN: 2,
			},
			{
				name:      "MinInt16",
				scalar:    func() { minInt16Scalar(ints16) },
				neon:      func() { minInt16NEON(ints16) },
				sve:       func() { minInt16SVE(ints16) },
				bytesPerN: 2,
			},
			{
				name:      "MaxInt16",
				scalar:    func() { maxInt16Scalar(ints16) },
				neon:      func() { maxInt16NEON(ints16) },
				sve:       func() { maxInt16SVE(ints16) },
				bytesPerN: 2,
			},
			{
				name:      "DotProductInt16",
				scalar:    func() { dotProductInt16Scalar(ints16, ints16) },
				neon:      func() { dotProductInt16NEON(ints16, ints16) },
				sve:       func() { dotProductInt16SVE(ints16, ints16) },
				sve2:      func() { dotProductInt16SVE2(ints16, ints16) },
				bytesPerN: 4, // reads two arrays
			},
			{
				name:      "SumSqInt16",
				scalar:    func() { sumSqInt16Scalar(ints16) },
				neon:      func() { sumSqInt16NEON(ints16) },
				sve:       func() { sumSqInt16SVE(ints16) },
				sve2:      func() { sumSqInt16SVE2(ints16) },
				bytesPerN: 2,
			},
			{
				name:      "AnyAbsGreaterThanInt16/worst",
				scalar:    func() { anyAbsGreaterThanInt16Scalar(noMatchInts16, 1000) },
				neon:      func() { anyAbsGreaterThanInt16NEON(noMatchInts16, 1000) },
				sve:       func() { anyAbsGreaterThanInt16SVE(noMatchInts16, 1000) },
				bytesPerN: 2,
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

			if HasSVE2() && bc.sve2 != nil {
				b.Run(fmt.Sprintf("fn=%s/impl=SVE2/n=%d", bc.name, size), func(b *testing.B) {
					if bytes > 0 {
						b.SetBytes(bytes)
					}
					for b.Loop() {
						bc.sve2()
					}
				})
			}
		}
	}
}
