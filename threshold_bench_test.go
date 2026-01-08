//go:build arm64

package simd

import (
	"fmt"
	"testing"
)

// Fine-grained sizes to find SIMD vs scalar crossover points
var thresholdSizes = []int{1, 2, 4, 8, 12, 16, 20, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512}

func BenchmarkThreshold(b *testing.B) {
	for _, size := range thresholdSizes {
		floats := makeFloatSlice(size)
		ints := makeInt64Slice(size)
		smallInts := makeSmallInt64Slice(size)

		// Float64 Sum
		b.Run(fmt.Sprintf("fn=SumFloat64/impl=Scalar/n=%d", size), func(b *testing.B) {
			for b.Loop() {
				sumFloat64Scalar(floats)
			}
		})
		b.Run(fmt.Sprintf("fn=SumFloat64/impl=NEON/n=%d", size), func(b *testing.B) {
			for b.Loop() {
				sumFloat64NEON(floats)
			}
		})
		if HasSVE() {
			b.Run(fmt.Sprintf("fn=SumFloat64/impl=SVE/n=%d", size), func(b *testing.B) {
				for b.Loop() {
					sumFloat64SVE(floats)
				}
			})
		}

		// Float64 Min
		b.Run(fmt.Sprintf("fn=MinFloat64/impl=Scalar/n=%d", size), func(b *testing.B) {
			for b.Loop() {
				minFloat64Scalar(floats)
			}
		})
		b.Run(fmt.Sprintf("fn=MinFloat64/impl=NEON/n=%d", size), func(b *testing.B) {
			for b.Loop() {
				minFloat64NEON(floats)
			}
		})
		if HasSVE() {
			b.Run(fmt.Sprintf("fn=MinFloat64/impl=SVE/n=%d", size), func(b *testing.B) {
				for b.Loop() {
					minFloat64SVE(floats)
				}
			})
		}

		// Float64 Max
		b.Run(fmt.Sprintf("fn=MaxFloat64/impl=Scalar/n=%d", size), func(b *testing.B) {
			for b.Loop() {
				maxFloat64Scalar(floats)
			}
		})
		b.Run(fmt.Sprintf("fn=MaxFloat64/impl=NEON/n=%d", size), func(b *testing.B) {
			for b.Loop() {
				maxFloat64NEON(floats)
			}
		})
		if HasSVE() {
			b.Run(fmt.Sprintf("fn=MaxFloat64/impl=SVE/n=%d", size), func(b *testing.B) {
				for b.Loop() {
					maxFloat64SVE(floats)
				}
			})
		}

		// Float64 DotProduct
		b.Run(fmt.Sprintf("fn=DotProductFloat64/impl=Scalar/n=%d", size), func(b *testing.B) {
			for b.Loop() {
				dotProductFloat64Scalar(floats, floats)
			}
		})
		b.Run(fmt.Sprintf("fn=DotProductFloat64/impl=NEON/n=%d", size), func(b *testing.B) {
			for b.Loop() {
				dotProductFloat64NEON(floats, floats)
			}
		})
		if HasSVE() {
			b.Run(fmt.Sprintf("fn=DotProductFloat64/impl=SVE/n=%d", size), func(b *testing.B) {
				for b.Loop() {
					dotProductFloat64SVE(floats, floats)
				}
			})
		}

		// Int64 Sum
		b.Run(fmt.Sprintf("fn=SumInt64/impl=Scalar/n=%d", size), func(b *testing.B) {
			for b.Loop() {
				sumInt64Scalar(ints)
			}
		})
		b.Run(fmt.Sprintf("fn=SumInt64/impl=NEON/n=%d", size), func(b *testing.B) {
			for b.Loop() {
				sumInt64NEON(ints)
			}
		})
		if HasSVE() {
			b.Run(fmt.Sprintf("fn=SumInt64/impl=SVE/n=%d", size), func(b *testing.B) {
				for b.Loop() {
					sumInt64SVE(ints)
				}
			})
		}

		// Int64 Min
		b.Run(fmt.Sprintf("fn=MinInt64/impl=Scalar/n=%d", size), func(b *testing.B) {
			for b.Loop() {
				minInt64Scalar(ints)
			}
		})
		b.Run(fmt.Sprintf("fn=MinInt64/impl=NEON/n=%d", size), func(b *testing.B) {
			for b.Loop() {
				minInt64NEON(ints)
			}
		})
		if HasSVE() {
			b.Run(fmt.Sprintf("fn=MinInt64/impl=SVE/n=%d", size), func(b *testing.B) {
				for b.Loop() {
					minInt64SVE(ints)
				}
			})
		}

		// Int64 Max
		b.Run(fmt.Sprintf("fn=MaxInt64/impl=Scalar/n=%d", size), func(b *testing.B) {
			for b.Loop() {
				maxInt64Scalar(ints)
			}
		})
		b.Run(fmt.Sprintf("fn=MaxInt64/impl=NEON/n=%d", size), func(b *testing.B) {
			for b.Loop() {
				maxInt64NEON(ints)
			}
		})
		if HasSVE() {
			b.Run(fmt.Sprintf("fn=MaxInt64/impl=SVE/n=%d", size), func(b *testing.B) {
				for b.Loop() {
					maxInt64SVE(ints)
				}
			})
		}

		// Int64 DotProduct (no NEON impl)
		b.Run(fmt.Sprintf("fn=DotProductInt64/impl=Scalar/n=%d", size), func(b *testing.B) {
			for b.Loop() {
				dotProductInt64Scalar(ints, ints)
			}
		})
		if HasSVE() {
			b.Run(fmt.Sprintf("fn=DotProductInt64/impl=SVE/n=%d", size), func(b *testing.B) {
				for b.Loop() {
					dotProductInt64SVE(ints, ints)
				}
			})
		}

		// Int64 SumSq
		b.Run(fmt.Sprintf("fn=SumSqInt64/impl=Scalar/n=%d", size), func(b *testing.B) {
			for b.Loop() {
				sumSqInt64Scalar(smallInts)
			}
		})
		b.Run(fmt.Sprintf("fn=SumSqInt64/impl=NEON/n=%d", size), func(b *testing.B) {
			for b.Loop() {
				sumSqInt64NEON(smallInts)
			}
		})
		if HasSVE() {
			b.Run(fmt.Sprintf("fn=SumSqInt64/impl=SVE/n=%d", size), func(b *testing.B) {
				for b.Loop() {
					sumSqInt64SVE(smallInts)
				}
			})
		}
	}
}
