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
		floats64 := makeFloatSlice(size)
		floats32 := makeFloat32Slice(size)
		ints64 := makeInt64Slice(size)
		ints32 := makeInt32Slice(size)
		ints16 := makeInt16Slice(size)
		smallInts64 := makeSmallInt64Slice(size)
		smallInts32 := makeSmallInt32Slice(size)

		// ════════════════════════════════════════════════════════════════════
		// Float64 operations
		// ════════════════════════════════════════════════════════════════════

		b.Run(fmt.Sprintf("fn=SumFloat64/impl=Scalar/n=%d", size), func(b *testing.B) {
			for b.Loop() {
				sumFloat64Scalar(floats64)
			}
		})
		b.Run(fmt.Sprintf("fn=SumFloat64/impl=NEON/n=%d", size), func(b *testing.B) {
			for b.Loop() {
				sumFloat64NEON(floats64)
			}
		})
		if HasSVE() {
			b.Run(fmt.Sprintf("fn=SumFloat64/impl=SVE/n=%d", size), func(b *testing.B) {
				for b.Loop() {
					sumFloat64SVE(floats64)
				}
			})
		}

		b.Run(fmt.Sprintf("fn=MinFloat64/impl=Scalar/n=%d", size), func(b *testing.B) {
			for b.Loop() {
				minFloat64Scalar(floats64)
			}
		})
		b.Run(fmt.Sprintf("fn=MinFloat64/impl=NEON/n=%d", size), func(b *testing.B) {
			for b.Loop() {
				minFloat64NEON(floats64)
			}
		})
		if HasSVE() {
			b.Run(fmt.Sprintf("fn=MinFloat64/impl=SVE/n=%d", size), func(b *testing.B) {
				for b.Loop() {
					minFloat64SVE(floats64)
				}
			})
		}

		b.Run(fmt.Sprintf("fn=MaxFloat64/impl=Scalar/n=%d", size), func(b *testing.B) {
			for b.Loop() {
				maxFloat64Scalar(floats64)
			}
		})
		b.Run(fmt.Sprintf("fn=MaxFloat64/impl=NEON/n=%d", size), func(b *testing.B) {
			for b.Loop() {
				maxFloat64NEON(floats64)
			}
		})
		if HasSVE() {
			b.Run(fmt.Sprintf("fn=MaxFloat64/impl=SVE/n=%d", size), func(b *testing.B) {
				for b.Loop() {
					maxFloat64SVE(floats64)
				}
			})
		}

		b.Run(fmt.Sprintf("fn=DotProductFloat64/impl=Scalar/n=%d", size), func(b *testing.B) {
			for b.Loop() {
				dotProductFloat64Scalar(floats64, floats64)
			}
		})
		b.Run(fmt.Sprintf("fn=DotProductFloat64/impl=NEON/n=%d", size), func(b *testing.B) {
			for b.Loop() {
				dotProductFloat64NEON(floats64, floats64)
			}
		})
		if HasSVE() {
			b.Run(fmt.Sprintf("fn=DotProductFloat64/impl=SVE/n=%d", size), func(b *testing.B) {
				for b.Loop() {
					dotProductFloat64SVE(floats64, floats64)
				}
			})
		}

		// ════════════════════════════════════════════════════════════════════
		// Float32 operations
		// ════════════════════════════════════════════════════════════════════

		b.Run(fmt.Sprintf("fn=SumFloat32/impl=Scalar/n=%d", size), func(b *testing.B) {
			for b.Loop() {
				sumFloat32Scalar(floats32)
			}
		})
		b.Run(fmt.Sprintf("fn=SumFloat32/impl=NEON/n=%d", size), func(b *testing.B) {
			for b.Loop() {
				sumFloat32NEON(floats32)
			}
		})
		if HasSVE() {
			b.Run(fmt.Sprintf("fn=SumFloat32/impl=SVE/n=%d", size), func(b *testing.B) {
				for b.Loop() {
					sumFloat32SVE(floats32)
				}
			})
		}

		b.Run(fmt.Sprintf("fn=MinFloat32/impl=Scalar/n=%d", size), func(b *testing.B) {
			for b.Loop() {
				minFloat32Scalar(floats32)
			}
		})
		b.Run(fmt.Sprintf("fn=MinFloat32/impl=NEON/n=%d", size), func(b *testing.B) {
			for b.Loop() {
				minFloat32NEON(floats32)
			}
		})
		if HasSVE() {
			b.Run(fmt.Sprintf("fn=MinFloat32/impl=SVE/n=%d", size), func(b *testing.B) {
				for b.Loop() {
					minFloat32SVE(floats32)
				}
			})
		}

		b.Run(fmt.Sprintf("fn=MaxFloat32/impl=Scalar/n=%d", size), func(b *testing.B) {
			for b.Loop() {
				maxFloat32Scalar(floats32)
			}
		})
		b.Run(fmt.Sprintf("fn=MaxFloat32/impl=NEON/n=%d", size), func(b *testing.B) {
			for b.Loop() {
				maxFloat32NEON(floats32)
			}
		})
		if HasSVE() {
			b.Run(fmt.Sprintf("fn=MaxFloat32/impl=SVE/n=%d", size), func(b *testing.B) {
				for b.Loop() {
					maxFloat32SVE(floats32)
				}
			})
		}

		b.Run(fmt.Sprintf("fn=DotProductFloat32/impl=Scalar/n=%d", size), func(b *testing.B) {
			for b.Loop() {
				dotProductFloat32Scalar(floats32, floats32)
			}
		})
		b.Run(fmt.Sprintf("fn=DotProductFloat32/impl=NEON/n=%d", size), func(b *testing.B) {
			for b.Loop() {
				dotProductFloat32NEON(floats32, floats32)
			}
		})
		if HasSVE() {
			b.Run(fmt.Sprintf("fn=DotProductFloat32/impl=SVE/n=%d", size), func(b *testing.B) {
				for b.Loop() {
					dotProductFloat32SVE(floats32, floats32)
				}
			})
		}

		// ════════════════════════════════════════════════════════════════════
		// Int64 operations
		// ════════════════════════════════════════════════════════════════════

		b.Run(fmt.Sprintf("fn=SumInt64/impl=Scalar/n=%d", size), func(b *testing.B) {
			for b.Loop() {
				sumInt64Scalar(ints64)
			}
		})
		b.Run(fmt.Sprintf("fn=SumInt64/impl=NEON/n=%d", size), func(b *testing.B) {
			for b.Loop() {
				sumInt64NEON(ints64)
			}
		})
		if HasSVE() {
			b.Run(fmt.Sprintf("fn=SumInt64/impl=SVE/n=%d", size), func(b *testing.B) {
				for b.Loop() {
					sumInt64SVE(ints64)
				}
			})
		}

		b.Run(fmt.Sprintf("fn=MinInt64/impl=Scalar/n=%d", size), func(b *testing.B) {
			for b.Loop() {
				minInt64Scalar(ints64)
			}
		})
		b.Run(fmt.Sprintf("fn=MinInt64/impl=NEON/n=%d", size), func(b *testing.B) {
			for b.Loop() {
				minInt64NEON(ints64)
			}
		})
		if HasSVE() {
			b.Run(fmt.Sprintf("fn=MinInt64/impl=SVE/n=%d", size), func(b *testing.B) {
				for b.Loop() {
					minInt64SVE(ints64)
				}
			})
		}

		b.Run(fmt.Sprintf("fn=MaxInt64/impl=Scalar/n=%d", size), func(b *testing.B) {
			for b.Loop() {
				maxInt64Scalar(ints64)
			}
		})
		b.Run(fmt.Sprintf("fn=MaxInt64/impl=NEON/n=%d", size), func(b *testing.B) {
			for b.Loop() {
				maxInt64NEON(ints64)
			}
		})
		if HasSVE() {
			b.Run(fmt.Sprintf("fn=MaxInt64/impl=SVE/n=%d", size), func(b *testing.B) {
				for b.Loop() {
					maxInt64SVE(ints64)
				}
			})
		}

		// Int64 DotProduct (no NEON impl - lacks 64-bit MUL)
		b.Run(fmt.Sprintf("fn=DotProductInt64/impl=Scalar/n=%d", size), func(b *testing.B) {
			for b.Loop() {
				dotProductInt64Scalar(ints64, ints64)
			}
		})
		if HasSVE() {
			b.Run(fmt.Sprintf("fn=DotProductInt64/impl=SVE/n=%d", size), func(b *testing.B) {
				for b.Loop() {
					dotProductInt64SVE(ints64, ints64)
				}
			})
		}

		b.Run(fmt.Sprintf("fn=SumSqInt64/impl=Scalar/n=%d", size), func(b *testing.B) {
			for b.Loop() {
				sumSqInt64Scalar(smallInts64)
			}
		})
		b.Run(fmt.Sprintf("fn=SumSqInt64/impl=NEON/n=%d", size), func(b *testing.B) {
			for b.Loop() {
				sumSqInt64NEON(smallInts64)
			}
		})
		if HasSVE() {
			b.Run(fmt.Sprintf("fn=SumSqInt64/impl=SVE/n=%d", size), func(b *testing.B) {
				for b.Loop() {
					sumSqInt64SVE(smallInts64)
				}
			})
		}

		b.Run(fmt.Sprintf("fn=AnyAbsGreaterThan/impl=Scalar/n=%d", size), func(b *testing.B) {
			for b.Loop() {
				anyAbsGreaterThanScalar(ints64, 1000000)
			}
		})
		b.Run(fmt.Sprintf("fn=AnyAbsGreaterThan/impl=NEON/n=%d", size), func(b *testing.B) {
			for b.Loop() {
				anyAbsGreaterThanNEON(ints64, 1000000)
			}
		})
		if HasSVE() {
			b.Run(fmt.Sprintf("fn=AnyAbsGreaterThan/impl=SVE/n=%d", size), func(b *testing.B) {
				for b.Loop() {
					anyAbsGreaterThanSVE(ints64, 1000000)
				}
			})
		}

		// ════════════════════════════════════════════════════════════════════
		// Int32 operations
		// ════════════════════════════════════════════════════════════════════

		b.Run(fmt.Sprintf("fn=SumInt32/impl=Scalar/n=%d", size), func(b *testing.B) {
			for b.Loop() {
				sumInt32Scalar(ints32)
			}
		})
		b.Run(fmt.Sprintf("fn=SumInt32/impl=NEON/n=%d", size), func(b *testing.B) {
			for b.Loop() {
				sumInt32NEON(ints32)
			}
		})
		if HasSVE() {
			b.Run(fmt.Sprintf("fn=SumInt32/impl=SVE/n=%d", size), func(b *testing.B) {
				for b.Loop() {
					sumInt32SVE(ints32)
				}
			})
		}
		if HasSVE2() {
			b.Run(fmt.Sprintf("fn=SumInt32/impl=SVE2/n=%d", size), func(b *testing.B) {
				for b.Loop() {
					sumInt32SVE2(ints32)
				}
			})
		}

		b.Run(fmt.Sprintf("fn=MinInt32/impl=Scalar/n=%d", size), func(b *testing.B) {
			for b.Loop() {
				minInt32Scalar(ints32)
			}
		})
		b.Run(fmt.Sprintf("fn=MinInt32/impl=NEON/n=%d", size), func(b *testing.B) {
			for b.Loop() {
				minInt32NEON(ints32)
			}
		})
		if HasSVE() {
			b.Run(fmt.Sprintf("fn=MinInt32/impl=SVE/n=%d", size), func(b *testing.B) {
				for b.Loop() {
					minInt32SVE(ints32)
				}
			})
		}

		b.Run(fmt.Sprintf("fn=MaxInt32/impl=Scalar/n=%d", size), func(b *testing.B) {
			for b.Loop() {
				maxInt32Scalar(ints32)
			}
		})
		b.Run(fmt.Sprintf("fn=MaxInt32/impl=NEON/n=%d", size), func(b *testing.B) {
			for b.Loop() {
				maxInt32NEON(ints32)
			}
		})
		if HasSVE() {
			b.Run(fmt.Sprintf("fn=MaxInt32/impl=SVE/n=%d", size), func(b *testing.B) {
				for b.Loop() {
					maxInt32SVE(ints32)
				}
			})
		}

		b.Run(fmt.Sprintf("fn=DotProductInt32/impl=Scalar/n=%d", size), func(b *testing.B) {
			for b.Loop() {
				dotProductInt32Scalar(ints32, ints32)
			}
		})
		b.Run(fmt.Sprintf("fn=DotProductInt32/impl=NEON/n=%d", size), func(b *testing.B) {
			for b.Loop() {
				dotProductInt32NEON(ints32, ints32)
			}
		})
		if HasSVE() {
			b.Run(fmt.Sprintf("fn=DotProductInt32/impl=SVE/n=%d", size), func(b *testing.B) {
				for b.Loop() {
					dotProductInt32SVE(ints32, ints32)
				}
			})
		}
		if HasSVE2() {
			b.Run(fmt.Sprintf("fn=DotProductInt32/impl=SVE2/n=%d", size), func(b *testing.B) {
				for b.Loop() {
					dotProductInt32SVE2(ints32, ints32)
				}
			})
		}

		b.Run(fmt.Sprintf("fn=SumSqInt32/impl=Scalar/n=%d", size), func(b *testing.B) {
			for b.Loop() {
				sumSqInt32Scalar(smallInts32)
			}
		})
		b.Run(fmt.Sprintf("fn=SumSqInt32/impl=NEON/n=%d", size), func(b *testing.B) {
			for b.Loop() {
				sumSqInt32NEON(smallInts32)
			}
		})
		if HasSVE() {
			b.Run(fmt.Sprintf("fn=SumSqInt32/impl=SVE/n=%d", size), func(b *testing.B) {
				for b.Loop() {
					sumSqInt32SVE(smallInts32)
				}
			})
		}
		if HasSVE2() {
			b.Run(fmt.Sprintf("fn=SumSqInt32/impl=SVE2/n=%d", size), func(b *testing.B) {
				for b.Loop() {
					sumSqInt32SVE2(smallInts32)
				}
			})
		}

		b.Run(fmt.Sprintf("fn=AnyAbsGreaterThanInt32/impl=Scalar/n=%d", size), func(b *testing.B) {
			for b.Loop() {
				anyAbsGreaterThanInt32Scalar(ints32, 1000000)
			}
		})
		b.Run(fmt.Sprintf("fn=AnyAbsGreaterThanInt32/impl=NEON/n=%d", size), func(b *testing.B) {
			for b.Loop() {
				anyAbsGreaterThanInt32NEON(ints32, 1000000)
			}
		})
		if HasSVE() {
			b.Run(fmt.Sprintf("fn=AnyAbsGreaterThanInt32/impl=SVE/n=%d", size), func(b *testing.B) {
				for b.Loop() {
					anyAbsGreaterThanInt32SVE(ints32, 1000000)
				}
			})
		}

		// ════════════════════════════════════════════════════════════════════
		// Int16 operations
		// ════════════════════════════════════════════════════════════════════

		b.Run(fmt.Sprintf("fn=SumInt16/impl=Scalar/n=%d", size), func(b *testing.B) {
			for b.Loop() {
				sumInt16Scalar(ints16)
			}
		})
		b.Run(fmt.Sprintf("fn=SumInt16/impl=NEON/n=%d", size), func(b *testing.B) {
			for b.Loop() {
				sumInt16NEON(ints16)
			}
		})
		if HasSVE() {
			b.Run(fmt.Sprintf("fn=SumInt16/impl=SVE/n=%d", size), func(b *testing.B) {
				for b.Loop() {
					sumInt16SVE(ints16)
				}
			})
		}

		b.Run(fmt.Sprintf("fn=MinInt16/impl=Scalar/n=%d", size), func(b *testing.B) {
			for b.Loop() {
				minInt16Scalar(ints16)
			}
		})
		b.Run(fmt.Sprintf("fn=MinInt16/impl=NEON/n=%d", size), func(b *testing.B) {
			for b.Loop() {
				minInt16NEON(ints16)
			}
		})
		if HasSVE() {
			b.Run(fmt.Sprintf("fn=MinInt16/impl=SVE/n=%d", size), func(b *testing.B) {
				for b.Loop() {
					minInt16SVE(ints16)
				}
			})
		}

		b.Run(fmt.Sprintf("fn=MaxInt16/impl=Scalar/n=%d", size), func(b *testing.B) {
			for b.Loop() {
				maxInt16Scalar(ints16)
			}
		})
		b.Run(fmt.Sprintf("fn=MaxInt16/impl=NEON/n=%d", size), func(b *testing.B) {
			for b.Loop() {
				maxInt16NEON(ints16)
			}
		})
		if HasSVE() {
			b.Run(fmt.Sprintf("fn=MaxInt16/impl=SVE/n=%d", size), func(b *testing.B) {
				for b.Loop() {
					maxInt16SVE(ints16)
				}
			})
		}

		b.Run(fmt.Sprintf("fn=DotProductInt16/impl=Scalar/n=%d", size), func(b *testing.B) {
			for b.Loop() {
				dotProductInt16Scalar(ints16, ints16)
			}
		})
		b.Run(fmt.Sprintf("fn=DotProductInt16/impl=NEON/n=%d", size), func(b *testing.B) {
			for b.Loop() {
				dotProductInt16NEON(ints16, ints16)
			}
		})
		if HasSVE() {
			b.Run(fmt.Sprintf("fn=DotProductInt16/impl=SVE/n=%d", size), func(b *testing.B) {
				for b.Loop() {
					dotProductInt16SVE(ints16, ints16)
				}
			})
		}

		b.Run(fmt.Sprintf("fn=SumSqInt16/impl=Scalar/n=%d", size), func(b *testing.B) {
			for b.Loop() {
				sumSqInt16Scalar(ints16)
			}
		})
		b.Run(fmt.Sprintf("fn=SumSqInt16/impl=NEON/n=%d", size), func(b *testing.B) {
			for b.Loop() {
				sumSqInt16NEON(ints16)
			}
		})
		if HasSVE() {
			b.Run(fmt.Sprintf("fn=SumSqInt16/impl=SVE/n=%d", size), func(b *testing.B) {
				for b.Loop() {
					sumSqInt16SVE(ints16)
				}
			})
		}

		b.Run(fmt.Sprintf("fn=AnyAbsGreaterThanInt16/impl=Scalar/n=%d", size), func(b *testing.B) {
			for b.Loop() {
				anyAbsGreaterThanInt16Scalar(ints16, 10000)
			}
		})
		b.Run(fmt.Sprintf("fn=AnyAbsGreaterThanInt16/impl=NEON/n=%d", size), func(b *testing.B) {
			for b.Loop() {
				anyAbsGreaterThanInt16NEON(ints16, 10000)
			}
		})
		if HasSVE() {
			b.Run(fmt.Sprintf("fn=AnyAbsGreaterThanInt16/impl=SVE/n=%d", size), func(b *testing.B) {
				for b.Loop() {
					anyAbsGreaterThanInt16SVE(ints16, 10000)
				}
			})
		}
	}
}
