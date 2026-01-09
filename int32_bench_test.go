package simd

import (
	"testing"
)

// Benchmark comparing int32 sum approaches to understand potential gains
// from native int32 SIMD vs converting to int64

func BenchmarkSumInt32ViaInt64(b *testing.B) {
	sizes := []int{100, 1000, 10000}
	for _, size := range sizes {
		data32 := make([]int32, size)
		data64 := make([]int64, size)
		for i := range data32 {
			data32[i] = int32(i % 1000)
		}

		b.Run(intToString(size), func(b *testing.B) {
			b.Run("convert+simd64", func(b *testing.B) {
				b.SetBytes(int64(size * 4))
				for i := 0; i < b.N; i++ {
					// Current approach: convert int32 to int64, then SIMD
					for j, v := range data32 {
						data64[j] = int64(v)
					}
					_ = SumInt64(data64)
				}
			})

			b.Run("scalar32", func(b *testing.B) {
				b.SetBytes(int64(size * 4))
				for i := 0; i < b.N; i++ {
					var sum int32
					for _, v := range data32 {
						sum += v
					}
					_ = sum
				}
			})

			b.Run("scalar32_unrolled8", func(b *testing.B) {
				b.SetBytes(int64(size * 4))
				for i := 0; i < b.N; i++ {
					_ = sumInt32Unrolled(data32)
				}
			})
		})
	}
}

func BenchmarkSumInt16ViaInt64(b *testing.B) {
	sizes := []int{100, 1000, 10000}
	for _, size := range sizes {
		data16 := make([]int16, size)
		data64 := make([]int64, size)
		for i := range data16 {
			data16[i] = int16(i % 1000)
		}

		b.Run(intToString(size), func(b *testing.B) {
			b.Run("convert+simd64", func(b *testing.B) {
				b.SetBytes(int64(size * 2))
				for i := 0; i < b.N; i++ {
					for j, v := range data16 {
						data64[j] = int64(v)
					}
					_ = SumInt64(data64)
				}
			})

			b.Run("scalar16", func(b *testing.B) {
				b.SetBytes(int64(size * 2))
				for i := 0; i < b.N; i++ {
					var sum int16
					for _, v := range data16 {
						sum += v
					}
					_ = sum
				}
			})

			b.Run("scalar16_to_int64", func(b *testing.B) {
				b.SetBytes(int64(size * 2))
				for i := 0; i < b.N; i++ {
					var sum int64
					for _, v := range data16 {
						sum += int64(v)
					}
					_ = sum
				}
			})
		})
	}
}

func sumInt32Unrolled(vals []int32) int64 {
	var s0, s1, s2, s3, s4, s5, s6, s7 int64
	n := len(vals)
	i := 0
	for ; i+8 <= n; i += 8 {
		s0 += int64(vals[i])
		s1 += int64(vals[i+1])
		s2 += int64(vals[i+2])
		s3 += int64(vals[i+3])
		s4 += int64(vals[i+4])
		s5 += int64(vals[i+5])
		s6 += int64(vals[i+6])
		s7 += int64(vals[i+7])
	}
	for ; i < n; i++ {
		s0 += int64(vals[i])
	}
	return s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7
}

func intToString(n int) string {
	switch n {
	case 100:
		return "100"
	case 1000:
		return "1000"
	case 10000:
		return "10000"
	default:
		return "unknown"
	}
}

// Simulate what native int32 SIMD could achieve by measuring int64 SIMD
// on same-sized data (since int32 processes 2x elements per vector)
func BenchmarkSumInt32Potential(b *testing.B) {
	sizes := []int{100, 1000, 10000}
	for _, size := range sizes {
		// For int32, we'd have size elements
		// Native SIMD processes 4 int32s per vector vs 2 int64s
		// So the SIMD portion would be ~2x faster
		
		data32 := make([]int32, size)
		data64 := make([]int64, size)
		data64half := make([]int64, size/2) // simulates 2x vectorization benefit
		
		for i := range data32 {
			data32[i] = int32(i % 1000)
		}
		for i := range data64 {
			data64[i] = int64(i % 1000)
		}
		for i := range data64half {
			data64half[i] = int64(i % 1000)
		}

		b.Run(intToString(size), func(b *testing.B) {
			b.Run("current_convert+simd64", func(b *testing.B) {
				b.SetBytes(int64(size * 4)) // reporting int32 bytes
				for i := 0; i < b.N; i++ {
					for j, v := range data32 {
						data64[j] = int64(v)
					}
					_ = SumInt64(data64)
				}
			})
			
			b.Run("potential_native_simd32", func(b *testing.B) {
				// Simulates native int32 SIMD: no conversion, 2x elements per vector
				// We use int64 SIMD on half-sized slice to simulate the 2x throughput
				b.SetBytes(int64(size * 4))
				for i := 0; i < b.N; i++ {
					_ = SumInt64(data64half)
					_ = SumInt64(data64half) // Two calls to simulate same work
				}
			})
			
			b.Run("pure_simd64_overhead", func(b *testing.B) {
				// Shows pure SIMD overhead without conversion
				b.SetBytes(int64(size * 8))
				for i := 0; i < b.N; i++ {
					_ = SumInt64(data64)
				}
			})
		})
	}
}
