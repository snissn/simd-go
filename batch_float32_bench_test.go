package simd

import (
	"fmt"
	"testing"
)

var (
	batchFloat32Sink float32
	batchBoolSink    bool
)

func BenchmarkDotProductFloat32Batch(b *testing.B) {
	dimsCases := []int{64, 128, 768, 2048}
	rowCases := []int{1, 2, 4, 8, 13, 16, 32, 64, 256}

	for _, dims := range dimsCases {
		query := makeBatchQuery(dims)
		for _, rows := range rowCases {
			benchmarkDotProductFloat32StridedCase(b, "contiguous", dims, rows, dims, query)
			benchmarkDotProductFloat32StridedCase(b, "fixed_stride", dims, rows, dims+13, query)
			benchmarkDotProductFloat32IndexedCase(b, "contiguous_ids", dims, rows, rows, query, false)
			benchmarkDotProductFloat32IndexedCase(b, "scattered_ids", dims, rows, rows*4+17, query, true)
		}
	}
}

func benchmarkDotProductFloat32StridedCase(b *testing.B, pattern string, dims, rows, stride int, query []float32) {
	base := makeBatchBase(rows, dims, stride)
	dst := make([]float32, rows)
	bytesPerIteration := int64(rows * dims * 8)

	b.Run(fmt.Sprintf("api=Strided/pattern=%s/impl=BatchAPI/dims=%d/rows=%d", pattern, dims, rows), func(b *testing.B) {
		b.ReportAllocs()
		b.SetBytes(bytesPerIteration)
		var optimized bool
		for i := 0; i < b.N; i++ {
			optimized = DotProductFloat32Strided(dst, base, query, rows, dims, stride)
		}
		batchBoolSink = optimized
		batchFloat32Sink = dst[rows-1]
	})

	b.Run(fmt.Sprintf("api=Strided/pattern=%s/impl=DotProductLoop/dims=%d/rows=%d", pattern, dims, rows), func(b *testing.B) {
		b.ReportAllocs()
		b.SetBytes(bytesPerIteration)
		for i := 0; i < b.N; i++ {
			for row := 0; row < rows; row++ {
				start := row * stride
				dst[row] = DotProductFloat32(base[start:start+dims], query)
			}
		}
		batchFloat32Sink = dst[rows-1]
	})

	b.Run(fmt.Sprintf("api=Strided/pattern=%s/impl=ScalarFallback/dims=%d/rows=%d", pattern, dims, rows), func(b *testing.B) {
		b.ReportAllocs()
		b.SetBytes(bytesPerIteration)
		for i := 0; i < b.N; i++ {
			dotProductFloat32StridedScalar(dst, base, query, rows, dims, stride)
		}
		batchFloat32Sink = dst[rows-1]
	})
}

func benchmarkDotProductFloat32IndexedCase(b *testing.B, pattern string, dims, rows, baseRows int, query []float32, scattered bool) {
	base := makeBatchBase(baseRows, dims, dims)
	rowIDs := make([]uint32, rows)
	for i := range rowIDs {
		if scattered {
			rowIDs[i] = uint32((i*131 + 7) % baseRows)
		} else {
			rowIDs[i] = uint32(i)
		}
	}
	dst := make([]float32, rows)
	bytesPerIteration := int64(rows * dims * 8)

	b.Run(fmt.Sprintf("api=Indexed/pattern=%s/impl=BatchAPI/dims=%d/rows=%d", pattern, dims, rows), func(b *testing.B) {
		b.ReportAllocs()
		b.SetBytes(bytesPerIteration)
		var optimized bool
		for i := 0; i < b.N; i++ {
			optimized = DotProductFloat32Indexed(dst, base, query, rowIDs, dims)
		}
		batchBoolSink = optimized
		batchFloat32Sink = dst[rows-1]
	})

	b.Run(fmt.Sprintf("api=Indexed/pattern=%s/impl=DotProductLoop/dims=%d/rows=%d", pattern, dims, rows), func(b *testing.B) {
		b.ReportAllocs()
		b.SetBytes(bytesPerIteration)
		for i := 0; i < b.N; i++ {
			for row, rowID := range rowIDs {
				start := int(rowID) * dims
				dst[row] = DotProductFloat32(base[start:start+dims], query)
			}
		}
		batchFloat32Sink = dst[rows-1]
	})

	b.Run(fmt.Sprintf("api=Indexed/pattern=%s/impl=ScalarFallback/dims=%d/rows=%d", pattern, dims, rows), func(b *testing.B) {
		b.ReportAllocs()
		b.SetBytes(bytesPerIteration)
		for i := 0; i < b.N; i++ {
			dotProductFloat32IndexedScalar(dst, base, query, rowIDs, dims)
		}
		batchFloat32Sink = dst[rows-1]
	})
}
