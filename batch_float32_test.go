package simd

import (
	"math"
	"testing"
)

func TestDotProductFloat32Indexed(t *testing.T) {
	t.Parallel()

	dimsCases := []int{1, 3, 4, 5, 15, 16, 17, 31, 32, 33, 64, 65, 128}
	rowCases := []int{0, 1, 2, 4, 5, 8, 13}
	for _, dims := range dimsCases {
		query := makeBatchQuery(dims)
		base := makeBatchBase(32, dims, dims)
		for _, rows := range rowCases {
			rowIDs := make([]uint32, rows)
			for i := range rowIDs {
				rowIDs[i] = uint32((i*7 + 3) % 32)
			}
			dst := make([]float32, rows)
			want := make([]float32, rows)
			dotProductFloat32IndexedScalar(want, base, query, rowIDs, dims)

			optimized := DotProductFloat32Indexed(dst, base, query, rowIDs, dims)
			wantOptimized := IsARM64() && HasNEON() && dotProductFloat32IndexedUseNEON(rowIDs, dims)
			if optimized != wantOptimized {
				t.Fatalf("dims=%d rows=%d optimized=%v want %v", dims, rows, optimized, wantOptimized)
			}
			assertFloat32SlicesClose(t, dst, want, "indexed")
		}
	}
}

func TestDotProductFloat32Strided(t *testing.T) {
	t.Parallel()

	dimsCases := []int{1, 3, 4, 5, 15, 16, 17, 31, 32, 33, 64, 65, 128}
	rowCases := []int{0, 1, 2, 4, 5, 8, 13}
	for _, dims := range dimsCases {
		query := makeBatchQuery(dims)
		stride := dims + 3
		base := makeBatchBase(32, dims, stride)
		for _, rows := range rowCases {
			dst := make([]float32, rows)
			want := make([]float32, rows)
			dotProductFloat32StridedScalar(want, base, query, rows, dims, stride)

			optimized := DotProductFloat32Strided(dst, base, query, rows, dims, stride)
			wantOptimized := IsARM64() && HasNEON() && dotProductFloat32StridedUseNEON(rows, dims)
			if optimized != wantOptimized {
				t.Fatalf("dims=%d rows=%d optimized=%v want %v", dims, rows, optimized, wantOptimized)
			}
			assertFloat32SlicesClose(t, dst, want, "strided")
		}
	}
}

func TestDotProductFloat32BatchTruncatesToDst(t *testing.T) {
	t.Parallel()

	const dims = 8
	query := makeBatchQuery(dims)
	base := makeBatchBase(4, dims, dims)

	indexedDst := []float32{0, 0}
	rowIDs := []uint32{0, 1, 2, 3}
	if DotProductFloat32Indexed(indexedDst, base, query, rowIDs, dims) {
		t.Fatal("indexed with two destination rows should use scalar fallback")
	}
	indexedWant := make([]float32, len(indexedDst))
	dotProductFloat32IndexedScalar(indexedWant, base, query, rowIDs[:len(indexedDst)], dims)
	assertFloat32SlicesClose(t, indexedDst, indexedWant, "indexed truncated")

	stridedDst := []float32{0, 0}
	if DotProductFloat32Strided(stridedDst, base, query, 4, dims, dims) {
		t.Fatal("strided with two destination rows should use scalar fallback")
	}
	stridedWant := make([]float32, len(stridedDst))
	dotProductFloat32StridedScalar(stridedWant, base, query, len(stridedDst), dims, dims)
	assertFloat32SlicesClose(t, stridedDst, stridedWant, "strided truncated")
}

func TestDotProductFloat32BatchInvalidShapesLeaveDst(t *testing.T) {
	t.Parallel()

	query := makeBatchQuery(4)
	base := makeBatchBase(2, 4, 4)

	tests := []struct {
		name string
		call func([]float32) bool
	}{
		{
			name: "indexed zero dims",
			call: func(dst []float32) bool {
				return DotProductFloat32Indexed(dst, base, query, []uint32{0, 1}, 0)
			},
		},
		{
			name: "indexed short query",
			call: func(dst []float32) bool {
				return DotProductFloat32Indexed(dst, base, query[:3], []uint32{0, 1}, 4)
			},
		},
		{
			name: "indexed row outside base",
			call: func(dst []float32) bool {
				return DotProductFloat32Indexed(dst, base, query, []uint32{0, 2}, 4)
			},
		},
		{
			name: "strided short query",
			call: func(dst []float32) bool {
				return DotProductFloat32Strided(dst, base, query[:3], 2, 4, 4)
			},
		},
		{
			name: "strided stride smaller than dims",
			call: func(dst []float32) bool {
				return DotProductFloat32Strided(dst, base, query, 2, 4, 3)
			},
		},
		{
			name: "strided rows outside base",
			call: func(dst []float32) bool {
				return DotProductFloat32Strided(dst, base, query, 3, 4, 4)
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			dst := []float32{101, 202, 303}
			before := append([]float32(nil), dst...)
			if optimized := tc.call(dst); optimized {
				t.Fatal("invalid shape reported optimized")
			}
			assertFloat32SlicesClose(t, dst, before, tc.name)
		})
	}
}

func TestDotProductFloat32BatchAllocs(t *testing.T) {
	const (
		rows = 8
		dims = 128
	)
	query := makeBatchQuery(dims)
	base := makeBatchBase(32, dims, dims+5)
	rowIDs := []uint32{0, 7, 3, 9, 12, 15, 2, 6}
	dst := make([]float32, rows)

	indexedAllocs := testing.AllocsPerRun(1000, func() {
		DotProductFloat32Indexed(dst, base, query, rowIDs, dims)
	})
	if indexedAllocs != 0 {
		t.Fatalf("DotProductFloat32Indexed allocated %v times", indexedAllocs)
	}

	stridedAllocs := testing.AllocsPerRun(1000, func() {
		DotProductFloat32Strided(dst, base, query, rows, dims, dims+5)
	})
	if stridedAllocs != 0 {
		t.Fatalf("DotProductFloat32Strided allocated %v times", stridedAllocs)
	}
}

func makeBatchQuery(dims int) []float32 {
	query := make([]float32, dims)
	for i := range query {
		query[i] = float32((i%11)-5) * 0.125
	}
	return query
}

func makeBatchBase(rows, dims, stride int) []float32 {
	base := make([]float32, (rows-1)*stride+dims)
	for r := 0; r < rows; r++ {
		for c := 0; c < dims; c++ {
			base[r*stride+c] = float32(((r+1)*(c+3))%23-11) * 0.0625
		}
	}
	return base
}

func assertFloat32SlicesClose(t *testing.T, got, want []float32, label string) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s length got %d want %d", label, len(got), len(want))
	}
	for i := range got {
		if !float32Close(got[i], want[i]) {
			t.Fatalf("%s[%d] got %g want %g diff %g", label, i, got[i], want[i], got[i]-want[i])
		}
	}
}

func float32Close(a, b float32) bool {
	if a == b {
		return true
	}
	diff := math.Abs(float64(a - b))
	if diff <= 1e-6 {
		return true
	}
	maxAbs := math.Max(math.Abs(float64(a)), math.Abs(float64(b)))
	if maxAbs == 0 {
		return diff < 1e-6
	}
	return diff/maxAbs < 1e-5
}
