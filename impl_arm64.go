//go:build arm64

package simd

// Dispatch functions select scalar, SVE, or NEON based on CPU capabilities and input size.
// Small arrays use scalar to avoid SIMD call overhead.

func sumFloat64Impl(vals []float64) float64 {
	n := len(vals)
	if hasSVE && n >= sveThresholds.SumFloat64 {
		return sumFloat64SVE(vals)
	}
	if n >= neonThresholds.SumFloat64 {
		return sumFloat64NEON(vals)
	}
	return sumFloat64Scalar(vals)
}

func minFloat64Impl(vals []float64) float64 {
	n := len(vals)
	if hasSVE && n >= sveThresholds.MinFloat64 {
		return minFloat64SVE(vals)
	}
	if n >= neonThresholds.MinFloat64 {
		return minFloat64NEON(vals)
	}
	return minFloat64Scalar(vals)
}

func maxFloat64Impl(vals []float64) float64 {
	n := len(vals)
	if hasSVE && n >= sveThresholds.MaxFloat64 {
		return maxFloat64SVE(vals)
	}
	if n >= neonThresholds.MaxFloat64 {
		return maxFloat64NEON(vals)
	}
	return maxFloat64Scalar(vals)
}

func dotProductFloat64Impl(a, b []float64) float64 {
	n := len(a)
	if hasSVE && n >= sveThresholds.DotProductFloat64 {
		return dotProductFloat64SVE(a, b)
	}
	if n >= neonThresholds.DotProductFloat64 {
		return dotProductFloat64NEON(a, b)
	}
	return dotProductFloat64Scalar(a, b)
}

func sumInt64Impl(vals []int64) int64 {
	n := len(vals)
	if hasSVE && n >= sveThresholds.SumInt64 {
		return sumInt64SVE(vals)
	}
	if n >= neonThresholds.SumInt64 {
		return sumInt64NEON(vals)
	}
	return sumInt64Scalar(vals)
}

func minInt64Impl(vals []int64) int64 {
	n := len(vals)
	if hasSVE && n >= sveThresholds.MinInt64 {
		return minInt64SVE(vals)
	}
	if n >= neonThresholds.MinInt64 {
		return minInt64NEON(vals)
	}
	return minInt64Scalar(vals)
}

func maxInt64Impl(vals []int64) int64 {
	n := len(vals)
	if hasSVE && n >= sveThresholds.MaxInt64 {
		return maxInt64SVE(vals)
	}
	if n >= neonThresholds.MaxInt64 {
		return maxInt64NEON(vals)
	}
	return maxInt64Scalar(vals)
}

func dotProductInt64Impl(a, b []int64) int64 {
	n := len(a)
	// No NEON impl for int64 dot product (lacks 64-bit MUL)
	if hasSVE && n >= sveThresholds.DotProductInt64 {
		return dotProductInt64SVE(a, b)
	}
	return dotProductInt64Scalar(a, b)
}

func sumSqInt64Impl(vals []int64) int64 {
	n := len(vals)
	if hasSVE && n >= sveThresholds.SumSqInt64 {
		return sumSqInt64SVE(vals)
	}
	if n >= neonThresholds.SumSqInt64 {
		return sumSqInt64NEON(vals)
	}
	return sumSqInt64Scalar(vals)
}

func anyAbsGreaterThanImpl(vals []int64, threshold int64) bool {
	n := len(vals)
	if hasSVE && n >= sveThresholds.AnyAbsGreaterThan {
		return anyAbsGreaterThanSVE(vals, threshold)
	}
	if n >= neonThresholds.AnyAbsGreaterThan {
		return anyAbsGreaterThanNEON(vals, threshold)
	}
	return anyAbsGreaterThanScalar(vals, threshold)
}

// NEON assembly implementations (in float64_arm64_neon.s and int64_arm64_neon.s)
func sumFloat64NEON(vals []float64) float64
func minFloat64NEON(vals []float64) float64
func maxFloat64NEON(vals []float64) float64
func dotProductFloat64NEON(a, b []float64) float64
func sumInt64NEON(vals []int64) int64
func minInt64NEON(vals []int64) int64
func maxInt64NEON(vals []int64) int64
func sumSqInt64NEON(vals []int64) int64
func anyAbsGreaterThanNEON(vals []int64, threshold int64) bool

// SVE assembly implementations (in float64_arm64_sve.s and int64_arm64_sve.s)
func sumFloat64SVE(vals []float64) float64
func minFloat64SVE(vals []float64) float64
func maxFloat64SVE(vals []float64) float64
func dotProductFloat64SVE(a, b []float64) float64
func sumInt64SVE(vals []int64) int64
func minInt64SVE(vals []int64) int64
func maxInt64SVE(vals []int64) int64
func dotProductInt64SVE(a, b []int64) int64
func sumSqInt64SVE(vals []int64) int64
func anyAbsGreaterThanSVE(vals []int64, threshold int64) bool

// Int32 dispatch functions

func sumInt32Impl(vals []int32) int64 {
	n := len(vals)
	// SVE2 is ~10% faster than NEON for int32 sum
	if hasSVE2 && n >= sveThresholds.SumInt32 {
		return sumInt32SVE2(vals)
	}
	if n >= neonThresholds.SumInt32 {
		return sumInt32NEON(vals)
	}
	return sumInt32Scalar(vals)
}

func minInt32Impl(vals []int32) int32 {
	n := len(vals)
	if hasSVE && n >= sveThresholds.MinInt32 {
		return minInt32SVE(vals)
	}
	if n >= neonThresholds.MinInt32 {
		return minInt32NEON(vals)
	}
	return minInt32Scalar(vals)
}

func maxInt32Impl(vals []int32) int32 {
	n := len(vals)
	if hasSVE && n >= sveThresholds.MaxInt32 {
		return maxInt32SVE(vals)
	}
	if n >= neonThresholds.MaxInt32 {
		return maxInt32NEON(vals)
	}
	return maxInt32Scalar(vals)
}

func dotProductInt32Impl(a, b []int32) int64 {
	n := len(a)
	if hasSVE && n >= sveThresholds.DotProductInt32 {
		return dotProductInt32SVE(a, b)
	}
	if n >= neonThresholds.DotProductInt32 {
		return dotProductInt32NEON(a, b)
	}
	return dotProductInt32Scalar(a, b)
}

func sumSqInt32Impl(vals []int32) int64 {
	n := len(vals)
	if hasSVE && n >= sveThresholds.SumSqInt32 {
		return sumSqInt32SVE(vals)
	}
	if n >= neonThresholds.SumSqInt32 {
		return sumSqInt32NEON(vals)
	}
	return sumSqInt32Scalar(vals)
}

// NEON int32 assembly implementations (in int32_arm64_neon.s)
func sumInt32NEON(vals []int32) int64
func minInt32NEON(vals []int32) int32
func maxInt32NEON(vals []int32) int32
func dotProductInt32NEON(a, b []int32) int64
func sumSqInt32NEON(vals []int32) int64

// SVE int32 assembly implementations (in int32_arm64_sve.s)
func sumInt32SVE(vals []int32) int64
func minInt32SVE(vals []int32) int32
func maxInt32SVE(vals []int32) int32
func dotProductInt32SVE(a, b []int32) int64
func sumSqInt32SVE(vals []int32) int64
func anyAbsGreaterThanInt32SVE(vals []int32, threshold int32) bool

// Int32 anyAbsGreaterThan dispatch function

func anyAbsGreaterThanInt32Impl(vals []int32, threshold int32) bool {
	n := len(vals)
	if hasSVE && n >= sveThresholds.AnyAbsGreaterThanInt32 {
		return anyAbsGreaterThanInt32SVE(vals, threshold)
	}
	if n >= neonThresholds.AnyAbsGreaterThanInt32 {
		return anyAbsGreaterThanInt32NEON(vals, threshold)
	}
	return anyAbsGreaterThanInt32Scalar(vals, threshold)
}

// NEON int32 anyAbsGreaterThan assembly implementation (in int32_arm64_neon.s)
func anyAbsGreaterThanInt32NEON(vals []int32, threshold int32) bool

// SVE2 int32 assembly implementations (in int32_arm64_sve2.s)
func sumInt32SVE2(vals []int32) int64
func dotProductInt32SVE2(a, b []int32) int64
func sumSqInt32SVE2(vals []int32) int64

// Float32 dispatch functions

func sumFloat32Impl(vals []float32) float32 {
	n := len(vals)
	if hasSVE && n >= sveThresholds.SumFloat32 {
		return sumFloat32SVE(vals)
	}
	if n >= neonThresholds.SumFloat32 {
		return sumFloat32NEON(vals)
	}
	return sumFloat32Scalar(vals)
}

func minFloat32Impl(vals []float32) float32 {
	n := len(vals)
	if hasSVE && n >= sveThresholds.MinFloat32 {
		return minFloat32SVE(vals)
	}
	if n >= neonThresholds.MinFloat32 {
		return minFloat32NEON(vals)
	}
	return minFloat32Scalar(vals)
}

func maxFloat32Impl(vals []float32) float32 {
	n := len(vals)
	if hasSVE && n >= sveThresholds.MaxFloat32 {
		return maxFloat32SVE(vals)
	}
	if n >= neonThresholds.MaxFloat32 {
		return maxFloat32NEON(vals)
	}
	return maxFloat32Scalar(vals)
}

func dotProductFloat32Impl(a, b []float32) float32 {
	n := len(a)
	if hasSVE && n >= sveThresholds.DotProductFloat32 {
		return dotProductFloat32SVE(a, b)
	}
	if n >= neonThresholds.DotProductFloat32 {
		return dotProductFloat32NEON(a, b)
	}
	return dotProductFloat32Scalar(a, b)
}

// NEON float32 assembly implementations (in float32_arm64_neon.s)
func sumFloat32NEON(vals []float32) float32
func minFloat32NEON(vals []float32) float32
func maxFloat32NEON(vals []float32) float32
func dotProductFloat32NEON(a, b []float32) float32

// SVE float32 assembly implementations (in float32_arm64_sve.s)
func sumFloat32SVE(vals []float32) float32
func minFloat32SVE(vals []float32) float32
func maxFloat32SVE(vals []float32) float32
func dotProductFloat32SVE(a, b []float32) float32

// Int16 dispatch functions

func sumInt16Impl(vals []int16) int64 {
	n := len(vals)
	// NOTE: SVE2's SADALP uses 32-bit accumulators which can overflow with
	// many large values. Use plain SVE which has 64-bit accumulators.
	if hasSVE && n >= sveThresholds.SumInt16 {
		return sumInt16SVE(vals)
	}
	if n >= neonThresholds.SumInt16 {
		return sumInt16NEON(vals)
	}
	return sumInt16Scalar(vals)
}

func minInt16Impl(vals []int16) int16 {
	n := len(vals)
	if hasSVE && n >= sveThresholds.MinInt16 {
		return minInt16SVE(vals)
	}
	if n >= neonThresholds.MinInt16 {
		return minInt16NEON(vals)
	}
	return minInt16Scalar(vals)
}

func maxInt16Impl(vals []int16) int16 {
	n := len(vals)
	if hasSVE && n >= sveThresholds.MaxInt16 {
		return maxInt16SVE(vals)
	}
	if n >= neonThresholds.MaxInt16 {
		return maxInt16NEON(vals)
	}
	return maxInt16Scalar(vals)
}

func dotProductInt16Impl(a, b []int16) int64 {
	n := len(a)
	// NOTE: SVE2's SMLALB/SMLALT use 32-bit accumulators which can overflow
	// with large int16 values. Use plain SVE which has 64-bit accumulators.
	if hasSVE && n >= sveThresholds.DotProductInt16 {
		return dotProductInt16SVE(a, b)
	}
	if n >= neonThresholds.DotProductInt16 {
		return dotProductInt16NEON(a, b)
	}
	return dotProductInt16Scalar(a, b)
}

// NEON int16 assembly implementations (in int16_arm64_neon.s)
func sumInt16NEON(vals []int16) int64
func minInt16NEON(vals []int16) int16
func maxInt16NEON(vals []int16) int16
func dotProductInt16NEON(a, b []int16) int64

// SVE int16 assembly implementations (in int16_arm64_sve.s)
func sumInt16SVE(vals []int16) int64
func minInt16SVE(vals []int16) int16
func maxInt16SVE(vals []int16) int16
func dotProductInt16SVE(a, b []int16) int64
func anyAbsGreaterThanInt16SVE(vals []int16, threshold int16) bool

// Int16 anyAbsGreaterThan dispatch function

func anyAbsGreaterThanInt16Impl(vals []int16, threshold int16) bool {
	n := len(vals)
	if hasSVE && n >= sveThresholds.AnyAbsGreaterThanInt16 {
		return anyAbsGreaterThanInt16SVE(vals, threshold)
	}
	if n >= neonThresholds.AnyAbsGreaterThanInt16 {
		return anyAbsGreaterThanInt16NEON(vals, threshold)
	}
	return anyAbsGreaterThanInt16Scalar(vals, threshold)
}

// NEON int16 anyAbsGreaterThan assembly implementation (in int16_arm64_neon.s)
func anyAbsGreaterThanInt16NEON(vals []int16, threshold int16) bool

// SVE2 int16 assembly implementations (in int16_arm64_sve2.s)
func sumInt16SVE2(vals []int16) int64
func dotProductInt16SVE2(a, b []int16) int64
func sumSqInt16SVE2(vals []int16) int64

// Int16 sumSq dispatch function

func sumSqInt16Impl(vals []int16) int64 {
	n := len(vals)
	// NOTE: SVE2's SMLALB/SMLALT use 32-bit accumulators which can overflow.
	// Use plain SVE which has 64-bit accumulators.
	if hasSVE && n >= sveThresholds.SumSqInt16 {
		return sumSqInt16SVE(vals)
	}
	if n >= neonThresholds.SumSqInt16 {
		return sumSqInt16NEON(vals)
	}
	return sumSqInt16Scalar(vals)
}

// NEON int16 sumSq assembly implementation (in int16_arm64_neon.s)
func sumSqInt16NEON(vals []int16) int64

// SVE int16 sumSq assembly implementation (in int16_arm64_sve.s)
func sumSqInt16SVE(vals []int16) int64
