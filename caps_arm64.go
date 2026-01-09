//go:build arm64

package simd

import (
	"os"
	"strings"

	"golang.org/x/sys/cpu"
)

// Runtime CPU feature detection for arm64.
var hasSVE = cpu.ARM64.HasSVE
var hasSVE2 = cpu.ARM64.HasSVE2

// CPU family detection for threshold tuning.
type cpuFamily int

const (
	cpuUnknown cpuFamily = iota
	cpuGraviton3
	cpuGraviton4
)

var detectedCPU = detectCPUFamily()

// detectCPUFamily attempts to identify the ARM CPU family.
// Returns cpuUnknown if detection fails.
func detectCPUFamily() cpuFamily {
	// Try MIDR_EL1 first (most reliable on Linux)
	if midr, err := os.ReadFile("/sys/devices/system/cpu/cpu0/regs/identification/midr_el1"); err == nil {
		s := strings.TrimSpace(string(midr))
		// MIDR format: 0x<impl><var><arch><partnum><rev>
		// Neoverse-V1 (Graviton3): part 0xD40, impl 0x41 -> 0x410FD400 - 0x410FD40F
		// Neoverse-V2 (Graviton4): part 0xD4F, impl 0x41 -> 0x410FD4F0 - 0x410FD4FF
		switch {
		case strings.HasPrefix(s, "0x410fd4f"):
			return cpuGraviton4
		case strings.HasPrefix(s, "0x410fd40"):
			return cpuGraviton3
		}
	}

	// Fallback: parse /proc/cpuinfo for CPU part
	if data, err := os.ReadFile("/proc/cpuinfo"); err == nil {
		content := string(data)
		for _, line := range strings.Split(content, "\n") {
			if strings.HasPrefix(line, "CPU part") {
				// "CPU part\t: 0xd4f" for Neoverse-V2
				// "CPU part\t: 0xd40" for Neoverse-V1
				if strings.Contains(line, "0xd4f") {
					return cpuGraviton4
				}
				if strings.Contains(line, "0xd40") {
					return cpuGraviton3
				}
				break
			}
		}
	}

	return cpuUnknown
}

// Thresholds define the minimum slice length for using SIMD.
// Below these thresholds, scalar is faster due to function call overhead.
type thresholds struct {
	// Float64 operations
	SumFloat64        int
	MinFloat64        int
	MaxFloat64        int
	DotProductFloat64 int

	// Int64 operations
	SumInt64          int
	MinInt64          int
	MaxInt64          int
	DotProductInt64   int
	SumSqInt64        int
	AnyAbsGreaterThan int

	// Int32 operations
	SumInt32               int
	MinInt32               int
	MaxInt32               int
	DotProductInt32        int
	SumSqInt32             int
	AnyAbsGreaterThanInt32 int

	// Int16 operations
	SumInt16               int
	MinInt16               int
	MaxInt16               int
	DotProductInt16        int
	SumSqInt16             int
	AnyAbsGreaterThanInt16 int

	// Float32 operations
	SumFloat32        int
	MinFloat32        int
	MaxFloat32        int
	DotProductFloat32 int
}

// NEON thresholds are consistent across processors (low overhead).
var neonThresholds = thresholds{
	SumFloat64:        32,
	MinFloat64:        8,
	MaxFloat64:        8,
	DotProductFloat64: 32,
	SumInt64:          8,
	MinInt64:          8,
	MaxInt64:          8,
	DotProductInt64:   0, // N/A - no NEON impl
	SumSqInt64:        8,
	AnyAbsGreaterThan: 8,
	// Int32: 4 elements per vector
	SumInt32:               32,
	MinInt32:               32,
	MaxInt32:               32,
	DotProductInt32:        32,
	SumSqInt32:             32,
	AnyAbsGreaterThanInt32: 8,
	// Int16: Conservative thresholds (2x elements per vector vs int32)
	SumInt16:               64,
	MinInt16:               64,
	MaxInt16:               64,
	DotProductInt16:        64,
	SumSqInt16:             64,
	AnyAbsGreaterThanInt16: 8,
	// Float32: 4 elements per vector
	SumFloat32:        32,
	MinFloat32:        8,
	MaxFloat32:        8,
	DotProductFloat32: 32,
}

// SVE thresholds vary by processor due to different overhead characteristics.
var sveThresholds = initSVEThresholds()

func initSVEThresholds() thresholds {
	switch detectedCPU {
	case cpuGraviton4:
		return thresholds{
			SumFloat64:        8,
			MinFloat64:        8,
			MaxFloat64:        8,
			DotProductFloat64: 16,
			SumInt64:          16,
			MinInt64:          32,
			MaxInt64:          32,
			DotProductInt64:   16,
			SumSqInt64:        32,
			AnyAbsGreaterThan: 16,
			// Int32: 2x elements per vector
			SumInt32:               8,
			MinInt32:               16,
			MaxInt32:               16,
			DotProductInt32:        8,
			SumSqInt32:             16,
			AnyAbsGreaterThanInt32: 16,
			// Float32: 2x elements per vector vs float64
			SumFloat32:        8,
			MinFloat32:        8,
			MaxFloat32:        8,
			DotProductFloat32: 8,
			// Int16: 4x elements per vector vs int64
			SumInt16:               8,
			MinInt16:               16,
			MaxInt16:               16,
			DotProductInt16:        8,
			SumSqInt16:             8,
			AnyAbsGreaterThanInt16: 16,
		}
	case cpuGraviton3:
		return thresholds{
			SumFloat64:        48,
			MinFloat64:        48,
			MaxFloat64:        48,
			DotProductFloat64: 32,
			SumInt64:          32,
			MinInt64:          48,
			MaxInt64:          48,
			DotProductInt64:   32,
			SumSqInt64:        32,
			AnyAbsGreaterThan: 32,
			// Int32
			SumInt32:               24,
			MinInt32:               24,
			MaxInt32:               24,
			DotProductInt32:        16,
			SumSqInt32:             16,
			AnyAbsGreaterThanInt32: 32,
			// Float32
			SumFloat32:        24,
			MinFloat32:        24,
			MaxFloat32:        24,
			DotProductFloat32: 16,
			// Int16
			SumInt16:               24,
			MinInt16:               24,
			MaxInt16:               24,
			DotProductInt16:        16,
			SumSqInt16:             16,
			AnyAbsGreaterThanInt16: 32,
		}
	default:
		// Conservative defaults for unknown SVE processors
		return thresholds{
			SumFloat64:        48,
			MinFloat64:        48,
			MaxFloat64:        48,
			DotProductFloat64: 32,
			SumInt64:          32,
			MinInt64:          48,
			MaxInt64:          48,
			DotProductInt64:   32,
			SumSqInt64:        32,
			AnyAbsGreaterThan: 32,
			// Int32
			SumInt32:               24,
			MinInt32:               24,
			MaxInt32:               24,
			DotProductInt32:        16,
			SumSqInt32:             16,
			AnyAbsGreaterThanInt32: 32,
			// Float32
			SumFloat32:        24,
			MinFloat32:        24,
			MaxFloat32:        24,
			DotProductFloat32: 16,
			// Int16
			SumInt16:               24,
			MinInt16:               24,
			MaxInt16:               24,
			DotProductInt16:        16,
			SumSqInt16:             16,
			AnyAbsGreaterThanInt16: 32,
		}
	}
}

// HasSVE reports whether the CPU supports SVE instructions.
func HasSVE() bool { return hasSVE }

// HasSVE2 reports whether the CPU supports SVE2 instructions.
func HasSVE2() bool { return hasSVE2 }

// HasNEON reports whether the CPU supports NEON instructions.
// All arm64 CPUs support NEON, so this always returns true on arm64.
func HasNEON() bool { return true }

// IsARM64 reports whether we're running on arm64 architecture.
func IsARM64() bool { return true }

// CPUName returns a human-readable name for the detected CPU.
func CPUName() string {
	switch detectedCPU {
	case cpuGraviton3:
		return "AWS Graviton3 (Neoverse-V1)"
	case cpuGraviton4:
		return "AWS Graviton4 (Neoverse-V2)"
	default:
		return "Unknown ARM64"
	}
}
