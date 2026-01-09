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
	cpuAppleM1
	cpuAppleM2
	cpuAppleM3
	cpuAppleM4
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

	// macOS: use sysctl to detect Apple Silicon
	if data, err := os.ReadFile("/usr/sbin/sysctl"); err == nil {
		_ = data // sysctl exists, we're on macOS
	}
	// On macOS, we can't read sysctl directly, so use a simpler approach:
	// Check if we're on Darwin (macOS) by trying to read a macOS-specific file
	if _, err := os.Stat("/System/Library/CoreServices/SystemVersion.plist"); err == nil {
		// We're on macOS, try to detect Apple Silicon variant
		return detectAppleSilicon()
	}

	return cpuUnknown
}

// detectAppleSilicon attempts to identify Apple Silicon variant on macOS.
func detectAppleSilicon() cpuFamily {
	// On macOS, we can use the hw.cpufamily sysctl to identify the chip.
	// However, reading sysctl from Go requires cgo or exec.
	// For simplicity, we'll use a conservative approach and detect based on
	// available features. Apple M1/M2/M3/M4 all have NEON but no SVE.
	//
	// CPU family values (from Apple headers):
	// M1: CPUFAMILY_ARM_FIRESTORM_ICESTORM (0x1B588BB3)
	// M2: CPUFAMILY_ARM_AVALANCHE_BLIZZARD (0xDA33D83D)
	// M3: CPUFAMILY_ARM_EVEREST_SAWTOOTH (0x8765EDEA)
	// M4: CPUFAMILY_ARM_EVEREST_SAWTOOTH2 (TBD)
	//
	// Since we can't easily read sysctl without cgo, we'll default to M3
	// for now on macOS arm64, as the NEON thresholds are similar across
	// Apple Silicon generations.
	return cpuAppleM3
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

// NEON thresholds vary by processor due to different microarchitectures.
var neonThresholds = initNEONThresholds()

func initNEONThresholds() thresholds {
	switch detectedCPU {
	case cpuAppleM1, cpuAppleM2, cpuAppleM3, cpuAppleM4:
		// Apple Silicon: Tuned from M3 Max benchmarks.
		// Note: Apple Silicon has no SVE, only NEON.
		return thresholds{
			// Float64: 2 elements per vector
			SumFloat64:        32, // NEON wins at n=32 (5.3ns vs 11.8ns scalar)
			MinFloat64:        4,  // NEON wins at n=4 (2.4ns vs 3.1ns scalar)
			MaxFloat64:        4,  // NEON wins at n=4 (2.4ns vs 2.9ns scalar)
			DotProductFloat64: 32, // NEON wins at n=32 (5.9ns vs 13.3ns scalar)
			// Int64: 2 elements per vector
			SumInt64:          4, // NEON wins at n=4 (2.7ns vs 3.4ns scalar)
			MinInt64:          4, // NEON wins at n=4 (2.7ns vs 3.1ns scalar)
			MaxInt64:          4, // Similar to MinInt64
			DotProductInt64:   0, // N/A - no NEON impl (SVE only)
			SumSqInt64:        4, // NEON wins at n=4 (3.2ns vs 3.4ns scalar)
			AnyAbsGreaterThan: 4, // NEON wins at n=4 (2.1ns vs 3.5ns scalar)
			// Int32: 4 elements per vector
			SumInt32:               4, // NEON wins at n=4 (2.9ns vs 3.5ns scalar)
			MinInt32:               8, // NEON wins at n=8 (3.5ns vs 4.2ns scalar)
			MaxInt32:               8, // Similar to MinInt32
			DotProductInt32:        4, // NEON wins at n=4 (2.4ns vs 3.6ns scalar)
			SumSqInt32:             4, // NEON wins at n=4 (2.4ns vs 3.6ns scalar)
			AnyAbsGreaterThanInt32: 4, // NEON wins at n=4 (2.1ns vs 3.2ns scalar)
			// Int16: 8 elements per vector
			SumInt16:               8,  // NEON wins at n=8 (2.9ns vs 4.4ns scalar)
			MinInt16:               12, // NEON wins at n=12 (3.5ns vs 5.3ns scalar)
			MaxInt16:               12, // Similar to MinInt16
			DotProductInt16:        8,  // NEON wins at n=8 (3.0ns vs 4.4ns scalar)
			SumSqInt16:             8,  // NEON wins at n=8 (3.0ns vs 4.5ns scalar)
			AnyAbsGreaterThanInt16: 8,  // NEON wins at n=8 (2.1ns vs 4.4ns scalar)
			// Float32: 4 elements per vector
			SumFloat32:        64, // NEON wins at n=64 (6.2ns vs 24ns scalar)
			MinFloat32:        16, // NEON wins at n=16 (4.1ns vs 7.0ns scalar)
			MaxFloat32:        16, // Similar to MinFloat32
			DotProductFloat32: 32, // NEON wins at n=32 (11.1ns vs 12.4ns scalar)
		}
	default:
		// Graviton and other ARM64: Measured on Graviton4
		return thresholds{
			// Float64: 2 elements per vector
			SumFloat64:        32, // NEON needs larger arrays to amortize overhead
			MinFloat64:        8,  // NEON: 12ns vs scalar: 16ns at n=8
			MaxFloat64:        8,  // Similar to MinFloat64
			DotProductFloat64: 16, // NEON: 18ns vs scalar: 14ns at n=8, wins at n=16
			// Int64: 2 elements per vector
			SumInt64:          4,  // NEON: 10ns vs scalar: 10ns at n=4
			MinInt64:          8,  // NEON: 12ns vs scalar: 12ns at n=8
			MaxInt64:          8,  // Similar to MinInt64
			DotProductInt64:   0,  // N/A - no NEON impl (SVE only)
			SumSqInt64:        4,  // NEON: 9.6ns vs scalar: 10.7ns at n=4
			AnyAbsGreaterThan: 8,  // NEON: 8.7ns vs scalar: 16ns at n=8
			// Int32: 4 elements per vector
			SumInt32:               8,  // NEON: 10.7ns vs scalar: 14ns at n=8
			MinInt32:               8,  // NEON: 12.7ns vs scalar: 14ns at n=8
			MaxInt32:               8,  // Similar to MinInt32
			DotProductInt32:        4,  // NEON: 10ns vs scalar: 9.6ns at n=4 (close)
			SumSqInt32:             4,  // NEON: 8.6ns vs scalar: 10.7ns at n=4
			AnyAbsGreaterThanInt32: 4,  // NEON: 8.1ns vs scalar: 9.7ns at n=4
			// Int16: 8 elements per vector
			SumInt16:               8,  // NEON: 14ns vs scalar: 22ns at n=8
			MinInt16:               12, // NEON: 17ns vs scalar: 29ns at n=12
			MaxInt16:               12, // Similar to MinInt16
			DotProductInt16:        8,
			SumSqInt16:             8,
			AnyAbsGreaterThanInt16: 8,
			// Float32: 4 elements per vector
			SumFloat32:        16, // NEON: 19ns vs scalar: 24ns at n=16
			MinFloat32:        8,  // NEON: 12.9ns vs scalar: 15ns at n=8
			MaxFloat32:        8,  // Similar to MinFloat32
			DotProductFloat32: 8,  // NEON: 17ns vs scalar: 14ns at n=8 (close)
		}
	}
}

// SVE thresholds vary by processor due to different overhead characteristics.
var sveThresholds = initSVEThresholds()

func initSVEThresholds() thresholds {
	switch detectedCPU {
	case cpuGraviton4:
		// Graviton4 (Neoverse-V2) with 256-bit SVE vectors
		// Thresholds indicate where SVE beats NEON (or scalar if no NEON impl)
		return thresholds{
			// Float64: 4 elements per 256-bit vector
			SumFloat64:        4,  // SVE: 10.8ns vs NEON: 11.9ns at n=4
			MinFloat64:        48, // SVE slower than NEON until n=48
			MaxFloat64:        48, // Similar to MinFloat64
			DotProductFloat64: 16, // SVE: 16.5ns vs scalar: 26ns at n=16
			// Int64: 4 elements per 256-bit vector
			SumInt64:          16, // SVE: 13ns vs NEON: 19ns at n=16
			MinInt64:          48, // SVE: 30ns vs NEON: 38ns at n=48
			MaxInt64:          48, // Similar to MinInt64
			DotProductInt64:   16, // SVE: 19.5ns vs scalar: 22.5ns at n=16
			SumSqInt64:        32, // SVE: 23ns vs NEON: 27ns at n=32
			AnyAbsGreaterThan: 16, // Conservative
			// Int32: 8 elements per 256-bit vector
			// Note: SVE2 is faster than SVE for many ops, but we tune for SVE
			SumInt32:               48, // SVE slower than NEON; NEON: 15ns vs SVE: 27ns at n=48
			MinInt32:               48, // SVE: 19ns vs NEON: 19ns at n=48
			MaxInt32:               48, // Similar to MinInt32
			DotProductInt32:        48, // SVE slower than NEON until larger sizes
			SumSqInt32:             64, // SVE slower than NEON
			AnyAbsGreaterThanInt32: 32, // SVE: 15ns vs NEON: 14ns at n=32
			// Float32: 8 elements per 256-bit vector
			SumFloat32:        8,  // SVE: 10.8ns vs NEON: 15ns at n=8
			MinFloat32:        48, // SVE: 20ns vs NEON: 19ns at n=48
			MaxFloat32:        48, // Similar to MinFloat32
			DotProductFloat32: 8,  // SVE: 12.7ns vs NEON: 17ns at n=8
			// Int16: 16 elements per 256-bit vector
			// Note: For Sum, NEON is faster than SVE (better pairwise reduction)
			// For Min/Max, SVE is faster (native SMINV/SMAXV horizontal reduction)
			SumInt16:               256, // NEON is faster, use high threshold to prefer NEON
			MinInt16:               8,   // SVE: 12.7ns vs NEON: 23ns at n=8
			MaxInt16:               8,   // Similar to MinInt16
			DotProductInt16:        256, // NEON is faster
			SumSqInt16:             256, // NEON is faster
			AnyAbsGreaterThanInt16: 16,
		}
	case cpuGraviton3:
		// Graviton3 (Neoverse-V1) with 256-bit SVE vectors
		return thresholds{
			SumFloat64:        12, // SVE wins at n=12 (6.1ns vs 6.6ns scalar)
			MinFloat64:        48, // SVE wins at n=48 (9.8ns vs NEON 11.4ns)
			MaxFloat64:        48, // Similar to MinFloat64
			DotProductFloat64: 16, // SVE wins at n=16 (7.5ns vs 7.7ns scalar)
			SumInt64:          16, // SVE wins at n=16 (7.0ns vs NEON 7.1ns)
			MinInt64:          48, // SVE wins at n=48 (9.3ns vs NEON 17.9ns)
			MaxInt64:          48, // Similar to MinInt64
			DotProductInt64:   32, // SVE wins at n=32 (10.7ns vs 13.9ns scalar)
			SumSqInt64:        32, // SVE wins at n=32 (10.0ns vs NEON 11.1ns)
			AnyAbsGreaterThan: 256, // SVE slower than NEON, use high threshold
			// Int32
			SumInt32:               256, // SVE slower than NEON, NEON wins at n=8
			MinInt32:               256, // SVE slower than NEON, NEON wins at n=8
			MaxInt32:               256, // Similar to MinInt32
			DotProductInt32:        256, // SVE slower than NEON, NEON wins at n=8
			SumSqInt32:             256, // SVE slower than NEON
			AnyAbsGreaterThanInt32: 256, // SVE slower than NEON
			// Float32
			SumFloat32:        16, // SVE wins at n=16 (5.0ns vs NEON 7.1ns)
			MinFloat32:        256, // SVE slower than NEON, NEON wins at n=8
			MaxFloat32:        256, // Similar to MinFloat32
			DotProductFloat32: 256, // SVE slower than NEON
			// Int16
			SumInt16:               256, // SVE slower than NEON, NEON wins at n=8
			MinInt16:               8,   // SVE wins at n=8 (5.3ns vs NEON 6.7ns)
			MaxInt16:               8,   // Similar to MinInt16
			DotProductInt16:        256, // SVE slower than NEON
			SumSqInt16:             256, // SVE slower than NEON
			AnyAbsGreaterThanInt16: 256, // SVE slower than NEON
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
	case cpuAppleM1:
		return "Apple M1"
	case cpuAppleM2:
		return "Apple M2"
	case cpuAppleM3:
		return "Apple M3"
	case cpuAppleM4:
		return "Apple M4"
	default:
		return "Unknown ARM64"
	}
}
