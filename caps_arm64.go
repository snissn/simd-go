//go:build arm64

package simd

import "golang.org/x/sys/cpu"

// Runtime CPU feature detection for arm64.
var hasSVE = cpu.ARM64.HasSVE

// HasSVE reports whether the CPU supports SVE instructions.
func HasSVE() bool { return hasSVE }

// HasNEON reports whether the CPU supports NEON instructions.
// All arm64 CPUs support NEON, so this always returns true on arm64.
func HasNEON() bool { return true }

// IsARM64 reports whether we're running on arm64 architecture.
func IsARM64() bool { return true }
