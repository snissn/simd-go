//go:build !arm64

package simd

// HasSVE reports whether the CPU supports SVE instructions.
// Returns false on generic platforms.
func HasSVE() bool { return false }

// HasNEON reports whether the CPU supports NEON instructions.
// Returns false on generic platforms.
func HasNEON() bool { return false }

// IsARM64 reports whether we're running on arm64 architecture.
func IsARM64() bool { return false }
