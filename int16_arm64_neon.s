//go:build arm64

#include "textflag.h"

// ╔══════════════════════════════════════════════════════════════════════════════╗
// ║                         NEON Int16 SIMD Operations                           ║
// ║                                                                              ║
// ║  NEON processes 8 x int16 per vector register (128-bit vectors)             ║
// ║                                                                              ║
// ║  Vector register layout:                                                     ║
// ║  ┌─────────────────────────────────────────────────────────────────────┐     ║
// ║  │  V0.H8 = [ h0 | h1 | h2 | h3 | h4 | h5 | h6 | h7 ]  (128 bits)     │     ║
// ║  └─────────────────────────────────────────────────────────────────────┘     ║
// ║                                                                              ║
// ║  Key advantage over int32:                                                   ║
// ║  • 2x elements per vector = 2x throughput potential                          ║
// ║  • Requires widening to avoid overflow in sum/dot product                    ║
// ╚══════════════════════════════════════════════════════════════════════════════╝

// ┌──────────────────────────────────────────────────────────────────────────────┐
// │ func sumInt16NEON(vals []int16) int64                                        │
// │                                                                              │
// │ Strategy: Load int16 vectors, widen to int32 with SADDLP, then to int64      │
// │ Processes 64 elements per iteration (8 vectors × 8 lanes)                    │
// └──────────────────────────────────────────────────────────────────────────────┘
TEXT ·sumInt16NEON(SB), NOSPLIT, $0-32
    MOVD vals_base+0(FP), R0
    MOVD vals_len+8(FP), R1
    
    // Initialize 8 int64 accumulators to zero (V0-V7.D2)
    VEOR V0.B16, V0.B16, V0.B16
    VEOR V1.B16, V1.B16, V1.B16
    VEOR V2.B16, V2.B16, V2.B16
    VEOR V3.B16, V3.B16, V3.B16
    VEOR V4.B16, V4.B16, V4.B16
    VEOR V5.B16, V5.B16, V5.B16
    VEOR V6.B16, V6.B16, V6.B16
    VEOR V7.B16, V7.B16, V7.B16
    
    // Process 64 elements at a time (8 vectors × 8 elements)
    CMP $64, R1
    BLT sum16_tail32

sum16_loop64:
    // Load 8 vectors of int16 (64 elements, 128 bytes)
    VLD1.P 64(R0), [V16.H8, V17.H8, V18.H8, V19.H8]
    VLD1.P 64(R0), [V20.H8, V21.H8, V22.H8, V23.H8]
    
    // SADDLP: Signed Add Long Pairwise - pairs of int16 -> int32
    // V16.H8 = [a,b,c,d,e,f,g,h] -> V24.S4 = [a+b, c+d, e+f, g+h]
    WORD $0x4E602A18              // SADDLP V24.4S, V16.8H
    WORD $0x4E602A39              // SADDLP V25.4S, V17.8H
    WORD $0x4E602A5A              // SADDLP V26.4S, V18.8H
    WORD $0x4E602A7B              // SADDLP V27.4S, V19.8H
    WORD $0x4E602A9C              // SADDLP V28.4S, V20.8H
    WORD $0x4E602ABD              // SADDLP V29.4S, V21.8H
    WORD $0x4E602ADE              // SADDLP V30.4S, V22.8H
    WORD $0x4E602AFF              // SADDLP V31.4S, V23.8H
    
    // SADDLP again: pairs of int32 -> int64
    // V24.S4 = [a,b,c,d] -> V24.D2 = [a+b, c+d]
    WORD $0x4EA02B18              // SADDLP V24.2D, V24.4S
    WORD $0x4EA02B39              // SADDLP V25.2D, V25.4S
    WORD $0x4EA02B5A              // SADDLP V26.2D, V26.4S
    WORD $0x4EA02B7B              // SADDLP V27.2D, V27.4S
    WORD $0x4EA02B9C              // SADDLP V28.2D, V28.4S
    WORD $0x4EA02BBD              // SADDLP V29.2D, V29.4S
    WORD $0x4EA02BDE              // SADDLP V30.2D, V30.4S
    WORD $0x4EA02BFF              // SADDLP V31.2D, V31.4S
    
    // Add widened results to int64 accumulators
    WORD $0x4EF88400              // ADD V0.2D, V0.2D, V24.2D
    WORD $0x4EF98421              // ADD V1.2D, V1.2D, V25.2D
    WORD $0x4EFA8442              // ADD V2.2D, V2.2D, V26.2D
    WORD $0x4EFB8463              // ADD V3.2D, V3.2D, V27.2D
    WORD $0x4EFC8484              // ADD V4.2D, V4.2D, V28.2D
    WORD $0x4EFD84A5              // ADD V5.2D, V5.2D, V29.2D
    WORD $0x4EFE84C6              // ADD V6.2D, V6.2D, V30.2D
    WORD $0x4EFF84E7              // ADD V7.2D, V7.2D, V31.2D
    
    SUB $64, R1
    CMP $64, R1
    BGE sum16_loop64

sum16_tail32:
    CMP $32, R1
    BLT sum16_tail16
    
    VLD1.P 64(R0), [V16.H8, V17.H8, V18.H8, V19.H8]
    WORD $0x4E602A18              // SADDLP V24.4S, V16.8H
    WORD $0x4E602A39              // SADDLP V25.4S, V17.8H
    WORD $0x4E602A5A              // SADDLP V26.4S, V18.8H
    WORD $0x4E602A7B              // SADDLP V27.4S, V19.8H
    WORD $0x4EA02B18              // SADDLP V24.2D, V24.4S
    WORD $0x4EA02B39              // SADDLP V25.2D, V25.4S
    WORD $0x4EA02B5A              // SADDLP V26.2D, V26.4S
    WORD $0x4EA02B7B              // SADDLP V27.2D, V27.4S
    WORD $0x4EF88400              // ADD V0.2D, V0.2D, V24.2D
    WORD $0x4EF98421              // ADD V1.2D, V1.2D, V25.2D
    WORD $0x4EFA8442              // ADD V2.2D, V2.2D, V26.2D
    WORD $0x4EFB8463              // ADD V3.2D, V3.2D, V27.2D
    SUB $32, R1

sum16_tail16:
    CMP $16, R1
    BLT sum16_tail8
    
    VLD1.P 32(R0), [V16.H8, V17.H8]
    WORD $0x4E602A18              // SADDLP V24.4S, V16.8H
    WORD $0x4E602A39              // SADDLP V25.4S, V17.8H
    WORD $0x4EA02B18              // SADDLP V24.2D, V24.4S
    WORD $0x4EA02B39              // SADDLP V25.2D, V25.4S
    WORD $0x4EF88400              // ADD V0.2D, V0.2D, V24.2D
    WORD $0x4EF98421              // ADD V1.2D, V1.2D, V25.2D
    SUB $16, R1

sum16_tail8:
    CMP $8, R1
    BLT sum16_reduce
    
    VLD1.P 16(R0), [V16.H8]
    WORD $0x4E602A18              // SADDLP V24.4S, V16.8H
    WORD $0x4EA02B18              // SADDLP V24.2D, V24.4S
    WORD $0x4EF88400              // ADD V0.2D, V0.2D, V24.2D
    SUB $8, R1

sum16_reduce:
    // Tree reduction: 8 -> 4 -> 2 -> 1
    WORD $0x4EE48400              // ADD V0.2D, V0.2D, V4.2D
    WORD $0x4EE58421              // ADD V1.2D, V1.2D, V5.2D
    WORD $0x4EE68442              // ADD V2.2D, V2.2D, V6.2D
    WORD $0x4EE78463              // ADD V3.2D, V3.2D, V7.2D
    WORD $0x4EE28400              // ADD V0.2D, V0.2D, V2.2D
    WORD $0x4EE38421              // ADD V1.2D, V1.2D, V3.2D
    WORD $0x4EE18400              // ADD V0.2D, V0.2D, V1.2D
    
    // Horizontal add of V0.D2
    VMOV V0.D[0], R2
    VMOV V0.D[1], R3
    ADD R2, R3, R3
    
    // Handle remaining 1-7 elements
    CBZ R1, sum16_done
    
sum16_scalar_loop:
    MOVH (R0), R2
    SXTH R2, R2                   // Sign extend to 64-bit
    ADD R2, R3, R3
    ADD $2, R0
    SUB $1, R1
    CBNZ R1, sum16_scalar_loop

sum16_done:
    MOVD R3, ret+24(FP)
    RET

// ┌──────────────────────────────────────────────────────────────────────────────┐
// │ func minInt16NEON(vals []int16) int16                                        │
// │                                                                              │
// │ Strategy: 8 accumulators with native SMIN.8H instruction                     │
// │ NEON has native SMIN for 16-bit!                                             │
// └──────────────────────────────────────────────────────────────────────────────┘
TEXT ·minInt16NEON(SB), NOSPLIT, $0-26
    MOVD vals_base+0(FP), R0
    MOVD vals_len+8(FP), R1
    
    CBZ R1, min16_empty
    
    // Load first element and broadcast to all lanes
    MOVH (R0), R2
    VDUP R2, V0.H8
    VMOV V0.B16, V1.B16
    VMOV V0.B16, V2.B16
    VMOV V0.B16, V3.B16
    VMOV V0.B16, V4.B16
    VMOV V0.B16, V5.B16
    VMOV V0.B16, V6.B16
    VMOV V0.B16, V7.B16
    ADD $2, R0
    SUB $1, R1
    
    CMP $64, R1
    BLT min16_tail32

min16_loop64:
    VLD1.P 64(R0), [V16.H8, V17.H8, V18.H8, V19.H8]
    VLD1.P 64(R0), [V20.H8, V21.H8, V22.H8, V23.H8]
    
    // Native SMIN.8H
    WORD $0x4E706C00              // SMIN V0.8H, V0.8H, V16.8H
    WORD $0x4E716C21              // SMIN V1.8H, V1.8H, V17.8H
    WORD $0x4E726C42              // SMIN V2.8H, V2.8H, V18.8H
    WORD $0x4E736C63              // SMIN V3.8H, V3.8H, V19.8H
    WORD $0x4E746C84              // SMIN V4.8H, V4.8H, V20.8H
    WORD $0x4E756CA5              // SMIN V5.8H, V5.8H, V21.8H
    WORD $0x4E766CC6              // SMIN V6.8H, V6.8H, V22.8H
    WORD $0x4E776CE7              // SMIN V7.8H, V7.8H, V23.8H
    
    SUB $64, R1
    CMP $64, R1
    BGE min16_loop64

min16_tail32:
    CMP $32, R1
    BLT min16_tail16
    
    VLD1.P 64(R0), [V16.H8, V17.H8, V18.H8, V19.H8]
    WORD $0x4E706C00              // SMIN V0.8H, V0.8H, V16.8H
    WORD $0x4E716C21              // SMIN V1.8H, V1.8H, V17.8H
    WORD $0x4E726C42              // SMIN V2.8H, V2.8H, V18.8H
    WORD $0x4E736C63              // SMIN V3.8H, V3.8H, V19.8H
    SUB $32, R1

min16_tail16:
    CMP $16, R1
    BLT min16_tail8
    
    VLD1.P 32(R0), [V16.H8, V17.H8]
    WORD $0x4E706C00              // SMIN V0.8H, V0.8H, V16.8H
    WORD $0x4E716C21              // SMIN V1.8H, V1.8H, V17.8H
    SUB $16, R1

min16_tail8:
    CMP $8, R1
    BLT min16_reduce
    
    VLD1.P 16(R0), [V16.H8]
    WORD $0x4E706C00              // SMIN V0.8H, V0.8H, V16.8H
    SUB $8, R1

min16_reduce:
    // Tree reduction: 8 -> 4 -> 2 -> 1
    WORD $0x4E746C00              // SMIN V0.8H, V0.8H, V4.8H
    WORD $0x4E756C21              // SMIN V1.8H, V1.8H, V5.8H
    WORD $0x4E766C42              // SMIN V2.8H, V2.8H, V6.8H
    WORD $0x4E776C63              // SMIN V3.8H, V3.8H, V7.8H
    WORD $0x4E726C00              // SMIN V0.8H, V0.8H, V2.8H
    WORD $0x4E736C21              // SMIN V1.8H, V1.8H, V3.8H
    WORD $0x4E716C00              // SMIN V0.8H, V0.8H, V1.8H
    
    // SMINV H0, V0.8H - horizontal minimum across all lanes
    WORD $0x4E71A800              // SMINV H0, V0.8H
    
    // Handle remaining 1-7 elements
    CBZ R1, min16_store
    FMOVS F0, R2
    SXTH R2, R2

min16_scalar_loop:
    MOVH (R0), R3
    SXTH R3, R3
    CMP R2, R3
    CSEL LT, R3, R2, R2
    ADD $2, R0
    SUB $1, R1
    CBNZ R1, min16_scalar_loop
    B min16_done

min16_store:
    FMOVS F0, R2
    B min16_done

min16_empty:
    MOVW $0, R2

min16_done:
    MOVH R2, ret+24(FP)
    RET

// ┌──────────────────────────────────────────────────────────────────────────────┐
// │ func maxInt16NEON(vals []int16) int16                                        │
// │                                                                              │
// │ Strategy: 8 accumulators with native SMAX.8H instruction                     │
// └──────────────────────────────────────────────────────────────────────────────┘
TEXT ·maxInt16NEON(SB), NOSPLIT, $0-26
    MOVD vals_base+0(FP), R0
    MOVD vals_len+8(FP), R1
    
    CBZ R1, max16_empty
    
    // Load first element and broadcast to all lanes
    MOVH (R0), R2
    VDUP R2, V0.H8
    VMOV V0.B16, V1.B16
    VMOV V0.B16, V2.B16
    VMOV V0.B16, V3.B16
    VMOV V0.B16, V4.B16
    VMOV V0.B16, V5.B16
    VMOV V0.B16, V6.B16
    VMOV V0.B16, V7.B16
    ADD $2, R0
    SUB $1, R1
    
    CMP $64, R1
    BLT max16_tail32

max16_loop64:
    VLD1.P 64(R0), [V16.H8, V17.H8, V18.H8, V19.H8]
    VLD1.P 64(R0), [V20.H8, V21.H8, V22.H8, V23.H8]
    
    // Native SMAX.8H
    WORD $0x4E706400              // SMAX V0.8H, V0.8H, V16.8H
    WORD $0x4E716421              // SMAX V1.8H, V1.8H, V17.8H
    WORD $0x4E726442              // SMAX V2.8H, V2.8H, V18.8H
    WORD $0x4E736463              // SMAX V3.8H, V3.8H, V19.8H
    WORD $0x4E746484              // SMAX V4.8H, V4.8H, V20.8H
    WORD $0x4E7564A5              // SMAX V5.8H, V5.8H, V21.8H
    WORD $0x4E7664C6              // SMAX V6.8H, V6.8H, V22.8H
    WORD $0x4E7764E7              // SMAX V7.8H, V7.8H, V23.8H
    
    SUB $64, R1
    CMP $64, R1
    BGE max16_loop64

max16_tail32:
    CMP $32, R1
    BLT max16_tail16
    
    VLD1.P 64(R0), [V16.H8, V17.H8, V18.H8, V19.H8]
    WORD $0x4E706400              // SMAX V0.8H, V0.8H, V16.8H
    WORD $0x4E716421              // SMAX V1.8H, V1.8H, V17.8H
    WORD $0x4E726442              // SMAX V2.8H, V2.8H, V18.8H
    WORD $0x4E736463              // SMAX V3.8H, V3.8H, V19.8H
    SUB $32, R1

max16_tail16:
    CMP $16, R1
    BLT max16_tail8
    
    VLD1.P 32(R0), [V16.H8, V17.H8]
    WORD $0x4E706400              // SMAX V0.8H, V0.8H, V16.8H
    WORD $0x4E716421              // SMAX V1.8H, V1.8H, V17.8H
    SUB $16, R1

max16_tail8:
    CMP $8, R1
    BLT max16_reduce
    
    VLD1.P 16(R0), [V16.H8]
    WORD $0x4E706400              // SMAX V0.8H, V0.8H, V16.8H
    SUB $8, R1

max16_reduce:
    // Tree reduction
    WORD $0x4E746400              // SMAX V0.8H, V0.8H, V4.8H
    WORD $0x4E756421              // SMAX V1.8H, V1.8H, V5.8H
    WORD $0x4E766442              // SMAX V2.8H, V2.8H, V6.8H
    WORD $0x4E776463              // SMAX V3.8H, V3.8H, V7.8H
    WORD $0x4E726400              // SMAX V0.8H, V0.8H, V2.8H
    WORD $0x4E736421              // SMAX V1.8H, V1.8H, V3.8H
    WORD $0x4E716400              // SMAX V0.8H, V0.8H, V1.8H
    
    // SMAXV H0, V0.8H - horizontal maximum
    WORD $0x4E70A800              // SMAXV H0, V0.8H
    
    CBZ R1, max16_store
    FMOVS F0, R2
    SXTH R2, R2

max16_scalar_loop:
    MOVH (R0), R3
    SXTH R3, R3
    CMP R2, R3
    CSEL GT, R3, R2, R2
    ADD $2, R0
    SUB $1, R1
    CBNZ R1, max16_scalar_loop
    B max16_done

max16_store:
    FMOVS F0, R2
    B max16_done

max16_empty:
    MOVW $0, R2

max16_done:
    MOVH R2, ret+24(FP)
    RET

// ┌──────────────────────────────────────────────────────────────────────────────┐
// │ func dotProductInt16NEON(a, b []int16) int64                                 │
// │                                                                              │
// │ Strategy: SMULL/SMLAL to multiply int16 pairs -> int32, then widen to int64 │
// │ Uses widening multiply to avoid overflow                                     │
// └──────────────────────────────────────────────────────────────────────────────┘
TEXT ·dotProductInt16NEON(SB), NOSPLIT, $0-56
    MOVD a_base+0(FP), R0
    MOVD a_len+8(FP), R1
    MOVD b_base+24(FP), R2
    
    // Initialize 8 int64 accumulators
    VEOR V0.B16, V0.B16, V0.B16
    VEOR V1.B16, V1.B16, V1.B16
    VEOR V2.B16, V2.B16, V2.B16
    VEOR V3.B16, V3.B16, V3.B16
    VEOR V4.B16, V4.B16, V4.B16
    VEOR V5.B16, V5.B16, V5.B16
    VEOR V6.B16, V6.B16, V6.B16
    VEOR V7.B16, V7.B16, V7.B16
    
    CMP $32, R1
    BLT dot16_tail16

dot16_loop32:
    // Load 32 elements from each array (64 bytes each)
    // a[0:16] -> V16,V17; a[16:32] -> V18,V19
    // b[0:16] -> V20,V21; b[16:32] -> V22,V23
    VLD1.P 32(R0), [V16.H8, V17.H8]
    VLD1.P 32(R2), [V20.H8, V21.H8]
    VLD1.P 32(R0), [V18.H8, V19.H8]
    VLD1.P 32(R2), [V22.H8, V23.H8]
    
    // SMULL Vd.4S, Vn.4H, Vm.4H: 0x0E60C000 | (Rm<<16) | (Rn<<5) | Rd
    // SMULL2 Vd.4S, Vn.8H, Vm.8H: 0x4E60C000 | (Rm<<16) | (Rn<<5) | Rd
    // V16 × V20 -> V24 (lower), V25 (upper)
    WORD $0x0E74C218              // SMULL V24.4S, V16.4H, V20.4H
    WORD $0x4E74C219              // SMULL2 V25.4S, V16.8H, V20.8H
    // V17 × V21 -> V26, V27
    WORD $0x0E75C23A              // SMULL V26.4S, V17.4H, V21.4H
    WORD $0x4E75C23B              // SMULL2 V27.4S, V17.8H, V21.8H
    // V18 × V22 -> V28, V29
    WORD $0x0E76C25C              // SMULL V28.4S, V18.4H, V22.4H
    WORD $0x4E76C25D              // SMULL2 V29.4S, V18.8H, V22.8H
    // V19 × V23 -> V30, V31
    WORD $0x0E77C27E              // SMULL V30.4S, V19.4H, V23.4H
    WORD $0x4E77C27F              // SMULL2 V31.4S, V19.8H, V23.8H
    
    // Now widen int32 -> int64 using SADDLP and add to accumulators
    // SADDLP Vd.2D, Vn.4S: 0x4EA02800 | (Rn<<5) | Rd
    // V24-V31 contain products, widen each to int64
    WORD $0x4EA02B18              // SADDLP V24.2D, V24.4S
    WORD $0x4EA02B39              // SADDLP V25.2D, V25.4S
    WORD $0x4EA02B5A              // SADDLP V26.2D, V26.4S
    WORD $0x4EA02B7B              // SADDLP V27.2D, V27.4S
    WORD $0x4EA02B9C              // SADDLP V28.2D, V28.4S
    WORD $0x4EA02BBD              // SADDLP V29.2D, V29.4S
    WORD $0x4EA02BDE              // SADDLP V30.2D, V30.4S
    WORD $0x4EA02BFF              // SADDLP V31.2D, V31.4S
    
    // Add to accumulators
    WORD $0x4EF88400              // ADD V0.2D, V0.2D, V24.2D
    WORD $0x4EF98421              // ADD V1.2D, V1.2D, V25.2D
    WORD $0x4EFA8442              // ADD V2.2D, V2.2D, V26.2D
    WORD $0x4EFB8463              // ADD V3.2D, V3.2D, V27.2D
    WORD $0x4EFC8484              // ADD V4.2D, V4.2D, V28.2D
    WORD $0x4EFD84A5              // ADD V5.2D, V5.2D, V29.2D
    WORD $0x4EFE84C6              // ADD V6.2D, V6.2D, V30.2D
    WORD $0x4EFF84E7              // ADD V7.2D, V7.2D, V31.2D
    
    SUB $32, R1
    CMP $32, R1
    BGE dot16_loop32

dot16_tail16:
    CMP $16, R1
    BLT dot16_tail8
    
    VLD1.P 32(R0), [V16.H8, V17.H8]
    VLD1.P 32(R2), [V20.H8, V21.H8]
    
    // V16 × V20 -> V24, V25; V17 × V21 -> V26, V27
    WORD $0x0E74C218              // SMULL V24.4S, V16.4H, V20.4H
    WORD $0x4E74C219              // SMULL2 V25.4S, V16.8H, V20.8H
    WORD $0x0E75C23A              // SMULL V26.4S, V17.4H, V21.4H
    WORD $0x4E75C23B              // SMULL2 V27.4S, V17.8H, V21.8H
    
    WORD $0x4EA02B18              // SADDLP V24.2D, V24.4S
    WORD $0x4EA02B39              // SADDLP V25.2D, V25.4S
    WORD $0x4EA02B5A              // SADDLP V26.2D, V26.4S
    WORD $0x4EA02B7B              // SADDLP V27.2D, V27.4S
    
    WORD $0x4EF88400              // ADD V0.2D, V0.2D, V24.2D
    WORD $0x4EF98421              // ADD V1.2D, V1.2D, V25.2D
    WORD $0x4EFA8442              // ADD V2.2D, V2.2D, V26.2D
    WORD $0x4EFB8463              // ADD V3.2D, V3.2D, V27.2D
    SUB $16, R1

dot16_tail8:
    CMP $8, R1
    BLT dot16_reduce
    
    VLD1.P 16(R0), [V16.H8]
    VLD1.P 16(R2), [V20.H8]
    
    // V16 × V20 -> V24 (lower), V25 (upper)
    WORD $0x0E74C218              // SMULL V24.4S, V16.4H, V20.4H
    WORD $0x4E74C219              // SMULL2 V25.4S, V16.8H, V20.8H
    
    WORD $0x4EA02B18              // SADDLP V24.2D, V24.4S
    WORD $0x4EA02B39              // SADDLP V25.2D, V25.4S
    
    WORD $0x4EF88400              // ADD V0.2D, V0.2D, V24.2D
    WORD $0x4EF98421              // ADD V1.2D, V1.2D, V25.2D
    SUB $8, R1

dot16_reduce:
    // Tree reduction
    WORD $0x4EE48400              // ADD V0.2D, V0.2D, V4.2D
    WORD $0x4EE58421              // ADD V1.2D, V1.2D, V5.2D
    WORD $0x4EE68442              // ADD V2.2D, V2.2D, V6.2D
    WORD $0x4EE78463              // ADD V3.2D, V3.2D, V7.2D
    WORD $0x4EE28400              // ADD V0.2D, V0.2D, V2.2D
    WORD $0x4EE38421              // ADD V1.2D, V1.2D, V3.2D
    WORD $0x4EE18400              // ADD V0.2D, V0.2D, V1.2D
    
    // Horizontal sum
    VMOV V0.D[0], R3
    VMOV V0.D[1], R4
    ADD R3, R4, R3
    
    // Handle remaining elements
    CBZ R1, dot16_done
    
dot16_scalar_loop:
    MOVH (R0), R4
    MOVH (R2), R5
    SXTH R4, R4
    SXTH R5, R5
    MUL R4, R5, R4
    ADD R4, R3, R3
    ADD $2, R0
    ADD $2, R2
    SUB $1, R1
    CBNZ R1, dot16_scalar_loop

dot16_done:
    MOVD R3, ret+48(FP)
    RET

// ┌──────────────────────────────────────────────────────────────────────────────┐
// │ func sumSqInt16NEON(vals []int16) int64                                      │
// │                                                                              │
// │ Strategy: SMULL/SMULL2 to square int16 pairs -> int32, then widen to int64  │
// │ Same as dotProduct but with same operand for both inputs (squaring)          │
// └──────────────────────────────────────────────────────────────────────────────┘
TEXT ·sumSqInt16NEON(SB), NOSPLIT, $0-32
    MOVD vals_base+0(FP), R0
    MOVD vals_len+8(FP), R1
    
    // Initialize 8 int64 accumulators
    VEOR V0.B16, V0.B16, V0.B16
    VEOR V1.B16, V1.B16, V1.B16
    VEOR V2.B16, V2.B16, V2.B16
    VEOR V3.B16, V3.B16, V3.B16
    VEOR V4.B16, V4.B16, V4.B16
    VEOR V5.B16, V5.B16, V5.B16
    VEOR V6.B16, V6.B16, V6.B16
    VEOR V7.B16, V7.B16, V7.B16
    
    CMP $32, R1
    BLT sumsq16_tail16

sumsq16_loop32:
    // Load 32 elements (64 bytes)
    VLD1.P 32(R0), [V16.H8, V17.H8]
    VLD1.P 32(R0), [V18.H8, V19.H8]
    
    // Square: SMULL Vd.4S, Vn.4H, Vn.4H (same operand for squaring)
    // V16^2 -> V24 (lower), V25 (upper)
    WORD $0x0E70C218              // SMULL V24.4S, V16.4H, V16.4H
    WORD $0x4E70C219              // SMULL2 V25.4S, V16.8H, V16.8H
    // V17^2 -> V26, V27
    WORD $0x0E71C23A              // SMULL V26.4S, V17.4H, V17.4H
    WORD $0x4E71C23B              // SMULL2 V27.4S, V17.8H, V17.8H
    // V18^2 -> V28, V29
    WORD $0x0E72C25C              // SMULL V28.4S, V18.4H, V18.4H
    WORD $0x4E72C25D              // SMULL2 V29.4S, V18.8H, V18.8H
    // V19^2 -> V30, V31
    WORD $0x0E73C27E              // SMULL V30.4S, V19.4H, V19.4H
    WORD $0x4E73C27F              // SMULL2 V31.4S, V19.8H, V19.8H
    
    // Widen int32 -> int64 using SADDLP and add to accumulators
    WORD $0x4EA02B18              // SADDLP V24.2D, V24.4S
    WORD $0x4EA02B39              // SADDLP V25.2D, V25.4S
    WORD $0x4EA02B5A              // SADDLP V26.2D, V26.4S
    WORD $0x4EA02B7B              // SADDLP V27.2D, V27.4S
    WORD $0x4EA02B9C              // SADDLP V28.2D, V28.4S
    WORD $0x4EA02BBD              // SADDLP V29.2D, V29.4S
    WORD $0x4EA02BDE              // SADDLP V30.2D, V30.4S
    WORD $0x4EA02BFF              // SADDLP V31.2D, V31.4S
    
    // Add to accumulators
    WORD $0x4EF88400              // ADD V0.2D, V0.2D, V24.2D
    WORD $0x4EF98421              // ADD V1.2D, V1.2D, V25.2D
    WORD $0x4EFA8442              // ADD V2.2D, V2.2D, V26.2D
    WORD $0x4EFB8463              // ADD V3.2D, V3.2D, V27.2D
    WORD $0x4EFC8484              // ADD V4.2D, V4.2D, V28.2D
    WORD $0x4EFD84A5              // ADD V5.2D, V5.2D, V29.2D
    WORD $0x4EFE84C6              // ADD V6.2D, V6.2D, V30.2D
    WORD $0x4EFF84E7              // ADD V7.2D, V7.2D, V31.2D
    
    SUB $32, R1
    CMP $32, R1
    BGE sumsq16_loop32

sumsq16_tail16:
    CMP $16, R1
    BLT sumsq16_tail8
    
    VLD1.P 32(R0), [V16.H8, V17.H8]
    
    WORD $0x0E70C218              // SMULL V24.4S, V16.4H, V16.4H
    WORD $0x4E70C219              // SMULL2 V25.4S, V16.8H, V16.8H
    WORD $0x0E71C23A              // SMULL V26.4S, V17.4H, V17.4H
    WORD $0x4E71C23B              // SMULL2 V27.4S, V17.8H, V17.8H
    
    WORD $0x4EA02B18              // SADDLP V24.2D, V24.4S
    WORD $0x4EA02B39              // SADDLP V25.2D, V25.4S
    WORD $0x4EA02B5A              // SADDLP V26.2D, V26.4S
    WORD $0x4EA02B7B              // SADDLP V27.2D, V27.4S
    
    WORD $0x4EF88400              // ADD V0.2D, V0.2D, V24.2D
    WORD $0x4EF98421              // ADD V1.2D, V1.2D, V25.2D
    WORD $0x4EFA8442              // ADD V2.2D, V2.2D, V26.2D
    WORD $0x4EFB8463              // ADD V3.2D, V3.2D, V27.2D
    SUB $16, R1

sumsq16_tail8:
    CMP $8, R1
    BLT sumsq16_reduce
    
    VLD1.P 16(R0), [V16.H8]
    
    WORD $0x0E70C218              // SMULL V24.4S, V16.4H, V16.4H
    WORD $0x4E70C219              // SMULL2 V25.4S, V16.8H, V16.8H
    
    WORD $0x4EA02B18              // SADDLP V24.2D, V24.4S
    WORD $0x4EA02B39              // SADDLP V25.2D, V25.4S
    
    WORD $0x4EF88400              // ADD V0.2D, V0.2D, V24.2D
    WORD $0x4EF98421              // ADD V1.2D, V1.2D, V25.2D
    SUB $8, R1

sumsq16_reduce:
    // Tree reduction
    WORD $0x4EE48400              // ADD V0.2D, V0.2D, V4.2D
    WORD $0x4EE58421              // ADD V1.2D, V1.2D, V5.2D
    WORD $0x4EE68442              // ADD V2.2D, V2.2D, V6.2D
    WORD $0x4EE78463              // ADD V3.2D, V3.2D, V7.2D
    WORD $0x4EE28400              // ADD V0.2D, V0.2D, V2.2D
    WORD $0x4EE38421              // ADD V1.2D, V1.2D, V3.2D
    WORD $0x4EE18400              // ADD V0.2D, V0.2D, V1.2D
    
    // Horizontal sum
    VMOV V0.D[0], R3
    VMOV V0.D[1], R4
    ADD R3, R4, R3
    
    // Handle remaining elements
    CBZ R1, sumsq16_done
    
sumsq16_scalar_loop:
    MOVH (R0), R4
    SXTH R4, R4
    MUL R4, R4, R4
    ADD R4, R3, R3
    ADD $2, R0
    SUB $1, R1
    CBNZ R1, sumsq16_scalar_loop

sumsq16_done:
    MOVD R3, ret+24(FP)
    RET

// ┌──────────────────────────────────────────────────────────────────────────────┐
// │ func anyAbsGreaterThanInt16NEON(vals []int16, threshold int16) bool          │
// │                                                                              │
// │ Strategy: Use ABS + CMGT for |val| > threshold                              │
// │ Early exit: Returns true immediately when found                              │
// └──────────────────────────────────────────────────────────────────────────────┘
TEXT ·anyAbsGreaterThanInt16NEON(SB), NOSPLIT, $0-41
    MOVD vals_base+0(FP), R0
    MOVD vals_len+8(FP), R1
    MOVH threshold+24(FP), R2
    
    CBZ R1, absgt16_notfound
    
    // Broadcast threshold to all lanes of V0.8H
    VDUP R2, V0.H8
    
    // Compute -threshold in V1.8H for the v < -threshold check
    WORD $0x6E60B801              // NEG V1.8H, V0.8H
    
    // Process 32 elements at a time (4 vectors × 8 lanes)
    CMP $32, R1
    BLT absgt16_tail16

absgt16_loop32:
    VLD1.P 64(R0), [V16.H8, V17.H8, V18.H8, V19.H8]
    
    // Check v > threshold: CMGT Vn.8H, Vn.8H, V0.8H
    WORD $0x4E603610              // CMGT V16.8H, V16.8H, V0.8H -> V16 = (val > threshold)
    WORD $0x4E603631              // CMGT V17.8H, V17.8H, V0.8H
    WORD $0x4E603652              // CMGT V18.8H, V18.8H, V0.8H
    WORD $0x4E603673              // CMGT V19.8H, V19.8H, V0.8H
    
    // For v < -threshold, we need to reload and check -threshold > v
    // Reload values for second comparison
    SUB $64, R0                    // Go back to reload
    VLD1 (R0), [V20.H8, V21.H8, V22.H8, V23.H8]
    ADD $64, R0                    // Move forward again
    
    // Check -threshold > v: CMGT Vn.8H, V1.8H, Vn.8H
    WORD $0x4E743434              // CMGT V20.8H, V1.8H, V20.8H -> V20 = (-threshold > val)
    WORD $0x4E753435              // CMGT V21.8H, V1.8H, V21.8H
    WORD $0x4E763436              // CMGT V22.8H, V1.8H, V22.8H
    WORD $0x4E773437              // CMGT V23.8H, V1.8H, V23.8H
    
    // ORR the two conditions: (v > threshold) || (v < -threshold)
    VORR V20.B16, V16.B16, V16.B16
    VORR V21.B16, V17.B16, V17.B16
    VORR V22.B16, V18.B16, V18.B16
    VORR V23.B16, V19.B16, V19.B16
    
    // ORR all comparison results together
    VORR V17.B16, V16.B16, V16.B16
    VORR V19.B16, V18.B16, V18.B16
    VORR V18.B16, V16.B16, V16.B16
    
    // UMAXV - horizontal max (non-zero means found)
    WORD $0x6E70AA10              // UMAXV H16, V16.8H
    FMOVS F16, R3
    CBNZ R3, absgt16_found
    
    SUB $32, R1
    CMP $32, R1
    BGE absgt16_loop32

absgt16_tail16:
    CMP $16, R1
    BLT absgt16_tail8
    
    VLD1 (R0), [V16.H8, V17.H8]
    
    // Check v > threshold
    WORD $0x4E603610              // CMGT V16.8H, V16.8H, V0.8H
    WORD $0x4E603631              // CMGT V17.8H, V17.8H, V0.8H
    
    // Reload and check -threshold > v
    VLD1 (R0), [V20.H8, V21.H8]
    ADD $32, R0
    WORD $0x4E743434              // CMGT V20.8H, V1.8H, V20.8H
    WORD $0x4E753435              // CMGT V21.8H, V1.8H, V21.8H
    
    // ORR conditions
    VORR V20.B16, V16.B16, V16.B16
    VORR V21.B16, V17.B16, V17.B16
    VORR V17.B16, V16.B16, V16.B16
    
    WORD $0x6E70AA10              // UMAXV H16, V16.8H
    FMOVS F16, R3
    CBNZ R3, absgt16_found
    
    SUB $16, R1

absgt16_tail8:
    CMP $8, R1
    BLT absgt16_scalar
    
    VLD1 (R0), [V16.H8]
    
    // Check v > threshold
    WORD $0x4E603610              // CMGT V16.8H, V16.8H, V0.8H
    
    // Reload and check -threshold > v
    VLD1 (R0), [V20.H8]
    ADD $16, R0
    WORD $0x4E743434              // CMGT V20.8H, V1.8H, V20.8H
    
    // ORR conditions
    VORR V20.B16, V16.B16, V16.B16
    
    WORD $0x6E70AA10              // UMAXV H16, V16.8H
    FMOVS F16, R3
    CBNZ R3, absgt16_found
    
    SUB $8, R1

absgt16_scalar:
    CBZ R1, absgt16_notfound
    NEG R2, R4
    SXTH R4, R4

absgt16_scalar_loop:
    MOVH (R0), R3
    SXTH R3, R3
    CMP R2, R3
    BGT absgt16_found
    CMP R4, R3
    BLT absgt16_found
    ADD $2, R0
    SUB $1, R1
    CBNZ R1, absgt16_scalar_loop
    B absgt16_notfound

absgt16_found:
    MOVD $1, R0
    MOVB R0, ret+32(FP)
    RET
    
absgt16_notfound:
    MOVD $0, R0
    MOVB R0, ret+32(FP)
    RET
