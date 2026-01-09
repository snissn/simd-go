//go:build arm64

#include "textflag.h"

// ╔══════════════════════════════════════════════════════════════════════════════╗
// ║                         NEON Int32 SIMD Operations                           ║
// ║                                                                              ║
// ║  NEON processes 4 x int32 per vector register (128-bit vectors)             ║
// ║                                                                              ║
// ║  Vector register layout:                                                     ║
// ║  ┌─────────────────────────────────────────────────────────────────────┐     ║
// ║  │  V0.S4 = [ lane0 | lane1 | lane2 | lane3 ]  (128 bits total)       │     ║
// ║  └─────────────────────────────────────────────────────────────────────┘     ║
// ║                                                                              ║
// ║  Key advantage over int64:                                                   ║
// ║  • NEON has native 32-bit MUL, SMIN, SMAX (unlike 64-bit!)                  ║
// ║  • 2x elements per vector = 2x throughput potential                          ║
// ╚══════════════════════════════════════════════════════════════════════════════╝

// ┌──────────────────────────────────────────────────────────────────────────────┐
// │ func sumInt32NEON(vals []int32) int64                                        │
// │                                                                              │
// │ Strategy: 16 accumulators (int64) to avoid overflow, then horizontal sum    │
// │ Processes 64 elements per iteration (16 vectors × 4 lanes)                   │
// │                                                                              │
// │ Uses SADDLP to widen pairs of int32 to int64 during accumulation            │
// └──────────────────────────────────────────────────────────────────────────────┘
TEXT ·sumInt32NEON(SB), NOSPLIT, $0-32
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
    
    // Process 32 elements at a time (8 vectors × 4 elements)
    CMP $32, R1
    BLT sum32_tail16

sum32_loop32:
    // Load 8 vectors of int32 (32 elements, 128 bytes)
    VLD1.P 64(R0), [V16.S4, V17.S4, V18.S4, V19.S4]
    VLD1.P 64(R0), [V20.S4, V21.S4, V22.S4, V23.S4]
    
    // SADDLP: Signed Add Long Pairwise - adds pairs of int32, widens to int64
    // V16.S4 = [a, b, c, d] -> V24.D2 = [a+b, c+d] as int64
    WORD $0x4EA02A18              // SADDLP V24.2D, V16.4S
    WORD $0x4EA02A39              // SADDLP V25.2D, V17.4S
    WORD $0x4EA02A5A              // SADDLP V26.2D, V18.4S
    WORD $0x4EA02A7B              // SADDLP V27.2D, V19.4S
    WORD $0x4EA02A9C              // SADDLP V28.2D, V20.4S
    WORD $0x4EA02ABD              // SADDLP V29.2D, V21.4S
    WORD $0x4EA02ADE              // SADDLP V30.2D, V22.4S
    WORD $0x4EA02AFF              // SADDLP V31.2D, V23.4S
    
    // Add widened results to int64 accumulators
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
    BGE sum32_loop32

sum32_tail16:
    CMP $16, R1
    BLT sum32_tail8
    
    VLD1.P 64(R0), [V16.S4, V17.S4, V18.S4, V19.S4]
    WORD $0x4EA02A18              // SADDLP V24.2D, V16.4S
    WORD $0x4EA02A39              // SADDLP V25.2D, V17.4S
    WORD $0x4EA02A5A              // SADDLP V26.2D, V18.4S
    WORD $0x4EA02A7B              // SADDLP V27.2D, V19.4S
    WORD $0x4EF88400              // ADD V0.2D, V0.2D, V24.2D
    WORD $0x4EF98421              // ADD V1.2D, V1.2D, V25.2D
    WORD $0x4EFA8442              // ADD V2.2D, V2.2D, V26.2D
    WORD $0x4EFB8463              // ADD V3.2D, V3.2D, V27.2D
    SUB $16, R1

sum32_tail8:
    CMP $8, R1
    BLT sum32_tail4
    
    VLD1.P 32(R0), [V16.S4, V17.S4]
    WORD $0x4EA02A18              // SADDLP V24.2D, V16.4S
    WORD $0x4EA02A39              // SADDLP V25.2D, V17.4S
    WORD $0x4EF88400              // ADD V0.2D, V0.2D, V24.2D
    WORD $0x4EF98421              // ADD V1.2D, V1.2D, V25.2D
    SUB $8, R1

sum32_tail4:
    CMP $4, R1
    BLT sum32_reduce
    
    VLD1.P 16(R0), [V16.S4]
    WORD $0x4EA02A18              // SADDLP V24.2D, V16.4S
    WORD $0x4EF88400              // ADD V0.2D, V0.2D, V24.2D
    SUB $4, R1

sum32_reduce:
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
    
    // Handle remaining 1-3 elements
    CBZ R1, sum32_done
    
sum32_scalar_loop:
    MOVW (R0), R2
    SXTW R2, R2                   // Sign extend to 64-bit
    ADD R2, R3, R3
    ADD $4, R0
    SUB $1, R1
    CBNZ R1, sum32_scalar_loop

sum32_done:
    MOVD R3, ret+24(FP)
    RET

// ┌──────────────────────────────────────────────────────────────────────────────┐
// │ func minInt32NEON(vals []int32) int32                                        │
// │                                                                              │
// │ Strategy: 8 accumulators with native SMIN instruction                        │
// │ Unlike int64, NEON has native SMIN for 32-bit!                               │
// └──────────────────────────────────────────────────────────────────────────────┘
TEXT ·minInt32NEON(SB), NOSPLIT, $0-28
    MOVD vals_base+0(FP), R0
    MOVD vals_len+8(FP), R1
    
    CBZ R1, min32_empty
    
    // Load first element and broadcast to all lanes
    MOVW (R0), R2
    VDUP R2, V0.S4
    VMOV V0.B16, V1.B16
    VMOV V0.B16, V2.B16
    VMOV V0.B16, V3.B16
    VMOV V0.B16, V4.B16
    VMOV V0.B16, V5.B16
    VMOV V0.B16, V6.B16
    VMOV V0.B16, V7.B16
    ADD $4, R0
    SUB $1, R1
    
    CMP $32, R1
    BLT min32_tail16

min32_loop32:
    VLD1.P 64(R0), [V16.S4, V17.S4, V18.S4, V19.S4]
    VLD1.P 64(R0), [V20.S4, V21.S4, V22.S4, V23.S4]
    
    // Native SMIN.4S - no emulation needed!
    WORD $0x4EB06C00              // SMIN V0.4S, V0.4S, V16.4S
    WORD $0x4EB16C21              // SMIN V1.4S, V1.4S, V17.4S
    WORD $0x4EB26C42              // SMIN V2.4S, V2.4S, V18.4S
    WORD $0x4EB36C63              // SMIN V3.4S, V3.4S, V19.4S
    WORD $0x4EB46C84              // SMIN V4.4S, V4.4S, V20.4S
    WORD $0x4EB56CA5              // SMIN V5.4S, V5.4S, V21.4S
    WORD $0x4EB66CC6              // SMIN V6.4S, V6.4S, V22.4S
    WORD $0x4EB76CE7              // SMIN V7.4S, V7.4S, V23.4S
    
    SUB $32, R1
    CMP $32, R1
    BGE min32_loop32

min32_tail16:
    CMP $16, R1
    BLT min32_tail8
    
    VLD1.P 64(R0), [V16.S4, V17.S4, V18.S4, V19.S4]
    WORD $0x4EB06C00              // SMIN V0.4S, V0.4S, V16.4S
    WORD $0x4EB16C21              // SMIN V1.4S, V1.4S, V17.4S
    WORD $0x4EB26C42              // SMIN V2.4S, V2.4S, V18.4S
    WORD $0x4EB36C63              // SMIN V3.4S, V3.4S, V19.4S
    SUB $16, R1

min32_tail8:
    CMP $8, R1
    BLT min32_tail4
    
    VLD1.P 32(R0), [V16.S4, V17.S4]
    WORD $0x4EB06C00              // SMIN V0.4S, V0.4S, V16.4S
    WORD $0x4EB16C21              // SMIN V1.4S, V1.4S, V17.4S
    SUB $8, R1

min32_tail4:
    CMP $4, R1
    BLT min32_reduce
    
    VLD1.P 16(R0), [V16.S4]
    WORD $0x4EB06C00              // SMIN V0.4S, V0.4S, V16.4S
    SUB $4, R1

min32_reduce:
    // Tree reduction: 8 -> 4 -> 2 -> 1
    WORD $0x4EA46C00              // SMIN V0.4S, V0.4S, V4.4S
    WORD $0x4EA56C21              // SMIN V1.4S, V1.4S, V5.4S
    WORD $0x4EA66C42              // SMIN V2.4S, V2.4S, V6.4S
    WORD $0x4EA76C63              // SMIN V3.4S, V3.4S, V7.4S
    WORD $0x4EA26C00              // SMIN V0.4S, V0.4S, V2.4S
    WORD $0x4EA36C21              // SMIN V1.4S, V1.4S, V3.4S
    WORD $0x4EA16C00              // SMIN V0.4S, V0.4S, V1.4S
    
    // SMINV: horizontal minimum across all lanes
    WORD $0x4EB1A800              // SMINV S0, V0.4S
    
    // Handle remaining 1-3 elements
    CBZ R1, min32_store
    FMOVS F0, R2
    SXTW R2, R2                   // Sign extend W2 to X2 for comparison

min32_scalar_loop:
    MOVW (R0), R3
    SXTW R3, R3
    CMP R2, R3
    CSEL LT, R3, R2, R2
    ADD $4, R0
    SUB $1, R1
    CBNZ R1, min32_scalar_loop
    B min32_done

min32_store:
    FMOVS F0, R2
    SXTW R2, R2                   // Sign extend for consistency
    B min32_done

min32_empty:
    MOVW $0, R2

min32_done:
    MOVW R2, ret+24(FP)
    RET

// ┌──────────────────────────────────────────────────────────────────────────────┐
// │ func maxInt32NEON(vals []int32) int32                                        │
// │                                                                              │
// │ Strategy: 8 accumulators with native SMAX instruction                        │
// └──────────────────────────────────────────────────────────────────────────────┘
TEXT ·maxInt32NEON(SB), NOSPLIT, $0-28
    MOVD vals_base+0(FP), R0
    MOVD vals_len+8(FP), R1
    
    CBZ R1, max32_empty
    
    // Load first element and broadcast to all lanes
    MOVW (R0), R2
    VDUP R2, V0.S4
    VMOV V0.B16, V1.B16
    VMOV V0.B16, V2.B16
    VMOV V0.B16, V3.B16
    VMOV V0.B16, V4.B16
    VMOV V0.B16, V5.B16
    VMOV V0.B16, V6.B16
    VMOV V0.B16, V7.B16
    ADD $4, R0
    SUB $1, R1
    
    CMP $32, R1
    BLT max32_tail16

max32_loop32:
    VLD1.P 64(R0), [V16.S4, V17.S4, V18.S4, V19.S4]
    VLD1.P 64(R0), [V20.S4, V21.S4, V22.S4, V23.S4]
    
    // Native SMAX.4S
    WORD $0x4EB06400              // SMAX V0.4S, V0.4S, V16.4S
    WORD $0x4EB16421              // SMAX V1.4S, V1.4S, V17.4S
    WORD $0x4EB26442              // SMAX V2.4S, V2.4S, V18.4S
    WORD $0x4EB36463              // SMAX V3.4S, V3.4S, V19.4S
    WORD $0x4EB46484              // SMAX V4.4S, V4.4S, V20.4S
    WORD $0x4EB564A5              // SMAX V5.4S, V5.4S, V21.4S
    WORD $0x4EB664C6              // SMAX V6.4S, V6.4S, V22.4S
    WORD $0x4EB764E7              // SMAX V7.4S, V7.4S, V23.4S
    
    SUB $32, R1
    CMP $32, R1
    BGE max32_loop32

max32_tail16:
    CMP $16, R1
    BLT max32_tail8
    
    VLD1.P 64(R0), [V16.S4, V17.S4, V18.S4, V19.S4]
    WORD $0x4EB06400              // SMAX V0.4S, V0.4S, V16.4S
    WORD $0x4EB16421              // SMAX V1.4S, V1.4S, V17.4S
    WORD $0x4EB26442              // SMAX V2.4S, V2.4S, V18.4S
    WORD $0x4EB36463              // SMAX V3.4S, V3.4S, V19.4S
    SUB $16, R1

max32_tail8:
    CMP $8, R1
    BLT max32_tail4
    
    VLD1.P 32(R0), [V16.S4, V17.S4]
    WORD $0x4EB06400              // SMAX V0.4S, V0.4S, V16.4S
    WORD $0x4EB16421              // SMAX V1.4S, V1.4S, V17.4S
    SUB $8, R1

max32_tail4:
    CMP $4, R1
    BLT max32_reduce
    
    VLD1.P 16(R0), [V16.S4]
    WORD $0x4EB06400              // SMAX V0.4S, V0.4S, V16.4S
    SUB $4, R1

max32_reduce:
    // Tree reduction
    WORD $0x4EA46400              // SMAX V0.4S, V0.4S, V4.4S
    WORD $0x4EA56421              // SMAX V1.4S, V1.4S, V5.4S
    WORD $0x4EA66442              // SMAX V2.4S, V2.4S, V6.4S
    WORD $0x4EA76463              // SMAX V3.4S, V3.4S, V7.4S
    WORD $0x4EA26400              // SMAX V0.4S, V0.4S, V2.4S
    WORD $0x4EA36421              // SMAX V1.4S, V1.4S, V3.4S
    WORD $0x4EA16400              // SMAX V0.4S, V0.4S, V1.4S
    
    // SMAXV: horizontal maximum
    WORD $0x4EB0A800              // SMAXV S0, V0.4S
    
    CBZ R1, max32_store
    FMOVS F0, R2
    SXTW R2, R2                   // Sign extend W2 to X2 for comparison

max32_scalar_loop:
    MOVW (R0), R3
    SXTW R3, R3
    CMP R2, R3
    CSEL GT, R3, R2, R2
    ADD $4, R0
    SUB $1, R1
    CBNZ R1, max32_scalar_loop
    B max32_done

max32_store:
    FMOVS F0, R2
    SXTW R2, R2                   // Sign extend for consistency
    B max32_done

max32_empty:
    MOVW $0, R2

max32_done:
    MOVW R2, ret+24(FP)
    RET

// ┌──────────────────────────────────────────────────────────────────────────────┐
// │ func dotProductInt32NEON(a, b []int32) int64                                 │
// │                                                                              │
// │ Strategy: SMULL/SMULL2 for widening multiply (int32×int32→int64)              │
// └──────────────────────────────────────────────────────────────────────────────┘
TEXT ·dotProductInt32NEON(SB), NOSPLIT, $0-56
    MOVD a_base+0(FP), R0
    MOVD a_len+8(FP), R1
    MOVD b_base+24(FP), R2
    
    // Initialize 4 int64 accumulators
    VEOR V0.B16, V0.B16, V0.B16
    VEOR V1.B16, V1.B16, V1.B16
    VEOR V2.B16, V2.B16, V2.B16
    VEOR V3.B16, V3.B16, V3.B16
    
    CMP $8, R1
    BLT dot32_tail4

dot32_loop8:
    // Load 8 elements from each array (2 vectors)
    VLD1.P 32(R0), [V16.S4, V17.S4]
    VLD1.P 32(R2), [V20.S4, V21.S4]
    
    // SMULL/SMULL2: Signed Multiply Long - int32×int32→int64
    WORD $0x0EB4C218              // SMULL V24.2D, V16.2S, V20.2S  (low halves)
    WORD $0x4EB4C219              // SMULL2 V25.2D, V16.4S, V20.4S (high halves)
    WORD $0x0EB5C23A              // SMULL V26.2D, V17.2S, V21.2S
    WORD $0x4EB5C23B              // SMULL2 V27.2D, V17.4S, V21.4S
    
    // Add to int64 accumulators
    WORD $0x4EF88400              // ADD V0.2D, V0.2D, V24.2D
    WORD $0x4EF98421              // ADD V1.2D, V1.2D, V25.2D
    WORD $0x4EFA8442              // ADD V2.2D, V2.2D, V26.2D
    WORD $0x4EFB8463              // ADD V3.2D, V3.2D, V27.2D
    
    SUB $8, R1
    CMP $8, R1
    BGE dot32_loop8

dot32_tail4:
    CMP $4, R1
    BLT dot32_tail2
    
    VLD1.P 16(R0), [V16.S4]
    VLD1.P 16(R2), [V20.S4]
    WORD $0x0EB4C218              // SMULL V24.2D, V16.2S, V20.2S
    WORD $0x4EB4C219              // SMULL2 V25.2D, V16.4S, V20.4S
    WORD $0x4EF88400              // ADD V0.2D, V0.2D, V24.2D
    WORD $0x4EF98421              // ADD V1.2D, V1.2D, V25.2D
    SUB $4, R1

dot32_tail2:
    CMP $2, R1
    BLT dot32_reduce
    
    // Load 2 elements from each array
    MOVW (R0), R3
    MOVW 4(R0), R4
    VMOV R3, V16.S[0]
    VMOV R4, V16.S[1]
    MOVW (R2), R3
    MOVW 4(R2), R4
    VMOV R3, V20.S[0]
    VMOV R4, V20.S[1]
    WORD $0x0EB4C218              // SMULL V24.2D, V16.2S, V20.2S
    WORD $0x4EF88400              // ADD V0.2D, V0.2D, V24.2D
    ADD $8, R0
    ADD $8, R2
    SUB $2, R1

dot32_reduce:
    // Tree reduction: 4 -> 2 -> 1
    WORD $0x4EE28400              // ADD V0.2D, V0.2D, V2.2D
    WORD $0x4EE38421              // ADD V1.2D, V1.2D, V3.2D
    WORD $0x4EE18400              // ADD V0.2D, V0.2D, V1.2D
    
    // Horizontal sum
    VMOV V0.D[0], R3
    VMOV V0.D[1], R4
    ADD R3, R4, R3
    
    // Handle remaining elements
    CBZ R1, dot32_done
    
dot32_scalar_loop:
    MOVW (R0), R4
    MOVW (R2), R5
    SXTW R4, R4
    SXTW R5, R5
    MUL R4, R5, R4
    ADD R4, R3, R3
    ADD $4, R0
    ADD $4, R2
    SUB $1, R1
    CBNZ R1, dot32_scalar_loop

dot32_done:
    MOVD R3, ret+48(FP)
    RET

// ┌──────────────────────────────────────────────────────────────────────────────┐
// │ func sumSqInt32NEON(vals []int32) int64                                      │
// │                                                                              │
// │ Strategy: SMULL/SMULL2 for widening square (int32×int32→int64)               │
// │           SMULL takes low 2 elements, SMULL2 takes high 2 elements           │
// └──────────────────────────────────────────────────────────────────────────────┘
TEXT ·sumSqInt32NEON(SB), NOSPLIT, $0-32
    MOVD vals_base+0(FP), R0
    MOVD vals_len+8(FP), R1
    
    // Initialize 4 int64 accumulators
    VEOR V0.B16, V0.B16, V0.B16
    VEOR V1.B16, V1.B16, V1.B16
    VEOR V2.B16, V2.B16, V2.B16
    VEOR V3.B16, V3.B16, V3.B16
    
    CMP $8, R1
    BLT sumsq32_tail4

sumsq32_loop8:
    // Load 2 vectors (8 elements)
    VLD1.P 32(R0), [V16.S4, V17.S4]
    
    // SMULL/SMULL2: Signed Multiply Long - int32×int32→int64
    // SMULL  Vd.2D, Vn.2S, Vm.2S - multiplies low 2 elements
    // SMULL2 Vd.2D, Vn.4S, Vm.4S - multiplies high 2 elements
    WORD $0x0EB0C218              // SMULL V24.2D, V16.2S, V16.2S  (low half of V16)
    WORD $0x4EB0C219              // SMULL2 V25.2D, V16.4S, V16.4S (high half of V16)
    WORD $0x0EB1C23A              // SMULL V26.2D, V17.2S, V17.2S  (low half of V17)
    WORD $0x4EB1C23B              // SMULL2 V27.2D, V17.4S, V17.4S (high half of V17)
    
    // Add to int64 accumulators
    WORD $0x4EF88400              // ADD V0.2D, V0.2D, V24.2D
    WORD $0x4EF98421              // ADD V1.2D, V1.2D, V25.2D
    WORD $0x4EFA8442              // ADD V2.2D, V2.2D, V26.2D
    WORD $0x4EFB8463              // ADD V3.2D, V3.2D, V27.2D
    
    SUB $8, R1
    CMP $8, R1
    BGE sumsq32_loop8

sumsq32_tail4:
    CMP $4, R1
    BLT sumsq32_tail2
    
    VLD1.P 16(R0), [V16.S4]
    WORD $0x0EB0C218              // SMULL V24.2D, V16.2S, V16.2S
    WORD $0x4EB0C219              // SMULL2 V25.2D, V16.4S, V16.4S
    WORD $0x4EF88400              // ADD V0.2D, V0.2D, V24.2D
    WORD $0x4EF98421              // ADD V1.2D, V1.2D, V25.2D
    SUB $4, R1

sumsq32_tail2:
    CMP $2, R1
    BLT sumsq32_reduce
    
    // Load 2 elements into low half of V16
    MOVW (R0), R2
    MOVW 4(R0), R3
    VMOV R2, V16.S[0]
    VMOV R3, V16.S[1]
    WORD $0x0EB0C218              // SMULL V24.2D, V16.2S, V16.2S
    WORD $0x4EF88400              // ADD V0.2D, V0.2D, V24.2D
    ADD $8, R0
    SUB $2, R1

sumsq32_reduce:
    // Tree reduction: 4 -> 2 -> 1
    WORD $0x4EE28400              // ADD V0.2D, V0.2D, V2.2D
    WORD $0x4EE38421              // ADD V1.2D, V1.2D, V3.2D
    WORD $0x4EE18400              // ADD V0.2D, V0.2D, V1.2D
    
    // Horizontal sum
    VMOV V0.D[0], R3
    VMOV V0.D[1], R4
    ADD R3, R4, R3
    
    CBZ R1, sumsq32_done
    
sumsq32_scalar_loop:
    MOVW (R0), R4
    SXTW R4, R4
    MUL R4, R4, R4
    ADD R4, R3, R3
    ADD $4, R0
    SUB $1, R1
    CBNZ R1, sumsq32_scalar_loop

sumsq32_done:
    MOVD R3, ret+24(FP)
    RET

// ┌──────────────────────────────────────────────────────────────────────────────┐
// │ func anyAbsGreaterThanInt32NEON(vals []int32, threshold int32) bool          │
// │                                                                              │
// │ Strategy: Use CMGT for |val| > threshold by checking val > threshold OR     │
// │           val < -threshold. Uses ABS + CMGT for simpler logic.               │
// │                                                                              │
// │ Early exit: Returns true immediately when found                              │
// └──────────────────────────────────────────────────────────────────────────────┘
TEXT ·anyAbsGreaterThanInt32NEON(SB), NOSPLIT, $0-25
    MOVD vals_base+0(FP), R0
    MOVD vals_len+8(FP), R1
    MOVW threshold+24(FP), R2
    
    CBZ R1, absgt32_notfound
    
    // Broadcast threshold to all lanes of V0
    VDUP R2, V0.S4
    
    // Process 16 elements at a time (4 vectors)
    CMP $16, R1
    BLT absgt32_tail8

absgt32_loop16:
    VLD1.P 64(R0), [V16.S4, V17.S4, V18.S4, V19.S4]
    
    // ABS - compute absolute values
    WORD $0x4EA0BA10              // ABS V16.4S, V16.4S
    WORD $0x4EA0BA31              // ABS V17.4S, V17.4S
    WORD $0x4EA0BA52              // ABS V18.4S, V18.4S
    WORD $0x4EA0BA73              // ABS V19.4S, V19.4S
    
    // CMGT - compare greater than threshold
    WORD $0x4EA03610              // CMGT V16.4S, V16.4S, V0.4S
    WORD $0x4EA03631              // CMGT V17.4S, V17.4S, V0.4S
    WORD $0x4EA03652              // CMGT V18.4S, V18.4S, V0.4S
    WORD $0x4EA03673              // CMGT V19.4S, V19.4S, V0.4S
    
    // ORR all comparison results
    VORR V17.B16, V16.B16, V16.B16
    VORR V19.B16, V18.B16, V18.B16
    VORR V18.B16, V16.B16, V16.B16
    
    // UMAXV - horizontal max (non-zero means found)
    WORD $0x6EB0A610              // UMAXV S16, V16.4S
    FMOVS F16, R3
    CBNZ R3, absgt32_found
    
    SUB $16, R1
    CMP $16, R1
    BGE absgt32_loop16

absgt32_tail8:
    CMP $8, R1
    BLT absgt32_tail4
    
    VLD1.P 32(R0), [V16.S4, V17.S4]
    
    WORD $0x4EA0BA10              // ABS V16.4S, V16.4S
    WORD $0x4EA0BA31              // ABS V17.4S, V17.4S
    WORD $0x4EA03610              // CMGT V16.4S, V16.4S, V0.4S
    WORD $0x4EA03631              // CMGT V17.4S, V17.4S, V0.4S
    VORR V17.B16, V16.B16, V16.B16
    
    WORD $0x6EB0A610              // UMAXV S16, V16.4S
    FMOVS F16, R3
    CBNZ R3, absgt32_found
    
    SUB $8, R1

absgt32_tail4:
    CMP $4, R1
    BLT absgt32_scalar
    
    VLD1.P 16(R0), [V16.S4]
    
    WORD $0x4EA0BA10              // ABS V16.4S, V16.4S
    WORD $0x4EA03610              // CMGT V16.4S, V16.4S, V0.4S
    
    WORD $0x6EB0A610              // UMAXV S16, V16.4S
    FMOVS F16, R3
    CBNZ R3, absgt32_found
    
    SUB $4, R1

absgt32_scalar:
    CBZ R1, absgt32_notfound
    NEG R2, R4

absgt32_scalar_loop:
    MOVW (R0), R3
    SXTW R3, R3
    CMP R2, R3
    BGT absgt32_found
    CMP R4, R3
    BLT absgt32_found
    ADD $4, R0
    SUB $1, R1
    CBNZ R1, absgt32_scalar_loop
    B absgt32_notfound

absgt32_found:
    MOVD $1, R0
    MOVB R0, ret+24(FP)
    RET
    
absgt32_notfound:
    MOVD $0, R0
    MOVB R0, ret+24(FP)
    RET
