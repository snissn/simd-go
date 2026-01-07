//go:build arm64

#include "textflag.h"

// ╔══════════════════════════════════════════════════════════════════════════════╗
// ║                         NEON Float64 SIMD Operations                         ║
// ║                                                                              ║
// ║  NEON processes 2 x float64 per vector register (128-bit vectors)           ║
// ║                                                                              ║
// ║  Vector register layout:                                                     ║
// ║  ┌─────────────────────────────────────────────────────────────────────┐     ║
// ║  │  V0.D2 = [ lane0: float64 | lane1: float64 ]  (128 bits total)     │     ║
// ║  └─────────────────────────────────────────────────────────────────────┘     ║
// ╚══════════════════════════════════════════════════════════════════════════════╝

// ┌──────────────────────────────────────────────────────────────────────────────┐
// │ func sumFloat64NEON(vals []float64) float64                                  │
// │                                                                              │
// │ Strategy: 16 parallel accumulators to hide memory latency                    │
// │ Processes 32 elements per iteration (16 vectors × 2 lanes)                   │
// │                                                                              │
// │ Memory layout for 32 elements:                                               │
// │ ┌────┬────┬────┬────┬────┬────┬────┬────┬─...─┬────┬────┬────┬────┐          │
// │ │ e0 │ e1 │ e2 │ e3 │ e4 │ e5 │ e6 │ e7 │     │e28 │e29 │e30 │e31 │          │
// │ └────┴────┴────┴────┴────┴────┴────┴────┴─...─┴────┴────┴────┴────┘          │
// │   ↓    ↓    ↓    ↓    ↓    ↓    ↓    ↓           ↓    ↓    ↓    ↓            │
// │  V16  V16  V17  V17  V18  V18  V19  V19  ...   V30  V30  V31  V31            │
// │  [0]  [1]  [0]  [1]  [0]  [1]  [0]  [1]        [0]  [1]  [0]  [1]            │
// └──────────────────────────────────────────────────────────────────────────────┘
TEXT ·sumFloat64NEON(SB), NOSPLIT, $0-32
    MOVD vals_base+0(FP), R0      // R0 = pointer to vals[0]
    MOVD vals_len+8(FP), R1       // R1 = len(vals)
    
    // ╔════════════════════════════════════════════════════════════════════════╗
    // ║ Initialize 16 accumulators to zero                                     ║
    // ║                                                                        ║
    // ║  V0  = [0.0 | 0.0]    V4  = [0.0 | 0.0]    V8  = [0.0 | 0.0]    V12 = [0.0 | 0.0] ║
    // ║  V1  = [0.0 | 0.0]    V5  = [0.0 | 0.0]    V9  = [0.0 | 0.0]    V13 = [0.0 | 0.0] ║
    // ║  V2  = [0.0 | 0.0]    V6  = [0.0 | 0.0]    V10 = [0.0 | 0.0]    V14 = [0.0 | 0.0] ║
    // ║  V3  = [0.0 | 0.0]    V7  = [0.0 | 0.0]    V11 = [0.0 | 0.0]    V15 = [0.0 | 0.0] ║
    // ╚════════════════════════════════════════════════════════════════════════╝
    VEOR V0.B16, V0.B16, V0.B16
    VEOR V1.B16, V1.B16, V1.B16
    VEOR V2.B16, V2.B16, V2.B16
    VEOR V3.B16, V3.B16, V3.B16
    VEOR V4.B16, V4.B16, V4.B16
    VEOR V5.B16, V5.B16, V5.B16
    VEOR V6.B16, V6.B16, V6.B16
    VEOR V7.B16, V7.B16, V7.B16
    VEOR V8.B16, V8.B16, V8.B16
    VEOR V9.B16, V9.B16, V9.B16
    VEOR V10.B16, V10.B16, V10.B16
    VEOR V11.B16, V11.B16, V11.B16
    VEOR V12.B16, V12.B16, V12.B16
    VEOR V13.B16, V13.B16, V13.B16
    VEOR V14.B16, V14.B16, V14.B16
    VEOR V15.B16, V15.B16, V15.B16
    
    CMP $32, R1
    BLT sum_tail16
    
// ┌──────────────────────────────────────────────────────────────────────────────┐
// │ Main loop: Process 32 elements per iteration                                 │
// │                                                                              │
// │ Load pattern (4 consecutive 64-byte loads = 32 elements):                    │
// │                                                                              │
// │   Memory:  ═══╦═══╦═══╦═══╦═══╦═══╦═══╦═══╦═══╦═══╦═══╦═══╦═══╦═══╦═══╦═══╗  │
// │               ║V16║V17║V18║V19║V20║V21║V22║V23║V24║V25║V26║V27║V28║V29║V30║V31║
// │            ═══╩═══╩═══╩═══╩═══╩═══╩═══╩═══╩═══╩═══╩═══╩═══╩═══╩═══╩═══╩═══╝  │
// │               │ 64 bytes │ 64 bytes │ 64 bytes │ 64 bytes │                  │
// │               └──────────┴──────────┴──────────┴──────────┘                  │
// │                                                                              │
// │ Accumulation (16 independent chains for ILP):                                │
// │   V0  += V16     V4  += V20     V8  += V24     V12 += V28                    │
// │   V1  += V17     V5  += V21     V9  += V25     V13 += V29                    │
// │   V2  += V18     V6  += V22     V10 += V26     V14 += V30                    │
// │   V3  += V19     V7  += V23     V11 += V27     V15 += V31                    │
// └──────────────────────────────────────────────────────────────────────────────┘
sum_loop32:
    // Load 32 elements (256 bytes total) into V16-V31
    VLD1.P 64(R0), [V16.D2, V17.D2, V18.D2, V19.D2]   // Load e[0:7]
    VLD1.P 64(R0), [V20.D2, V21.D2, V22.D2, V23.D2]   // Load e[8:15]
    VLD1.P 64(R0), [V24.D2, V25.D2, V26.D2, V27.D2]   // Load e[16:23]
    VLD1.P 64(R0), [V28.D2, V29.D2, V30.D2, V31.D2]   // Load e[24:31]
    
    // ┌────────────────────────────────────────────────────────────────────┐
    // │ FADD: Accumulate loaded values into 16 independent accumulators   │
    // │                                                                    │
    // │   V0  ← V0  + V16      Each accumulator is independent,           │
    // │   V1  ← V1  + V17      allowing out-of-order execution            │
    // │   V2  ← V2  + V18      to process them in parallel                │
    // │   ...                                                              │
    // │   V15 ← V15 + V31                                                  │
    // └────────────────────────────────────────────────────────────────────┘
    WORD $0x4E70D400              // FADD V0.2D, V0.2D, V16.2D
    WORD $0x4E71D421              // FADD V1.2D, V1.2D, V17.2D
    WORD $0x4E72D442              // FADD V2.2D, V2.2D, V18.2D
    WORD $0x4E73D463              // FADD V3.2D, V3.2D, V19.2D
    WORD $0x4E74D484              // FADD V4.2D, V4.2D, V20.2D
    WORD $0x4E75D4A5              // FADD V5.2D, V5.2D, V21.2D
    WORD $0x4E76D4C6              // FADD V6.2D, V6.2D, V22.2D
    WORD $0x4E77D4E7              // FADD V7.2D, V7.2D, V23.2D
    WORD $0x4E78D508              // FADD V8.2D, V8.2D, V24.2D
    WORD $0x4E79D529              // FADD V9.2D, V9.2D, V25.2D
    WORD $0x4E7AD54A              // FADD V10.2D, V10.2D, V26.2D
    WORD $0x4E7BD56B              // FADD V11.2D, V11.2D, V27.2D
    WORD $0x4E7CD58C              // FADD V12.2D, V12.2D, V28.2D
    WORD $0x4E7DD5AD              // FADD V13.2D, V13.2D, V29.2D
    WORD $0x4E7ED5CE              // FADD V14.2D, V14.2D, V30.2D
    WORD $0x4E7FD5EF              // FADD V15.2D, V15.2D, V31.2D
    
    SUB $32, R1
    CMP $32, R1
    BGE sum_loop32
    
    // ╔════════════════════════════════════════════════════════════════════════╗
    // ║ Tree reduction: Combine 16 accumulators → 1                            ║
    // ║                                                                        ║
    // ║ Step 1: 16 → 8                                                         ║
    // ║   V0 ═╦═ V8      V1 ═╦═ V9      V2 ═╦═ V10     V3 ═╦═ V11               ║
    // ║      ╚═► V0         ╚═► V1         ╚═► V2         ╚═► V3               ║
    // ║   V4 ═╦═ V12     V5 ═╦═ V13     V6 ═╦═ V14     V7 ═╦═ V15              ║
    // ║      ╚═► V4         ╚═► V5         ╚═► V6         ╚═► V7               ║
    // ║                                                                        ║
    // ║ Step 2: 8 → 4                                                          ║
    // ║   V0 ═══╦═══ V4     V1 ═══╦═══ V5     V2 ═══╦═══ V6     V3 ═══╦═══ V7  ║
    // ║        ╚════► V0         ╚════► V1         ╚════► V2         ╚════► V3 ║
    // ║                                                                        ║
    // ║ Step 3: 4 → 2                                                          ║
    // ║   V0 ═══════╦═══════ V2          V1 ═══════╦═══════ V3                 ║
    // ║            ╚════════► V0                  ╚════════► V1                ║
    // ║                                                                        ║
    // ║ Step 4: 2 → 1                                                          ║
    // ║   V0 ═══════════════╦═══════════════ V1                                ║
    // ║                    ╚════════════════► V0                               ║
    // ╚════════════════════════════════════════════════════════════════════════╝
    
    // 16 → 8
    WORD $0x4E68D400              // FADD V0.2D, V0.2D, V8.2D
    WORD $0x4E69D421              // FADD V1.2D, V1.2D, V9.2D
    WORD $0x4E6AD442              // FADD V2.2D, V2.2D, V10.2D
    WORD $0x4E6BD463              // FADD V3.2D, V3.2D, V11.2D
    WORD $0x4E6CD484              // FADD V4.2D, V4.2D, V12.2D
    WORD $0x4E6DD4A5              // FADD V5.2D, V5.2D, V13.2D
    WORD $0x4E6ED4C6              // FADD V6.2D, V6.2D, V14.2D
    WORD $0x4E6FD4E7              // FADD V7.2D, V7.2D, V15.2D
    // 8 → 4
    WORD $0x4E64D400              // FADD V0.2D, V0.2D, V4.2D
    WORD $0x4E65D421              // FADD V1.2D, V1.2D, V5.2D
    WORD $0x4E66D442              // FADD V2.2D, V2.2D, V6.2D
    WORD $0x4E67D463              // FADD V3.2D, V3.2D, V7.2D
    // 4 → 2
    WORD $0x4E62D400              // FADD V0.2D, V0.2D, V2.2D
    WORD $0x4E63D421              // FADD V1.2D, V1.2D, V3.2D
    // 2 → 1
    WORD $0x4E61D400              // FADD V0.2D, V0.2D, V1.2D
    
// ┌──────────────────────────────────────────────────────────────────────────────┐
// │ Tail handling: Process remaining 16, 8, 4, 2, 1 elements                     │
// └──────────────────────────────────────────────────────────────────────────────┘
sum_tail16:
    CMP $16, R1
    BLT sum_tail8
    
    VLD1.P 64(R0), [V16.D2, V17.D2, V18.D2, V19.D2]
    VLD1.P 64(R0), [V20.D2, V21.D2, V22.D2, V23.D2]
    WORD $0x4E70D400              // FADD V0.2D, V0.2D, V16.2D
    WORD $0x4E71D400              // FADD V0.2D, V0.2D, V17.2D
    WORD $0x4E72D400              // FADD V0.2D, V0.2D, V18.2D
    WORD $0x4E73D400              // FADD V0.2D, V0.2D, V19.2D
    WORD $0x4E74D400              // FADD V0.2D, V0.2D, V20.2D
    WORD $0x4E75D400              // FADD V0.2D, V0.2D, V21.2D
    WORD $0x4E76D400              // FADD V0.2D, V0.2D, V22.2D
    WORD $0x4E77D400              // FADD V0.2D, V0.2D, V23.2D
    SUB $16, R1

sum_tail8:
    CMP $8, R1
    BLT sum_tail4
    
    VLD1.P 64(R0), [V16.D2, V17.D2, V18.D2, V19.D2]
    WORD $0x4E70D400              // FADD V0.2D, V0.2D, V16.2D
    WORD $0x4E71D400              // FADD V0.2D, V0.2D, V17.2D
    WORD $0x4E72D400              // FADD V0.2D, V0.2D, V18.2D
    WORD $0x4E73D400              // FADD V0.2D, V0.2D, V19.2D
    SUB $8, R1

sum_tail4:
    CMP $4, R1
    BLT sum_tail2
    
    VLD1.P 32(R0), [V16.D2, V17.D2]
    WORD $0x4E70D400              // FADD V0.2D, V0.2D, V16.2D
    WORD $0x4E71D400              // FADD V0.2D, V0.2D, V17.2D
    SUB $4, R1
    
sum_tail2:
    CMP $2, R1
    BLT sum_reduce
    
    VLD1.P 16(R0), [V16.D2]
    WORD $0x4E70D400              // FADD V0.2D, V0.2D, V16.2D
    SUB $2, R1
    
sum_reduce:
    // ┌────────────────────────────────────────────────────────────────────┐
    // │ Horizontal reduction: Add both lanes of V0                        │
    // │                                                                    │
    // │   V0 = [ a | b ]                                                   │
    // │          ╲   ╱                                                     │
    // │           ╲ ╱   FADDP (pairwise add)                               │
    // │            ╳                                                       │
    // │           ╱ ╲                                                      │
    // │   D0 = [ a + b ]                                                   │
    // └────────────────────────────────────────────────────────────────────┘
    WORD $0x7E70D800              // FADDP D0, V0.2D
    
    // Handle remaining 1 element (scalar add)
    CMP $1, R1
    BLT sum_done
    FMOVD (R0), F1
    FADDD F0, F1, F0
    
sum_done:
    FMOVD F0, ret+24(FP)
    RET

// ┌──────────────────────────────────────────────────────────────────────────────┐
// │ func minFloat64NEON(vals []float64) float64                                  │
// │                                                                              │
// │ Strategy: Same 16-accumulator pattern as sum, but with FMIN                  │
// │ Initialize all accumulators with first element, then take pairwise min       │
// └──────────────────────────────────────────────────────────────────────────────┘
TEXT ·minFloat64NEON(SB), NOSPLIT, $0-32
    MOVD vals_base+0(FP), R0
    MOVD vals_len+8(FP), R1
    
    // ┌────────────────────────────────────────────────────────────────────┐
    // │ Broadcast first element to all 16 accumulators                    │
    // │                                                                    │
    // │   vals[0] ──────────────────────────────────────────────┐          │
    // │            ╔═══════╦═══════╦═══════╦═══════╦═══════╦═══╧═══╗      │
    // │            ║  V0   ║  V1   ║  V2   ║  ...  ║  V14  ║  V15  ║      │
    // │            ║[v|v]  ║[v|v]  ║[v|v]  ║       ║[v|v]  ║[v|v]  ║      │
    // │            ╚═══════╩═══════╩═══════╩═══════╩═══════╩═══════╝      │
    // └────────────────────────────────────────────────────────────────────┘
    FMOVD (R0), F0
    VDUP V0.D[0], V0.D2
    VMOV V0.B16, V1.B16
    VMOV V0.B16, V2.B16
    VMOV V0.B16, V3.B16
    VMOV V0.B16, V4.B16
    VMOV V0.B16, V5.B16
    VMOV V0.B16, V6.B16
    VMOV V0.B16, V7.B16
    VMOV V0.B16, V8.B16
    VMOV V0.B16, V9.B16
    VMOV V0.B16, V10.B16
    VMOV V0.B16, V11.B16
    VMOV V0.B16, V12.B16
    VMOV V0.B16, V13.B16
    VMOV V0.B16, V14.B16
    VMOV V0.B16, V15.B16
    ADD $8, R0
    SUB $1, R1
    
    CMP $32, R1
    BLT min_tail16
    
// ┌──────────────────────────────────────────────────────────────────────────────┐
// │ Main loop: FMIN operation per accumulator                                    │
// │                                                                              │
// │   Before:  V0 = [ 5.0 | 3.0 ]    V16 = [ 2.0 | 7.0 ]                         │
// │                    │       │             │       │                           │
// │   FMIN:          min(5,2) min(3,7)                                           │
// │                    │       │                                                 │
// │   After:   V0 = [ 2.0 | 3.0 ]                                                │
// └──────────────────────────────────────────────────────────────────────────────┘
min_loop32:
    VLD1.P 64(R0), [V16.D2, V17.D2, V18.D2, V19.D2]
    VLD1.P 64(R0), [V20.D2, V21.D2, V22.D2, V23.D2]
    VLD1.P 64(R0), [V24.D2, V25.D2, V26.D2, V27.D2]
    VLD1.P 64(R0), [V28.D2, V29.D2, V30.D2, V31.D2]
    
    WORD $0x4EF0F400              // FMIN V0.2D, V0.2D, V16.2D
    WORD $0x4EF1F421              // FMIN V1.2D, V1.2D, V17.2D
    WORD $0x4EF2F442              // FMIN V2.2D, V2.2D, V18.2D
    WORD $0x4EF3F463              // FMIN V3.2D, V3.2D, V19.2D
    WORD $0x4EF4F484              // FMIN V4.2D, V4.2D, V20.2D
    WORD $0x4EF5F4A5              // FMIN V5.2D, V5.2D, V21.2D
    WORD $0x4EF6F4C6              // FMIN V6.2D, V6.2D, V22.2D
    WORD $0x4EF7F4E7              // FMIN V7.2D, V7.2D, V23.2D
    WORD $0x4EF8F508              // FMIN V8.2D, V8.2D, V24.2D
    WORD $0x4EF9F529              // FMIN V9.2D, V9.2D, V25.2D
    WORD $0x4EFAF54A              // FMIN V10.2D, V10.2D, V26.2D
    WORD $0x4EFBF56B              // FMIN V11.2D, V11.2D, V27.2D
    WORD $0x4EFCF58C              // FMIN V12.2D, V12.2D, V28.2D
    WORD $0x4EFDF5AD              // FMIN V13.2D, V13.2D, V29.2D
    WORD $0x4EFEF5CE              // FMIN V14.2D, V14.2D, V30.2D
    WORD $0x4EFFF5EF              // FMIN V15.2D, V15.2D, V31.2D
    
    SUB $32, R1
    CMP $32, R1
    BGE min_loop32
    
    // Tree reduction: 16 → 8 → 4 → 2 → 1 (same pattern as sum, but FMIN)
    WORD $0x4EE8F400              // FMIN V0.2D, V0.2D, V8.2D
    WORD $0x4EE9F421              // FMIN V1.2D, V1.2D, V9.2D
    WORD $0x4EEAF442              // FMIN V2.2D, V2.2D, V10.2D
    WORD $0x4EEBF463              // FMIN V3.2D, V3.2D, V11.2D
    WORD $0x4EECF484              // FMIN V4.2D, V4.2D, V12.2D
    WORD $0x4EEDF4A5              // FMIN V5.2D, V5.2D, V13.2D
    WORD $0x4EEEF4C6              // FMIN V6.2D, V6.2D, V14.2D
    WORD $0x4EEFF4E7              // FMIN V7.2D, V7.2D, V15.2D
    WORD $0x4EE4F400              // FMIN V0.2D, V0.2D, V4.2D
    WORD $0x4EE5F421              // FMIN V1.2D, V1.2D, V5.2D
    WORD $0x4EE6F442              // FMIN V2.2D, V2.2D, V6.2D
    WORD $0x4EE7F463              // FMIN V3.2D, V3.2D, V7.2D
    WORD $0x4EE2F400              // FMIN V0.2D, V0.2D, V2.2D
    WORD $0x4EE3F421              // FMIN V1.2D, V1.2D, V3.2D
    WORD $0x4EE1F400              // FMIN V0.2D, V0.2D, V1.2D
    
min_tail16:
    CMP $16, R1
    BLT min_tail8
    
    VLD1.P 64(R0), [V16.D2, V17.D2, V18.D2, V19.D2]
    VLD1.P 64(R0), [V20.D2, V21.D2, V22.D2, V23.D2]
    WORD $0x4EF0F400              // FMIN V0.2D, V0.2D, V16.2D
    WORD $0x4EF1F400              // FMIN V0.2D, V0.2D, V17.2D
    WORD $0x4EF2F400              // FMIN V0.2D, V0.2D, V18.2D
    WORD $0x4EF3F400              // FMIN V0.2D, V0.2D, V19.2D
    WORD $0x4EF4F400              // FMIN V0.2D, V0.2D, V20.2D
    WORD $0x4EF5F400              // FMIN V0.2D, V0.2D, V21.2D
    WORD $0x4EF6F400              // FMIN V0.2D, V0.2D, V22.2D
    WORD $0x4EF7F400              // FMIN V0.2D, V0.2D, V23.2D
    SUB $16, R1

min_tail8:
    CMP $8, R1
    BLT min_tail4
    
    VLD1.P 64(R0), [V16.D2, V17.D2, V18.D2, V19.D2]
    WORD $0x4EF0F400              // FMIN V0.2D, V0.2D, V16.2D
    WORD $0x4EF1F400              // FMIN V0.2D, V0.2D, V17.2D
    WORD $0x4EF2F400              // FMIN V0.2D, V0.2D, V18.2D
    WORD $0x4EF3F400              // FMIN V0.2D, V0.2D, V19.2D
    SUB $8, R1
    
min_tail4:
    CMP $4, R1
    BLT min_tail2
    
    VLD1.P 32(R0), [V16.D2, V17.D2]
    WORD $0x4EF0F400              // FMIN V0.2D, V0.2D, V16.2D
    WORD $0x4EF1F400              // FMIN V0.2D, V0.2D, V17.2D
    SUB $4, R1
    
min_tail2:
    CMP $2, R1
    BLT min_reduce
    
    VLD1.P 16(R0), [V16.D2]
    WORD $0x4EF0F400              // FMIN V0.2D, V0.2D, V16.2D
    SUB $2, R1
    
min_reduce:
    // ┌────────────────────────────────────────────────────────────────────┐
    // │ Horizontal min: Compare both lanes of V0                          │
    // │                                                                    │
    // │   V0 = [ a | b ]                                                   │
    // │          │   │                                                     │
    // │   V1 ←───┘   │  (extract lane 1)                                   │
    // │          ╲   ╱                                                     │
    // │         FMIND                                                      │
    // │            │                                                       │
    // │   D0 = min(a, b)                                                   │
    // └────────────────────────────────────────────────────────────────────┘
    VMOV V0.D[1], R2
    VMOV R2, V1.D[0]
    FMIND F0, F1, F0
    
    CMP $1, R1
    BLT min_done
    FMOVD (R0), F1
    FMIND F0, F1, F0
    
min_done:
    FMOVD F0, ret+24(FP)
    RET

// ┌──────────────────────────────────────────────────────────────────────────────┐
// │ func maxFloat64NEON(vals []float64) float64                                  │
// │                                                                              │
// │ Strategy: Identical to min but uses FMAX instead of FMIN                     │
// └──────────────────────────────────────────────────────────────────────────────┘
TEXT ·maxFloat64NEON(SB), NOSPLIT, $0-32
    MOVD vals_base+0(FP), R0
    MOVD vals_len+8(FP), R1
    
    // Broadcast first element to all 16 accumulators
    FMOVD (R0), F0
    VDUP V0.D[0], V0.D2
    VMOV V0.B16, V1.B16
    VMOV V0.B16, V2.B16
    VMOV V0.B16, V3.B16
    VMOV V0.B16, V4.B16
    VMOV V0.B16, V5.B16
    VMOV V0.B16, V6.B16
    VMOV V0.B16, V7.B16
    VMOV V0.B16, V8.B16
    VMOV V0.B16, V9.B16
    VMOV V0.B16, V10.B16
    VMOV V0.B16, V11.B16
    VMOV V0.B16, V12.B16
    VMOV V0.B16, V13.B16
    VMOV V0.B16, V14.B16
    VMOV V0.B16, V15.B16
    ADD $8, R0
    SUB $1, R1
    
    CMP $32, R1
    BLT max_tail16
    
max_loop32:
    VLD1.P 64(R0), [V16.D2, V17.D2, V18.D2, V19.D2]
    VLD1.P 64(R0), [V20.D2, V21.D2, V22.D2, V23.D2]
    VLD1.P 64(R0), [V24.D2, V25.D2, V26.D2, V27.D2]
    VLD1.P 64(R0), [V28.D2, V29.D2, V30.D2, V31.D2]
    
    WORD $0x4E70F400              // FMAX V0.2D, V0.2D, V16.2D
    WORD $0x4E71F421              // FMAX V1.2D, V1.2D, V17.2D
    WORD $0x4E72F442              // FMAX V2.2D, V2.2D, V18.2D
    WORD $0x4E73F463              // FMAX V3.2D, V3.2D, V19.2D
    WORD $0x4E74F484              // FMAX V4.2D, V4.2D, V20.2D
    WORD $0x4E75F4A5              // FMAX V5.2D, V5.2D, V21.2D
    WORD $0x4E76F4C6              // FMAX V6.2D, V6.2D, V22.2D
    WORD $0x4E77F4E7              // FMAX V7.2D, V7.2D, V23.2D
    WORD $0x4E78F508              // FMAX V8.2D, V8.2D, V24.2D
    WORD $0x4E79F529              // FMAX V9.2D, V9.2D, V25.2D
    WORD $0x4E7AF54A              // FMAX V10.2D, V10.2D, V26.2D
    WORD $0x4E7BF56B              // FMAX V11.2D, V11.2D, V27.2D
    WORD $0x4E7CF58C              // FMAX V12.2D, V12.2D, V28.2D
    WORD $0x4E7DF5AD              // FMAX V13.2D, V13.2D, V29.2D
    WORD $0x4E7EF5CE              // FMAX V14.2D, V14.2D, V30.2D
    WORD $0x4E7FF5EF              // FMAX V15.2D, V15.2D, V31.2D
    
    SUB $32, R1
    CMP $32, R1
    BGE max_loop32
    
    // Tree reduction: 16 → 8 → 4 → 2 → 1
    WORD $0x4E68F400              // FMAX V0.2D, V0.2D, V8.2D
    WORD $0x4E69F421              // FMAX V1.2D, V1.2D, V9.2D
    WORD $0x4E6AF442              // FMAX V2.2D, V2.2D, V10.2D
    WORD $0x4E6BF463              // FMAX V3.2D, V3.2D, V11.2D
    WORD $0x4E6CF484              // FMAX V4.2D, V4.2D, V12.2D
    WORD $0x4E6DF4A5              // FMAX V5.2D, V5.2D, V13.2D
    WORD $0x4E6EF4C6              // FMAX V6.2D, V6.2D, V14.2D
    WORD $0x4E6FF4E7              // FMAX V7.2D, V7.2D, V15.2D
    WORD $0x4E64F400              // FMAX V0.2D, V0.2D, V4.2D
    WORD $0x4E65F421              // FMAX V1.2D, V1.2D, V5.2D
    WORD $0x4E66F442              // FMAX V2.2D, V2.2D, V6.2D
    WORD $0x4E67F463              // FMAX V3.2D, V3.2D, V7.2D
    WORD $0x4E62F400              // FMAX V0.2D, V0.2D, V2.2D
    WORD $0x4E63F421              // FMAX V1.2D, V1.2D, V3.2D
    WORD $0x4E61F400              // FMAX V0.2D, V0.2D, V1.2D
    
max_tail16:
    CMP $16, R1
    BLT max_tail8
    
    VLD1.P 64(R0), [V16.D2, V17.D2, V18.D2, V19.D2]
    VLD1.P 64(R0), [V20.D2, V21.D2, V22.D2, V23.D2]
    WORD $0x4E70F400              // FMAX V0.2D, V0.2D, V16.2D
    WORD $0x4E71F400              // FMAX V0.2D, V0.2D, V17.2D
    WORD $0x4E72F400              // FMAX V0.2D, V0.2D, V18.2D
    WORD $0x4E73F400              // FMAX V0.2D, V0.2D, V19.2D
    WORD $0x4E74F400              // FMAX V0.2D, V0.2D, V20.2D
    WORD $0x4E75F400              // FMAX V0.2D, V0.2D, V21.2D
    WORD $0x4E76F400              // FMAX V0.2D, V0.2D, V22.2D
    WORD $0x4E77F400              // FMAX V0.2D, V0.2D, V23.2D
    SUB $16, R1

max_tail8:
    CMP $8, R1
    BLT max_tail4
    
    VLD1.P 64(R0), [V16.D2, V17.D2, V18.D2, V19.D2]
    WORD $0x4E70F400              // FMAX V0.2D, V0.2D, V16.2D
    WORD $0x4E71F400              // FMAX V0.2D, V0.2D, V17.2D
    WORD $0x4E72F400              // FMAX V0.2D, V0.2D, V18.2D
    WORD $0x4E73F400              // FMAX V0.2D, V0.2D, V19.2D
    SUB $8, R1
    
max_tail4:
    CMP $4, R1
    BLT max_tail2
    
    VLD1.P 32(R0), [V16.D2, V17.D2]
    WORD $0x4E70F400              // FMAX V0.2D, V0.2D, V16.2D
    WORD $0x4E71F400              // FMAX V0.2D, V0.2D, V17.2D
    SUB $4, R1
    
max_tail2:
    CMP $2, R1
    BLT max_reduce
    
    VLD1.P 16(R0), [V16.D2]
    WORD $0x4E70F400              // FMAX V0.2D, V0.2D, V16.2D
    SUB $2, R1
    
max_reduce:
    // Horizontal max
    VMOV V0.D[1], R2
    VMOV R2, V1.D[0]
    FMAXD F0, F1, F0
    
    CMP $1, R1
    BLT max_done
    FMOVD (R0), F1
    FMAXD F0, F1, F0
    
max_done:
    FMOVD F0, ret+24(FP)
    RET

// ┌──────────────────────────────────────────────────────────────────────────────┐
// │ func dotProductFloat64NEON(a, b []float64) float64                           │
// │                                                                              │
// │ Strategy: Use FMLA (fused multiply-add) for better accuracy                  │
// │           acc += a[i] * b[i]                                                 │
// │                                                                              │
// │ Dot product visualization:                                                   │
// │                                                                              │
// │   Array a:    [ a0 | a1 | a2 | a3 | a4 | a5 | ... ]                          │
// │                 ×    ×    ×    ×    ×    ×                                   │
// │   Array b:    [ b0 | b1 | b2 | b3 | b4 | b5 | ... ]                          │
// │                 ↓    ↓    ↓    ↓    ↓    ↓                                   │
// │   Products:   [a0b0|a1b1|a2b2|a3b3|a4b4|a5b5| ... ]                          │
// │                 └────┴────┴────┴────┴────┴──── + ─────► result               │
// │                                                                              │
// │ FMLA operation (fused multiply-add, single rounding):                        │
// │   acc = acc + (a * b)                                                        │
// └──────────────────────────────────────────────────────────────────────────────┘
TEXT ·dotProductFloat64NEON(SB), NOSPLIT, $0-56
    MOVD a_base+0(FP), R0         // R0 = pointer to a[0]
    MOVD a_len+8(FP), R1          // R1 = len(a)
    MOVD b_base+24(FP), R2        // R2 = pointer to b[0]
    
    // Initialize 16 accumulators to zero
    VEOR V0.B16, V0.B16, V0.B16
    VEOR V1.B16, V1.B16, V1.B16
    VEOR V2.B16, V2.B16, V2.B16
    VEOR V3.B16, V3.B16, V3.B16
    VEOR V4.B16, V4.B16, V4.B16
    VEOR V5.B16, V5.B16, V5.B16
    VEOR V6.B16, V6.B16, V6.B16
    VEOR V7.B16, V7.B16, V7.B16
    VEOR V8.B16, V8.B16, V8.B16
    VEOR V9.B16, V9.B16, V9.B16
    VEOR V10.B16, V10.B16, V10.B16
    VEOR V11.B16, V11.B16, V11.B16
    VEOR V12.B16, V12.B16, V12.B16
    VEOR V13.B16, V13.B16, V13.B16
    VEOR V14.B16, V14.B16, V14.B16
    VEOR V15.B16, V15.B16, V15.B16
    
    CMP $32, R1
    BLT dot_tail16
    
// ┌──────────────────────────────────────────────────────────────────────────────┐
// │ Main loop: FMLA (fused multiply-add)                                         │
// │                                                                              │
// │   Load from a[]: V16-V23 (16 elements)                                       │
// │   Load from b[]: V24-V31 (16 elements)                                       │
// │                                                                              │
// │   FMLA V0, V16, V24:                                                         │
// │   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐                     │
// │   │ V0 [acc|acc] │ + │V16 [a0 | a1] │ × │V24 [b0 | b1] │                     │
// │   └──────────────┘   └──────────────┘   └──────────────┘                     │
// │          ║                  ║                  ║                             │
// │          ╚══════════════════╬══════════════════╝                             │
// │                             ↓                                                │
// │   ┌────────────────────────────────────────────┐                             │
// │   │ V0 = [acc + a0*b0 | acc + a1*b1]           │                             │
// │   └────────────────────────────────────────────┘                             │
// └──────────────────────────────────────────────────────────────────────────────┘
dot_loop32:
    // Load 16 elements from a into V16-V23
    VLD1.P 64(R0), [V16.D2, V17.D2, V18.D2, V19.D2]
    // Load corresponding 16 elements from b into V24-V31
    VLD1.P 64(R2), [V24.D2, V25.D2, V26.D2, V27.D2]
    VLD1.P 64(R0), [V20.D2, V21.D2, V22.D2, V23.D2]
    VLD1.P 64(R2), [V28.D2, V29.D2, V30.D2, V31.D2]
    
    // FMLA: acc[i] += a[i] * b[i] (fused, single rounding)
    WORD $0x4E78CE00              // FMLA V0.2D, V16.2D, V24.2D
    WORD $0x4E79CE21              // FMLA V1.2D, V17.2D, V25.2D
    WORD $0x4E7ACE42              // FMLA V2.2D, V18.2D, V26.2D
    WORD $0x4E7BCE63              // FMLA V3.2D, V19.2D, V27.2D
    WORD $0x4E7CCE84              // FMLA V4.2D, V20.2D, V28.2D
    WORD $0x4E7DCEA5              // FMLA V5.2D, V21.2D, V29.2D
    WORD $0x4E7ECEC6              // FMLA V6.2D, V22.2D, V30.2D
    WORD $0x4E7FCEE7              // FMLA V7.2D, V23.2D, V31.2D
    
    // Load next 16 elements and accumulate into V8-V15
    VLD1.P 64(R0), [V16.D2, V17.D2, V18.D2, V19.D2]
    VLD1.P 64(R2), [V24.D2, V25.D2, V26.D2, V27.D2]
    VLD1.P 64(R0), [V20.D2, V21.D2, V22.D2, V23.D2]
    VLD1.P 64(R2), [V28.D2, V29.D2, V30.D2, V31.D2]
    
    WORD $0x4E78CE08              // FMLA V8.2D, V16.2D, V24.2D
    WORD $0x4E79CE29              // FMLA V9.2D, V17.2D, V25.2D
    WORD $0x4E7ACE4A              // FMLA V10.2D, V18.2D, V26.2D
    WORD $0x4E7BCE6B              // FMLA V11.2D, V19.2D, V27.2D
    WORD $0x4E7CCE8C              // FMLA V12.2D, V20.2D, V28.2D
    WORD $0x4E7DCEAD              // FMLA V13.2D, V21.2D, V29.2D
    WORD $0x4E7ECECE              // FMLA V14.2D, V22.2D, V30.2D
    WORD $0x4E7FCEEF              // FMLA V15.2D, V23.2D, V31.2D
    
    SUB $32, R1
    CMP $32, R1
    BGE dot_loop32
    
    // Tree reduction: combine 16 accumulators into 1
    WORD $0x4E68D400              // FADD V0.2D, V0.2D, V8.2D
    WORD $0x4E69D421              // FADD V1.2D, V1.2D, V9.2D
    WORD $0x4E6AD442              // FADD V2.2D, V2.2D, V10.2D
    WORD $0x4E6BD463              // FADD V3.2D, V3.2D, V11.2D
    WORD $0x4E6CD484              // FADD V4.2D, V4.2D, V12.2D
    WORD $0x4E6DD4A5              // FADD V5.2D, V5.2D, V13.2D
    WORD $0x4E6ED4C6              // FADD V6.2D, V6.2D, V14.2D
    WORD $0x4E6FD4E7              // FADD V7.2D, V7.2D, V15.2D
    WORD $0x4E64D400              // FADD V0.2D, V0.2D, V4.2D
    WORD $0x4E65D421              // FADD V1.2D, V1.2D, V5.2D
    WORD $0x4E66D442              // FADD V2.2D, V2.2D, V6.2D
    WORD $0x4E67D463              // FADD V3.2D, V3.2D, V7.2D
    WORD $0x4E62D400              // FADD V0.2D, V0.2D, V2.2D
    WORD $0x4E63D421              // FADD V1.2D, V1.2D, V3.2D
    WORD $0x4E61D400              // FADD V0.2D, V0.2D, V1.2D
    
dot_tail16:
    CMP $16, R1
    BLT dot_tail8
    
    VLD1.P 64(R0), [V16.D2, V17.D2, V18.D2, V19.D2]
    VLD1.P 64(R2), [V24.D2, V25.D2, V26.D2, V27.D2]
    VLD1.P 64(R0), [V20.D2, V21.D2, V22.D2, V23.D2]
    VLD1.P 64(R2), [V28.D2, V29.D2, V30.D2, V31.D2]
    WORD $0x4E78CE00              // FMLA V0.2D, V16.2D, V24.2D
    WORD $0x4E79CE20              // FMLA V0.2D, V17.2D, V25.2D
    WORD $0x4E7ACE40              // FMLA V0.2D, V18.2D, V26.2D
    WORD $0x4E7BCE60              // FMLA V0.2D, V19.2D, V27.2D
    WORD $0x4E7CCE80              // FMLA V0.2D, V20.2D, V28.2D
    WORD $0x4E7DCEA0              // FMLA V0.2D, V21.2D, V29.2D
    WORD $0x4E7ECEC0              // FMLA V0.2D, V22.2D, V30.2D
    WORD $0x4E7FCEE0              // FMLA V0.2D, V23.2D, V31.2D
    SUB $16, R1

dot_tail8:
    CMP $8, R1
    BLT dot_tail4
    
    VLD1.P 64(R0), [V16.D2, V17.D2, V18.D2, V19.D2]
    VLD1.P 64(R2), [V24.D2, V25.D2, V26.D2, V27.D2]
    WORD $0x4E78CE00              // FMLA V0.2D, V16.2D, V24.2D
    WORD $0x4E79CE20              // FMLA V0.2D, V17.2D, V25.2D
    WORD $0x4E7ACE40              // FMLA V0.2D, V18.2D, V26.2D
    WORD $0x4E7BCE60              // FMLA V0.2D, V19.2D, V27.2D
    SUB $8, R1

dot_tail4:
    CMP $4, R1
    BLT dot_tail2
    
    VLD1.P 32(R0), [V16.D2, V17.D2]
    VLD1.P 32(R2), [V24.D2, V25.D2]
    WORD $0x4E78CE00              // FMLA V0.2D, V16.2D, V24.2D
    WORD $0x4E79CE20              // FMLA V0.2D, V17.2D, V25.2D
    SUB $4, R1
    
dot_tail2:
    CMP $2, R1
    BLT dot_reduce
    
    VLD1.P 16(R0), [V16.D2]
    VLD1.P 16(R2), [V24.D2]
    WORD $0x4E78CE00              // FMLA V0.2D, V16.2D, V24.2D
    SUB $2, R1
    
dot_reduce:
    // Horizontal sum: add both lanes
    WORD $0x7E70D800              // FADDP D0, V0.2D
    
    // Handle remaining 1 element (scalar multiply-add)
    CMP $1, R1
    BLT dot_done
    FMOVD (R0), F1
    FMOVD (R2), F2
    FMULD F1, F2, F1
    FADDD F0, F1, F0
    
dot_done:
    FMOVD F0, ret+48(FP)
    RET
