//go:build arm64

#include "textflag.h"

// ╔══════════════════════════════════════════════════════════════════════════════╗
// ║                           SVE Int64 SIMD Operations                          ║
// ║                                                                              ║
// ║  SVE (Scalable Vector Extension) - Variable-width vector processing         ║
// ║                                                                              ║
// ║  Key advantages over NEON for int64:                                         ║
// ║  • Native 64-bit integer MUL instruction (NEON lacks this!)                  ║
// ║  • Native SMIN/SMAX instructions                                             ║
// ║  • SMINV/SMAXV for horizontal reduction                                      ║
// ║  • UADDV for horizontal sum                                                  ║
// ║                                                                              ║
// ║  SVE Instruction Reference:                                                  ║
// ║  PTRUE P0.D             : 0x2518E3E0 - All predicate lanes true              ║
// ║  DUP Zd.D, #0           : 0x25F8C000 - Broadcast zero                        ║
// ║  LD1D {Zt}, Pg/Z, [Xn]  : 0xA5E0A000 - Predicated load                       ║
// ║  ADD Zdn, Pg/M, Zdn, Zm : 0x04C00000 - Integer add                           ║
// ║  MUL Zdn, Pg/M, Zdn, Zm : 0x04D00000 - Integer multiply (64-bit!)            ║
// ║  SMIN Zdn, Pg/M, Zdn, Zm: 0x04CA0000 - Signed minimum                        ║
// ║  SMAX Zdn, Pg/M, Zdn, Zm: 0x04C80000 - Signed maximum                        ║
// ║  UADDV Dd, Pg, Zn.D     : 0x04C12000 - Horizontal unsigned sum               ║
// ║  SMINV Dd, Pg, Zn.D     : 0x04CA2000 - Horizontal signed min                 ║
// ║  SMAXV Dd, Pg, Zn.D     : 0x04C82000 - Horizontal signed max                 ║
// ╚══════════════════════════════════════════════════════════════════════════════╝

// ┌──────────────────────────────────────────────────────────────────────────────┐
// │ func sumInt64SVE(vals []int64) int64                                         │
// │                                                                              │
// │ Strategy: 8 vector accumulators, UADDV for horizontal reduction              │
// └──────────────────────────────────────────────────────────────────────────────┘
TEXT ·sumInt64SVE(SB), NOSPLIT, $0-32
    MOVD vals_base+0(FP), R0      // R0 = pointer to vals
    MOVD vals_len+8(FP), R1       // R1 = len(vals)
    
    // ╔════════════════════════════════════════════════════════════════════════╗
    // ║ PTRUE P0.D - Enable all 64-bit lanes                                   ║
    // ║                                                                        ║
    // ║   P0 = [ 1 | 1 | 1 | 1 | ... ]  (VL-dependent number of lanes)        ║
    // ╚════════════════════════════════════════════════════════════════════════╝
    WORD $0x2518E3E0              // PTRUE P0.D
    
    // Zero 8 accumulators
    WORD $0x25F8C000              // DUP Z0.D, #0
    WORD $0x25F8C001              // DUP Z1.D, #0
    WORD $0x25F8C002              // DUP Z2.D, #0
    WORD $0x25F8C003              // DUP Z3.D, #0
    WORD $0x25F8C004              // DUP Z4.D, #0
    WORD $0x25F8C005              // DUP Z5.D, #0
    WORD $0x25F8C006              // DUP Z6.D, #0
    WORD $0x25F8C007              // DUP Z7.D, #0
    
    // ┌────────────────────────────────────────────────────────────────────┐
    // │ CNTD - Get number of 64-bit elements per vector                   │
    // │                                                                    │
    // │   R2 = elements_per_vector (2, 4, or 8 depending on VL)            │
    // │   R3 = R2 * 8 = elements per main loop iteration                   │
    // └────────────────────────────────────────────────────────────────────┘
    WORD $0x04E0E3E2              // CNTD X2
    
    LSL $3, R2, R3                // R3 = R2 * 8 (8 vectors per iteration)
    
    CMP R3, R1
    BLT sve_isum_tail

// ┌──────────────────────────────────────────────────────────────────────────────┐
// │ Main loop: Load 8 vectors, add to 8 accumulators                             │
// │                                                                              │
// │ Memory layout (256-bit SVE example, 4 int64 per vector):                     │
// │                                                                              │
// │   Memory: ┌────┬────┬────┬────┬────┬────┬────┬────┬────...────┬────┐         │
// │           │ v0 │ v1 │ v2 │ v3 │ v4 │ v5 │ v6 │ v7 │    ...   │v31 │         │
// │           └────┴────┴────┴────┴────┴────┴────┴────┴────...────┴────┘         │
// │           ╰───── Z8 ─────╯╰───── Z9 ─────╯         ╰──── Z15 ────╯           │
// │                #0              #1                       #7                   │
// └──────────────────────────────────────────────────────────────────────────────┘
sve_isum_loop8:
    // Load 8 vectors (contiguous, VL-offset addressing)
    WORD $0xA5E0A008              // LD1D {Z8.D}, P0/Z, [R0, #0, MUL VL]
    WORD $0xA5E1A009              // LD1D {Z9.D}, P0/Z, [R0, #1, MUL VL]
    WORD $0xA5E2A00A              // LD1D {Z10.D}, P0/Z, [R0, #2, MUL VL]
    WORD $0xA5E3A00B              // LD1D {Z11.D}, P0/Z, [R0, #3, MUL VL]
    WORD $0xA5E4A00C              // LD1D {Z12.D}, P0/Z, [R0, #4, MUL VL]
    WORD $0xA5E5A00D              // LD1D {Z13.D}, P0/Z, [R0, #5, MUL VL]
    WORD $0xA5E6A00E              // LD1D {Z14.D}, P0/Z, [R0, #6, MUL VL]
    WORD $0xA5E7A00F              // LD1D {Z15.D}, P0/Z, [R0, #7, MUL VL]
    
    // ┌────────────────────────────────────────────────────────────────────┐
    // │ ADD with predicate: Z0-Z7 += Z8-Z15                               │
    // │                                                                    │
    // │   Before: Z0 = [ s0 | s1 | s2 | s3 ]   Z8 = [ a | b | c | d ]      │
    // │   P0/M:        [  1 |  1 |  1 |  1 ]                               │
    // │   After:  Z0 = [ s0+a | s1+b | s2+c | s3+d ]                       │
    // └────────────────────────────────────────────────────────────────────┘
    WORD $0x04C00100              // ADD Z0.D, P0/M, Z0.D, Z8.D
    WORD $0x04C00121              // ADD Z1.D, P0/M, Z1.D, Z9.D
    WORD $0x04C00142              // ADD Z2.D, P0/M, Z2.D, Z10.D
    WORD $0x04C00163              // ADD Z3.D, P0/M, Z3.D, Z11.D
    WORD $0x04C00184              // ADD Z4.D, P0/M, Z4.D, Z12.D
    WORD $0x04C001A5              // ADD Z5.D, P0/M, Z5.D, Z13.D
    WORD $0x04C001C6              // ADD Z6.D, P0/M, Z6.D, Z14.D
    WORD $0x04C001E7              // ADD Z7.D, P0/M, Z7.D, Z15.D
    
    LSL $3, R3, R5
    ADD R5, R0, R0
    SUBS R3, R1, R1
    CMP R3, R1
    BGE sve_isum_loop8

// ┌──────────────────────────────────────────────────────────────────────────────┐
// │ Tail: Use WHILELO for automatic partial vector masking                       │
// └──────────────────────────────────────────────────────────────────────────────┘
sve_isum_tail:
    MOVD ZR, R4                   // R4 = 0 (loop index)
    CBZ R1, sve_isum_reduce
    
sve_isum_tail_loop:
    WORD $0x25E11C81              // WHILELO P1.D, X4, X1 (64-bit compare)
    BEQ sve_isum_reduce           // Exit if no active lanes
    
    WORD $0xA5E0A408              // LD1D {Z8.D}, P1/Z, [R0, #0, MUL VL]
    WORD $0x04C00500              // ADD Z0.D, P1/M, Z0.D, Z8.D
    
    WORD $0x04F0E3E4              // INCD X4
    LSL $3, R2, R5
    ADD R5, R0, R0
    B sve_isum_tail_loop

sve_isum_reduce:
    // ╔════════════════════════════════════════════════════════════════════════╗
    // ║ Tree reduction: 8 → 4 → 2 → 1 vectors                                  ║
    // ╚════════════════════════════════════════════════════════════════════════╝
    WORD $0x04C00080              // ADD Z0.D, P0/M, Z0.D, Z4.D
    WORD $0x04C000A1              // ADD Z1.D, P0/M, Z1.D, Z5.D
    WORD $0x04C000C2              // ADD Z2.D, P0/M, Z2.D, Z6.D
    WORD $0x04C000E3              // ADD Z3.D, P0/M, Z3.D, Z7.D
    WORD $0x04C00040              // ADD Z0.D, P0/M, Z0.D, Z2.D
    WORD $0x04C00061              // ADD Z1.D, P0/M, Z1.D, Z3.D
    WORD $0x04C00020              // ADD Z0.D, P0/M, Z0.D, Z1.D
    
    // ┌────────────────────────────────────────────────────────────────────┐
    // │ UADDV - Horizontal unsigned sum                                   │
    // │                                                                    │
    // │   Z0 = [ a | b | c | d ]                                           │
    // │          │   │   │   │                                             │
    // │          └───┴───┴───┴───────► D0 = a + b + c + d                  │
    // └────────────────────────────────────────────────────────────────────┘
    WORD $0x04C12000              // UADDV D0, P0, Z0.D
    
    FMOVD F0, R0
    MOVD R0, ret+24(FP)
    RET

// ┌──────────────────────────────────────────────────────────────────────────────┐
// │ func minInt64SVE(vals []int64) int64                                         │
// │                                                                              │
// │ Strategy: Native SMIN instruction (unlike NEON which needs CMGT+BIT)         │
// │           SMINV for horizontal reduction                                     │
// │                                                                              │
// │ SVE SMIN operation (signed minimum):                                         │
// │   ┌────────────────────────────────────────────────────────────────────┐     │
// │   │  Z0 = [ 5 | -3 |  8 | 2 ]    Z8 = [ 2 | 7 | -1 | 9 ]               │     │
// │   │        ───────────────────────────────────────                     │     │
// │   │  SMIN P0/M                                                         │     │
// │   │        ↓   ↓    ↓   ↓                                              │     │
// │   │  Z0 = [ 2 | -3 | -1 | 2 ]   (element-wise signed min)              │     │
// │   └────────────────────────────────────────────────────────────────────┘     │
// └──────────────────────────────────────────────────────────────────────────────┘
TEXT ·minInt64SVE(SB), NOSPLIT, $0-32
    MOVD vals_base+0(FP), R0
    MOVD vals_len+8(FP), R1
    
    CBZ R1, sve_min_empty
    
    WORD $0x2518E3E0              // PTRUE P0.D
    WORD $0x04E0E3E2              // CNTD X2
    
    // ┌────────────────────────────────────────────────────────────────────┐
    // │ Broadcast first element to all 8 accumulators                     │
    // │                                                                    │
    // │   DUP Zd.D, Xn broadcasts scalar register to all lanes            │
    // │                                                                    │
    // │   vals[0] ────────────────────────────────────────────┐            │
    // │            ╔═══════╦═══════╦═══════╦═══════╦═══════╦══╧════╗       │
    // │            ║  Z0   ║  Z1   ║  Z2   ║ ...   ║  Z6   ║  Z7   ║       │
    // │            ║[v|v|v]║[v|v|v]║[v|v|v]║       ║[v|v|v]║[v|v|v]║       │
    // │            ╚═══════╩═══════╩═══════╩═══════╩═══════╩═══════╝       │
    // └────────────────────────────────────────────────────────────────────┘
    MOVD (R0), R3
    WORD $0x05E03860              // DUP Z0.D, X3
    WORD $0x05E03861              // DUP Z1.D, X3
    WORD $0x05E03862              // DUP Z2.D, X3
    WORD $0x05E03863              // DUP Z3.D, X3
    WORD $0x05E03864              // DUP Z4.D, X3
    WORD $0x05E03865              // DUP Z5.D, X3
    WORD $0x05E03866              // DUP Z6.D, X3
    WORD $0x05E03867              // DUP Z7.D, X3
    
    ADD $8, R0
    SUB $1, R1
    
    LSL $3, R2, R3
    CMP R3, R1
    BLT sve_min_tail

sve_min_loop8:
    // Load 8 vectors
    WORD $0xA5E0A008              // LD1D {Z8.D}, P0/Z, [R0, #0, MUL VL]
    WORD $0xA5E1A009              // LD1D {Z9.D}, P0/Z, [R0, #1, MUL VL]
    WORD $0xA5E2A00A              // LD1D {Z10.D}, P0/Z, [R0, #2, MUL VL]
    WORD $0xA5E3A00B              // LD1D {Z11.D}, P0/Z, [R0, #3, MUL VL]
    WORD $0xA5E4A00C              // LD1D {Z12.D}, P0/Z, [R0, #4, MUL VL]
    WORD $0xA5E5A00D              // LD1D {Z13.D}, P0/Z, [R0, #5, MUL VL]
    WORD $0xA5E6A00E              // LD1D {Z14.D}, P0/Z, [R0, #6, MUL VL]
    WORD $0xA5E7A00F              // LD1D {Z15.D}, P0/Z, [R0, #7, MUL VL]
    
    // SMIN - Native signed minimum (much simpler than NEON's CMGT+BIT)
    WORD $0x04CA0100              // SMIN Z0.D, P0/M, Z0.D, Z8.D
    WORD $0x04CA0121              // SMIN Z1.D, P0/M, Z1.D, Z9.D
    WORD $0x04CA0142              // SMIN Z2.D, P0/M, Z2.D, Z10.D
    WORD $0x04CA0163              // SMIN Z3.D, P0/M, Z3.D, Z11.D
    WORD $0x04CA0184              // SMIN Z4.D, P0/M, Z4.D, Z12.D
    WORD $0x04CA01A5              // SMIN Z5.D, P0/M, Z5.D, Z13.D
    WORD $0x04CA01C6              // SMIN Z6.D, P0/M, Z6.D, Z14.D
    WORD $0x04CA01E7              // SMIN Z7.D, P0/M, Z7.D, Z15.D
    
    LSL $3, R3, R5
    ADD R5, R0, R0
    SUBS R3, R1, R1
    CMP R3, R1
    BGE sve_min_loop8

sve_min_tail:
    MOVD ZR, R4
    CBZ R1, sve_min_reduce
    
sve_min_tail_loop:
    WORD $0x25E11C81              // WHILELO P1.D, X4, X1
    BEQ sve_min_reduce
    
    WORD $0xA5E0A408              // LD1D {Z8.D}, P1/Z, [R0, #0, MUL VL]
    WORD $0x04CA0500              // SMIN Z0.D, P1/M, Z0.D, Z8.D
    
    WORD $0x04F0E3E4              // INCD X4
    LSL $3, R2, R5
    ADD R5, R0, R0
    B sve_min_tail_loop

sve_min_reduce:
    // Tree reduction: 8 → 4 → 2 → 1
    WORD $0x04CA0080              // SMIN Z0.D, P0/M, Z0.D, Z4.D
    WORD $0x04CA00A1              // SMIN Z1.D, P0/M, Z1.D, Z5.D
    WORD $0x04CA00C2              // SMIN Z2.D, P0/M, Z2.D, Z6.D
    WORD $0x04CA00E3              // SMIN Z3.D, P0/M, Z3.D, Z7.D
    WORD $0x04CA0040              // SMIN Z0.D, P0/M, Z0.D, Z2.D
    WORD $0x04CA0061              // SMIN Z1.D, P0/M, Z1.D, Z3.D
    WORD $0x04CA0020              // SMIN Z0.D, P0/M, Z0.D, Z1.D
    
    // ┌────────────────────────────────────────────────────────────────────┐
    // │ SMINV - Horizontal signed minimum                                 │
    // │                                                                    │
    // │   Z0 = [ 5 | -3 | 8 | 2 ]                                          │
    // │          │    │   │   │                                            │
    // │          └────┴───┴───┴────────► D0 = min(5, -3, 8, 2) = -3        │
    // └────────────────────────────────────────────────────────────────────┘
    WORD $0x04CA2000              // SMINV D0, P0, Z0.D
    
    FMOVD F0, R0
    MOVD R0, ret+24(FP)
    RET

sve_min_empty:
    MOVD $0, R0
    MOVD R0, ret+24(FP)
    RET

// ┌──────────────────────────────────────────────────────────────────────────────┐
// │ func maxInt64SVE(vals []int64) int64                                         │
// │                                                                              │
// │ Strategy: Native SMAX instruction, SMAXV for horizontal reduction            │
// └──────────────────────────────────────────────────────────────────────────────┘
TEXT ·maxInt64SVE(SB), NOSPLIT, $0-32
    MOVD vals_base+0(FP), R0
    MOVD vals_len+8(FP), R1
    
    CBZ R1, sve_max_empty
    
    WORD $0x2518E3E0              // PTRUE P0.D
    WORD $0x04E0E3E2              // CNTD X2
    
    // Broadcast first element
    MOVD (R0), R3
    WORD $0x05E03860              // DUP Z0.D, X3
    WORD $0x05E03861              // DUP Z1.D, X3
    WORD $0x05E03862              // DUP Z2.D, X3
    WORD $0x05E03863              // DUP Z3.D, X3
    WORD $0x05E03864              // DUP Z4.D, X3
    WORD $0x05E03865              // DUP Z5.D, X3
    WORD $0x05E03866              // DUP Z6.D, X3
    WORD $0x05E03867              // DUP Z7.D, X3
    
    ADD $8, R0
    SUB $1, R1
    
    LSL $3, R2, R3
    CMP R3, R1
    BLT sve_max_tail

sve_max_loop8:
    WORD $0xA5E0A008              // LD1D {Z8.D}, P0/Z, [R0, #0, MUL VL]
    WORD $0xA5E1A009              // LD1D {Z9.D}, P0/Z, [R0, #1, MUL VL]
    WORD $0xA5E2A00A              // LD1D {Z10.D}, P0/Z, [R0, #2, MUL VL]
    WORD $0xA5E3A00B              // LD1D {Z11.D}, P0/Z, [R0, #3, MUL VL]
    WORD $0xA5E4A00C              // LD1D {Z12.D}, P0/Z, [R0, #4, MUL VL]
    WORD $0xA5E5A00D              // LD1D {Z13.D}, P0/Z, [R0, #5, MUL VL]
    WORD $0xA5E6A00E              // LD1D {Z14.D}, P0/Z, [R0, #6, MUL VL]
    WORD $0xA5E7A00F              // LD1D {Z15.D}, P0/Z, [R0, #7, MUL VL]
    
    // SMAX - Native signed maximum
    WORD $0x04C80100              // SMAX Z0.D, P0/M, Z0.D, Z8.D
    WORD $0x04C80121              // SMAX Z1.D, P0/M, Z1.D, Z9.D
    WORD $0x04C80142              // SMAX Z2.D, P0/M, Z2.D, Z10.D
    WORD $0x04C80163              // SMAX Z3.D, P0/M, Z3.D, Z11.D
    WORD $0x04C80184              // SMAX Z4.D, P0/M, Z4.D, Z12.D
    WORD $0x04C801A5              // SMAX Z5.D, P0/M, Z5.D, Z13.D
    WORD $0x04C801C6              // SMAX Z6.D, P0/M, Z6.D, Z14.D
    WORD $0x04C801E7              // SMAX Z7.D, P0/M, Z7.D, Z15.D
    
    LSL $3, R3, R5
    ADD R5, R0, R0
    SUBS R3, R1, R1
    CMP R3, R1
    BGE sve_max_loop8

sve_max_tail:
    MOVD ZR, R4
    CBZ R1, sve_max_reduce
    
sve_max_tail_loop:
    WORD $0x25E11C81              // WHILELO P1.D, X4, X1
    BEQ sve_max_reduce
    
    WORD $0xA5E0A408              // LD1D {Z8.D}, P1/Z, [R0, #0, MUL VL]
    WORD $0x04C80500              // SMAX Z0.D, P1/M, Z0.D, Z8.D
    
    WORD $0x04F0E3E4              // INCD X4
    LSL $3, R2, R5
    ADD R5, R0, R0
    B sve_max_tail_loop

sve_max_reduce:
    WORD $0x04C80080              // SMAX Z0.D, P0/M, Z0.D, Z4.D
    WORD $0x04C800A1              // SMAX Z1.D, P0/M, Z1.D, Z5.D
    WORD $0x04C800C2              // SMAX Z2.D, P0/M, Z2.D, Z6.D
    WORD $0x04C800E3              // SMAX Z3.D, P0/M, Z3.D, Z7.D
    WORD $0x04C80040              // SMAX Z0.D, P0/M, Z0.D, Z2.D
    WORD $0x04C80061              // SMAX Z1.D, P0/M, Z1.D, Z3.D
    WORD $0x04C80020              // SMAX Z0.D, P0/M, Z0.D, Z1.D
    
    // SMAXV - Horizontal signed maximum
    WORD $0x04C82000              // SMAXV D0, P0, Z0.D
    
    FMOVD F0, R0
    MOVD R0, ret+24(FP)
    RET

sve_max_empty:
    MOVD $0, R0
    MOVD R0, ret+24(FP)
    RET

// ┌──────────────────────────────────────────────────────────────────────────────┐
// │ func dotProductInt64SVE(a, b []int64) int64                                  │
// │                                                                              │
// │ ⭐ SVE HAS 64-BIT INTEGER MUL! (NEON doesn't)                                │
// │                                                                              │
// │ Strategy: MUL + ADD in vector form, UADDV for horizontal sum                 │
// │                                                                              │
// │   a:    [ a0 | a1 | a2 | a3 ]                                                │
// │           ×    ×    ×    ×     MUL (native 64-bit!)                          │
// │   b:    [ b0 | b1 | b2 | b3 ]                                                │
// │           ↓    ↓    ↓    ↓                                                   │
// │   prod: [a0b0|a1b1|a2b2|a3b3]                                                │
// │           +    +    +    +     ADD to accumulator                            │
// │   acc:  [ s0 | s1 | s2 | s3 ]                                                │
// └──────────────────────────────────────────────────────────────────────────────┘
TEXT ·dotProductInt64SVE(SB), NOSPLIT, $0-56
    MOVD a_base+0(FP), R0         // R0 = pointer to a
    MOVD a_len+8(FP), R1          // R1 = len(a)
    MOVD b_base+24(FP), R2        // R2 = pointer to b
    
    WORD $0x2518E3E0              // PTRUE P0.D
    
    // Zero 8 accumulators
    WORD $0x25F8C000              // DUP Z0.D, #0
    WORD $0x25F8C001              // DUP Z1.D, #0
    WORD $0x25F8C002              // DUP Z2.D, #0
    WORD $0x25F8C003              // DUP Z3.D, #0
    WORD $0x25F8C004              // DUP Z4.D, #0
    WORD $0x25F8C005              // DUP Z5.D, #0
    WORD $0x25F8C006              // DUP Z6.D, #0
    WORD $0x25F8C007              // DUP Z7.D, #0
    
    WORD $0x04E0E3E3              // CNTD X3
    
    LSL $3, R3, R4                // R4 = elements per iteration
    
    CMP R4, R1
    BLT sve_idot_tail

sve_idot_loop8:
    // Load 8 vectors from a into Z8-Z15
    WORD $0xA5E0A008              // LD1D {Z8.D}, P0/Z, [R0, #0, MUL VL]
    WORD $0xA5E1A009              // LD1D {Z9.D}, P0/Z, [R0, #1, MUL VL]
    WORD $0xA5E2A00A              // LD1D {Z10.D}, P0/Z, [R0, #2, MUL VL]
    WORD $0xA5E3A00B              // LD1D {Z11.D}, P0/Z, [R0, #3, MUL VL]
    WORD $0xA5E4A00C              // LD1D {Z12.D}, P0/Z, [R0, #4, MUL VL]
    WORD $0xA5E5A00D              // LD1D {Z13.D}, P0/Z, [R0, #5, MUL VL]
    WORD $0xA5E6A00E              // LD1D {Z14.D}, P0/Z, [R0, #6, MUL VL]
    WORD $0xA5E7A00F              // LD1D {Z15.D}, P0/Z, [R0, #7, MUL VL]
    
    // Load 8 vectors from b into Z16-Z23
    WORD $0xA5E0A050              // LD1D {Z16.D}, P0/Z, [R2, #0, MUL VL]
    WORD $0xA5E1A051              // LD1D {Z17.D}, P0/Z, [R2, #1, MUL VL]
    WORD $0xA5E2A052              // LD1D {Z18.D}, P0/Z, [R2, #2, MUL VL]
    WORD $0xA5E3A053              // LD1D {Z19.D}, P0/Z, [R2, #3, MUL VL]
    WORD $0xA5E4A054              // LD1D {Z20.D}, P0/Z, [R2, #4, MUL VL]
    WORD $0xA5E5A055              // LD1D {Z21.D}, P0/Z, [R2, #5, MUL VL]
    WORD $0xA5E6A056              // LD1D {Z22.D}, P0/Z, [R2, #6, MUL VL]
    WORD $0xA5E7A057              // LD1D {Z23.D}, P0/Z, [R2, #7, MUL VL]
    
    // ┌────────────────────────────────────────────────────────────────────┐
    // │ MUL: Z8-Z15 *= Z16-Z23 (in-place element-wise multiply)           │
    // │                                                                    │
    // │   Z8 = [ a0 | a1 | a2 | a3 ]                                       │
    // │         × × × ×                                                    │
    // │  Z16 = [ b0 | b1 | b2 | b3 ]                                       │
    // │         ↓   ↓   ↓   ↓                                              │
    // │   Z8 = [a0b0|a1b1|a2b2|a3b3]                                       │
    // └────────────────────────────────────────────────────────────────────┘
    WORD $0x04D00208              // MUL Z8.D, P0/M, Z8.D, Z16.D
    WORD $0x04D00229              // MUL Z9.D, P0/M, Z9.D, Z17.D
    WORD $0x04D0024A              // MUL Z10.D, P0/M, Z10.D, Z18.D
    WORD $0x04D0026B              // MUL Z11.D, P0/M, Z11.D, Z19.D
    WORD $0x04D0028C              // MUL Z12.D, P0/M, Z12.D, Z20.D
    WORD $0x04D002AD              // MUL Z13.D, P0/M, Z13.D, Z21.D
    WORD $0x04D002CE              // MUL Z14.D, P0/M, Z14.D, Z22.D
    WORD $0x04D002EF              // MUL Z15.D, P0/M, Z15.D, Z23.D
    
    // ADD accumulators Z0-Z7 += Z8-Z15 (products)
    WORD $0x04C00100              // ADD Z0.D, P0/M, Z0.D, Z8.D
    WORD $0x04C00121              // ADD Z1.D, P0/M, Z1.D, Z9.D
    WORD $0x04C00142              // ADD Z2.D, P0/M, Z2.D, Z10.D
    WORD $0x04C00163              // ADD Z3.D, P0/M, Z3.D, Z11.D
    WORD $0x04C00184              // ADD Z4.D, P0/M, Z4.D, Z12.D
    WORD $0x04C001A5              // ADD Z5.D, P0/M, Z5.D, Z13.D
    WORD $0x04C001C6              // ADD Z6.D, P0/M, Z6.D, Z14.D
    WORD $0x04C001E7              // ADD Z7.D, P0/M, Z7.D, Z15.D
    
    // Advance both pointers
    LSL $3, R4, R5
    ADD R5, R0, R0
    ADD R5, R2, R2
    SUBS R4, R1, R1
    CMP R4, R1
    BGE sve_idot_loop8

sve_idot_tail:
    MOVD ZR, R6
    CBZ R1, sve_idot_reduce
    
sve_idot_tail_loop:
    WORD $0x25E11CC1              // WHILELO P1.D, X6, X1
    BEQ sve_idot_reduce
    
    WORD $0xA5E0A408              // LD1D {Z8.D}, P1/Z, [R0, #0, MUL VL]
    WORD $0xA5E0A450              // LD1D {Z16.D}, P1/Z, [R2, #0, MUL VL]
    
    WORD $0x04D00608              // MUL Z8.D, P1/M, Z8.D, Z16.D
    WORD $0x04C00500              // ADD Z0.D, P1/M, Z0.D, Z8.D
    
    WORD $0x04F0E3E6              // INCD X6
    LSL $3, R3, R5
    ADD R5, R0, R0
    ADD R5, R2, R2
    B sve_idot_tail_loop

sve_idot_reduce:
    // Combine 8 → 4 → 2 → 1
    WORD $0x04C00080              // ADD Z0.D, P0/M, Z0.D, Z4.D
    WORD $0x04C000A1              // ADD Z1.D, P0/M, Z1.D, Z5.D
    WORD $0x04C000C2              // ADD Z2.D, P0/M, Z2.D, Z6.D
    WORD $0x04C000E3              // ADD Z3.D, P0/M, Z3.D, Z7.D
    WORD $0x04C00040              // ADD Z0.D, P0/M, Z0.D, Z2.D
    WORD $0x04C00061              // ADD Z1.D, P0/M, Z1.D, Z3.D
    WORD $0x04C00020              // ADD Z0.D, P0/M, Z0.D, Z1.D
    
    // UADDV for horizontal sum
    WORD $0x04C12000              // UADDV D0, P0, Z0.D
    
    FMOVD F0, R0
    MOVD R0, ret+48(FP)
    RET

// ┌──────────────────────────────────────────────────────────────────────────────┐
// │ func sumSqInt64SVE(vals []int64) int64                                       │
// │                                                                              │
// │ ⭐ Uses native SVE MUL for 64-bit integer square!                            │
// │    (NEON version must use scalar MUL)                                        │
// │                                                                              │
// │   Load:   Z8 = [ v0 | v1 | v2 | v3 ]                                         │
// │                   ×    ×    ×    ×    MUL Z8, Z8                              │
// │   Square: Z8 = [ v0² | v1² | v2² | v3² ]                                     │
// │                   +    +    +    +    ADD to accumulator                     │
// │   Acc:    Z0 = [ s0+v0² | s1+v1² | s2+v2² | s3+v3² ]                         │
// └──────────────────────────────────────────────────────────────────────────────┘
TEXT ·sumSqInt64SVE(SB), NOSPLIT, $0-32
    MOVD vals_base+0(FP), R0
    MOVD vals_len+8(FP), R1
    
    WORD $0x2518E3E0              // PTRUE P0.D
    
    // Zero 8 accumulators
    WORD $0x25F8C000              // DUP Z0.D, #0
    WORD $0x25F8C001              // DUP Z1.D, #0
    WORD $0x25F8C002              // DUP Z2.D, #0
    WORD $0x25F8C003              // DUP Z3.D, #0
    WORD $0x25F8C004              // DUP Z4.D, #0
    WORD $0x25F8C005              // DUP Z5.D, #0
    WORD $0x25F8C006              // DUP Z6.D, #0
    WORD $0x25F8C007              // DUP Z7.D, #0
    
    WORD $0x04E0E3E2              // CNTD X2
    
    LSL $3, R2, R3                // R3 = R2 * 8
    
    CMP R3, R1
    BLT sve_sumsq_tail

sve_sumsq_loop8:
    // Load 8 vectors
    WORD $0xA5E0A008              // LD1D {Z8.D}, P0/Z, [R0, #0, MUL VL]
    WORD $0xA5E1A009              // LD1D {Z9.D}, P0/Z, [R0, #1, MUL VL]
    WORD $0xA5E2A00A              // LD1D {Z10.D}, P0/Z, [R0, #2, MUL VL]
    WORD $0xA5E3A00B              // LD1D {Z11.D}, P0/Z, [R0, #3, MUL VL]
    WORD $0xA5E4A00C              // LD1D {Z12.D}, P0/Z, [R0, #4, MUL VL]
    WORD $0xA5E5A00D              // LD1D {Z13.D}, P0/Z, [R0, #5, MUL VL]
    WORD $0xA5E6A00E              // LD1D {Z14.D}, P0/Z, [R0, #6, MUL VL]
    WORD $0xA5E7A00F              // LD1D {Z15.D}, P0/Z, [R0, #7, MUL VL]
    
    // ┌────────────────────────────────────────────────────────────────────┐
    // │ Square: MUL Zn, Zn, Zn (v² = v × v)                               │
    // │                                                                    │
    // │   Z8 = [ 3 | -2 | 5 | -1 ]                                         │
    // │         ×     ×   ×    ×                                           │
    // │   Z8 = [ 9 |  4 | 25|  1 ]                                         │
    // └────────────────────────────────────────────────────────────────────┘
    WORD $0x04D00108              // MUL Z8.D, P0/M, Z8.D, Z8.D
    WORD $0x04D00129              // MUL Z9.D, P0/M, Z9.D, Z9.D
    WORD $0x04D0014A              // MUL Z10.D, P0/M, Z10.D, Z10.D
    WORD $0x04D0016B              // MUL Z11.D, P0/M, Z11.D, Z11.D
    WORD $0x04D0018C              // MUL Z12.D, P0/M, Z12.D, Z12.D
    WORD $0x04D001AD              // MUL Z13.D, P0/M, Z13.D, Z13.D
    WORD $0x04D001CE              // MUL Z14.D, P0/M, Z14.D, Z14.D
    WORD $0x04D001EF              // MUL Z15.D, P0/M, Z15.D, Z15.D
    
    // Accumulate: Z0-Z7 += Z8-Z15 (squared values)
    WORD $0x04C00100              // ADD Z0.D, P0/M, Z0.D, Z8.D
    WORD $0x04C00121              // ADD Z1.D, P0/M, Z1.D, Z9.D
    WORD $0x04C00142              // ADD Z2.D, P0/M, Z2.D, Z10.D
    WORD $0x04C00163              // ADD Z3.D, P0/M, Z3.D, Z11.D
    WORD $0x04C00184              // ADD Z4.D, P0/M, Z4.D, Z12.D
    WORD $0x04C001A5              // ADD Z5.D, P0/M, Z5.D, Z13.D
    WORD $0x04C001C6              // ADD Z6.D, P0/M, Z6.D, Z14.D
    WORD $0x04C001E7              // ADD Z7.D, P0/M, Z7.D, Z15.D
    
    LSL $3, R3, R5
    ADD R5, R0, R0
    SUBS R3, R1, R1
    CMP R3, R1
    BGE sve_sumsq_loop8

sve_sumsq_tail:
    MOVD ZR, R4
    CBZ R1, sve_sumsq_reduce
    
sve_sumsq_tail_loop:
    WORD $0x25E11C81              // WHILELO P1.D, X4, X1
    BEQ sve_sumsq_reduce
    
    WORD $0xA5E0A408              // LD1D {Z8.D}, P1/Z, [R0, #0, MUL VL]
    WORD $0x04D00508              // MUL Z8.D, P1/M, Z8.D, Z8.D
    WORD $0x04C00500              // ADD Z0.D, P1/M, Z0.D, Z8.D
    
    WORD $0x04F0E3E4              // INCD X4
    LSL $3, R2, R5
    ADD R5, R0, R0
    B sve_sumsq_tail_loop

sve_sumsq_reduce:
    // Combine accumulators: 8 → 4 → 2 → 1
    WORD $0x04C00080              // ADD Z0.D, P0/M, Z0.D, Z4.D
    WORD $0x04C000A1              // ADD Z1.D, P0/M, Z1.D, Z5.D
    WORD $0x04C000C2              // ADD Z2.D, P0/M, Z2.D, Z6.D
    WORD $0x04C000E3              // ADD Z3.D, P0/M, Z3.D, Z7.D
    WORD $0x04C00040              // ADD Z0.D, P0/M, Z0.D, Z2.D
    WORD $0x04C00061              // ADD Z1.D, P0/M, Z1.D, Z3.D
    WORD $0x04C00020              // ADD Z0.D, P0/M, Z0.D, Z1.D
    
    // UADDV - horizontal sum
    WORD $0x04C12000              // UADDV D0, P0, Z0.D
    
    FMOVD F0, R0
    MOVD R0, ret+24(FP)
    RET

// ┌──────────────────────────────────────────────────────────────────────────────┐
// │ func anyAbsGreaterThanSVE(vals []int64, threshold int64) bool                │
// │                                                                              │
// │ Strategy: Check (v > threshold) OR (v < -threshold) using predicates         │
// │                                                                              │
// │ ⚡ SHORT-CIRCUIT: Uses PTEST to check if any lane matched after each block   │
// │    Returns immediately when a match is found.                                │
// │                                                                              │
// │ Logic visualization:                                                         │
// │   ┌─────────────────────────────────────────────────────────────────┐        │
// │   │  threshold = 100      -threshold = -100                        │        │
// │   │                                                                 │        │
// │   │     ←── MATCH ──┤ no match ├←── MATCH ──→                       │        │
// │   │  ───────────────┼──────────┼────────────────                    │        │
// │   │           -100     0      100                                   │        │
// │   │                                                                 │        │
// │   │  CMPGT Z8, Z0 → P1 = (vals > threshold)                         │        │
// │   │  CMPGT Z1, Z8 → P2 = (-threshold > vals) = (vals < -threshold)  │        │
// │   │  ORR P1, P2 → any_match                                         │        │
// │   │  PTEST → early exit if match found                              │        │
// │   └─────────────────────────────────────────────────────────────────┘        │
// └──────────────────────────────────────────────────────────────────────────────┘
TEXT ·anyAbsGreaterThanSVE(SB), NOSPLIT, $0-33
    MOVD vals_base+0(FP), R0
    MOVD vals_len+8(FP), R1
    MOVD threshold+24(FP), R2     // R2 = threshold
    
    CBZ R1, sve_absgt_notfound
    
    // PTRUE P0.D (doubleword granularity)
    WORD $0x25D8E3E0
    
    // ┌────────────────────────────────────────────────────────────────────┐
    // │ Broadcast threshold and -threshold to vectors                    │
    // │                                                                    │
    // │   Z0 = [ threshold | threshold | threshold | ... ]                 │
    // │   Z1 = [-threshold |-threshold |-threshold | ... ]                 │
    // └────────────────────────────────────────────────────────────────────┘
    WORD $0x05E03840              // DUP Z0.D, X2 (threshold)
    
    NEG R2, R4                    // R4 = -threshold
    WORD $0x05E03881              // DUP Z1.D, X4 (-threshold)
    
    WORD $0x04E0E3E2              // CNTD X2
    
    LSL $3, R2, R3                // R3 = elements per iteration
    
    CMP R3, R1
    BLT sve_absgt_tail

// ┌──────────────────────────────────────────────────────────────────────────────┐
// │ Main loop: Check 8 vectors per iteration                                     │
// │                                                                              │
// │ For each vector:                                                             │
// │   1. CMPGT P.D, Pg/Z, Zn.D, Z0.D → P = (vals > threshold)                   │
// │   2. CMPGT P.D, Pg/Z, Z1.D, Zn.D → P' = (-thresh > vals) = (vals < -thresh) │
// │   3. ORR predicates together                                                 │
// │   4. Tree-reduce all 8 predicates into one                                   │
// │   5. PTEST: If any bit set → FOUND!                                          │
// └──────────────────────────────────────────────────────────────────────────────┘
sve_absgt_loop8:
    // Load 8 vectors
    WORD $0xA5E0A008              // LD1D {Z8.D}, P0/Z, [R0, #0, MUL VL]
    WORD $0xA5E1A009              // LD1D {Z9.D}, P0/Z, [R0, #1, MUL VL]
    WORD $0xA5E2A00A              // LD1D {Z10.D}, P0/Z, [R0, #2, MUL VL]
    WORD $0xA5E3A00B              // LD1D {Z11.D}, P0/Z, [R0, #3, MUL VL]
    WORD $0xA5E4A00C              // LD1D {Z12.D}, P0/Z, [R0, #4, MUL VL]
    WORD $0xA5E5A00D              // LD1D {Z13.D}, P0/Z, [R0, #5, MUL VL]
    WORD $0xA5E6A00E              // LD1D {Z14.D}, P0/Z, [R0, #6, MUL VL]
    WORD $0xA5E7A00F              // LD1D {Z15.D}, P0/Z, [R0, #7, MUL VL]
    
    // Check v > threshold: CMPGT Pd.D, P0/Z, Zn.D, Z0.D → P1-P8
    WORD $0x24C08111              // CMPGT P1.D, P0/Z, Z8.D, Z0.D
    WORD $0x24C08132              // CMPGT P2.D, P0/Z, Z9.D, Z0.D
    WORD $0x24C08153              // CMPGT P3.D, P0/Z, Z10.D, Z0.D
    WORD $0x24C08174              // CMPGT P4.D, P0/Z, Z11.D, Z0.D
    WORD $0x24C08195              // CMPGT P5.D, P0/Z, Z12.D, Z0.D
    WORD $0x24C081B6              // CMPGT P6.D, P0/Z, Z13.D, Z0.D
    WORD $0x24C081D7              // CMPGT P7.D, P0/Z, Z14.D, Z0.D
    WORD $0x24C081F8              // CMPGT P8.D, P0/Z, Z15.D, Z0.D
    
    // ┌────────────────────────────────────────────────────────────────────┐
    // │ Check v < -threshold (via reversed comparison):                   │
    // │   CMPGT P9, P0/Z, Z1, Z8 → P9 = (-threshold > val) = (val < -thr) │
    // │   ORR P1, P1, P9 → combine with "v > threshold" result            │
    // └────────────────────────────────────────────────────────────────────┘
    WORD $0x24C88039              // CMPGT P9.D, P0/Z, Z1.D, Z8.D
    WORD $0x25894021              // ORR P1.B, P0/Z, P1.B, P9.B
    WORD $0x24C9803A              // CMPGT P10.D, P0/Z, Z1.D, Z9.D
    WORD $0x258A4042              // ORR P2.B, P0/Z, P2.B, P10.B
    WORD $0x24CA803B              // CMPGT P11.D, P0/Z, Z1.D, Z10.D
    WORD $0x258B4063              // ORR P3.B, P0/Z, P3.B, P11.B
    WORD $0x24CB803C              // CMPGT P12.D, P0/Z, Z1.D, Z11.D
    WORD $0x258C4084              // ORR P4.B, P0/Z, P4.B, P12.B
    WORD $0x24CC803D              // CMPGT P13.D, P0/Z, Z1.D, Z12.D
    WORD $0x258D40A5              // ORR P5.B, P0/Z, P5.B, P13.B
    WORD $0x24CD803E              // CMPGT P14.D, P0/Z, Z1.D, Z13.D
    WORD $0x258E40C6              // ORR P6.B, P0/Z, P6.B, P14.B
    WORD $0x24CE803F              // CMPGT P15.D, P0/Z, Z1.D, Z14.D
    WORD $0x258F40E7              // ORR P7.B, P0/Z, P7.B, P15.B
    // Reuse P9 for Z15
    WORD $0x24CF8039              // CMPGT P9.D, P0/Z, Z1.D, Z15.D
    WORD $0x25894108              // ORR P8.B, P0/Z, P8.B, P9.B
    
    // ╔════════════════════════════════════════════════════════════════════════╗
    // ║ Tree reduction of predicates: 8 → 4 → 2 → 1                            ║
    // ║                                                                        ║
    // ║   P1 ═╦═ P2       P3 ═╦═ P4       P5 ═╦═ P6       P7 ═╦═ P8            ║
    // ║      ╚═► P1          ╚═► P3          ╚═► P5          ╚═► P7            ║
    // ║                                                                        ║
    // ║   P1 ═══════╦═══════ P3          P5 ═══════╦═══════ P7                 ║
    // ║            ╚════════► P1                  ╚════════► P5                ║
    // ║                                                                        ║
    // ║   P1 ═══════════════════════════════╦═══════════════════ P5            ║
    // ║                                    ╚════════════════════► P1           ║
    // ╚════════════════════════════════════════════════════════════════════════╝
    WORD $0x25824021              // ORR P1.B, P0/Z, P1.B, P2.B
    WORD $0x25844063              // ORR P3.B, P0/Z, P3.B, P4.B
    WORD $0x258640A5              // ORR P5.B, P0/Z, P5.B, P6.B
    WORD $0x258840E7              // ORR P7.B, P0/Z, P7.B, P8.B
    WORD $0x25834021              // ORR P1.B, P0/Z, P1.B, P3.B
    WORD $0x258740A5              // ORR P5.B, P0/Z, P5.B, P7.B
    WORD $0x25854021              // ORR P1.B, P0/Z, P1.B, P5.B
    
    // ┌────────────────────────────────────────────────────────────────────┐
    // │ PTEST: Test if any predicate bit is set → early exit if found     │
    // │                                                                    │
    // │   PTEST P0, P1.B sets Z flag if P1 is all zeros                   │
    // │   BNE → branch if NOT equal (i.e., some bit was set = FOUND)      │
    // └────────────────────────────────────────────────────────────────────┘
    WORD $0x2550C020              // PTEST P0, P1.B
    BNE sve_absgt_found           // ⚡ Early exit if match found!
    
    // Advance pointer
    LSL $3, R3, R5
    ADD R5, R0, R0
    SUB R3, R1, R1
    CMP R3, R1
    BGE sve_absgt_loop8

sve_absgt_tail:
    CBZ R1, sve_absgt_notfound

sve_absgt_tail_loop:
    // Create predicate for remaining elements
    WORD $0x25E11FE3              // WHILELO P3.D, XZR, X1
    BEQ sve_absgt_notfound
    
    WORD $0xA5E0AC08              // LD1D {Z8.D}, P3/Z, [R0, #0, MUL VL]
    
    // Check v > threshold
    WORD $0x24C08D12              // CMPGT P2.D, P3/Z, Z8.D, Z0.D
    
    // Check v < -threshold
    WORD $0x24C88C31              // CMPGT P1.D, P3/Z, Z1.D, Z8.D
    
    // Combine results
    WORD $0x25824C21              // ORR P1.B, P3/Z, P1.B, P2.B
    
    // Check if any lane matched
    WORD $0x2550CC20              // PTEST P3, P1.B
    BNE sve_absgt_found
    
    // Advance
    LSL $3, R2, R5
    ADD R5, R0, R0
    SUBS R2, R1, R1
    BGT sve_absgt_tail_loop
    B sve_absgt_notfound

sve_absgt_found:
    MOVD $1, R0
    MOVB R0, ret+32(FP)
    RET

sve_absgt_notfound:
    MOVD $0, R0
    MOVB R0, ret+32(FP)
    RET
