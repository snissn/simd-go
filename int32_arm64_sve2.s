//go:build arm64

#include "textflag.h"

// ╔══════════════════════════════════════════════════════════════════════════════╗
// ║                          SVE2 Int32 SIMD Operations                          ║
// ║                                                                              ║
// ║  SVE2 adds fused widening multiply-accumulate instructions:                  ║
// ║  • SMLALB/SMLALT - Signed Multiply-Add Long (bottom/top)                     ║
// ║  • SMULLB/SMULLT - Signed Multiply Long (bottom/top)                         ║
// ║  • SADALP - Signed Add and Accumulate Long Pairwise                          ║
// ║                                                                              ║
// ║  These replace the SVE pattern of SUNPKLO/SUNPKHI + MUL + ADD               ║
// ╚══════════════════════════════════════════════════════════════════════════════╝

// ┌──────────────────────────────────────────────────────────────────────────────┐
// │ func dotProductInt32SVE2(a, b []int32) int64                                 │
// │                                                                              │
// │ Strategy: Use SMLALB/SMLALT for fused widen+multiply+accumulate             │
// │                                                                              │
// │ SVE pattern (10 instructions per pair of vectors):                           │
// │   SUNPKLO z20, z8 → SUNPKHI z21, z8 → SUNPKLO z22, z16 → SUNPKHI z23, z16   │
// │   → MUL z20, z22 → MUL z21, z23 → ADD acc, z20 → ADD acc, z21               │
// │                                                                              │
// │ SVE2 pattern (2 instructions per pair of vectors):                           │
// │   SMLALB acc, z8, z16  (bottom halves: widen + multiply + accumulate)        │
// │   SMLALT acc, z8, z16  (top halves: widen + multiply + accumulate)           │
// └──────────────────────────────────────────────────────────────────────────────┘
TEXT ·dotProductInt32SVE2(SB), NOSPLIT, $0-56
    MOVD a_base+0(FP), R0
    MOVD a_len+8(FP), R1
    MOVD b_base+24(FP), R2
    
    // PTRUE P0.S - all 32-bit lanes active
    WORD $0x2518E3E0              // PTRUE P0.S
    
    // Zero 4 int64 accumulators (Z0-Z3)
    // Each accumulator holds results from bottom/top halves
    WORD $0x25F8C000              // DUP Z0.D, #0
    WORD $0x25F8C001              // DUP Z1.D, #0
    WORD $0x25F8C002              // DUP Z2.D, #0
    WORD $0x25F8C003              // DUP Z3.D, #0
    
    // CNTW - count 32-bit elements per vector
    WORD $0x04A0E3E3              // CNTW X3
    LSL $2, R3, R4                // R4 = R3 * 4 (4 vectors per iteration)
    
    CMP R4, R1
    BLT sve2_dot32_tail

sve2_dot32_loop4:
    // Load 4 vectors from each array
    WORD $0xA540A008              // LD1W {Z8.S}, P0/Z, [R0, #0, MUL VL]
    WORD $0xA541A009              // LD1W {Z9.S}, P0/Z, [R0, #1, MUL VL]
    WORD $0xA542A00A              // LD1W {Z10.S}, P0/Z, [R0, #2, MUL VL]
    WORD $0xA543A00B              // LD1W {Z11.S}, P0/Z, [R0, #3, MUL VL]
    
    WORD $0xA540A050              // LD1W {Z16.S}, P0/Z, [R2, #0, MUL VL]
    WORD $0xA541A051              // LD1W {Z17.S}, P0/Z, [R2, #1, MUL VL]
    WORD $0xA542A052              // LD1W {Z18.S}, P0/Z, [R2, #2, MUL VL]
    WORD $0xA543A053              // LD1W {Z19.S}, P0/Z, [R2, #3, MUL VL]
    
    // SMLALB/SMLALT: Signed Multiply-Add Long (bottom/top)
    // SMLALB Zda.D, Zn.S, Zm.S: Zda += Zn_bottom * Zm_bottom (widening to int64)
    // SMLALT Zda.D, Zn.S, Zm.S: Zda += Zn_top * Zm_top (widening to int64)
    
    // Z8 * Z16 -> accumulate into Z0 (bottom) and Z1 (top)
    WORD $0x44D04100              // SMLALB Z0.D, Z8.S, Z16.S
    WORD $0x44D04501              // SMLALT Z1.D, Z8.S, Z16.S
    
    // Z9 * Z17 -> accumulate into Z2 (bottom) and Z3 (top)
    WORD $0x44D14122              // SMLALB Z2.D, Z9.S, Z17.S
    WORD $0x44D14523              // SMLALT Z3.D, Z9.S, Z17.S
    
    // Z10 * Z18 -> accumulate into Z0, Z1
    WORD $0x44D24140              // SMLALB Z0.D, Z10.S, Z18.S
    WORD $0x44D24541              // SMLALT Z1.D, Z10.S, Z18.S
    
    // Z11 * Z19 -> accumulate into Z2, Z3
    WORD $0x44D34162              // SMLALB Z2.D, Z11.S, Z19.S
    WORD $0x44D34563              // SMLALT Z3.D, Z11.S, Z19.S
    
    // Advance pointers
    LSL $2, R4, R5                // R5 = R4 * 4 bytes
    ADD R5, R0, R0
    ADD R5, R2, R2
    SUB R4, R1, R1
    CMP R4, R1
    BGE sve2_dot32_loop4

sve2_dot32_tail:
    CBZ R1, sve2_dot32_reduce
    MOVD ZR, R5                   // R5 = 0 (loop index)

sve2_dot32_tail_loop:
    WORD $0x25A11CA2              // WHILELO P2.S, X5, X1
    BEQ sve2_dot32_reduce
    
    WORD $0xA540A808              // LD1W {Z8.S}, P2/Z, [R0]
    WORD $0xA540A850              // LD1W {Z16.S}, P2/Z, [R2]
    
    // SMLALB/SMLALT for tail
    WORD $0x44D04100              // SMLALB Z0.D, Z8.S, Z16.S
    WORD $0x44D04501              // SMLALT Z1.D, Z8.S, Z16.S
    
    WORD $0x04B0E3E5              // INCW X5
    LSL $2, R3, R6                // bytes per vector = R3 * 4
    ADD R6, R0, R0
    ADD R6, R2, R2
    B sve2_dot32_tail_loop

sve2_dot32_reduce:
    // Combine accumulators: Z0 += Z2, Z1 += Z3
    WORD $0x2518E3E1              // PTRUE P1.D
    WORD $0x04C00440              // ADD Z0.D, P1/M, Z0.D, Z2.D
    WORD $0x04C00461              // ADD Z1.D, P1/M, Z1.D, Z3.D
    WORD $0x04C00420              // ADD Z0.D, P1/M, Z0.D, Z1.D
    
    // UADDV - horizontal sum
    WORD $0x04C12001              // UADDV D1, P1, Z0.D
    FMOVD F1, R0
    
    MOVD R0, ret+48(FP)
    RET

// ┌──────────────────────────────────────────────────────────────────────────────┐
// │ func sumSqInt32SVE2(vals []int32) int64                                      │
// │                                                                              │
// │ Strategy: Use SMLALB/SMLALT with same operand for squaring                  │
// │           Zda += Zn * Zn (widening square + accumulate)                      │
// └──────────────────────────────────────────────────────────────────────────────┘
TEXT ·sumSqInt32SVE2(SB), NOSPLIT, $0-32
    MOVD vals_base+0(FP), R0
    MOVD vals_len+8(FP), R1
    
    WORD $0x2518E3E0              // PTRUE P0.S
    
    // Zero 4 int64 accumulators
    WORD $0x25F8C000              // DUP Z0.D, #0
    WORD $0x25F8C001              // DUP Z1.D, #0
    WORD $0x25F8C002              // DUP Z2.D, #0
    WORD $0x25F8C003              // DUP Z3.D, #0
    
    WORD $0x04A0E3E3              // CNTW X3
    LSL $2, R3, R4                // R4 = R3 * 4 (4 vectors per iteration)
    
    CMP R4, R1
    BLT sve2_sumsq32_tail

sve2_sumsq32_loop4:
    // Load 4 vectors
    WORD $0xA540A008              // LD1W {Z8.S}, P0/Z, [R0, #0, MUL VL]
    WORD $0xA541A009              // LD1W {Z9.S}, P0/Z, [R0, #1, MUL VL]
    WORD $0xA542A00A              // LD1W {Z10.S}, P0/Z, [R0, #2, MUL VL]
    WORD $0xA543A00B              // LD1W {Z11.S}, P0/Z, [R0, #3, MUL VL]
    
    // Square using SMLALB/SMLALT with same operand
    // Z8^2 -> Z0 (bottom), Z1 (top)
    WORD $0x44C84100              // SMLALB Z0.D, Z8.S, Z8.S
    WORD $0x44C84501              // SMLALT Z1.D, Z8.S, Z8.S
    
    // Z9^2 -> Z2, Z3
    WORD $0x44C94122              // SMLALB Z2.D, Z9.S, Z9.S
    WORD $0x44C94523              // SMLALT Z3.D, Z9.S, Z9.S
    
    // Z10^2 -> Z0, Z1
    WORD $0x44CA4140              // SMLALB Z0.D, Z10.S, Z10.S
    WORD $0x44CA4541              // SMLALT Z1.D, Z10.S, Z10.S
    
    // Z11^2 -> Z2, Z3
    WORD $0x44CB4162              // SMLALB Z2.D, Z11.S, Z11.S
    WORD $0x44CB4563              // SMLALT Z3.D, Z11.S, Z11.S
    
    // Advance pointer
    LSL $2, R4, R5
    ADD R5, R0, R0
    SUB R4, R1, R1
    CMP R4, R1
    BGE sve2_sumsq32_loop4

sve2_sumsq32_tail:
    CBZ R1, sve2_sumsq32_reduce
    MOVD ZR, R5

sve2_sumsq32_tail_loop:
    WORD $0x25A11CA2              // WHILELO P2.S, X5, X1
    BEQ sve2_sumsq32_reduce
    
    WORD $0xA540A808              // LD1W {Z8.S}, P2/Z, [R0]
    
    WORD $0x44C84100              // SMLALB Z0.D, Z8.S, Z8.S
    WORD $0x44C84501              // SMLALT Z1.D, Z8.S, Z8.S
    
    WORD $0x04B0E3E5              // INCW X5
    LSL $2, R3, R6
    ADD R6, R0, R0
    B sve2_sumsq32_tail_loop

sve2_sumsq32_reduce:
    WORD $0x2518E3E1              // PTRUE P1.D
    WORD $0x04C00440              // ADD Z0.D, P1/M, Z0.D, Z2.D
    WORD $0x04C00461              // ADD Z1.D, P1/M, Z1.D, Z3.D
    WORD $0x04C00420              // ADD Z0.D, P1/M, Z0.D, Z1.D
    
    WORD $0x04C12001              // UADDV D1, P1, Z0.D
    FMOVD F1, R0
    
    MOVD R0, ret+24(FP)
    RET

// ┌──────────────────────────────────────────────────────────────────────────────┐
// │ func sumInt32SVE2(vals []int32) int64                                        │
// │                                                                              │
// │ Strategy: Use SADALP for pairwise add with widening accumulation            │
// │           This replaces SUNPKLO/SUNPKHI + ADD pattern                        │
// │                                                                              │
// │ SADALP Zda.D, Pg/M, Zn.S:                                                   │
// │   For each pair of 32-bit elements in Zn, add them and accumulate           │
// │   into corresponding 64-bit element in Zda                                   │
// └──────────────────────────────────────────────────────────────────────────────┘
TEXT ·sumInt32SVE2(SB), NOSPLIT, $0-32
    MOVD vals_base+0(FP), R0
    MOVD vals_len+8(FP), R1
    
    // PTRUE P0.S for loads, P1.D for 64-bit operations
    WORD $0x2518E3E0              // PTRUE P0.S
    WORD $0x2518E3E1              // PTRUE P1.D
    
    // Zero 4 int64 accumulators
    WORD $0x25F8C000              // DUP Z0.D, #0
    WORD $0x25F8C001              // DUP Z1.D, #0
    WORD $0x25F8C002              // DUP Z2.D, #0
    WORD $0x25F8C003              // DUP Z3.D, #0
    
    WORD $0x04A0E3E3              // CNTW X3
    LSL $3, R3, R4                // R4 = R3 * 8 (8 vectors per iteration)
    
    CMP R4, R1
    BLT sve2_sum32_tail

sve2_sum32_loop8:
    // Load 8 vectors
    WORD $0xA540A008              // LD1W {Z8.S}, P0/Z, [R0, #0, MUL VL]
    WORD $0xA541A009              // LD1W {Z9.S}, P0/Z, [R0, #1, MUL VL]
    WORD $0xA542A00A              // LD1W {Z10.S}, P0/Z, [R0, #2, MUL VL]
    WORD $0xA543A00B              // LD1W {Z11.S}, P0/Z, [R0, #3, MUL VL]
    WORD $0xA544A00C              // LD1W {Z12.S}, P0/Z, [R0, #4, MUL VL]
    WORD $0xA545A00D              // LD1W {Z13.S}, P0/Z, [R0, #5, MUL VL]
    WORD $0xA546A00E              // LD1W {Z14.S}, P0/Z, [R0, #6, MUL VL]
    WORD $0xA547A00F              // LD1W {Z15.S}, P0/Z, [R0, #7, MUL VL]
    
    // SADALP: Signed Add and Accumulate Long Pairwise
    // SADALP Zda.D, P/M, Zn.S: pairs of int32 -> widened sum accumulated to int64
    // Encoding: 0x44c4a0XX for .D,.S with P0
    WORD $0x44C4A100              // SADALP Z0.D, P0/M, Z8.S
    WORD $0x44C4A121              // SADALP Z1.D, P0/M, Z9.S
    WORD $0x44C4A142              // SADALP Z2.D, P0/M, Z10.S
    WORD $0x44C4A163              // SADALP Z3.D, P0/M, Z11.S
    WORD $0x44C4A180              // SADALP Z0.D, P0/M, Z12.S
    WORD $0x44C4A1A1              // SADALP Z1.D, P0/M, Z13.S
    WORD $0x44C4A1C2              // SADALP Z2.D, P0/M, Z14.S
    WORD $0x44C4A1E3              // SADALP Z3.D, P0/M, Z15.S
    
    // Advance pointer
    LSL $2, R4, R5                // R5 = R4 * 4 bytes
    ADD R5, R0, R0
    SUB R4, R1, R1
    CMP R4, R1
    BGE sve2_sum32_loop8

sve2_sum32_tail:
    CBZ R1, sve2_sum32_reduce
    MOVD ZR, R5

sve2_sum32_tail_loop:
    WORD $0x25A11CA2              // WHILELO P2.S, X5, X1
    BEQ sve2_sum32_reduce
    
    WORD $0xA540A808              // LD1W {Z8.S}, P2/Z, [R0]
    
    // SADALP with P2 predicate - need correct encoding
    // For tail, we need to be careful with partial vectors
    // Use P0 since we loaded with P2 (inactive lanes are zero)
    WORD $0x44C4A100              // SADALP Z0.D, P0/M, Z8.S
    
    WORD $0x04B0E3E5              // INCW X5
    LSL $2, R3, R6
    ADD R6, R0, R0
    B sve2_sum32_tail_loop

sve2_sum32_reduce:
    // Tree reduction: 4 -> 2 -> 1
    WORD $0x04C00440              // ADD Z0.D, P1/M, Z0.D, Z2.D
    WORD $0x04C00461              // ADD Z1.D, P1/M, Z1.D, Z3.D
    WORD $0x04C00420              // ADD Z0.D, P1/M, Z0.D, Z1.D
    
    // UADDV - horizontal sum
    WORD $0x04C12001              // UADDV D1, P1, Z0.D
    FMOVD F1, R0
    
    MOVD R0, ret+24(FP)
    RET
