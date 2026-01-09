//go:build arm64

#include "textflag.h"

// ╔══════════════════════════════════════════════════════════════════════════════╗
// ║                           SVE Int32 SIMD Operations                          ║
// ║                                                                              ║
// ║  SVE processes variable number of int32 per vector (VL-dependent)           ║
// ║  At 256-bit: 8 x int32 per vector                                            ║
// ║                                                                              ║
// ║  Key SVE int32 instructions:                                                 ║
// ║  • LD1W - load 32-bit elements                                               ║
// ║  • ADD, MUL, SMIN, SMAX - native operations                                  ║
// ║  • SADDV - horizontal sum with widening to 64-bit                            ║
// ║  • SMINV, SMAXV - horizontal min/max                                         ║
// ╚══════════════════════════════════════════════════════════════════════════════╝

// ┌──────────────────────────────────────────────────────────────────────────────┐
// │ func sumInt32SVE(vals []int32) int64                                         │
// │                                                                              │
// │ Strategy: 8 vector accumulators, UADDV for horizontal reduction              │
// │ Accumulates in 64-bit to avoid overflow                                      │
// └──────────────────────────────────────────────────────────────────────────────┘
TEXT ·sumInt32SVE(SB), NOSPLIT, $0-32
    MOVD vals_base+0(FP), R0
    MOVD vals_len+8(FP), R1
    
    // PTRUE P0.S - Enable all 32-bit lanes
    WORD $0x2598E3E0              // PTRUE P0.S
    
    // Zero 8 accumulators (using 64-bit lanes for accumulation)
    WORD $0x25F8C000              // DUP Z0.D, #0
    WORD $0x25F8C001              // DUP Z1.D, #0
    WORD $0x25F8C002              // DUP Z2.D, #0
    WORD $0x25F8C003              // DUP Z3.D, #0
    WORD $0x25F8C004              // DUP Z4.D, #0
    WORD $0x25F8C005              // DUP Z5.D, #0
    WORD $0x25F8C006              // DUP Z6.D, #0
    WORD $0x25F8C007              // DUP Z7.D, #0
    
    // CNTW - count 32-bit elements per vector
    WORD $0x04A0E3E2              // CNTW X2
    
    LSL $3, R2, R3                // R3 = R2 * 8 (8 vectors per iteration)
    
    CMP R3, R1
    BLT sve_sum32_tail

sve_sum32_loop8:
    // Load 8 vectors of int32
    WORD $0xA540A008              // LD1W {Z8.S}, P0/Z, [R0, #0, MUL VL]
    WORD $0xA541A009              // LD1W {Z9.S}, P0/Z, [R0, #1, MUL VL]
    WORD $0xA542A00A              // LD1W {Z10.S}, P0/Z, [R0, #2, MUL VL]
    WORD $0xA543A00B              // LD1W {Z11.S}, P0/Z, [R0, #3, MUL VL]
    WORD $0xA544A00C              // LD1W {Z12.S}, P0/Z, [R0, #4, MUL VL]
    WORD $0xA545A00D              // LD1W {Z13.S}, P0/Z, [R0, #5, MUL VL]
    WORD $0xA546A00E              // LD1W {Z14.S}, P0/Z, [R0, #6, MUL VL]
    WORD $0xA547A00F              // LD1W {Z15.S}, P0/Z, [R0, #7, MUL VL]
    
    // Sign-extend and add: use SADDV to widen each vector, then scalar add
    // Alternative: SUNPKLO/SUNPKHI to widen, then ADD
    // For simplicity, widen to 64-bit accumulators using unpacks
    
    // SUNPKLO/SUNPKHI: sign-extend 32-bit to 64-bit
    // Encoding: 00000101 size[1:0] [fixed 12-bit] Zn[4:0] Zd[4:0]
    // For S->D: size=11, SUNPKLO fixed=0xC0E, SUNPKHI fixed=0xC4E
    WORD $0x05F03910              // SUNPKLO Z16.D, Z8.S
    WORD $0x05F13911              // SUNPKHI Z17.D, Z8.S
    WORD $0x05F03932              // SUNPKLO Z18.D, Z9.S
    WORD $0x05F13933              // SUNPKHI Z19.D, Z9.S
    WORD $0x05F03954              // SUNPKLO Z20.D, Z10.S
    WORD $0x05F13955              // SUNPKHI Z21.D, Z10.S
    WORD $0x05F03976              // SUNPKLO Z22.D, Z11.S
    WORD $0x05F13977              // SUNPKHI Z23.D, Z11.S
    
    // Add to accumulators (using PTRUE P0.D for 64-bit ops)
    WORD $0x25D8E3E1              // PTRUE P1.D
    WORD $0x04C00600              // ADD Z0.D, P1/M, Z0.D, Z16.D
    WORD $0x04C00621              // ADD Z1.D, P1/M, Z1.D, Z17.D
    WORD $0x04C00642              // ADD Z2.D, P1/M, Z2.D, Z18.D
    WORD $0x04C00663              // ADD Z3.D, P1/M, Z3.D, Z19.D
    WORD $0x04C00684              // ADD Z4.D, P1/M, Z4.D, Z20.D
    WORD $0x04C006A5              // ADD Z5.D, P1/M, Z5.D, Z21.D
    WORD $0x04C006C6              // ADD Z6.D, P1/M, Z6.D, Z22.D
    WORD $0x04C006E7              // ADD Z7.D, P1/M, Z7.D, Z23.D
    
    // Process remaining 4 vectors from Z12-Z15
    WORD $0x05F03990              // SUNPKLO Z16.D, Z12.S
    WORD $0x05F13991              // SUNPKHI Z17.D, Z12.S
    WORD $0x05F039B2              // SUNPKLO Z18.D, Z13.S
    WORD $0x05F139B3              // SUNPKHI Z19.D, Z13.S
    WORD $0x05F039D4              // SUNPKLO Z20.D, Z14.S
    WORD $0x05F139D5              // SUNPKHI Z21.D, Z14.S
    WORD $0x05F039F6              // SUNPKLO Z22.D, Z15.S
    WORD $0x05F139F7              // SUNPKHI Z23.D, Z15.S
    
    WORD $0x04C00600              // ADD Z0.D, P1/M, Z0.D, Z16.D
    WORD $0x04C00621              // ADD Z1.D, P1/M, Z1.D, Z17.D
    WORD $0x04C00642              // ADD Z2.D, P1/M, Z2.D, Z18.D
    WORD $0x04C00663              // ADD Z3.D, P1/M, Z3.D, Z19.D
    WORD $0x04C00684              // ADD Z4.D, P1/M, Z4.D, Z20.D
    WORD $0x04C006A5              // ADD Z5.D, P1/M, Z5.D, Z21.D
    WORD $0x04C006C6              // ADD Z6.D, P1/M, Z6.D, Z22.D
    WORD $0x04C006E7              // ADD Z7.D, P1/M, Z7.D, Z23.D
    
    // Advance pointer
    LSL $2, R3, R4                // R4 = R3 * 4 (bytes)
    ADD R4, R0, R0
    SUB R3, R1, R1
    CMP R3, R1
    BGE sve_sum32_loop8

sve_sum32_tail:
    MOVD ZR, R4                   // R4 = 0 (loop index)
    CBZ R1, sve_sum32_reduce
    
sve_sum32_tail_loop:
    // Use WHILELO for predicated tail loop
    WORD $0x25A11C81              // WHILELO P1.S, X4, X1
    BEQ sve_sum32_reduce          // Exit if no active lanes
    
    WORD $0xA540A408              // LD1W {Z8.S}, P1/Z, [R0]
    
    WORD $0x05F03910              // SUNPKLO Z16.D, Z8.S
    WORD $0x05F13911              // SUNPKHI Z17.D, Z8.S
    WORD $0x25D8E3E2              // PTRUE P2.D
    WORD $0x04C00A00              // ADD Z0.D, P2/M, Z0.D, Z16.D
    WORD $0x04C00A21              // ADD Z1.D, P2/M, Z1.D, Z17.D
    
    WORD $0x04B0E3E4              // INCW X4
    LSL $2, R2, R5                // bytes per vector = R2 * 4
    ADD R5, R0, R0
    B sve_sum32_tail_loop

sve_sum32_reduce:
    // Tree reduction: 8 -> 4 -> 2 -> 1
    WORD $0x25D8E3E1              // PTRUE P1.D
    WORD $0x04C00480              // ADD Z0.D, P1/M, Z0.D, Z4.D
    WORD $0x04C004A1              // ADD Z1.D, P1/M, Z1.D, Z5.D
    WORD $0x04C004C2              // ADD Z2.D, P1/M, Z2.D, Z6.D
    WORD $0x04C004E3              // ADD Z3.D, P1/M, Z3.D, Z7.D
    WORD $0x04C00440              // ADD Z0.D, P1/M, Z0.D, Z2.D
    WORD $0x04C00461              // ADD Z1.D, P1/M, Z1.D, Z3.D
    WORD $0x04C00420              // ADD Z0.D, P1/M, Z0.D, Z1.D
    
    // UADDV - unsigned horizontal sum
    WORD $0x04C12401              // UADDV D1, P1, Z0.D
    FMOVD F1, R0
    
    MOVD R0, ret+24(FP)
    RET

// ┌──────────────────────────────────────────────────────────────────────────────┐
// │ func minInt32SVE(vals []int32) int32                                         │
// │                                                                              │
// │ Strategy: Native SMIN.S and SMINV for horizontal reduction                   │
// └──────────────────────────────────────────────────────────────────────────────┘
TEXT ·minInt32SVE(SB), NOSPLIT, $0-28
    MOVD vals_base+0(FP), R0
    MOVD vals_len+8(FP), R1
    
    CBZ R1, sve_min32_empty
    
    WORD $0x2598E3E0              // PTRUE P0.S
    
    WORD $0x04A0E3E2              // CNTW X2
    
    // Initialize with first vector
    WORD $0xA540A000              // LD1W {Z0.S}, P0/Z, [R0]
    // MOV Z1-Z7 = Z0 using ORR Zd.D, Zn.D, Zn.D
    WORD $0x04603001              // ORR Z1.D, Z0.D, Z0.D
    WORD $0x04603002              // ORR Z2.D, Z0.D, Z0.D
    WORD $0x04603003              // ORR Z3.D, Z0.D, Z0.D
    WORD $0x04603004              // ORR Z4.D, Z0.D, Z0.D
    WORD $0x04603005              // ORR Z5.D, Z0.D, Z0.D
    WORD $0x04603006              // ORR Z6.D, Z0.D, Z0.D
    WORD $0x04603007              // ORR Z7.D, Z0.D, Z0.D
    
    // Advance past initialized vector
    LSL $2, R2, R4                // bytes per vector = R2 * 4
    ADD R4, R0, R0
    SUB R2, R1, R1
    
    LSL $3, R2, R3
    
    CMP R3, R1
    BLT sve_min32_tail

sve_min32_loop8:
    WORD $0xA540A008              // LD1W {Z8.S}, P0/Z, [R0, #0, MUL VL]
    WORD $0xA541A009              // LD1W {Z9.S}, P0/Z, [R0, #1, MUL VL]
    WORD $0xA542A00A              // LD1W {Z10.S}, P0/Z, [R0, #2, MUL VL]
    WORD $0xA543A00B              // LD1W {Z11.S}, P0/Z, [R0, #3, MUL VL]
    WORD $0xA544A00C              // LD1W {Z12.S}, P0/Z, [R0, #4, MUL VL]
    WORD $0xA545A00D              // LD1W {Z13.S}, P0/Z, [R0, #5, MUL VL]
    WORD $0xA546A00E              // LD1W {Z14.S}, P0/Z, [R0, #6, MUL VL]
    WORD $0xA547A00F              // LD1W {Z15.S}, P0/Z, [R0, #7, MUL VL]
    
    // SMIN.S
    WORD $0x048A0100              // SMIN Z0.S, P0/M, Z0.S, Z8.S
    WORD $0x048A0121              // SMIN Z1.S, P0/M, Z1.S, Z9.S
    WORD $0x048A0142              // SMIN Z2.S, P0/M, Z2.S, Z10.S
    WORD $0x048A0163              // SMIN Z3.S, P0/M, Z3.S, Z11.S
    WORD $0x048A0184              // SMIN Z4.S, P0/M, Z4.S, Z12.S
    WORD $0x048A01A5              // SMIN Z5.S, P0/M, Z5.S, Z13.S
    WORD $0x048A01C6              // SMIN Z6.S, P0/M, Z6.S, Z14.S
    WORD $0x048A01E7              // SMIN Z7.S, P0/M, Z7.S, Z15.S
    
    LSL $2, R3, R4
    ADD R4, R0, R0
    SUB R3, R1, R1
    CMP R3, R1
    BGE sve_min32_loop8

sve_min32_tail:
    MOVD ZR, R4                   // R4 = 0 (loop index)
    CBZ R1, sve_min32_reduce
    
sve_min32_tail_loop:
    WORD $0x25A11C82              // WHILELO P2.S, X4, X1
    BEQ sve_min32_reduce          // Exit if no active lanes
    
    WORD $0xA540A808              // LD1W {Z8.S}, P2/Z, [R0]
    
    // SMIN with predicate
    WORD $0x048A0900              // SMIN Z0.S, P2/M, Z0.S, Z8.S
    
    WORD $0x04B0E3E4              // INCW X4
    LSL $2, R2, R5                // bytes per vector = R2 * 4
    ADD R5, R0, R0
    B sve_min32_tail_loop

sve_min32_reduce:
    // Tree reduction
    WORD $0x048A0080              // SMIN Z0.S, P0/M, Z0.S, Z4.S
    WORD $0x048A00A1              // SMIN Z1.S, P0/M, Z1.S, Z5.S
    WORD $0x048A00C2              // SMIN Z2.S, P0/M, Z2.S, Z6.S
    WORD $0x048A00E3              // SMIN Z3.S, P0/M, Z3.S, Z7.S
    WORD $0x048A0040              // SMIN Z0.S, P0/M, Z0.S, Z2.S
    WORD $0x048A0061              // SMIN Z1.S, P0/M, Z1.S, Z3.S
    WORD $0x048A0020              // SMIN Z0.S, P0/M, Z0.S, Z1.S
    
    // SMINV - horizontal min
    WORD $0x048A2001              // SMINV S1, P0, Z0.S
    FMOVS F1, R0
    B sve_min32_done

sve_min32_empty:
    MOVW $0, R0

sve_min32_done:
    MOVW R0, ret+24(FP)
    RET

// ┌──────────────────────────────────────────────────────────────────────────────┐
// │ func maxInt32SVE(vals []int32) int32                                         │
// │                                                                              │
// │ Strategy: Native SMAX.S and SMAXV for horizontal reduction                   │
// └──────────────────────────────────────────────────────────────────────────────┘
TEXT ·maxInt32SVE(SB), NOSPLIT, $0-28
    MOVD vals_base+0(FP), R0
    MOVD vals_len+8(FP), R1
    
    CBZ R1, sve_max32_empty
    
    WORD $0x2598E3E0              // PTRUE P0.S
    
    WORD $0x04A0E3E2              // CNTW X2
    
    // Initialize with first vector
    WORD $0xA540A000              // LD1W {Z0.S}, P0/Z, [R0]
    // MOV Z1-Z7 = Z0 using ORR Zd.D, Zn.D, Zn.D
    WORD $0x04603001              // ORR Z1.D, Z0.D, Z0.D
    WORD $0x04603002              // ORR Z2.D, Z0.D, Z0.D
    WORD $0x04603003              // ORR Z3.D, Z0.D, Z0.D
    WORD $0x04603004              // ORR Z4.D, Z0.D, Z0.D
    WORD $0x04603005              // ORR Z5.D, Z0.D, Z0.D
    WORD $0x04603006              // ORR Z6.D, Z0.D, Z0.D
    WORD $0x04603007              // ORR Z7.D, Z0.D, Z0.D
    
    // Advance past initialized vector
    LSL $2, R2, R4                // bytes per vector = R2 * 4
    ADD R4, R0, R0
    SUB R2, R1, R1
    
    LSL $3, R2, R3
    
    CMP R3, R1
    BLT sve_max32_tail

sve_max32_loop8:
    WORD $0xA540A008              // LD1W {Z8.S}, P0/Z, [R0, #0, MUL VL]
    WORD $0xA541A009              // LD1W {Z9.S}, P0/Z, [R0, #1, MUL VL]
    WORD $0xA542A00A              // LD1W {Z10.S}, P0/Z, [R0, #2, MUL VL]
    WORD $0xA543A00B              // LD1W {Z11.S}, P0/Z, [R0, #3, MUL VL]
    WORD $0xA544A00C              // LD1W {Z12.S}, P0/Z, [R0, #4, MUL VL]
    WORD $0xA545A00D              // LD1W {Z13.S}, P0/Z, [R0, #5, MUL VL]
    WORD $0xA546A00E              // LD1W {Z14.S}, P0/Z, [R0, #6, MUL VL]
    WORD $0xA547A00F              // LD1W {Z15.S}, P0/Z, [R0, #7, MUL VL]
    
    // SMAX.S
    WORD $0x04880100              // SMAX Z0.S, P0/M, Z0.S, Z8.S
    WORD $0x04880121              // SMAX Z1.S, P0/M, Z1.S, Z9.S
    WORD $0x04880142              // SMAX Z2.S, P0/M, Z2.S, Z10.S
    WORD $0x04880163              // SMAX Z3.S, P0/M, Z3.S, Z11.S
    WORD $0x04880184              // SMAX Z4.S, P0/M, Z4.S, Z12.S
    WORD $0x048801A5              // SMAX Z5.S, P0/M, Z5.S, Z13.S
    WORD $0x048801C6              // SMAX Z6.S, P0/M, Z6.S, Z14.S
    WORD $0x048801E7              // SMAX Z7.S, P0/M, Z7.S, Z15.S
    
    LSL $2, R3, R4
    ADD R4, R0, R0
    SUB R3, R1, R1
    CMP R3, R1
    BGE sve_max32_loop8

sve_max32_tail:
    MOVD ZR, R4                   // R4 = 0 (loop index)
    CBZ R1, sve_max32_reduce
    
sve_max32_tail_loop:
    WORD $0x25A11C82              // WHILELO P2.S, X4, X1
    BEQ sve_max32_reduce          // Exit if no active lanes
    
    WORD $0xA540A808              // LD1W {Z8.S}, P2/Z, [R0]
    
    WORD $0x04880900              // SMAX Z0.S, P2/M, Z0.S, Z8.S
    
    WORD $0x04B0E3E4              // INCW X4
    LSL $2, R2, R5                // bytes per vector = R2 * 4
    ADD R5, R0, R0
    B sve_max32_tail_loop

sve_max32_reduce:
    // Tree reduction
    WORD $0x04880080              // SMAX Z0.S, P0/M, Z0.S, Z4.S
    WORD $0x048800A1              // SMAX Z1.S, P0/M, Z1.S, Z5.S
    WORD $0x048800C2              // SMAX Z2.S, P0/M, Z2.S, Z6.S
    WORD $0x048800E3              // SMAX Z3.S, P0/M, Z3.S, Z7.S
    WORD $0x04880040              // SMAX Z0.S, P0/M, Z0.S, Z2.S
    WORD $0x04880061              // SMAX Z1.S, P0/M, Z1.S, Z3.S
    WORD $0x04880020              // SMAX Z0.S, P0/M, Z0.S, Z1.S
    
    // SMAXV - horizontal max
    WORD $0x04882001              // SMAXV S1, P0, Z0.S
    FMOVS F1, R0
    B sve_max32_done

sve_max32_empty:
    MOVW $0, R0

sve_max32_done:
    MOVW R0, ret+24(FP)
    RET

// ┌──────────────────────────────────────────────────────────────────────────────┐
// │ func dotProductInt32SVE(a, b []int32) int64                                  │
// │                                                                              │
// │ Strategy: Native MUL.S then widen and accumulate                             │
// └──────────────────────────────────────────────────────────────────────────────┘
TEXT ·dotProductInt32SVE(SB), NOSPLIT, $0-56
    MOVD a_base+0(FP), R0
    MOVD a_len+8(FP), R1
    MOVD b_base+24(FP), R2
    
    WORD $0x2598E3E0              // PTRUE P0.S
    WORD $0x25D8E3E1              // PTRUE P1.D
    
    // Zero 4 int64 accumulators (only need 4 for 2-vector loop)
    WORD $0x25F8C000              // DUP Z0.D, #0
    WORD $0x25F8C001              // DUP Z1.D, #0
    WORD $0x25F8C002              // DUP Z2.D, #0
    WORD $0x25F8C003              // DUP Z3.D, #0
    
    WORD $0x04A0E3E3              // CNTW X3
    LSL $1, R3, R4                // R4 = R3 * 2 (2 vectors per iteration)
    
    CMP R4, R1
    BLT sve_dot32_tail

sve_dot32_loop2:
    // Load 2 vectors from each array
    WORD $0xA540A008              // LD1W {Z8.S}, P0/Z, [R0, #0, MUL VL]
    WORD $0xA541A009              // LD1W {Z9.S}, P0/Z, [R0, #1, MUL VL]
    
    WORD $0xA540A050              // LD1W {Z16.S}, P0/Z, [R2, #0, MUL VL]
    WORD $0xA541A051              // LD1W {Z17.S}, P0/Z, [R2, #1, MUL VL]
    
    // Widen BEFORE multiply to avoid truncation
    // Z8.S -> Z20.D (lo), Z21.D (hi)
    WORD $0x05F03914              // SUNPKLO Z20.D, Z8.S
    WORD $0x05F13915              // SUNPKHI Z21.D, Z8.S
    // Z16.S -> Z24.D (lo), Z25.D (hi)
    WORD $0x05F03A18              // SUNPKLO Z24.D, Z16.S
    WORD $0x05F13A19              // SUNPKHI Z25.D, Z16.S
    
    // Z9.S -> Z22.D (lo), Z23.D (hi)
    WORD $0x05F03936              // SUNPKLO Z22.D, Z9.S
    WORD $0x05F13937              // SUNPKHI Z23.D, Z9.S
    // Z17.S -> Z26.D (lo), Z27.D (hi)
    WORD $0x05F03A3A              // SUNPKLO Z26.D, Z17.S
    WORD $0x05F13A3B              // SUNPKHI Z27.D, Z17.S
    
    // Multiply 64-bit values: MUL Zdn.D, Pg/M, Zdn.D, Zm.D
    WORD $0x04D00714              // MUL Z20.D, P1/M, Z20.D, Z24.D
    WORD $0x04D00735              // MUL Z21.D, P1/M, Z21.D, Z25.D
    WORD $0x04D00756              // MUL Z22.D, P1/M, Z22.D, Z26.D
    WORD $0x04D00777              // MUL Z23.D, P1/M, Z23.D, Z27.D
    
    // Accumulate
    WORD $0x04C00680              // ADD Z0.D, P1/M, Z0.D, Z20.D
    WORD $0x04C006A1              // ADD Z1.D, P1/M, Z1.D, Z21.D
    WORD $0x04C006C2              // ADD Z2.D, P1/M, Z2.D, Z22.D
    WORD $0x04C006E3              // ADD Z3.D, P1/M, Z3.D, Z23.D
    
    // Advance pointers (R4 elements = R4*4 bytes)
    LSL $2, R4, R5
    ADD R5, R0, R0
    ADD R5, R2, R2
    SUB R4, R1, R1
    CMP R4, R1
    BGE sve_dot32_loop2

sve_dot32_tail:
    MOVD ZR, R7                   // R7 = scalar accumulator
    CBZ R1, sve_dot32_reduce
    
sve_dot32_tail_loop:
    // Process remaining elements one at a time via scalar
    MOVW (R0), R5
    MOVW (R2), R6
    SXTW R5, R5
    SXTW R6, R6
    MUL R5, R6, R5
    ADD R5, R7, R7                // Accumulate in R7, not Z0
    
    ADD $4, R0
    ADD $4, R2
    SUB $1, R1
    CBNZ R1, sve_dot32_tail_loop

sve_dot32_reduce:
    // Tree reduction: only Z0-Z3 are used now
    WORD $0x04C00440              // ADD Z0.D, P1/M, Z0.D, Z2.D
    WORD $0x04C00461              // ADD Z1.D, P1/M, Z1.D, Z3.D
    WORD $0x04C00420              // ADD Z0.D, P1/M, Z0.D, Z1.D
    
    WORD $0x04C12401              // UADDV D1, P1, Z0.D
    FMOVD F1, R0
    ADD R7, R0, R0                // Add scalar tail accumulator
    
    MOVD R0, ret+48(FP)
    RET

// ┌──────────────────────────────────────────────────────────────────────────────┐
// │ func sumSqInt32SVE(vals []int32) int64                                       │
// │                                                                              │
// │ Strategy: Native MUL.S (square), widen, accumulate                           │
// └──────────────────────────────────────────────────────────────────────────────┘
TEXT ·sumSqInt32SVE(SB), NOSPLIT, $0-32
    MOVD vals_base+0(FP), R0
    MOVD vals_len+8(FP), R1
    
    WORD $0x2598E3E0              // PTRUE P0.S
    WORD $0x25D8E3E1              // PTRUE P1.D
    
    // Zero 4 int64 accumulators (only need 4 for 2-vector loop)
    WORD $0x25F8C000              // DUP Z0.D, #0
    WORD $0x25F8C001              // DUP Z1.D, #0
    WORD $0x25F8C002              // DUP Z2.D, #0
    WORD $0x25F8C003              // DUP Z3.D, #0
    
    WORD $0x04A0E3E3              // CNTW X3
    LSL $1, R3, R4                // R4 = R3 * 2 (2 vectors per iteration)
    
    CMP R4, R1
    BLT sve_sumsq32_tail

sve_sumsq32_loop2:
    WORD $0xA540A008              // LD1W {Z8.S}, P0/Z, [R0, #0, MUL VL]
    WORD $0xA541A009              // LD1W {Z9.S}, P0/Z, [R0, #1, MUL VL]
    
    // Widen BEFORE squaring to avoid truncation
    WORD $0x05F03914              // SUNPKLO Z20.D, Z8.S
    WORD $0x05F13915              // SUNPKHI Z21.D, Z8.S
    WORD $0x05F03936              // SUNPKLO Z22.D, Z9.S
    WORD $0x05F13937              // SUNPKHI Z23.D, Z9.S
    
    // Square the 64-bit values: MUL Zdn.D, Pg/M, Zdn.D, Zdn.D
    WORD $0x04D00694              // MUL Z20.D, P1/M, Z20.D, Z20.D
    WORD $0x04D006B5              // MUL Z21.D, P1/M, Z21.D, Z21.D
    WORD $0x04D006D6              // MUL Z22.D, P1/M, Z22.D, Z22.D
    WORD $0x04D006F7              // MUL Z23.D, P1/M, Z23.D, Z23.D
    
    // Accumulate
    WORD $0x04C00680              // ADD Z0.D, P1/M, Z0.D, Z20.D
    WORD $0x04C006A1              // ADD Z1.D, P1/M, Z1.D, Z21.D
    WORD $0x04C006C2              // ADD Z2.D, P1/M, Z2.D, Z22.D
    WORD $0x04C006E3              // ADD Z3.D, P1/M, Z3.D, Z23.D
    
    LSL $2, R4, R5
    ADD R5, R0, R0
    SUB R4, R1, R1
    CMP R4, R1
    BGE sve_sumsq32_loop2

sve_sumsq32_tail:
    MOVD ZR, R6                   // R6 = scalar accumulator
    CBZ R1, sve_sumsq32_reduce
    
sve_sumsq32_tail_loop:
    // Process remaining elements one at a time via scalar
    MOVW (R0), R5
    SXTW R5, R5
    MUL R5, R5, R5
    ADD R5, R6, R6                // Accumulate in R6, not Z0
    
    ADD $4, R0
    SUB $1, R1
    CBNZ R1, sve_sumsq32_tail_loop

sve_sumsq32_reduce:
    // Tree reduction: only Z0-Z3 are used now
    WORD $0x04C00440              // ADD Z0.D, P1/M, Z0.D, Z2.D
    WORD $0x04C00461              // ADD Z1.D, P1/M, Z1.D, Z3.D
    WORD $0x04C00420              // ADD Z0.D, P1/M, Z0.D, Z1.D
    
    WORD $0x04C12401              // UADDV D1, P1, Z0.D
    FMOVD F1, R0
    ADD R6, R0, R0                // Add scalar tail accumulator
    
    MOVD R0, ret+24(FP)
    RET

// ┌──────────────────────────────────────────────────────────────────────────────┐
// │ func anyAbsGreaterThanInt32SVE(vals []int32, threshold int32) bool           │
// │                                                                              │
// │ Strategy: Use ABS + CMPGT with early exit via PTEST                         │
// └──────────────────────────────────────────────────────────────────────────────┘
TEXT ·anyAbsGreaterThanInt32SVE(SB), NOSPLIT, $0-25
    MOVD vals_base+0(FP), R0
    MOVD vals_len+8(FP), R1
    MOVW threshold+24(FP), R2
    
    CBZ R1, sve_absgt32_notfound
    
    WORD $0x2598E3E0              // PTRUE P0.S
    
    // Broadcast threshold to Z0.S
    WORD $0x05A03840              // DUP Z0.S, W2
    
    // CNTW - count 32-bit elements per vector
    WORD $0x04A0E3E3              // CNTW X3
    LSL $2, R3, R4                // R4 = R3 * 4 (4 vectors per iteration)
    
    CMP R4, R1
    BLT sve_absgt32_tail

sve_absgt32_loop4:
    // Load 4 vectors
    WORD $0xA540A008              // LD1W {Z8.S}, P0/Z, [R0, #0, MUL VL]
    WORD $0xA541A009              // LD1W {Z9.S}, P0/Z, [R0, #1, MUL VL]
    WORD $0xA542A00A              // LD1W {Z10.S}, P0/Z, [R0, #2, MUL VL]
    WORD $0xA543A00B              // LD1W {Z11.S}, P0/Z, [R0, #3, MUL VL]
    
    // ABS - compute absolute values
    WORD $0x0496A108              // ABS Z8.S, P0/M, Z8.S
    WORD $0x0496A129              // ABS Z9.S, P0/M, Z9.S
    WORD $0x0496A14A              // ABS Z10.S, P0/M, Z10.S
    WORD $0x0496A16B              // ABS Z11.S, P0/M, Z11.S
    
    // CMPGT P1, P0/Z, Zn.S, Z0.S - compare greater than threshold
    WORD $0x24808111              // CMPGT P1.S, P0/Z, Z8.S, Z0.S
    WORD $0x2550C020              // PTEST P0, P1.B  (test if any bits set)
    BNE sve_absgt32_found
    
    WORD $0x24808131              // CMPGT P1.S, P0/Z, Z9.S, Z0.S
    WORD $0x2550C020              // PTEST P0, P1.B
    BNE sve_absgt32_found
    
    WORD $0x24808151              // CMPGT P1.S, P0/Z, Z10.S, Z0.S
    WORD $0x2550C020              // PTEST P0, P1.B
    BNE sve_absgt32_found
    
    WORD $0x24808171              // CMPGT P1.S, P0/Z, Z11.S, Z0.S
    WORD $0x2550C020              // PTEST P0, P1.B
    BNE sve_absgt32_found
    
    // Advance pointer
    LSL $2, R4, R5
    ADD R5, R0, R0
    SUB R4, R1, R1
    CMP R4, R1
    BGE sve_absgt32_loop4

sve_absgt32_tail:
    CBZ R1, sve_absgt32_notfound
    MOVD ZR, R5

sve_absgt32_tail_loop:
    WORD $0x25A11CA2              // WHILELO P2.S, X5, X1
    BEQ sve_absgt32_notfound
    
    WORD $0xA540A808              // LD1W {Z8.S}, P2/Z, [R0]
    
    WORD $0x0496A908              // ABS Z8.S, P2/M, Z8.S
    
    WORD $0x24808111              // CMPGT P1.S, P0/Z, Z8.S, Z0.S
    WORD $0x25024021              // AND P1.B, P0/Z, P1.B, P2.B
    WORD $0x2550C020              // PTEST P0, P1.B
    BNE sve_absgt32_found
    
    WORD $0x04B0E3E5              // INCW X5
    LSL $2, R3, R6
    ADD R6, R0, R0
    B sve_absgt32_tail_loop

sve_absgt32_found:
    MOVD $1, R0
    MOVB R0, ret+24(FP)
    RET
    
sve_absgt32_notfound:
    MOVD $0, R0
    MOVB R0, ret+24(FP)
    RET
