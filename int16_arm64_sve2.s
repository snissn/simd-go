//go:build arm64

#include "textflag.h"

// ╔══════════════════════════════════════════════════════════════════════════════╗
// ║                          SVE2 Int16 SIMD Operations                          ║
// ║                                                                              ║
// ║  SVE2 instructions for int16:                                                ║
// ║  • SMLALB/SMLALT Zda.S, Zn.H, Zm.H - int16×int16→int32 with accumulate      ║
// ║  • SMULLB/SMULLT Zd.S, Zn.H, Zm.H - int16×int16→int32 widening multiply     ║
// ║  • SADALP Zda.S, Pg/M, Zn.H - pairwise add int16→int32 with accumulate      ║
// ╚══════════════════════════════════════════════════════════════════════════════╝

// ┌──────────────────────────────────────────────────────────────────────────────┐
// │ func dotProductInt16SVE2(a, b []int16) int64                                 │
// │                                                                              │
// │ Strategy: SMLALB/SMLALT for int16×int16→int32, then widen to int64          │
// │           Much fewer instructions than SVE's widen+multiply+add pattern      │
// └──────────────────────────────────────────────────────────────────────────────┘
TEXT ·dotProductInt16SVE2(SB), NOSPLIT, $0-56
    MOVD a_base+0(FP), R0
    MOVD a_len+8(FP), R1
    MOVD b_base+24(FP), R2
    
    // PTRUE with ALL pattern - sets all predicate bits true
    WORD $0x2518E3E0              // PTRUE P0.B, ALL
    WORD $0x2518E3E1              // PTRUE P1.B, ALL
    WORD $0x2518E3E2              // PTRUE P2.B, ALL
    
    // Zero 4 int32 accumulators (will sum to int64 at end)
    WORD $0x25B8C000              // DUP Z0.S, #0
    WORD $0x25B8C001              // DUP Z1.S, #0
    WORD $0x25B8C002              // DUP Z2.S, #0
    WORD $0x25B8C003              // DUP Z3.S, #0
    
    // CNTH - count 16-bit elements per vector
    WORD $0x0460E3E3              // CNTH X3
    LSL $2, R3, R4                // R4 = R3 * 4 (4 vectors per iteration)
    
    CMP R4, R1
    BLT sve2_dot16_tail

sve2_dot16_loop4:
    // Load 4 vectors from each array
    WORD $0xA4A0A008              // LD1H {Z8.H}, P0/Z, [R0, #0, MUL VL]
    WORD $0xA4A1A009              // LD1H {Z9.H}, P0/Z, [R0, #1, MUL VL]
    WORD $0xA4A2A00A              // LD1H {Z10.H}, P0/Z, [R0, #2, MUL VL]
    WORD $0xA4A3A00B              // LD1H {Z11.H}, P0/Z, [R0, #3, MUL VL]
    
    WORD $0xA4A0A050              // LD1H {Z16.H}, P0/Z, [R2, #0, MUL VL]
    WORD $0xA4A1A051              // LD1H {Z17.H}, P0/Z, [R2, #1, MUL VL]
    WORD $0xA4A2A052              // LD1H {Z18.H}, P0/Z, [R2, #2, MUL VL]
    WORD $0xA4A3A053              // LD1H {Z19.H}, P0/Z, [R2, #3, MUL VL]
    
    // SMLALB/SMLALT: int16×int16→int32 fused multiply-accumulate
    // SMLALB Zda.S, Zn.H, Zm.H: Zda += Zn_bottom * Zm_bottom
    // SMLALT Zda.S, Zn.H, Zm.H: Zda += Zn_top * Zm_top
    
    // Z8 * Z16 -> Z0 (bottom), Z1 (top)
    WORD $0x44904100              // SMLALB Z0.S, Z8.H, Z16.H
    WORD $0x44904501              // SMLALT Z1.S, Z8.H, Z16.H
    
    // Z9 * Z17 -> Z2, Z3
    WORD $0x44914122              // SMLALB Z2.S, Z9.H, Z17.H
    WORD $0x44914523              // SMLALT Z3.S, Z9.H, Z17.H
    
    // Z10 * Z18 -> Z0, Z1
    WORD $0x44924140              // SMLALB Z0.S, Z10.H, Z18.H
    WORD $0x44924541              // SMLALT Z1.S, Z10.H, Z18.H
    
    // Z11 * Z19 -> Z2, Z3
    WORD $0x44934162              // SMLALB Z2.S, Z11.H, Z19.H
    WORD $0x44934563              // SMLALT Z3.S, Z11.H, Z19.H
    
    // Advance pointers
    LSL $1, R4, R5                // R5 = R4 * 2 bytes
    ADD R5, R0, R0
    ADD R5, R2, R2
    SUB R4, R1, R1
    CMP R4, R1
    BGE sve2_dot16_loop4

sve2_dot16_tail:
    CBZ R1, sve2_dot16_reduce
    MOVD ZR, R5

sve2_dot16_tail_loop:
    WORD $0x25611CA3              // WHILELO P3.H, X5, X1
    BEQ sve2_dot16_reduce
    
    WORD $0xA4A0AC08              // LD1H {Z8.H}, P3/Z, [R0]
    WORD $0xA4A0AC50              // LD1H {Z16.H}, P3/Z, [R2]
    
    WORD $0x44904100              // SMLALB Z0.S, Z8.H, Z16.H
    WORD $0x44904501              // SMLALT Z1.S, Z8.H, Z16.H
    
    WORD $0x0470E3E5              // INCH X5
    LSL $1, R3, R6
    ADD R6, R0, R0
    ADD R6, R2, R2
    B sve2_dot16_tail_loop

sve2_dot16_reduce:
    // Widen each 32-bit accumulator to 64-bit FIRST to avoid overflow during combine
    // Z0.S -> Z16.D (lo), Z17.D (hi)
    WORD $0x05F03810              // SUNPKLO Z16.D, Z0.S
    WORD $0x05F13811              // SUNPKHI Z17.D, Z0.S
    // Z1.S -> Z18.D (lo), Z19.D (hi)
    WORD $0x05F03832              // SUNPKLO Z18.D, Z1.S
    WORD $0x05F13833              // SUNPKHI Z19.D, Z1.S
    // Z2.S -> Z20.D (lo), Z21.D (hi)
    WORD $0x05F03854              // SUNPKLO Z20.D, Z2.S
    WORD $0x05F13855              // SUNPKHI Z21.D, Z2.S
    // Z3.S -> Z22.D (lo), Z23.D (hi)
    WORD $0x05F03876              // SUNPKLO Z22.D, Z3.S
    WORD $0x05F13877              // SUNPKHI Z23.D, Z3.S
    
    // Now combine all 64-bit values: Z16 += Z17 + Z18 + Z19 + Z20 + Z21 + Z22 + Z23
    WORD $0x04C00A30              // ADD Z16.D, P2/M, Z16.D, Z17.D
    WORD $0x04C00A50              // ADD Z16.D, P2/M, Z16.D, Z18.D
    WORD $0x04C00A70              // ADD Z16.D, P2/M, Z16.D, Z19.D
    WORD $0x04C00A90              // ADD Z16.D, P2/M, Z16.D, Z20.D
    WORD $0x04C00AB0              // ADD Z16.D, P2/M, Z16.D, Z21.D
    WORD $0x04C00AD0              // ADD Z16.D, P2/M, Z16.D, Z22.D
    WORD $0x04C00AF0              // ADD Z16.D, P2/M, Z16.D, Z23.D
    
    // UADDV - horizontal sum
    WORD $0x04C12A01              // UADDV D1, P2, Z16.D
    FMOVD F1, R0
    
    MOVD R0, ret+48(FP)
    RET

// ┌──────────────────────────────────────────────────────────────────────────────┐
// │ func sumInt16SVE2(vals []int16) int64                                        │
// │                                                                              │
// │ Strategy: Use SADALP for pairwise add with widening to int32                │
// │           Then widen int32 to int64 for final sum                            │
// └──────────────────────────────────────────────────────────────────────────────┘
TEXT ·sumInt16SVE2(SB), NOSPLIT, $0-32
    MOVD vals_base+0(FP), R0
    MOVD vals_len+8(FP), R1
    
    WORD $0x2518E3E0              // PTRUE P0.B, ALL
    WORD $0x2518E3E1              // PTRUE P1.B, ALL
    WORD $0x2518E3E2              // PTRUE P2.B, ALL
    
    // Zero 4 int32 accumulators
    WORD $0x25B8C000              // DUP Z0.S, #0
    WORD $0x25B8C001              // DUP Z1.S, #0
    WORD $0x25B8C002              // DUP Z2.S, #0
    WORD $0x25B8C003              // DUP Z3.S, #0
    
    WORD $0x0460E3E3              // CNTH X3
    LSL $3, R3, R4                // R4 = R3 * 8 (8 vectors per iteration)
    
    CMP R4, R1
    BLT sve2_sum16_tail

sve2_sum16_loop8:
    // Load 8 vectors
    WORD $0xA4A0A008              // LD1H {Z8.H}, P0/Z, [R0, #0, MUL VL]
    WORD $0xA4A1A009              // LD1H {Z9.H}, P0/Z, [R0, #1, MUL VL]
    WORD $0xA4A2A00A              // LD1H {Z10.H}, P0/Z, [R0, #2, MUL VL]
    WORD $0xA4A3A00B              // LD1H {Z11.H}, P0/Z, [R0, #3, MUL VL]
    WORD $0xA4A4A00C              // LD1H {Z12.H}, P0/Z, [R0, #4, MUL VL]
    WORD $0xA4A5A00D              // LD1H {Z13.H}, P0/Z, [R0, #5, MUL VL]
    WORD $0xA4A6A00E              // LD1H {Z14.H}, P0/Z, [R0, #6, MUL VL]
    WORD $0xA4A7A00F              // LD1H {Z15.H}, P0/Z, [R0, #7, MUL VL]
    
    // SADALP: int16 pairs -> int32 with accumulate
    // SADALP Zda.S, Pg/M, Zn.H
    WORD $0x4484A100              // SADALP Z0.S, P0/M, Z8.H
    WORD $0x4484A121              // SADALP Z1.S, P0/M, Z9.H
    WORD $0x4484A142              // SADALP Z2.S, P0/M, Z10.H
    WORD $0x4484A163              // SADALP Z3.S, P0/M, Z11.H
    WORD $0x4484A180              // SADALP Z0.S, P0/M, Z12.H
    WORD $0x4484A1A1              // SADALP Z1.S, P0/M, Z13.H
    WORD $0x4484A1C2              // SADALP Z2.S, P0/M, Z14.H
    WORD $0x4484A1E3              // SADALP Z3.S, P0/M, Z15.H
    
    // Advance pointer
    LSL $1, R4, R5
    ADD R5, R0, R0
    SUB R4, R1, R1
    CMP R4, R1
    BGE sve2_sum16_loop8

sve2_sum16_tail:
    CBZ R1, sve2_sum16_reduce
    MOVD ZR, R5

sve2_sum16_tail_loop:
    WORD $0x25611CA3              // WHILELO P3.H, X5, X1
    BEQ sve2_sum16_reduce
    
    WORD $0xA4A0AC08              // LD1H {Z8.H}, P3/Z, [R0]
    WORD $0x4484A100              // SADALP Z0.S, P0/M, Z8.H
    
    WORD $0x0470E3E5              // INCH X5
    LSL $1, R3, R6
    ADD R6, R0, R0
    B sve2_sum16_tail_loop

sve2_sum16_reduce:
    // Widen each 32-bit accumulator to 64-bit FIRST to avoid overflow during combine
    // Z0.S -> Z16.D (lo), Z17.D (hi)
    WORD $0x05F03810              // SUNPKLO Z16.D, Z0.S
    WORD $0x05F13811              // SUNPKHI Z17.D, Z0.S
    // Z1.S -> Z18.D (lo), Z19.D (hi)
    WORD $0x05F03832              // SUNPKLO Z18.D, Z1.S
    WORD $0x05F13833              // SUNPKHI Z19.D, Z1.S
    // Z2.S -> Z20.D (lo), Z21.D (hi)
    WORD $0x05F03854              // SUNPKLO Z20.D, Z2.S
    WORD $0x05F13855              // SUNPKHI Z21.D, Z2.S
    // Z3.S -> Z22.D (lo), Z23.D (hi)
    WORD $0x05F03876              // SUNPKLO Z22.D, Z3.S
    WORD $0x05F13877              // SUNPKHI Z23.D, Z3.S
    
    // Now combine all 64-bit values: Z16 += Z17 + Z18 + Z19 + Z20 + Z21 + Z22 + Z23
    WORD $0x04C00A30              // ADD Z16.D, P2/M, Z16.D, Z17.D
    WORD $0x04C00A50              // ADD Z16.D, P2/M, Z16.D, Z18.D
    WORD $0x04C00A70              // ADD Z16.D, P2/M, Z16.D, Z19.D
    WORD $0x04C00A90              // ADD Z16.D, P2/M, Z16.D, Z20.D
    WORD $0x04C00AB0              // ADD Z16.D, P2/M, Z16.D, Z21.D
    WORD $0x04C00AD0              // ADD Z16.D, P2/M, Z16.D, Z22.D
    WORD $0x04C00AF0              // ADD Z16.D, P2/M, Z16.D, Z23.D
    
    WORD $0x04C12A01              // UADDV D1, P2, Z16.D
    FMOVD F1, R0
    
    MOVD R0, ret+24(FP)
    RET

// ┌──────────────────────────────────────────────────────────────────────────────┐
// │ func sumSqInt16SVE2(vals []int16) int64                                      │
// │                                                                              │
// │ Strategy: Use SMLALB/SMLALT with same operand for fused squaring            │
// │           int16×int16→int32 with accumulate, then widen to int64            │
// └──────────────────────────────────────────────────────────────────────────────┘
TEXT ·sumSqInt16SVE2(SB), NOSPLIT, $0-32
    MOVD vals_base+0(FP), R0
    MOVD vals_len+8(FP), R1
    
    // PTRUE with ALL pattern - sets all predicate bits true
    WORD $0x2518E3E0              // PTRUE P0.B, ALL
    WORD $0x2518E3E1              // PTRUE P1.B, ALL
    WORD $0x2518E3E2              // PTRUE P2.B, ALL
    
    // Zero 4 int32 accumulators (will sum to int64 at end)
    WORD $0x25B8C000              // DUP Z0.S, #0
    WORD $0x25B8C001              // DUP Z1.S, #0
    WORD $0x25B8C002              // DUP Z2.S, #0
    WORD $0x25B8C003              // DUP Z3.S, #0
    
    // CNTH - count 16-bit elements per vector
    WORD $0x0460E3E3              // CNTH X3
    LSL $2, R3, R4                // R4 = R3 * 4 (4 vectors per iteration)
    
    CMP R4, R1
    BLT sve2_sumsq16_tail

sve2_sumsq16_loop4:
    // Load 4 vectors
    WORD $0xA4A0A008              // LD1H {Z8.H}, P0/Z, [R0, #0, MUL VL]
    WORD $0xA4A1A009              // LD1H {Z9.H}, P0/Z, [R0, #1, MUL VL]
    WORD $0xA4A2A00A              // LD1H {Z10.H}, P0/Z, [R0, #2, MUL VL]
    WORD $0xA4A3A00B              // LD1H {Z11.H}, P0/Z, [R0, #3, MUL VL]
    
    // SMLALB/SMLALT: int16×int16→int32 fused multiply-accumulate
    // Using same operand for squaring: z8*z8
    
    // Z8^2 -> Z0 (bottom), Z1 (top)
    WORD $0x44884100              // SMLALB Z0.S, Z8.H, Z8.H
    WORD $0x44884501              // SMLALT Z1.S, Z8.H, Z8.H
    
    // Z9^2 -> Z2, Z3
    WORD $0x44894122              // SMLALB Z2.S, Z9.H, Z9.H
    WORD $0x44894523              // SMLALT Z3.S, Z9.H, Z9.H
    
    // Z10^2 -> Z0, Z1
    WORD $0x448A4140              // SMLALB Z0.S, Z10.H, Z10.H
    WORD $0x448A4541              // SMLALT Z1.S, Z10.H, Z10.H
    
    // Z11^2 -> Z2, Z3
    WORD $0x448B4162              // SMLALB Z2.S, Z11.H, Z11.H
    WORD $0x448B4563              // SMLALT Z3.S, Z11.H, Z11.H
    
    // Advance pointer
    LSL $1, R4, R5                // R5 = R4 * 2 bytes
    ADD R5, R0, R0
    SUB R4, R1, R1
    CMP R4, R1
    BGE sve2_sumsq16_loop4

sve2_sumsq16_tail:
    CBZ R1, sve2_sumsq16_reduce
    MOVD ZR, R5

sve2_sumsq16_tail_loop:
    WORD $0x25611CA3              // WHILELO P3.H, X5, X1
    BEQ sve2_sumsq16_reduce
    
    WORD $0xA4A0AC08              // LD1H {Z8.H}, P3/Z, [R0]
    
    WORD $0x44884100              // SMLALB Z0.S, Z8.H, Z8.H
    WORD $0x44884501              // SMLALT Z1.S, Z8.H, Z8.H
    
    WORD $0x0470E3E5              // INCH X5
    LSL $1, R3, R6
    ADD R6, R0, R0
    B sve2_sumsq16_tail_loop

sve2_sumsq16_reduce:
    // Widen each 32-bit accumulator to 64-bit FIRST to avoid overflow during combine
    // Z0.S -> Z16.D (lo), Z17.D (hi)
    WORD $0x05F03810              // SUNPKLO Z16.D, Z0.S
    WORD $0x05F13811              // SUNPKHI Z17.D, Z0.S
    // Z1.S -> Z18.D (lo), Z19.D (hi)
    WORD $0x05F03832              // SUNPKLO Z18.D, Z1.S
    WORD $0x05F13833              // SUNPKHI Z19.D, Z1.S
    // Z2.S -> Z20.D (lo), Z21.D (hi)
    WORD $0x05F03854              // SUNPKLO Z20.D, Z2.S
    WORD $0x05F13855              // SUNPKHI Z21.D, Z2.S
    // Z3.S -> Z22.D (lo), Z23.D (hi)
    WORD $0x05F03876              // SUNPKLO Z22.D, Z3.S
    WORD $0x05F13877              // SUNPKHI Z23.D, Z3.S
    
    // Now combine all 64-bit values: Z16 += Z17 + Z18 + Z19 + Z20 + Z21 + Z22 + Z23
    WORD $0x04C00A30              // ADD Z16.D, P2/M, Z16.D, Z17.D
    WORD $0x04C00A50              // ADD Z16.D, P2/M, Z16.D, Z18.D
    WORD $0x04C00A70              // ADD Z16.D, P2/M, Z16.D, Z19.D
    WORD $0x04C00A90              // ADD Z16.D, P2/M, Z16.D, Z20.D
    WORD $0x04C00AB0              // ADD Z16.D, P2/M, Z16.D, Z21.D
    WORD $0x04C00AD0              // ADD Z16.D, P2/M, Z16.D, Z22.D
    WORD $0x04C00AF0              // ADD Z16.D, P2/M, Z16.D, Z23.D
    
    // UADDV - horizontal sum
    WORD $0x04C12A01              // UADDV D1, P2, Z16.D
    FMOVD F1, R0
    
    MOVD R0, ret+24(FP)
    RET
