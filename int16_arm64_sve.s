//go:build arm64

#include "textflag.h"

// ╔══════════════════════════════════════════════════════════════════════════════╗
// ║                           SVE Int16 SIMD Operations                          ║
// ║                                                                              ║
// ║  SVE processes variable number of int16 per vector (VL-dependent)           ║
// ║  At 256-bit: 16 x int16 per vector                                           ║
// ║                                                                              ║
// ║  Key SVE int16 instructions:                                                 ║
// ║  • LD1H - load 16-bit elements                                               ║
// ║  • SMIN, SMAX - native operations                                            ║
// ║  • SADDV - horizontal sum with widening to 64-bit                            ║
// ║  • SMINV, SMAXV - horizontal min/max                                         ║
// ║  • SUNPKLO/SUNPKHI - sign-extend to wider type                               ║
// ╚══════════════════════════════════════════════════════════════════════════════╝

// ┌──────────────────────────────────────────────────────────────────────────────┐
// │ func sumInt16SVE(vals []int16) int64                                         │
// │                                                                              │
// │ Strategy: Load int16, widen to int32, then to int64, accumulate              │
// └──────────────────────────────────────────────────────────────────────────────┘
TEXT ·sumInt16SVE(SB), NOSPLIT, $0-32
    MOVD vals_base+0(FP), R0
    MOVD vals_len+8(FP), R1
    
    // PTRUE P0.H - Enable all 16-bit lanes
    WORD $0x2558E3E0              // PTRUE P0.H
    
    // Zero 8 int64 accumulators
    WORD $0x25F8C000              // DUP Z0.D, #0
    WORD $0x25F8C001              // DUP Z1.D, #0
    WORD $0x25F8C002              // DUP Z2.D, #0
    WORD $0x25F8C003              // DUP Z3.D, #0
    WORD $0x25F8C004              // DUP Z4.D, #0
    WORD $0x25F8C005              // DUP Z5.D, #0
    WORD $0x25F8C006              // DUP Z6.D, #0
    WORD $0x25F8C007              // DUP Z7.D, #0
    
    // CNTH - count 16-bit elements per vector
    WORD $0x0460E3E2              // CNTH X2
    
    LSL $3, R2, R3                // R3 = R2 * 8 (8 vectors per iteration)
    
    CMP R3, R1
    BLT sve_sum16_tail

sve_sum16_loop8:
    // Load 8 vectors of int16
    WORD $0xA4A0A008              // LD1H {Z8.H}, P0/Z, [R0, #0, MUL VL]
    WORD $0xA4A1A009              // LD1H {Z9.H}, P0/Z, [R0, #1, MUL VL]
    WORD $0xA4A2A00A              // LD1H {Z10.H}, P0/Z, [R0, #2, MUL VL]
    WORD $0xA4A3A00B              // LD1H {Z11.H}, P0/Z, [R0, #3, MUL VL]
    WORD $0xA4A4A00C              // LD1H {Z12.H}, P0/Z, [R0, #4, MUL VL]
    WORD $0xA4A5A00D              // LD1H {Z13.H}, P0/Z, [R0, #5, MUL VL]
    WORD $0xA4A6A00E              // LD1H {Z14.H}, P0/Z, [R0, #6, MUL VL]
    WORD $0xA4A7A00F              // LD1H {Z15.H}, P0/Z, [R0, #7, MUL VL]
    
    // Widen int16 -> int32 using SUNPKLO/SUNPKHI
    // Then widen int32 -> int64 using SUNPKLO/SUNPKHI again
    // Z8.H -> Z16.S (lo), Z17.S (hi) -> Z16.D, Z17.D, Z18.D, Z19.D
    
    // For Z8: H -> S
    // Encoding: size=10 for S dest, SUNPKLO=0x05B039, SUNPKHI=0x05B139
    WORD $0x05B03910              // SUNPKLO Z16.S, Z8.H
    WORD $0x05B13911              // SUNPKHI Z17.S, Z8.H
    // S -> D
    // Encoding: size=11 for D dest, SUNPKLO=0x05F039, SUNPKHI=0x05F139
    WORD $0x05F03A12              // SUNPKLO Z18.D, Z16.S
    WORD $0x05F13A13              // SUNPKHI Z19.D, Z16.S
    WORD $0x05F03A34              // SUNPKLO Z20.D, Z17.S
    WORD $0x05F13A35              // SUNPKHI Z21.D, Z17.S
    
    // Add to accumulators
    WORD $0x25D8E3E1              // PTRUE P1.D
    WORD $0x04C00640              // ADD Z0.D, P1/M, Z0.D, Z18.D
    WORD $0x04C00661              // ADD Z1.D, P1/M, Z1.D, Z19.D
    WORD $0x04C00682              // ADD Z2.D, P1/M, Z2.D, Z20.D
    WORD $0x04C006A3              // ADD Z3.D, P1/M, Z3.D, Z21.D
    
    // Repeat for Z9
    WORD $0x05B03930              // SUNPKLO Z16.S, Z9.H
    WORD $0x05B13931              // SUNPKHI Z17.S, Z9.H
    WORD $0x05F03A12              // SUNPKLO Z18.D, Z16.S
    WORD $0x05F13A13              // SUNPKHI Z19.D, Z16.S
    WORD $0x05F03A34              // SUNPKLO Z20.D, Z17.S
    WORD $0x05F13A35              // SUNPKHI Z21.D, Z17.S
    WORD $0x04C00644              // ADD Z4.D, P1/M, Z4.D, Z18.D
    WORD $0x04C00665              // ADD Z5.D, P1/M, Z5.D, Z19.D
    WORD $0x04C00686              // ADD Z6.D, P1/M, Z6.D, Z20.D
    WORD $0x04C006A7              // ADD Z7.D, P1/M, Z7.D, Z21.D
    
    // Process remaining 6 vectors (Z10-Z15) similarly - accumulate into Z0-Z7
    // Z10
    WORD $0x05B03950              // SUNPKLO Z16.S, Z10.H
    WORD $0x05B13951              // SUNPKHI Z17.S, Z10.H
    WORD $0x05F03A12              // SUNPKLO Z18.D, Z16.S
    WORD $0x05F13A13              // SUNPKHI Z19.D, Z16.S
    WORD $0x05F03A34              // SUNPKLO Z20.D, Z17.S
    WORD $0x05F13A35              // SUNPKHI Z21.D, Z17.S
    WORD $0x04C00640              // ADD Z0.D, P1/M, Z0.D, Z18.D
    WORD $0x04C00661              // ADD Z1.D, P1/M, Z1.D, Z19.D
    WORD $0x04C00682              // ADD Z2.D, P1/M, Z2.D, Z20.D
    WORD $0x04C006A3              // ADD Z3.D, P1/M, Z3.D, Z21.D
    
    // Z11
    WORD $0x05B03970              // SUNPKLO Z16.S, Z11.H
    WORD $0x05B13971              // SUNPKHI Z17.S, Z11.H
    WORD $0x05F03A12              // SUNPKLO Z18.D, Z16.S
    WORD $0x05F13A13              // SUNPKHI Z19.D, Z16.S
    WORD $0x05F03A34              // SUNPKLO Z20.D, Z17.S
    WORD $0x05F13A35              // SUNPKHI Z21.D, Z17.S
    WORD $0x04C00644              // ADD Z4.D, P1/M, Z4.D, Z18.D
    WORD $0x04C00665              // ADD Z5.D, P1/M, Z5.D, Z19.D
    WORD $0x04C00686              // ADD Z6.D, P1/M, Z6.D, Z20.D
    WORD $0x04C006A7              // ADD Z7.D, P1/M, Z7.D, Z21.D
    
    // Z12
    WORD $0x05B03990              // SUNPKLO Z16.S, Z12.H
    WORD $0x05B13991              // SUNPKHI Z17.S, Z12.H
    WORD $0x05F03A12              // SUNPKLO Z18.D, Z16.S
    WORD $0x05F13A13              // SUNPKHI Z19.D, Z16.S
    WORD $0x05F03A34              // SUNPKLO Z20.D, Z17.S
    WORD $0x05F13A35              // SUNPKHI Z21.D, Z17.S
    WORD $0x04C00640              // ADD Z0.D, P1/M, Z0.D, Z18.D
    WORD $0x04C00661              // ADD Z1.D, P1/M, Z1.D, Z19.D
    WORD $0x04C00682              // ADD Z2.D, P1/M, Z2.D, Z20.D
    WORD $0x04C006A3              // ADD Z3.D, P1/M, Z3.D, Z21.D
    
    // Z13
    WORD $0x05B039B0              // SUNPKLO Z16.S, Z13.H
    WORD $0x05B139B1              // SUNPKHI Z17.S, Z13.H
    WORD $0x05F03A12              // SUNPKLO Z18.D, Z16.S
    WORD $0x05F13A13              // SUNPKHI Z19.D, Z16.S
    WORD $0x05F03A34              // SUNPKLO Z20.D, Z17.S
    WORD $0x05F13A35              // SUNPKHI Z21.D, Z17.S
    WORD $0x04C00644              // ADD Z4.D, P1/M, Z4.D, Z18.D
    WORD $0x04C00665              // ADD Z5.D, P1/M, Z5.D, Z19.D
    WORD $0x04C00686              // ADD Z6.D, P1/M, Z6.D, Z20.D
    WORD $0x04C006A7              // ADD Z7.D, P1/M, Z7.D, Z21.D
    
    // Z14
    WORD $0x05B039D0              // SUNPKLO Z16.S, Z14.H
    WORD $0x05B139D1              // SUNPKHI Z17.S, Z14.H
    WORD $0x05F03A12              // SUNPKLO Z18.D, Z16.S
    WORD $0x05F13A13              // SUNPKHI Z19.D, Z16.S
    WORD $0x05F03A34              // SUNPKLO Z20.D, Z17.S
    WORD $0x05F13A35              // SUNPKHI Z21.D, Z17.S
    WORD $0x04C00640              // ADD Z0.D, P1/M, Z0.D, Z18.D
    WORD $0x04C00661              // ADD Z1.D, P1/M, Z1.D, Z19.D
    WORD $0x04C00682              // ADD Z2.D, P1/M, Z2.D, Z20.D
    WORD $0x04C006A3              // ADD Z3.D, P1/M, Z3.D, Z21.D
    
    // Z15
    WORD $0x05B039F0              // SUNPKLO Z16.S, Z15.H
    WORD $0x05B139F1              // SUNPKHI Z17.S, Z15.H
    WORD $0x05F03A12              // SUNPKLO Z18.D, Z16.S
    WORD $0x05F13A13              // SUNPKHI Z19.D, Z16.S
    WORD $0x05F03A34              // SUNPKLO Z20.D, Z17.S
    WORD $0x05F13A35              // SUNPKHI Z21.D, Z17.S
    WORD $0x04C00644              // ADD Z4.D, P1/M, Z4.D, Z18.D
    WORD $0x04C00665              // ADD Z5.D, P1/M, Z5.D, Z19.D
    WORD $0x04C00686              // ADD Z6.D, P1/M, Z6.D, Z20.D
    WORD $0x04C006A7              // ADD Z7.D, P1/M, Z7.D, Z21.D
    
    // Advance pointer: R3 elements * 2 bytes
    LSL $1, R3, R4
    ADD R4, R0, R0
    SUB R3, R1, R1
    CMP R3, R1
    BGE sve_sum16_loop8

sve_sum16_tail:
    MOVD ZR, R4                   // R4 = 0 (loop index)
    CBZ R1, sve_sum16_reduce
    
sve_sum16_tail_loop:
    // Use WHILELO for predicated tail
    WORD $0x25611C82              // WHILELO P2.H, X4, X1
    BEQ sve_sum16_reduce          // Exit if no active lanes
    
    WORD $0xA4A0A808              // LD1H {Z8.H}, P2/Z, [R0]
    
    // Widen and add
    WORD $0x05B03910              // SUNPKLO Z16.S, Z8.H
    WORD $0x05B13911              // SUNPKHI Z17.S, Z8.H
    WORD $0x05F03A12              // SUNPKLO Z18.D, Z16.S
    WORD $0x05F13A13              // SUNPKHI Z19.D, Z16.S
    WORD $0x05F03A34              // SUNPKLO Z20.D, Z17.S
    WORD $0x05F13A35              // SUNPKHI Z21.D, Z17.S
    WORD $0x25D8E3E1              // PTRUE P1.D
    WORD $0x04C00640              // ADD Z0.D, P1/M, Z0.D, Z18.D
    WORD $0x04C00661              // ADD Z1.D, P1/M, Z1.D, Z19.D
    WORD $0x04C00682              // ADD Z2.D, P1/M, Z2.D, Z20.D
    WORD $0x04C006A3              // ADD Z3.D, P1/M, Z3.D, Z21.D
    
    WORD $0x0470E3E4              // INCH X4
    LSL $1, R2, R5                // bytes per vector = R2 * 2
    ADD R5, R0, R0
    B sve_sum16_tail_loop

sve_sum16_reduce:
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
// │ func minInt16SVE(vals []int16) int16                                         │
// │                                                                              │
// │ Strategy: Native SMIN.H and SMINV for horizontal reduction                   │
// └──────────────────────────────────────────────────────────────────────────────┘
TEXT ·minInt16SVE(SB), NOSPLIT, $0-26
    MOVD vals_base+0(FP), R0
    MOVD vals_len+8(FP), R1
    
    CBZ R1, sve_min16_empty
    
    WORD $0x2558E3E0              // PTRUE P0.H
    
    WORD $0x0460E3E2              // CNTH X2
    
    // Check if we have at least one full vector
    CMP R2, R1
    BLT sve_min16_small           // If len < vector_size, handle specially
    
    // Initialize with first full vector
    WORD $0xA4A0A000              // LD1H {Z0.H}, P0/Z, [R0]
    // MOV Z1-Z7 = Z0
    WORD $0x04603001              // ORR Z1.D, Z0.D, Z0.D
    WORD $0x04603002              // ORR Z2.D, Z0.D, Z0.D
    WORD $0x04603003              // ORR Z3.D, Z0.D, Z0.D
    WORD $0x04603004              // ORR Z4.D, Z0.D, Z0.D
    WORD $0x04603005              // ORR Z5.D, Z0.D, Z0.D
    WORD $0x04603006              // ORR Z6.D, Z0.D, Z0.D
    WORD $0x04603007              // ORR Z7.D, Z0.D, Z0.D
    
    // Advance past initialized vector
    LSL $1, R2, R4                // bytes per vector = R2 * 2
    ADD R4, R0, R0
    SUB R2, R1, R1
    
    LSL $3, R2, R3
    
    CMP R3, R1
    BLT sve_min16_tail
    B sve_min16_loop8

sve_min16_small:
    // Handle arrays smaller than one vector
    // Load first element and broadcast to all accumulators
    MOVH (R0), R3
    SXTH R3, R3
    WORD $0x05603860              // DUP Z0.H, W3
    WORD $0x04603001              // ORR Z1.D, Z0.D, Z0.D
    WORD $0x04603002              // ORR Z2.D, Z0.D, Z0.D
    WORD $0x04603003              // ORR Z3.D, Z0.D, Z0.D
    WORD $0x04603004              // ORR Z4.D, Z0.D, Z0.D
    WORD $0x04603005              // ORR Z5.D, Z0.D, Z0.D
    WORD $0x04603006              // ORR Z6.D, Z0.D, Z0.D
    WORD $0x04603007              // ORR Z7.D, Z0.D, Z0.D
    B sve_min16_tail

sve_min16_loop8:
    WORD $0xA4A0A008              // LD1H {Z8.H}, P0/Z, [R0, #0, MUL VL]
    WORD $0xA4A1A009              // LD1H {Z9.H}, P0/Z, [R0, #1, MUL VL]
    WORD $0xA4A2A00A              // LD1H {Z10.H}, P0/Z, [R0, #2, MUL VL]
    WORD $0xA4A3A00B              // LD1H {Z11.H}, P0/Z, [R0, #3, MUL VL]
    WORD $0xA4A4A00C              // LD1H {Z12.H}, P0/Z, [R0, #4, MUL VL]
    WORD $0xA4A5A00D              // LD1H {Z13.H}, P0/Z, [R0, #5, MUL VL]
    WORD $0xA4A6A00E              // LD1H {Z14.H}, P0/Z, [R0, #6, MUL VL]
    WORD $0xA4A7A00F              // LD1H {Z15.H}, P0/Z, [R0, #7, MUL VL]
    
    // SMIN.H
    WORD $0x044A0100              // SMIN Z0.H, P0/M, Z0.H, Z8.H
    WORD $0x044A0121              // SMIN Z1.H, P0/M, Z1.H, Z9.H
    WORD $0x044A0142              // SMIN Z2.H, P0/M, Z2.H, Z10.H
    WORD $0x044A0163              // SMIN Z3.H, P0/M, Z3.H, Z11.H
    WORD $0x044A0184              // SMIN Z4.H, P0/M, Z4.H, Z12.H
    WORD $0x044A01A5              // SMIN Z5.H, P0/M, Z5.H, Z13.H
    WORD $0x044A01C6              // SMIN Z6.H, P0/M, Z6.H, Z14.H
    WORD $0x044A01E7              // SMIN Z7.H, P0/M, Z7.H, Z15.H
    
    LSL $1, R3, R4
    ADD R4, R0, R0
    SUB R3, R1, R1
    CMP R3, R1
    BGE sve_min16_loop8

sve_min16_tail:
    MOVD ZR, R4                   // R4 = 0 (loop index)
    CBZ R1, sve_min16_reduce
    
sve_min16_tail_loop:
    WORD $0x25611C82              // WHILELO P2.H, X4, X1
    BEQ sve_min16_reduce          // Exit if no active lanes
    
    WORD $0xA4A0A808              // LD1H {Z8.H}, P2/Z, [R0]
    
    WORD $0x044A0900              // SMIN Z0.H, P2/M, Z0.H, Z8.H
    
    WORD $0x0470E3E4              // INCH X4
    LSL $1, R2, R5                // bytes per vector = R2 * 2
    ADD R5, R0, R0
    B sve_min16_tail_loop

sve_min16_reduce:
    // Tree reduction
    WORD $0x044A0080              // SMIN Z0.H, P0/M, Z0.H, Z4.H
    WORD $0x044A00A1              // SMIN Z1.H, P0/M, Z1.H, Z5.H
    WORD $0x044A00C2              // SMIN Z2.H, P0/M, Z2.H, Z6.H
    WORD $0x044A00E3              // SMIN Z3.H, P0/M, Z3.H, Z7.H
    WORD $0x044A0040              // SMIN Z0.H, P0/M, Z0.H, Z2.H
    WORD $0x044A0061              // SMIN Z1.H, P0/M, Z1.H, Z3.H
    WORD $0x044A0020              // SMIN Z0.H, P0/M, Z0.H, Z1.H
    
    // SMINV H1, P0, Z0.H - horizontal min
    WORD $0x044A2001              // SMINV H1, P0, Z0.H
    FMOVS F1, R0
    SXTH R0, R0
    B sve_min16_done

sve_min16_empty:
    MOVW $0, R0

sve_min16_done:
    MOVH R0, ret+24(FP)
    RET

// ┌──────────────────────────────────────────────────────────────────────────────┐
// │ func maxInt16SVE(vals []int16) int16                                         │
// │                                                                              │
// │ Strategy: Native SMAX.H and SMAXV for horizontal reduction                   │
// └──────────────────────────────────────────────────────────────────────────────┘
TEXT ·maxInt16SVE(SB), NOSPLIT, $0-26
    MOVD vals_base+0(FP), R0
    MOVD vals_len+8(FP), R1
    
    CBZ R1, sve_max16_empty
    
    WORD $0x2558E3E0              // PTRUE P0.H
    
    WORD $0x0460E3E2              // CNTH X2
    
    // Check if we have at least one full vector
    CMP R2, R1
    BLT sve_max16_small           // If len < vector_size, handle specially
    
    // Initialize with first full vector
    WORD $0xA4A0A000              // LD1H {Z0.H}, P0/Z, [R0]
    // MOV Z1-Z7 = Z0
    WORD $0x04603001              // ORR Z1.D, Z0.D, Z0.D
    WORD $0x04603002              // ORR Z2.D, Z0.D, Z0.D
    WORD $0x04603003              // ORR Z3.D, Z0.D, Z0.D
    WORD $0x04603004              // ORR Z4.D, Z0.D, Z0.D
    WORD $0x04603005              // ORR Z5.D, Z0.D, Z0.D
    WORD $0x04603006              // ORR Z6.D, Z0.D, Z0.D
    WORD $0x04603007              // ORR Z7.D, Z0.D, Z0.D
    
    // Advance past initialized vector
    LSL $1, R2, R4                // bytes per vector = R2 * 2
    ADD R4, R0, R0
    SUB R2, R1, R1
    
    LSL $3, R2, R3
    
    CMP R3, R1
    BLT sve_max16_tail
    B sve_max16_loop8

sve_max16_small:
    // Handle arrays smaller than one vector
    // Load first element and broadcast to all accumulators
    MOVH (R0), R3
    SXTH R3, R3
    WORD $0x05603860              // DUP Z0.H, W3
    WORD $0x04603001              // ORR Z1.D, Z0.D, Z0.D
    WORD $0x04603002              // ORR Z2.D, Z0.D, Z0.D
    WORD $0x04603003              // ORR Z3.D, Z0.D, Z0.D
    WORD $0x04603004              // ORR Z4.D, Z0.D, Z0.D
    WORD $0x04603005              // ORR Z5.D, Z0.D, Z0.D
    WORD $0x04603006              // ORR Z6.D, Z0.D, Z0.D
    WORD $0x04603007              // ORR Z7.D, Z0.D, Z0.D
    B sve_max16_tail

sve_max16_loop8:
    WORD $0xA4A0A008              // LD1H {Z8.H}, P0/Z, [R0, #0, MUL VL]
    WORD $0xA4A1A009              // LD1H {Z9.H}, P0/Z, [R0, #1, MUL VL]
    WORD $0xA4A2A00A              // LD1H {Z10.H}, P0/Z, [R0, #2, MUL VL]
    WORD $0xA4A3A00B              // LD1H {Z11.H}, P0/Z, [R0, #3, MUL VL]
    WORD $0xA4A4A00C              // LD1H {Z12.H}, P0/Z, [R0, #4, MUL VL]
    WORD $0xA4A5A00D              // LD1H {Z13.H}, P0/Z, [R0, #5, MUL VL]
    WORD $0xA4A6A00E              // LD1H {Z14.H}, P0/Z, [R0, #6, MUL VL]
    WORD $0xA4A7A00F              // LD1H {Z15.H}, P0/Z, [R0, #7, MUL VL]
    
    // SMAX.H - encoding: 04 48 00+pred<<1 dst | 04 48 01 src<<5|dst
    WORD $0x04480100              // SMAX Z0.H, P0/M, Z0.H, Z8.H
    WORD $0x04480121              // SMAX Z1.H, P0/M, Z1.H, Z9.H
    WORD $0x04480142              // SMAX Z2.H, P0/M, Z2.H, Z10.H
    WORD $0x04480163              // SMAX Z3.H, P0/M, Z3.H, Z11.H
    WORD $0x04480184              // SMAX Z4.H, P0/M, Z4.H, Z12.H
    WORD $0x044801A5              // SMAX Z5.H, P0/M, Z5.H, Z13.H
    WORD $0x044801C6              // SMAX Z6.H, P0/M, Z6.H, Z14.H
    WORD $0x044801E7              // SMAX Z7.H, P0/M, Z7.H, Z15.H
    
    LSL $1, R3, R4
    ADD R4, R0, R0
    SUB R3, R1, R1
    CMP R3, R1
    BGE sve_max16_loop8

sve_max16_tail:
    MOVD ZR, R4                   // R4 = 0 (loop index)
    CBZ R1, sve_max16_reduce
    
sve_max16_tail_loop:
    WORD $0x25611C82              // WHILELO P2.H, X4, X1
    BEQ sve_max16_reduce          // Exit if no active lanes
    
    WORD $0xA4A0A808              // LD1H {Z8.H}, P2/Z, [R0]
    
    WORD $0x04480900              // SMAX Z0.H, P2/M, Z0.H, Z8.H
    
    WORD $0x0470E3E4              // INCH X4
    LSL $1, R2, R5                // bytes per vector = R2 * 2
    ADD R5, R0, R0
    B sve_max16_tail_loop

sve_max16_reduce:
    // Tree reduction
    WORD $0x04480080              // SMAX Z0.H, P0/M, Z0.H, Z4.H
    WORD $0x044800A1              // SMAX Z1.H, P0/M, Z1.H, Z5.H
    WORD $0x044800C2              // SMAX Z2.H, P0/M, Z2.H, Z6.H
    WORD $0x044800E3              // SMAX Z3.H, P0/M, Z3.H, Z7.H
    WORD $0x04480040              // SMAX Z0.H, P0/M, Z0.H, Z2.H
    WORD $0x04480061              // SMAX Z1.H, P0/M, Z1.H, Z3.H
    WORD $0x04480020              // SMAX Z0.H, P0/M, Z0.H, Z1.H
    
    // SMAXV H1, P0, Z0.H - horizontal max
    WORD $0x04482001              // SMAXV H1, P0, Z0.H
    FMOVS F1, R0
    SXTH R0, R0
    B sve_max16_done

sve_max16_empty:
    MOVW $0, R0

sve_max16_done:
    MOVH R0, ret+24(FP)
    RET

// ┌──────────────────────────────────────────────────────────────────────────────┐
// │ func dotProductInt16SVE(a, b []int16) int64                                  │
// │                                                                              │
// │ Strategy: Multiply int16 pairs, widen to int64, accumulate                   │
// └──────────────────────────────────────────────────────────────────────────────┘
TEXT ·dotProductInt16SVE(SB), NOSPLIT, $0-56
    MOVD a_base+0(FP), R0
    MOVD a_len+8(FP), R1
    MOVD b_base+24(FP), R2
    
    WORD $0x2558E3E0              // PTRUE P0.H
    WORD $0x25D8E3E1              // PTRUE P1.D
    
    // Zero 8 int64 accumulators
    WORD $0x25F8C000              // DUP Z0.D, #0
    WORD $0x25F8C001              // DUP Z1.D, #0
    WORD $0x25F8C002              // DUP Z2.D, #0
    WORD $0x25F8C003              // DUP Z3.D, #0
    WORD $0x25F8C004              // DUP Z4.D, #0
    WORD $0x25F8C005              // DUP Z5.D, #0
    WORD $0x25F8C006              // DUP Z6.D, #0
    WORD $0x25F8C007              // DUP Z7.D, #0
    
    WORD $0x0460E3E3              // CNTH X3
    LSL $2, R3, R4                // R4 = R3 * 4 (4 vectors per iteration)
    
    CMP R4, R1
    BLT sve_dot16_tail

sve_dot16_loop4:
    // Load 4 vectors from each array
    WORD $0xA4A0A008              // LD1H {Z8.H}, P0/Z, [R0, #0, MUL VL]
    WORD $0xA4A1A009              // LD1H {Z9.H}, P0/Z, [R0, #1, MUL VL]
    WORD $0xA4A2A00A              // LD1H {Z10.H}, P0/Z, [R0, #2, MUL VL]
    WORD $0xA4A3A00B              // LD1H {Z11.H}, P0/Z, [R0, #3, MUL VL]
    
    WORD $0xA4A0A050              // LD1H {Z16.H}, P0/Z, [R2, #0, MUL VL]
    WORD $0xA4A1A051              // LD1H {Z17.H}, P0/Z, [R2, #1, MUL VL]
    WORD $0xA4A2A052              // LD1H {Z18.H}, P0/Z, [R2, #2, MUL VL]
    WORD $0xA4A3A053              // LD1H {Z19.H}, P0/Z, [R2, #3, MUL VL]
    
    // Multiply (16-bit × 16-bit = 16-bit, truncated)
    // We need widening multiply: use MUL then widen
    // Actually SVE doesn't have direct widening multiply like NEON SMULL
    // Strategy: widen both operands to 32-bit, multiply, then widen result to 64-bit
    
    // Widen Z8.H -> Z20.S (lo), Z21.S (hi)
    WORD $0x2598E3E2              // PTRUE P2.S
    WORD $0x05B03914              // SUNPKLO Z20.S, Z8.H
    WORD $0x05B13915              // SUNPKHI Z21.S, Z8.H
    // Widen Z16.H -> Z22.S, Z23.S
    WORD $0x05B03A16              // SUNPKLO Z22.S, Z16.H
    WORD $0x05B13A17              // SUNPKHI Z23.S, Z16.H
    
    // Multiply 32-bit
    WORD $0x04900AD4              // MUL Z20.S, P2/M, Z20.S, Z22.S
    WORD $0x04900AF5              // MUL Z21.S, P2/M, Z21.S, Z23.S
    
    // Widen 32-bit products to 64-bit and add
    WORD $0x05F03A98              // SUNPKLO Z24.D, Z20.S
    WORD $0x05F13A99              // SUNPKHI Z25.D, Z20.S
    WORD $0x05F03ABA              // SUNPKLO Z26.D, Z21.S
    WORD $0x05F13ABB              // SUNPKHI Z27.D, Z21.S
    
    WORD $0x04C00700              // ADD Z0.D, P1/M, Z0.D, Z24.D
    WORD $0x04C00721              // ADD Z1.D, P1/M, Z1.D, Z25.D
    WORD $0x04C00742              // ADD Z2.D, P1/M, Z2.D, Z26.D
    WORD $0x04C00763              // ADD Z3.D, P1/M, Z3.D, Z27.D
    
    // Process Z9 × Z17
    WORD $0x05B03934              // SUNPKLO Z20.S, Z9.H
    WORD $0x05B13935              // SUNPKHI Z21.S, Z9.H
    WORD $0x05B03A36              // SUNPKLO Z22.S, Z17.H
    WORD $0x05B13A37              // SUNPKHI Z23.S, Z17.H
    WORD $0x04900AD4              // MUL Z20.S, P2/M, Z20.S, Z22.S
    WORD $0x04900AF5              // MUL Z21.S, P2/M, Z21.S, Z23.S
    WORD $0x05F03A98              // SUNPKLO Z24.D, Z20.S
    WORD $0x05F13A99              // SUNPKHI Z25.D, Z20.S
    WORD $0x05F03ABA              // SUNPKLO Z26.D, Z21.S
    WORD $0x05F13ABB              // SUNPKHI Z27.D, Z21.S
    WORD $0x04C00704              // ADD Z4.D, P1/M, Z4.D, Z24.D
    WORD $0x04C00725              // ADD Z5.D, P1/M, Z5.D, Z25.D
    WORD $0x04C00746              // ADD Z6.D, P1/M, Z6.D, Z26.D
    WORD $0x04C00767              // ADD Z7.D, P1/M, Z7.D, Z27.D
    
    // Process Z10 × Z18
    WORD $0x05B03954              // SUNPKLO Z20.S, Z10.H
    WORD $0x05B13955              // SUNPKHI Z21.S, Z10.H
    WORD $0x05B03A56              // SUNPKLO Z22.S, Z18.H
    WORD $0x05B13A57              // SUNPKHI Z23.S, Z18.H
    WORD $0x04900AD4              // MUL Z20.S, P2/M, Z20.S, Z22.S
    WORD $0x04900AF5              // MUL Z21.S, P2/M, Z21.S, Z23.S
    WORD $0x05F03A98              // SUNPKLO Z24.D, Z20.S
    WORD $0x05F13A99              // SUNPKHI Z25.D, Z20.S
    WORD $0x05F03ABA              // SUNPKLO Z26.D, Z21.S
    WORD $0x05F13ABB              // SUNPKHI Z27.D, Z21.S
    WORD $0x04C00700              // ADD Z0.D, P1/M, Z0.D, Z24.D
    WORD $0x04C00721              // ADD Z1.D, P1/M, Z1.D, Z25.D
    WORD $0x04C00742              // ADD Z2.D, P1/M, Z2.D, Z26.D
    WORD $0x04C00763              // ADD Z3.D, P1/M, Z3.D, Z27.D
    
    // Process Z11 × Z19
    WORD $0x05B03974              // SUNPKLO Z20.S, Z11.H
    WORD $0x05B13975              // SUNPKHI Z21.S, Z11.H
    WORD $0x05B03A76              // SUNPKLO Z22.S, Z19.H
    WORD $0x05B13A77              // SUNPKHI Z23.S, Z19.H
    WORD $0x04900AD4              // MUL Z20.S, P2/M, Z20.S, Z22.S
    WORD $0x04900AF5              // MUL Z21.S, P2/M, Z21.S, Z23.S
    WORD $0x05F03A98              // SUNPKLO Z24.D, Z20.S
    WORD $0x05F13A99              // SUNPKHI Z25.D, Z20.S
    WORD $0x05F03ABA              // SUNPKLO Z26.D, Z21.S
    WORD $0x05F13ABB              // SUNPKHI Z27.D, Z21.S
    WORD $0x04C00704              // ADD Z4.D, P1/M, Z4.D, Z24.D
    WORD $0x04C00725              // ADD Z5.D, P1/M, Z5.D, Z25.D
    WORD $0x04C00746              // ADD Z6.D, P1/M, Z6.D, Z26.D
    WORD $0x04C00767              // ADD Z7.D, P1/M, Z7.D, Z27.D
    
    // Advance pointers
    LSL $1, R4, R5
    ADD R5, R0, R0
    ADD R5, R2, R2
    SUB R4, R1, R1
    CMP R4, R1
    BGE sve_dot16_loop4

sve_dot16_tail:
    MOVD ZR, R5                   // R5 = 0 (loop index)
    CBZ R1, sve_dot16_reduce
    
sve_dot16_tail_loop:
    WORD $0x25611CA2              // WHILELO P2.H, X5, X1
    BEQ sve_dot16_reduce          // Exit if no active lanes
    
    WORD $0xA4A0A808              // LD1H {Z8.H}, P2/Z, [R0]
    WORD $0xA4A0A850              // LD1H {Z16.H}, P2/Z, [R2]
    
    // Widen and multiply
    WORD $0x2598E3E3              // PTRUE P3.S
    WORD $0x05B03914              // SUNPKLO Z20.S, Z8.H
    WORD $0x05B13915              // SUNPKHI Z21.S, Z8.H
    WORD $0x05B03A16              // SUNPKLO Z22.S, Z16.H
    WORD $0x05B13A17              // SUNPKHI Z23.S, Z16.H
    WORD $0x04900ED4              // MUL Z20.S, P3/M, Z20.S, Z22.S
    WORD $0x04900EF5              // MUL Z21.S, P3/M, Z21.S, Z23.S
    WORD $0x05F03A98              // SUNPKLO Z24.D, Z20.S
    WORD $0x05F13A99              // SUNPKHI Z25.D, Z20.S
    WORD $0x05F03ABA              // SUNPKLO Z26.D, Z21.S
    WORD $0x05F13ABB              // SUNPKHI Z27.D, Z21.S
    WORD $0x04C00700              // ADD Z0.D, P1/M, Z0.D, Z24.D
    WORD $0x04C00721              // ADD Z1.D, P1/M, Z1.D, Z25.D
    WORD $0x04C00742              // ADD Z2.D, P1/M, Z2.D, Z26.D
    WORD $0x04C00763              // ADD Z3.D, P1/M, Z3.D, Z27.D
    
    WORD $0x0470E3E5              // INCH X5
    LSL $1, R3, R6                // bytes per vector = R3 * 2
    ADD R6, R0, R0
    ADD R6, R2, R2
    B sve_dot16_tail_loop

sve_dot16_reduce:
    // Tree reduction
    WORD $0x04C00480              // ADD Z0.D, P1/M, Z0.D, Z4.D
    WORD $0x04C004A1              // ADD Z1.D, P1/M, Z1.D, Z5.D
    WORD $0x04C004C2              // ADD Z2.D, P1/M, Z2.D, Z6.D
    WORD $0x04C004E3              // ADD Z3.D, P1/M, Z3.D, Z7.D
    WORD $0x04C00440              // ADD Z0.D, P1/M, Z0.D, Z2.D
    WORD $0x04C00461              // ADD Z1.D, P1/M, Z1.D, Z3.D
    WORD $0x04C00420              // ADD Z0.D, P1/M, Z0.D, Z1.D
    
    WORD $0x04C12401              // UADDV D1, P1, Z0.D
    FMOVD F1, R0
    
    MOVD R0, ret+48(FP)
    RET

// ┌──────────────────────────────────────────────────────────────────────────────┐
// │ func sumSqInt16SVE(vals []int16) int64                                       │
// │                                                                              │
// │ Strategy: Square int16 values, widen to int64, accumulate                    │
// │ Same as dotProduct but with same operand for squaring                        │
// └──────────────────────────────────────────────────────────────────────────────┘
TEXT ·sumSqInt16SVE(SB), NOSPLIT, $0-32
    MOVD vals_base+0(FP), R0
    MOVD vals_len+8(FP), R1
    
    WORD $0x2558E3E0              // PTRUE P0.H
    WORD $0x25D8E3E1              // PTRUE P1.D
    
    // Zero 8 int64 accumulators
    WORD $0x25F8C000              // DUP Z0.D, #0
    WORD $0x25F8C001              // DUP Z1.D, #0
    WORD $0x25F8C002              // DUP Z2.D, #0
    WORD $0x25F8C003              // DUP Z3.D, #0
    WORD $0x25F8C004              // DUP Z4.D, #0
    WORD $0x25F8C005              // DUP Z5.D, #0
    WORD $0x25F8C006              // DUP Z6.D, #0
    WORD $0x25F8C007              // DUP Z7.D, #0
    
    WORD $0x0460E3E3              // CNTH X3
    LSL $2, R3, R4                // R4 = R3 * 4 (4 vectors per iteration)
    
    CMP R4, R1
    BLT sve_sumsq16_tail

sve_sumsq16_loop4:
    // Load 4 vectors
    WORD $0xA4A0A008              // LD1H {Z8.H}, P0/Z, [R0, #0, MUL VL]
    WORD $0xA4A1A009              // LD1H {Z9.H}, P0/Z, [R0, #1, MUL VL]
    WORD $0xA4A2A00A              // LD1H {Z10.H}, P0/Z, [R0, #2, MUL VL]
    WORD $0xA4A3A00B              // LD1H {Z11.H}, P0/Z, [R0, #3, MUL VL]
    
    // Widen Z8.H -> Z20.S (lo), Z21.S (hi), then square
    WORD $0x2598E3E2              // PTRUE P2.S
    WORD $0x05B03914              // SUNPKLO Z20.S, Z8.H
    WORD $0x05B13915              // SUNPKHI Z21.S, Z8.H
    
    // Square 32-bit: Z20 = Z20 * Z20
    WORD $0x04900A94              // MUL Z20.S, P2/M, Z20.S, Z20.S
    WORD $0x04900AB5              // MUL Z21.S, P2/M, Z21.S, Z21.S
    
    // Widen 32-bit products to 64-bit and add
    WORD $0x05F03A98              // SUNPKLO Z24.D, Z20.S
    WORD $0x05F13A99              // SUNPKHI Z25.D, Z20.S
    WORD $0x05F03ABA              // SUNPKLO Z26.D, Z21.S
    WORD $0x05F13ABB              // SUNPKHI Z27.D, Z21.S
    
    WORD $0x04C00700              // ADD Z0.D, P1/M, Z0.D, Z24.D
    WORD $0x04C00721              // ADD Z1.D, P1/M, Z1.D, Z25.D
    WORD $0x04C00742              // ADD Z2.D, P1/M, Z2.D, Z26.D
    WORD $0x04C00763              // ADD Z3.D, P1/M, Z3.D, Z27.D
    
    // Process Z9
    WORD $0x05B03934              // SUNPKLO Z20.S, Z9.H
    WORD $0x05B13935              // SUNPKHI Z21.S, Z9.H
    WORD $0x04900A94              // MUL Z20.S, P2/M, Z20.S, Z20.S
    WORD $0x04900AB5              // MUL Z21.S, P2/M, Z21.S, Z21.S
    WORD $0x05F03A98              // SUNPKLO Z24.D, Z20.S
    WORD $0x05F13A99              // SUNPKHI Z25.D, Z20.S
    WORD $0x05F03ABA              // SUNPKLO Z26.D, Z21.S
    WORD $0x05F13ABB              // SUNPKHI Z27.D, Z21.S
    WORD $0x04C00704              // ADD Z4.D, P1/M, Z4.D, Z24.D
    WORD $0x04C00725              // ADD Z5.D, P1/M, Z5.D, Z25.D
    WORD $0x04C00746              // ADD Z6.D, P1/M, Z6.D, Z26.D
    WORD $0x04C00767              // ADD Z7.D, P1/M, Z7.D, Z27.D
    
    // Process Z10
    WORD $0x05B03954              // SUNPKLO Z20.S, Z10.H
    WORD $0x05B13955              // SUNPKHI Z21.S, Z10.H
    WORD $0x04900A94              // MUL Z20.S, P2/M, Z20.S, Z20.S
    WORD $0x04900AB5              // MUL Z21.S, P2/M, Z21.S, Z21.S
    WORD $0x05F03A98              // SUNPKLO Z24.D, Z20.S
    WORD $0x05F13A99              // SUNPKHI Z25.D, Z20.S
    WORD $0x05F03ABA              // SUNPKLO Z26.D, Z21.S
    WORD $0x05F13ABB              // SUNPKHI Z27.D, Z21.S
    WORD $0x04C00700              // ADD Z0.D, P1/M, Z0.D, Z24.D
    WORD $0x04C00721              // ADD Z1.D, P1/M, Z1.D, Z25.D
    WORD $0x04C00742              // ADD Z2.D, P1/M, Z2.D, Z26.D
    WORD $0x04C00763              // ADD Z3.D, P1/M, Z3.D, Z27.D
    
    // Process Z11
    WORD $0x05B03974              // SUNPKLO Z20.S, Z11.H
    WORD $0x05B13975              // SUNPKHI Z21.S, Z11.H
    WORD $0x04900A94              // MUL Z20.S, P2/M, Z20.S, Z20.S
    WORD $0x04900AB5              // MUL Z21.S, P2/M, Z21.S, Z21.S
    WORD $0x05F03A98              // SUNPKLO Z24.D, Z20.S
    WORD $0x05F13A99              // SUNPKHI Z25.D, Z20.S
    WORD $0x05F03ABA              // SUNPKLO Z26.D, Z21.S
    WORD $0x05F13ABB              // SUNPKHI Z27.D, Z21.S
    WORD $0x04C00704              // ADD Z4.D, P1/M, Z4.D, Z24.D
    WORD $0x04C00725              // ADD Z5.D, P1/M, Z5.D, Z25.D
    WORD $0x04C00746              // ADD Z6.D, P1/M, Z6.D, Z26.D
    WORD $0x04C00767              // ADD Z7.D, P1/M, Z7.D, Z27.D
    
    // Advance pointer
    LSL $1, R4, R5
    ADD R5, R0, R0
    SUB R4, R1, R1
    CMP R4, R1
    BGE sve_sumsq16_loop4

sve_sumsq16_tail:
    MOVD ZR, R5                   // R5 = 0 (loop index)
    CBZ R1, sve_sumsq16_reduce
    
sve_sumsq16_tail_loop:
    WORD $0x25611CA2              // WHILELO P2.H, X5, X1
    BEQ sve_sumsq16_reduce        // Exit if no active lanes
    
    WORD $0xA4A0A808              // LD1H {Z8.H}, P2/Z, [R0]
    
    // Widen and square
    WORD $0x2598E3E3              // PTRUE P3.S
    WORD $0x05B03914              // SUNPKLO Z20.S, Z8.H
    WORD $0x05B13915              // SUNPKHI Z21.S, Z8.H
    WORD $0x04900E94              // MUL Z20.S, P3/M, Z20.S, Z20.S
    WORD $0x04900EB5              // MUL Z21.S, P3/M, Z21.S, Z21.S
    WORD $0x05F03A98              // SUNPKLO Z24.D, Z20.S
    WORD $0x05F13A99              // SUNPKHI Z25.D, Z20.S
    WORD $0x05F03ABA              // SUNPKLO Z26.D, Z21.S
    WORD $0x05F13ABB              // SUNPKHI Z27.D, Z21.S
    WORD $0x04C00700              // ADD Z0.D, P1/M, Z0.D, Z24.D
    WORD $0x04C00721              // ADD Z1.D, P1/M, Z1.D, Z25.D
    WORD $0x04C00742              // ADD Z2.D, P1/M, Z2.D, Z26.D
    WORD $0x04C00763              // ADD Z3.D, P1/M, Z3.D, Z27.D
    
    WORD $0x0470E3E5              // INCH X5
    LSL $1, R3, R6                // bytes per vector = R3 * 2
    ADD R6, R0, R0
    B sve_sumsq16_tail_loop

sve_sumsq16_reduce:
    // Tree reduction
    WORD $0x04C00480              // ADD Z0.D, P1/M, Z0.D, Z4.D
    WORD $0x04C004A1              // ADD Z1.D, P1/M, Z1.D, Z5.D
    WORD $0x04C004C2              // ADD Z2.D, P1/M, Z2.D, Z6.D
    WORD $0x04C004E3              // ADD Z3.D, P1/M, Z3.D, Z7.D
    WORD $0x04C00440              // ADD Z0.D, P1/M, Z0.D, Z2.D
    WORD $0x04C00461              // ADD Z1.D, P1/M, Z1.D, Z3.D
    WORD $0x04C00420              // ADD Z0.D, P1/M, Z0.D, Z1.D
    
    WORD $0x04C12401              // UADDV D1, P1, Z0.D
    FMOVD F1, R0
    
    MOVD R0, ret+24(FP)
    RET

// ┌──────────────────────────────────────────────────────────────────────────────┐
// │ func anyAbsGreaterThanInt16SVE(vals []int16, threshold int16) bool           │
// │                                                                              │
// │ Strategy: Use ABS + CMPGT with early exit via PTEST                         │
// └──────────────────────────────────────────────────────────────────────────────┘
TEXT ·anyAbsGreaterThanInt16SVE(SB), NOSPLIT, $0-41
    MOVD vals_base+0(FP), R0
    MOVD vals_len+8(FP), R1
    MOVH threshold+24(FP), R2
    
    CBZ R1, sve_absgt16_notfound
    
    WORD $0x2558E3E0              // PTRUE P0.H
    
    // Broadcast threshold to Z0.H
    WORD $0x05603840              // DUP Z0.H, W2
    
    // Compute -threshold in Z1.H for the v < -threshold check
    WORD $0x0457A001              // NEG Z1.H, P0/M, Z0.H
    
    // CNTH - count 16-bit elements per vector
    WORD $0x0460E3E3              // CNTH X3
    LSL $2, R3, R4                // R4 = R3 * 4 (4 vectors per iteration)
    
    CMP R4, R1
    BLT sve_absgt16_tail

sve_absgt16_loop4:
    // Load 4 vectors
    WORD $0xA4A0A008              // LD1H {Z8.H}, P0/Z, [R0, #0, MUL VL]
    WORD $0xA4A1A009              // LD1H {Z9.H}, P0/Z, [R0, #1, MUL VL]
    WORD $0xA4A2A00A              // LD1H {Z10.H}, P0/Z, [R0, #2, MUL VL]
    WORD $0xA4A3A00B              // LD1H {Z11.H}, P0/Z, [R0, #3, MUL VL]
    
    // For each vector: check (v > threshold) || (v < -threshold)
    // = (v > threshold) || (-threshold > v)
    
    // Z8: CMPGT P1 = (Z8 > Z0), CMPGT P2 = (Z1 > Z8), ORR P1 = P1 | P2
    WORD $0x24408111              // CMPGT P1.H, P0/Z, Z8.H, Z0.H
    WORD $0x24488032              // CMPGT P2.H, P0/Z, Z1.H, Z8.H
    WORD $0x25824021              // ORR P1.B, P0/Z, P1.B, P2.B
    WORD $0x2550C020              // PTEST P0, P1.B
    BNE sve_absgt16_found
    
    // Z9
    WORD $0x24408131              // CMPGT P1.H, P0/Z, Z9.H, Z0.H
    WORD $0x24498032              // CMPGT P2.H, P0/Z, Z1.H, Z9.H
    WORD $0x25824021              // ORR P1.B, P0/Z, P1.B, P2.B
    WORD $0x2550C020              // PTEST P0, P1.B
    BNE sve_absgt16_found
    
    // Z10
    WORD $0x24408151              // CMPGT P1.H, P0/Z, Z10.H, Z0.H
    WORD $0x244A8032              // CMPGT P2.H, P0/Z, Z1.H, Z10.H
    WORD $0x25824021              // ORR P1.B, P0/Z, P1.B, P2.B
    WORD $0x2550C020              // PTEST P0, P1.B
    BNE sve_absgt16_found
    
    // Z11
    WORD $0x24408171              // CMPGT P1.H, P0/Z, Z11.H, Z0.H
    WORD $0x244B8032              // CMPGT P2.H, P0/Z, Z1.H, Z11.H
    WORD $0x25824021              // ORR P1.B, P0/Z, P1.B, P2.B
    WORD $0x2550C020              // PTEST P0, P1.B
    BNE sve_absgt16_found
    
    // Advance pointer
    LSL $1, R4, R5
    ADD R5, R0, R0
    SUB R4, R1, R1
    CMP R4, R1
    BGE sve_absgt16_loop4

sve_absgt16_tail:
    CBZ R1, sve_absgt16_notfound
    MOVD ZR, R5

sve_absgt16_tail_loop:
    WORD $0x25611CA2              // WHILELO P2.H, X5, X1
    BEQ sve_absgt16_notfound
    
    WORD $0xA4A0A808              // LD1H {Z8.H}, P2/Z, [R0]
    
    // Check (v > threshold) || (-threshold > v)
    WORD $0x24408111              // CMPGT P1.H, P0/Z, Z8.H, Z0.H
    WORD $0x25024821              // AND P1.B, P2/Z, P1.B, P2.B  (mask with active lanes)
    WORD $0x24488033              // CMPGT P3.H, P0/Z, Z1.H, Z8.H
    WORD $0x25024863              // AND P3.B, P2/Z, P3.B, P2.B  (mask with active lanes)
    WORD $0x25834021              // ORR P1.B, P0/Z, P1.B, P3.B
    WORD $0x2550C020              // PTEST P0, P1.B
    BNE sve_absgt16_found
    
    WORD $0x0470E3E5              // INCH X5
    LSL $1, R3, R6
    ADD R6, R0, R0
    B sve_absgt16_tail_loop

sve_absgt16_found:
    MOVD $1, R0
    MOVB R0, ret+32(FP)
    RET
    
sve_absgt16_notfound:
    MOVD $0, R0
    MOVB R0, ret+32(FP)
    RET
