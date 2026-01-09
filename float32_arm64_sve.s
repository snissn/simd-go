//go:build arm64

#include "textflag.h"

// ╔══════════════════════════════════════════════════════════════════════════════╗
// ║                          SVE Float32 SIMD Operations                         ║
// ║                                                                              ║
// ║  SVE (Scalable Vector Extension) uses variable-width vectors                 ║
// ║  Vector length is discovered at runtime via CNTW instruction                 ║
// ║                                                                              ║
// ║  Common SVE vector lengths for float32:                                      ║
// ║  ┌─────────────────────────────────────────────────────────────────────┐     ║
// ║  │  128-bit:  4 x float32   (Apple M1/M2, AWS Graviton2)               │     ║
// ║  │  256-bit:  8 x float32   (AWS Graviton3, Fujitsu A64FX)             │     ║
// ║  │  512-bit: 16 x float32   (Fujitsu A64FX, future chips)              │     ║
// ║  └─────────────────────────────────────────────────────────────────────┘     ║
// ╚══════════════════════════════════════════════════════════════════════════════╝

// ┌──────────────────────────────────────────────────────────────────────────────┐
// │ func sumFloat32SVE(vals []float32) float32                                    │
// │                                                                              │
// │ Strategy: 8 vector accumulators, SVE handles variable vector lengths         │
// └──────────────────────────────────────────────────────────────────────────────┘
TEXT ·sumFloat32SVE(SB), NOSPLIT, $0-28
    MOVD vals_base+0(FP), R0      // R0 = pointer to vals
    MOVD vals_len+8(FP), R1       // R1 = len(vals)
    
    // PTRUE P0.S - Enable all 32-bit lanes
    WORD $0x2598E3E0              // PTRUE P0.S
    
    // Zero 8 accumulator vectors
    WORD $0x25F8C000              // DUP Z0.S, #0
    WORD $0x25F8C001              // DUP Z1.S, #0
    WORD $0x25F8C002              // DUP Z2.S, #0
    WORD $0x25F8C003              // DUP Z3.S, #0
    WORD $0x25F8C004              // DUP Z4.S, #0
    WORD $0x25F8C005              // DUP Z5.S, #0
    WORD $0x25F8C006              // DUP Z6.S, #0
    WORD $0x25F8C007              // DUP Z7.S, #0
    
    // CNTW - Count number of 32-bit elements per vector
    WORD $0x04A0E3E2              // CNTW X2
    
    LSL $3, R2, R3                // R3 = R2 * 8 (process 8 vectors per iteration)
    
    CMP R3, R1
    BLT sve_f32sum_tail

sve_f32sum_loop8:
    // Load 8 vectors using LD1W (32-bit elements)
    WORD $0xA540A008              // LD1W {Z8.S}, P0/Z, [R0, #0, MUL VL]
    WORD $0xA541A009              // LD1W {Z9.S}, P0/Z, [R0, #1, MUL VL]
    WORD $0xA542A00A              // LD1W {Z10.S}, P0/Z, [R0, #2, MUL VL]
    WORD $0xA543A00B              // LD1W {Z11.S}, P0/Z, [R0, #3, MUL VL]
    WORD $0xA544A00C              // LD1W {Z12.S}, P0/Z, [R0, #4, MUL VL]
    WORD $0xA545A00D              // LD1W {Z13.S}, P0/Z, [R0, #5, MUL VL]
    WORD $0xA546A00E              // LD1W {Z14.S}, P0/Z, [R0, #6, MUL VL]
    WORD $0xA547A00F              // LD1W {Z15.S}, P0/Z, [R0, #7, MUL VL]
    
    // FADD Z.S, P0/M, Z.S, Z.S
    WORD $0x65808100              // FADD Z0.S, P0/M, Z0.S, Z8.S
    WORD $0x65808121              // FADD Z1.S, P0/M, Z1.S, Z9.S
    WORD $0x65808142              // FADD Z2.S, P0/M, Z2.S, Z10.S
    WORD $0x65808163              // FADD Z3.S, P0/M, Z3.S, Z11.S
    WORD $0x65808184              // FADD Z4.S, P0/M, Z4.S, Z12.S
    WORD $0x658081A5              // FADD Z5.S, P0/M, Z5.S, Z13.S
    WORD $0x658081C6              // FADD Z6.S, P0/M, Z6.S, Z14.S
    WORD $0x658081E7              // FADD Z7.S, P0/M, Z7.S, Z15.S
    
    // Advance pointer by 8 * VL bytes (VL in bytes = R2 * 4)
    LSL $2, R3, R5                // R5 = R3 * 4 (bytes per iteration)
    ADD R5, R0, R0
    SUBS R3, R1, R1
    CMP R3, R1
    BGE sve_f32sum_loop8

sve_f32sum_tail:
    MOVD ZR, R4                   // R4 = 0 (loop index)
    CBZ R1, sve_f32sum_reduce
    
sve_f32sum_tail_loop:
    // WHILELO P1.S, X4, X1
    WORD $0x25A11C81              // WHILELO P1.S, X4, X1
    BEQ sve_f32sum_reduce
    
    WORD $0xA540A408              // LD1W {Z8.S}, P1/Z, [R0, #0, MUL VL]
    WORD $0x65808500              // FADD Z0.S, P1/M, Z0.S, Z8.S
    
    // INCW X4
    WORD $0x04B0E3E4              // INCW X4
    LSL $2, R2, R5                // bytes per vector = R2 * 4
    ADD R5, R0, R0
    B sve_f32sum_tail_loop

sve_f32sum_reduce:
    // Tree reduction: 8 → 4 → 2 → 1
    WORD $0x65808080              // FADD Z0.S, P0/M, Z0.S, Z4.S
    WORD $0x658080A1              // FADD Z1.S, P0/M, Z1.S, Z5.S
    WORD $0x658080C2              // FADD Z2.S, P0/M, Z2.S, Z6.S
    WORD $0x658080E3              // FADD Z3.S, P0/M, Z3.S, Z7.S
    WORD $0x65808040              // FADD Z0.S, P0/M, Z0.S, Z2.S
    WORD $0x65808061              // FADD Z1.S, P0/M, Z1.S, Z3.S
    WORD $0x65808020              // FADD Z0.S, P0/M, Z0.S, Z1.S
    
    // FADDV S16, P0, Z0.S - horizontal sum
    FMOVS $0.0, F16
    WORD $0x65982010              // FADDV S16, P0, Z0.S
    
    FMOVS F16, ret+24(FP)
    RET

// ┌──────────────────────────────────────────────────────────────────────────────┐
// │ func minFloat32SVE(vals []float32) float32                                    │
// └──────────────────────────────────────────────────────────────────────────────┘
TEXT ·minFloat32SVE(SB), NOSPLIT, $0-28
    MOVD vals_base+0(FP), R0
    MOVD vals_len+8(FP), R1
    
    CBZ R1, sve_f32min_empty
    
    WORD $0x2598E3E0              // PTRUE P0.S
    WORD $0x04A0E3E2              // CNTW X2
    
    // Initialize with first element
    FMOVS (R0), F3
    WORD $0x05242060              // DUP Z0.S, Z3.S[0]
    WORD $0x05242061              // DUP Z1.S, Z3.S[0]
    WORD $0x05242062              // DUP Z2.S, Z3.S[0]
    WORD $0x05242063              // DUP Z3.S, Z3.S[0]
    WORD $0x05242064              // DUP Z4.S, Z3.S[0]
    WORD $0x05242065              // DUP Z5.S, Z3.S[0]
    WORD $0x05242066              // DUP Z6.S, Z3.S[0]
    WORD $0x05242067              // DUP Z7.S, Z3.S[0]
    
    ADD $4, R0
    SUB $1, R1
    
    LSL $3, R2, R3
    CMP R3, R1
    BLT sve_f32min_tail

sve_f32min_loop8:
    WORD $0xA540A008              // LD1W Z8
    WORD $0xA541A009              // LD1W Z9
    WORD $0xA542A00A              // LD1W Z10
    WORD $0xA543A00B              // LD1W Z11
    WORD $0xA544A00C              // LD1W Z12
    WORD $0xA545A00D              // LD1W Z13
    WORD $0xA546A00E              // LD1W Z14
    WORD $0xA547A00F              // LD1W Z15
    
    // FMIN Z.S, P0/M, Z.S, Z.S
    WORD $0x65878100              // FMIN Z0.S, P0/M, Z0.S, Z8.S
    WORD $0x65878121              // FMIN Z1.S, P0/M, Z1.S, Z9.S
    WORD $0x65878142              // FMIN Z2.S, P0/M, Z2.S, Z10.S
    WORD $0x65878163              // FMIN Z3.S, P0/M, Z3.S, Z11.S
    WORD $0x65878184              // FMIN Z4.S, P0/M, Z4.S, Z12.S
    WORD $0x658781A5              // FMIN Z5.S, P0/M, Z5.S, Z13.S
    WORD $0x658781C6              // FMIN Z6.S, P0/M, Z6.S, Z14.S
    WORD $0x658781E7              // FMIN Z7.S, P0/M, Z7.S, Z15.S
    
    LSL $2, R3, R5
    ADD R5, R0, R0
    SUBS R3, R1, R1
    CMP R3, R1
    BGE sve_f32min_loop8

sve_f32min_tail:
    MOVD ZR, R4
    CBZ R1, sve_f32min_reduce
    
sve_f32min_tail_loop:
    WORD $0x25A11C81              // WHILELO P1.S, X4, X1
    BEQ sve_f32min_reduce
    
    WORD $0xA540A408              // LD1W {Z8.S}, P1/Z, [R0, #0, MUL VL]
    WORD $0x65878500              // FMIN Z0.S, P1/M, Z0.S, Z8.S
    
    WORD $0x04B0E3E4              // INCW X4
    LSL $2, R2, R5
    ADD R5, R0, R0
    B sve_f32min_tail_loop

sve_f32min_reduce:
    WORD $0x65878080              // FMIN Z0.S, P0/M, Z0.S, Z4.S
    WORD $0x658780A1              // FMIN Z1.S, P0/M, Z1.S, Z5.S
    WORD $0x658780C2              // FMIN Z2.S, P0/M, Z2.S, Z6.S
    WORD $0x658780E3              // FMIN Z3.S, P0/M, Z3.S, Z7.S
    WORD $0x65878040              // FMIN Z0.S, P0/M, Z0.S, Z2.S
    WORD $0x65878061              // FMIN Z1.S, P0/M, Z1.S, Z3.S
    WORD $0x65878020              // FMIN Z0.S, P0/M, Z0.S, Z1.S
    
    // FMINV S0, P0, Z0.S
    WORD $0x65872000              // FMINV S0, P0, Z0.S
    
    FMOVS F0, ret+24(FP)
    RET

sve_f32min_empty:
    FMOVS $0.0, F0
    FMOVS F0, ret+24(FP)
    RET

// ┌──────────────────────────────────────────────────────────────────────────────┐
// │ func maxFloat32SVE(vals []float32) float32                                    │
// └──────────────────────────────────────────────────────────────────────────────┘
TEXT ·maxFloat32SVE(SB), NOSPLIT, $0-28
    MOVD vals_base+0(FP), R0
    MOVD vals_len+8(FP), R1
    
    CBZ R1, sve_f32max_empty
    
    WORD $0x2598E3E0              // PTRUE P0.S
    WORD $0x04A0E3E2              // CNTW X2
    
    // Initialize with first element
    FMOVS (R0), F3
    WORD $0x05242060              // DUP Z0.S, Z3.S[0]
    WORD $0x05242061              // DUP Z1.S, Z3.S[0]
    WORD $0x05242062              // DUP Z2.S, Z3.S[0]
    WORD $0x05242063              // DUP Z3.S, Z3.S[0]
    WORD $0x05242064              // DUP Z4.S, Z3.S[0]
    WORD $0x05242065              // DUP Z5.S, Z3.S[0]
    WORD $0x05242066              // DUP Z6.S, Z3.S[0]
    WORD $0x05242067              // DUP Z7.S, Z3.S[0]
    
    ADD $4, R0
    SUB $1, R1
    
    LSL $3, R2, R3
    CMP R3, R1
    BLT sve_f32max_tail

sve_f32max_loop8:
    WORD $0xA540A008              // LD1W Z8
    WORD $0xA541A009              // LD1W Z9
    WORD $0xA542A00A              // LD1W Z10
    WORD $0xA543A00B              // LD1W Z11
    WORD $0xA544A00C              // LD1W Z12
    WORD $0xA545A00D              // LD1W Z13
    WORD $0xA546A00E              // LD1W Z14
    WORD $0xA547A00F              // LD1W Z15
    
    // FMAX Z.S, P0/M, Z.S, Z.S
    WORD $0x65868100              // FMAX Z0.S, P0/M, Z0.S, Z8.S
    WORD $0x65868121              // FMAX Z1.S, P0/M, Z1.S, Z9.S
    WORD $0x65868142              // FMAX Z2.S, P0/M, Z2.S, Z10.S
    WORD $0x65868163              // FMAX Z3.S, P0/M, Z3.S, Z11.S
    WORD $0x65868184              // FMAX Z4.S, P0/M, Z4.S, Z12.S
    WORD $0x658681A5              // FMAX Z5.S, P0/M, Z5.S, Z13.S
    WORD $0x658681C6              // FMAX Z6.S, P0/M, Z6.S, Z14.S
    WORD $0x658681E7              // FMAX Z7.S, P0/M, Z7.S, Z15.S
    
    LSL $2, R3, R5
    ADD R5, R0, R0
    SUBS R3, R1, R1
    CMP R3, R1
    BGE sve_f32max_loop8

sve_f32max_tail:
    MOVD ZR, R4
    CBZ R1, sve_f32max_reduce
    
sve_f32max_tail_loop:
    WORD $0x25A11C81              // WHILELO P1.S, X4, X1
    BEQ sve_f32max_reduce
    
    WORD $0xA540A408              // LD1W {Z8.S}, P1/Z, [R0, #0, MUL VL]
    WORD $0x65868500              // FMAX Z0.S, P1/M, Z0.S, Z8.S
    
    WORD $0x04B0E3E4              // INCW X4
    LSL $2, R2, R5
    ADD R5, R0, R0
    B sve_f32max_tail_loop

sve_f32max_reduce:
    WORD $0x65868080              // FMAX Z0.S, P0/M, Z0.S, Z4.S
    WORD $0x658680A1              // FMAX Z1.S, P0/M, Z1.S, Z5.S
    WORD $0x658680C2              // FMAX Z2.S, P0/M, Z2.S, Z6.S
    WORD $0x658680E3              // FMAX Z3.S, P0/M, Z3.S, Z7.S
    WORD $0x65868040              // FMAX Z0.S, P0/M, Z0.S, Z2.S
    WORD $0x65868061              // FMAX Z1.S, P0/M, Z1.S, Z3.S
    WORD $0x65868020              // FMAX Z0.S, P0/M, Z0.S, Z1.S
    
    // FMAXV S0, P0, Z0.S
    WORD $0x65862000              // FMAXV S0, P0, Z0.S
    
    FMOVS F0, ret+24(FP)
    RET

sve_f32max_empty:
    FMOVS $0.0, F0
    FMOVS F0, ret+24(FP)
    RET

// ┌──────────────────────────────────────────────────────────────────────────────┐
// │ func dotProductFloat32SVE(a, b []float32) float32                             │
// │                                                                              │
// │ Strategy: Use FMLA (fused multiply-add) for accumulation                     │
// └──────────────────────────────────────────────────────────────────────────────┘
TEXT ·dotProductFloat32SVE(SB), NOSPLIT, $0-52
    MOVD a_base+0(FP), R0         // R0 = pointer to a
    MOVD a_len+8(FP), R1          // R1 = len(a)
    MOVD b_base+24(FP), R2        // R2 = pointer to b
    
    WORD $0x2598E3E0              // PTRUE P0.S
    
    // Zero 8 accumulators
    WORD $0x25F8C000              // DUP Z0.S, #0
    WORD $0x25F8C001              // DUP Z1.S, #0
    WORD $0x25F8C002              // DUP Z2.S, #0
    WORD $0x25F8C003              // DUP Z3.S, #0
    WORD $0x25F8C004              // DUP Z4.S, #0
    WORD $0x25F8C005              // DUP Z5.S, #0
    WORD $0x25F8C006              // DUP Z6.S, #0
    WORD $0x25F8C007              // DUP Z7.S, #0
    
    WORD $0x04A0E3E3              // CNTW X3
    
    LSL $3, R3, R4                // R4 = elements per iteration = R3 * 8
    
    CMP R4, R1
    BLT sve_f32dot_tail

sve_f32dot_loop8:
    // Load 8 vectors from a into Z8-Z15
    WORD $0xA540A008              // LD1W {Z8.S}, P0/Z, [R0, #0, MUL VL]
    WORD $0xA541A009              // LD1W {Z9.S}, P0/Z, [R0, #1, MUL VL]
    WORD $0xA542A00A              // LD1W {Z10.S}, P0/Z, [R0, #2, MUL VL]
    WORD $0xA543A00B              // LD1W {Z11.S}, P0/Z, [R0, #3, MUL VL]
    WORD $0xA544A00C              // LD1W {Z12.S}, P0/Z, [R0, #4, MUL VL]
    WORD $0xA545A00D              // LD1W {Z13.S}, P0/Z, [R0, #5, MUL VL]
    WORD $0xA546A00E              // LD1W {Z14.S}, P0/Z, [R0, #6, MUL VL]
    WORD $0xA547A00F              // LD1W {Z15.S}, P0/Z, [R0, #7, MUL VL]
    
    // Load 8 vectors from b into Z16-Z23
    WORD $0xA540A050              // LD1W {Z16.S}, P0/Z, [R2, #0, MUL VL]
    WORD $0xA541A051              // LD1W {Z17.S}, P0/Z, [R2, #1, MUL VL]
    WORD $0xA542A052              // LD1W {Z18.S}, P0/Z, [R2, #2, MUL VL]
    WORD $0xA543A053              // LD1W {Z19.S}, P0/Z, [R2, #3, MUL VL]
    WORD $0xA544A054              // LD1W {Z20.S}, P0/Z, [R2, #4, MUL VL]
    WORD $0xA545A055              // LD1W {Z21.S}, P0/Z, [R2, #5, MUL VL]
    WORD $0xA546A056              // LD1W {Z22.S}, P0/Z, [R2, #6, MUL VL]
    WORD $0xA547A057              // LD1W {Z23.S}, P0/Z, [R2, #7, MUL VL]
    
    // FMLA Z.S, P0/M, Z.S, Z.S (Zda = Zda + Zn * Zm)
    WORD $0x65B00100              // FMLA Z0.S, P0/M, Z8.S, Z16.S
    WORD $0x65B10121              // FMLA Z1.S, P0/M, Z9.S, Z17.S
    WORD $0x65B20142              // FMLA Z2.S, P0/M, Z10.S, Z18.S
    WORD $0x65B30163              // FMLA Z3.S, P0/M, Z11.S, Z19.S
    WORD $0x65B40184              // FMLA Z4.S, P0/M, Z12.S, Z20.S
    WORD $0x65B501A5              // FMLA Z5.S, P0/M, Z13.S, Z21.S
    WORD $0x65B601C6              // FMLA Z6.S, P0/M, Z14.S, Z22.S
    WORD $0x65B701E7              // FMLA Z7.S, P0/M, Z15.S, Z23.S
    
    // Advance both pointers
    LSL $2, R4, R5                // bytes = R4 * 4
    ADD R5, R0, R0
    ADD R5, R2, R2
    SUBS R4, R1, R1
    CMP R4, R1
    BGE sve_f32dot_loop8

sve_f32dot_tail:
    MOVD ZR, R6                   // Index for WHILELO
    CBZ R1, sve_f32dot_reduce
    
sve_f32dot_tail_loop:
    WORD $0x25A11CC1              // WHILELO P1.S, X6, X1
    BEQ sve_f32dot_reduce
    
    WORD $0xA540A408              // LD1W {Z8.S}, P1/Z, [R0, #0, MUL VL]
    WORD $0xA540A450              // LD1W {Z16.S}, P1/Z, [R2, #0, MUL VL]
    
    WORD $0x65B00500              // FMLA Z0.S, P1/M, Z8.S, Z16.S
    
    WORD $0x04B0E3E6              // INCW X6
    LSL $2, R3, R5
    ADD R5, R0, R0
    ADD R5, R2, R2
    B sve_f32dot_tail_loop

sve_f32dot_reduce:
    // Combine accumulators: 8 → 4 → 2 → 1
    WORD $0x65808080              // FADD Z0.S, P0/M, Z0.S, Z4.S
    WORD $0x658080A1              // FADD Z1.S, P0/M, Z1.S, Z5.S
    WORD $0x658080C2              // FADD Z2.S, P0/M, Z2.S, Z6.S
    WORD $0x658080E3              // FADD Z3.S, P0/M, Z3.S, Z7.S
    WORD $0x65808040              // FADD Z0.S, P0/M, Z0.S, Z2.S
    WORD $0x65808061              // FADD Z1.S, P0/M, Z1.S, Z3.S
    WORD $0x65808020              // FADD Z0.S, P0/M, Z0.S, Z1.S
    
    // FADDV S16, P0, Z0.S
    FMOVS $0.0, F16
    WORD $0x65982010              // FADDV S16, P0, Z0.S
    
    FMOVS F16, ret+48(FP)
    RET
