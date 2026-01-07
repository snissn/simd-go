//go:build arm64

#include "textflag.h"

// ╔══════════════════════════════════════════════════════════════════════════════╗
// ║                          SVE Float64 SIMD Operations                         ║
// ║                                                                              ║
// ║  SVE (Scalable Vector Extension) uses variable-width vectors                 ║
// ║  Vector length is discovered at runtime via CNTD instruction                 ║
// ║                                                                              ║
// ║  Key advantages over NEON:                                                   ║
// ║  • Vector length agnostic (VLA) programming model                            ║
// ║  • Predicated operations (masked processing)                                 ║
// ║  • WHILELO for automatic tail handling                                       ║
// ║  • Native horizontal reductions (FADDA, FMINV, FMAXV)                        ║
// ║                                                                              ║
// ║  Common SVE vector lengths:                                                  ║
// ║  ┌─────────────────────────────────────────────────────────────────────┐     ║
// ║  │  128-bit:  2 x float64   (Apple M1/M2, AWS Graviton2)               │     ║
// ║  │  256-bit:  4 x float64   (AWS Graviton3, Fujitsu A64FX)             │     ║
// ║  │  512-bit:  8 x float64   (Fujitsu A64FX, future chips)              │     ║
// ║  └─────────────────────────────────────────────────────────────────────┘     ║
// ╚══════════════════════════════════════════════════════════════════════════════╝

// SVE Instruction Reference:
// PTRUE P0.D             : 0x2518E3E0 | Pd        - Set all predicate lanes true
// DUP Zd.D, #0           : 0x25F8C000 | Zd        - Broadcast immediate to vector
// LD1D {Zt.D}, Pg/Z, [Xn, #imm, MUL VL] : 0xA5E0A000  - Predicated load
// FADD Zdn.D, Pg/M, Zdn.D, Zm.D : 0x65C08000      - Predicated float add
// FMLA Zda.D, Pg/M, Zn.D, Zm.D : 0x65E00000       - Fused multiply-add
// FADDA Dd, Pg, Dn, Zm.D : 0x65D82000             - Horizontal add (strictly ordered)
// FMINV Dd, Pg, Zn.D     : 0x65C72000             - Horizontal min
// FMAXV Dd, Pg, Zn.D     : 0x65C62000             - Horizontal max
// WHILELO Pd.D, Xn, Xm   : Loop predicate generation

// ┌──────────────────────────────────────────────────────────────────────────────┐
// │ func sumFloat64SVE(vals []float64) float64                                   │
// │                                                                              │
// │ Strategy: 8 vector accumulators, SVE handles variable vector lengths         │
// │           Uses FADDA for final horizontal reduction (strictly ordered)       │
// │                                                                              │
// │ SVE vector layout (example: 256-bit = 4 x float64):                          │
// │ ┌─────────────────────────────────────────────────────────────────────┐      │
// │ │  Z0.D = [ lane0 | lane1 | lane2 | lane3 ]  (256 bits)               │      │
// │ └─────────────────────────────────────────────────────────────────────┘      │
// │                                                                              │
// │ Predicate register (controls which lanes are active):                        │
// │ ┌─────────────────────────────────────────────────────────────────────┐      │
// │ │  P0.D = [ 1 | 1 | 1 | 1 ]  (all lanes active for PTRUE)             │      │
// │ └─────────────────────────────────────────────────────────────────────┘      │
// └──────────────────────────────────────────────────────────────────────────────┘
TEXT ·sumFloat64SVE(SB), NOSPLIT, $0-32
    MOVD vals_base+0(FP), R0      // R0 = pointer to vals
    MOVD vals_len+8(FP), R1       // R1 = len(vals)
    
    // ╔════════════════════════════════════════════════════════════════════════╗
    // ║ PTRUE P0.D - Set all predicate lanes to true (for 64-bit elements)    ║
    // ║                                                                        ║
    // ║   P0 = [ 1 | 1 | 1 | 1 | ... ]  (number depends on vector length)     ║
    // ╚════════════════════════════════════════════════════════════════════════╝
    WORD $0x2518E3E0              // PTRUE P0.D
    
    // ╔════════════════════════════════════════════════════════════════════════╗
    // ║ Zero 8 accumulator vectors                                             ║
    // ║                                                                        ║
    // ║   Z0 = [0|0|0|0...]    Z1 = [0|0|0|0...]    ... Z7 = [0|0|0|0...]      ║
    // ╚════════════════════════════════════════════════════════════════════════╝
    WORD $0x25F8C000              // DUP Z0.D, #0
    WORD $0x25F8C001              // DUP Z1.D, #0
    WORD $0x25F8C002              // DUP Z2.D, #0
    WORD $0x25F8C003              // DUP Z3.D, #0
    WORD $0x25F8C004              // DUP Z4.D, #0
    WORD $0x25F8C005              // DUP Z5.D, #0
    WORD $0x25F8C006              // DUP Z6.D, #0
    WORD $0x25F8C007              // DUP Z7.D, #0
    
    // ┌────────────────────────────────────────────────────────────────────┐
    // │ CNTD - Count number of 64-bit elements per vector                 │
    // │                                                                    │
    // │   Example results:                                                 │
    // │   • 128-bit SVE: CNTD returns 2                                    │
    // │   • 256-bit SVE: CNTD returns 4                                    │
    // │   • 512-bit SVE: CNTD returns 8                                    │
    // │                                                                    │
    // │   R2 = elements_per_vector                                         │
    // │   R3 = R2 * 8 = elements per iteration (8 vectors)                 │
    // └────────────────────────────────────────────────────────────────────┘
    WORD $0x04E0E3E2              // CNTD X2 (count doublewords per vector)
    
    LSL $3, R2, R3                // R3 = R2 * 8 (process 8 vectors per iteration)
    
    CMP R3, R1
    BLT sve_fsum_tail

// ┌──────────────────────────────────────────────────────────────────────────────┐
// │ Main loop: Process 8 vectors per iteration                                   │
// │                                                                              │
// │ SVE load with offset (MUL VL = multiply by vector length):                   │
// │ ┌─────────────────────────────────────────────────────────────────────┐      │
// │ │  LD1D {Z8.D}, P0/Z, [R0, #0, MUL VL]  → Load at R0 + 0*VL           │      │
// │ │  LD1D {Z9.D}, P0/Z, [R0, #1, MUL VL]  → Load at R0 + 1*VL           │      │
// │ │  ...                                                                 │      │
// │ │  LD1D {Z15.D}, P0/Z, [R0, #7, MUL VL] → Load at R0 + 7*VL           │      │
// │ └─────────────────────────────────────────────────────────────────────┘      │
// │                                                                              │
// │ Memory layout (for 256-bit SVE, 4 elements per vector):                      │
// │                                                                              │
// │   R0 ──►┌────┬────┬────┬────┬────┬────┬────┬────┬─...─┬────┬────┬────┬────┐  │
// │         │ e0 │ e1 │ e2 │ e3 │ e4 │ e5 │ e6 │ e7 │     │e28 │e29 │e30 │e31 │  │
// │         └────┴────┴────┴────┴────┴────┴────┴────┴─...─┴────┴────┴────┴────┘  │
// │         ╰───── Z8 ─────╯╰───── Z9 ─────╯         ╰──── Z15 ────╯             │
// │              #0             #1                        #7                     │
// └──────────────────────────────────────────────────────────────────────────────┘
sve_fsum_loop8:
    // Load 8 vectors (contiguous, using VL offsets)
    WORD $0xA5E0A008              // LD1D {Z8.D}, P0/Z, [R0, #0, MUL VL]
    WORD $0xA5E1A009              // LD1D {Z9.D}, P0/Z, [R0, #1, MUL VL]
    WORD $0xA5E2A00A              // LD1D {Z10.D}, P0/Z, [R0, #2, MUL VL]
    WORD $0xA5E3A00B              // LD1D {Z11.D}, P0/Z, [R0, #3, MUL VL]
    WORD $0xA5E4A00C              // LD1D {Z12.D}, P0/Z, [R0, #4, MUL VL]
    WORD $0xA5E5A00D              // LD1D {Z13.D}, P0/Z, [R0, #5, MUL VL]
    WORD $0xA5E6A00E              // LD1D {Z14.D}, P0/Z, [R0, #6, MUL VL]
    WORD $0xA5E7A00F              // LD1D {Z15.D}, P0/Z, [R0, #7, MUL VL]
    
    // ┌────────────────────────────────────────────────────────────────────┐
    // │ FADD with predicate: Only active lanes are updated                │
    // │                                                                    │
    // │   Before:  Z0 = [ s0 | s1 | s2 | s3 ]   Z8 = [ a | b | c | d ]     │
    // │   P0/M:         [  1 |  1 |  1 |  1 ]                              │
    // │   After:   Z0 = [ s0+a | s1+b | s2+c | s3+d ]                      │
    // └────────────────────────────────────────────────────────────────────┘
    WORD $0x65C08100              // FADD Z0.D, P0/M, Z0.D, Z8.D
    WORD $0x65C08121              // FADD Z1.D, P0/M, Z1.D, Z9.D
    WORD $0x65C08142              // FADD Z2.D, P0/M, Z2.D, Z10.D
    WORD $0x65C08163              // FADD Z3.D, P0/M, Z3.D, Z11.D
    WORD $0x65C08184              // FADD Z4.D, P0/M, Z4.D, Z12.D
    WORD $0x65C081A5              // FADD Z5.D, P0/M, Z5.D, Z13.D
    WORD $0x65C081C6              // FADD Z6.D, P0/M, Z6.D, Z14.D
    WORD $0x65C081E7              // FADD Z7.D, P0/M, Z7.D, Z15.D
    
    // Advance pointer by 8 * VL bytes
    LSL $3, R3, R5                // R5 = R3 * 8 (bytes per iteration)
    ADD R5, R0, R0
    SUBS R3, R1, R1
    CMP R3, R1
    BGE sve_fsum_loop8

// ┌──────────────────────────────────────────────────────────────────────────────┐
// │ Tail handling with WHILELO                                                   │
// │                                                                              │
// │ WHILELO generates a predicate based on loop index vs limit:                  │
// │                                                                              │
// │   WHILELO P1.D, X4, X1   (X4 = current index, X1 = total remaining)          │
// │                                                                              │
// │   Example: X4=0, X1=3, VL=4                                                  │
// │   ┌─────────────────────────────────────────────────────────────────────┐    │
// │   │  Index:    [ 0 | 1 | 2 | 3 ]                                        │    │
// │   │  Compare:  [ 0<3? | 1<3? | 2<3? | 3<3? ]                            │    │
// │   │  P1:       [  1   |  1   |  1   |  0   ]   ← Only first 3 active   │    │
// │   └─────────────────────────────────────────────────────────────────────┘    │
// │                                                                              │
// │ This elegantly handles partial vectors without explicit masking!             │
// └──────────────────────────────────────────────────────────────────────────────┘
sve_fsum_tail:
    MOVD ZR, R4                   // R4 = 0 (loop index)
    CBZ R1, sve_fsum_reduce
    
sve_fsum_tail_loop:
    WORD $0x25E11C81              // WHILELO P1.D, X4, X1 (compare index < remaining)
    BEQ sve_fsum_reduce           // Z flag set if no active lanes
    
    WORD $0xA5E0A408              // LD1D {Z8.D}, P1/Z, [R0, #0, MUL VL]
    WORD $0x65C08500              // FADD Z0.D, P1/M, Z0.D, Z8.D
    
    WORD $0x04F0E3E4              // INCD X4 (increment by vector length)
    LSL $3, R2, R5
    ADD R5, R0, R0
    B sve_fsum_tail_loop

sve_fsum_reduce:
    // ╔════════════════════════════════════════════════════════════════════════╗
    // ║ Tree reduction: 8 → 4 → 2 → 1 vectors                                  ║
    // ║                                                                        ║
    // ║   Z0 ═╦═ Z4       Z1 ═╦═ Z5       Z2 ═╦═ Z6       Z3 ═╦═ Z7            ║
    // ║      ╚═► Z0          ╚═► Z1          ╚═► Z2          ╚═► Z3            ║
    // ║                                                                        ║
    // ║   Z0 ═══════╦═══════ Z2       Z1 ═══════╦═══════ Z3                    ║
    // ║            ╚════════► Z0               ╚════════► Z1                   ║
    // ║                                                                        ║
    // ║   Z0 ═══════════════════╦═══════════════════ Z1                        ║
    // ║                        ╚════════════════════► Z0                       ║
    // ╚════════════════════════════════════════════════════════════════════════╝
    WORD $0x65C08080              // FADD Z0.D, P0/M, Z0.D, Z4.D
    WORD $0x65C080A1              // FADD Z1.D, P0/M, Z1.D, Z5.D
    WORD $0x65C080C2              // FADD Z2.D, P0/M, Z2.D, Z6.D
    WORD $0x65C080E3              // FADD Z3.D, P0/M, Z3.D, Z7.D
    WORD $0x65C08040              // FADD Z0.D, P0/M, Z0.D, Z2.D
    WORD $0x65C08061              // FADD Z1.D, P0/M, Z1.D, Z3.D
    WORD $0x65C08020              // FADD Z0.D, P0/M, Z0.D, Z1.D
    
    // ┌────────────────────────────────────────────────────────────────────┐
    // │ FADDA - Horizontal add with strict ordering                       │
    // │                                                                    │
    // │ FADDA Dd, Pg, Dn, Zm.D                                             │
    // │   D16 = D16 + Z0[0] + Z0[1] + Z0[2] + ... (left to right)          │
    // │                                                                    │
    // │   Z0 = [ a | b | c | d ]                                           │
    // │          │   │   │   │                                             │
    // │          └───┴───┴───┴──────────────►  D16 = a + b + c + d         │
    // │                                                                    │
    // │ Note: FADDA maintains strict ordering for reproducible results    │
    // └────────────────────────────────────────────────────────────────────┘
    FMOVD $0.0, F16
    WORD $0x65D82010              // FADDA D16, P0, D16, Z0.D
    
    FMOVD F16, ret+24(FP)
    RET

// ┌──────────────────────────────────────────────────────────────────────────────┐
// │ func minFloat64SVE(vals []float64) float64                                   │
// │                                                                              │
// │ Strategy: 8 accumulators with FMIN, then FMINV for horizontal reduction      │
// │           FMINV = single instruction to find min across entire vector        │
// └──────────────────────────────────────────────────────────────────────────────┘
TEXT ·minFloat64SVE(SB), NOSPLIT, $0-32
    MOVD vals_base+0(FP), R0
    MOVD vals_len+8(FP), R1
    
    CBZ R1, sve_fmin_empty
    
    WORD $0x2518E3E0              // PTRUE P0.D
    WORD $0x04E0E3E2              // CNTD X2
    
    // ┌────────────────────────────────────────────────────────────────────┐
    // │ Initialize 8 accumulators with first element (broadcast)          │
    // │                                                                    │
    // │   DUP Zd.D, Zn.D[0] broadcasts lane 0 of Zn to all lanes of Zd    │
    // │                                                                    │
    // │   vals[0] ─────────────────────────────────────────────┐           │
    // │            ╔═══════╦═══════╦═══════╦═══════╦═══════╦═══╧═══╗       │
    // │            ║ Z0    ║ Z1    ║ Z2    ║ ...   ║ Z6    ║ Z7    ║       │
    // │            ║[v|v|v]║[v|v|v]║[v|v|v]║       ║[v|v|v]║[v|v|v]║       │
    // │            ╚═══════╩═══════╩═══════╩═══════╩═══════╩═══════╝       │
    // └────────────────────────────────────────────────────────────────────┘
    FMOVD (R0), F3                // Load first element into D3
    WORD $0x05282060              // DUP Z0.D, Z3.D[0]
    WORD $0x05282061              // DUP Z1.D, Z3.D[0]
    WORD $0x05282062              // DUP Z2.D, Z3.D[0]
    WORD $0x05282063              // DUP Z3.D, Z3.D[0]
    WORD $0x05282064              // DUP Z4.D, Z3.D[0]
    WORD $0x05282065              // DUP Z5.D, Z3.D[0]
    WORD $0x05282066              // DUP Z6.D, Z3.D[0]
    WORD $0x05282067              // DUP Z7.D, Z3.D[0]
    
    ADD $8, R0
    SUB $1, R1
    
    LSL $3, R2, R3
    CMP R3, R1
    BLT sve_fmin_tail

sve_fmin_loop8:
    WORD $0xA5E0A008              // LD1D {Z8.D}, P0/Z, [R0, #0, MUL VL]
    WORD $0xA5E1A009              // LD1D {Z9.D}, P0/Z, [R0, #1, MUL VL]
    WORD $0xA5E2A00A              // LD1D {Z10.D}, P0/Z, [R0, #2, MUL VL]
    WORD $0xA5E3A00B              // LD1D {Z11.D}, P0/Z, [R0, #3, MUL VL]
    WORD $0xA5E4A00C              // LD1D {Z12.D}, P0/Z, [R0, #4, MUL VL]
    WORD $0xA5E5A00D              // LD1D {Z13.D}, P0/Z, [R0, #5, MUL VL]
    WORD $0xA5E6A00E              // LD1D {Z14.D}, P0/Z, [R0, #6, MUL VL]
    WORD $0xA5E7A00F              // LD1D {Z15.D}, P0/Z, [R0, #7, MUL VL]
    
    // ┌────────────────────────────────────────────────────────────────────┐
    // │ FMIN: Element-wise minimum                                        │
    // │                                                                    │
    // │   Before: Z0 = [ 5 | 3 | 8 | 2 ]    Z8 = [ 2 | 7 | 1 | 9 ]         │
    // │   After:  Z0 = [ 2 | 3 | 1 | 2 ]    (min of each lane)            │
    // └────────────────────────────────────────────────────────────────────┘
    WORD $0x65C78100              // FMIN Z0.D, P0/M, Z0.D, Z8.D
    WORD $0x65C78121              // FMIN Z1.D, P0/M, Z1.D, Z9.D
    WORD $0x65C78142              // FMIN Z2.D, P0/M, Z2.D, Z10.D
    WORD $0x65C78163              // FMIN Z3.D, P0/M, Z3.D, Z11.D
    WORD $0x65C78184              // FMIN Z4.D, P0/M, Z4.D, Z12.D
    WORD $0x65C781A5              // FMIN Z5.D, P0/M, Z5.D, Z13.D
    WORD $0x65C781C6              // FMIN Z6.D, P0/M, Z6.D, Z14.D
    WORD $0x65C781E7              // FMIN Z7.D, P0/M, Z7.D, Z15.D
    
    LSL $3, R3, R5
    ADD R5, R0, R0
    SUBS R3, R1, R1
    CMP R3, R1
    BGE sve_fmin_loop8

sve_fmin_tail:
    MOVD ZR, R4
    CBZ R1, sve_fmin_reduce
    
sve_fmin_tail_loop:
    WORD $0x25E11C81              // WHILELO P1.D, X4, X1
    BEQ sve_fmin_reduce
    
    WORD $0xA5E0A408              // LD1D {Z8.D}, P1/Z, [R0, #0, MUL VL]
    WORD $0x65C78500              // FMIN Z0.D, P1/M, Z0.D, Z8.D
    
    WORD $0x04F0E3E4              // INCD X4
    LSL $3, R2, R5
    ADD R5, R0, R0
    B sve_fmin_tail_loop

sve_fmin_reduce:
    // Tree reduction: 8 → 4 → 2 → 1
    WORD $0x65C78080              // FMIN Z0.D, P0/M, Z0.D, Z4.D
    WORD $0x65C780A1              // FMIN Z1.D, P0/M, Z1.D, Z5.D
    WORD $0x65C780C2              // FMIN Z2.D, P0/M, Z2.D, Z6.D
    WORD $0x65C780E3              // FMIN Z3.D, P0/M, Z3.D, Z7.D
    WORD $0x65C78040              // FMIN Z0.D, P0/M, Z0.D, Z2.D
    WORD $0x65C78061              // FMIN Z1.D, P0/M, Z1.D, Z3.D
    WORD $0x65C78020              // FMIN Z0.D, P0/M, Z0.D, Z1.D
    
    // ┌────────────────────────────────────────────────────────────────────┐
    // │ FMINV - Horizontal minimum across all vector lanes                │
    // │                                                                    │
    // │   Z0 = [ 2.5 | 1.3 | 4.7 | 3.2 ]                                   │
    // │          │     │     │     │                                       │
    // │          └─────┴─────┴─────┴───────► D0 = min(2.5, 1.3, 4.7, 3.2)  │
    // │                                           = 1.3                    │
    // └────────────────────────────────────────────────────────────────────┘
    WORD $0x65C72000              // FMINV D0, P0, Z0.D
    
    FMOVD F0, ret+24(FP)
    RET

sve_fmin_empty:
    FMOVD $0.0, F0
    FMOVD F0, ret+24(FP)
    RET

// ┌──────────────────────────────────────────────────────────────────────────────┐
// │ func maxFloat64SVE(vals []float64) float64                                   │
// │                                                                              │
// │ Strategy: Identical to min but uses FMAX and FMAXV                           │
// └──────────────────────────────────────────────────────────────────────────────┘
TEXT ·maxFloat64SVE(SB), NOSPLIT, $0-32
    MOVD vals_base+0(FP), R0
    MOVD vals_len+8(FP), R1
    
    CBZ R1, sve_fmax_empty
    
    WORD $0x2518E3E0              // PTRUE P0.D
    WORD $0x04E0E3E2              // CNTD X2
    
    FMOVD (R0), F3
    WORD $0x05282060              // DUP Z0.D, Z3.D[0]
    WORD $0x05282061              // DUP Z1.D, Z3.D[0]
    WORD $0x05282062              // DUP Z2.D, Z3.D[0]
    WORD $0x05282063              // DUP Z3.D, Z3.D[0]
    WORD $0x05282064              // DUP Z4.D, Z3.D[0]
    WORD $0x05282065              // DUP Z5.D, Z3.D[0]
    WORD $0x05282066              // DUP Z6.D, Z3.D[0]
    WORD $0x05282067              // DUP Z7.D, Z3.D[0]
    
    ADD $8, R0
    SUB $1, R1
    
    LSL $3, R2, R3
    CMP R3, R1
    BLT sve_fmax_tail

sve_fmax_loop8:
    WORD $0xA5E0A008              // LD1D Z8
    WORD $0xA5E1A009              // LD1D Z9
    WORD $0xA5E2A00A              // LD1D Z10
    WORD $0xA5E3A00B              // LD1D Z11
    WORD $0xA5E4A00C              // LD1D Z12
    WORD $0xA5E5A00D              // LD1D Z13
    WORD $0xA5E6A00E              // LD1D Z14
    WORD $0xA5E7A00F              // LD1D Z15
    
    WORD $0x65C68100              // FMAX Z0.D, P0/M, Z0.D, Z8.D
    WORD $0x65C68121              // FMAX Z1.D, P0/M, Z1.D, Z9.D
    WORD $0x65C68142              // FMAX Z2.D, P0/M, Z2.D, Z10.D
    WORD $0x65C68163              // FMAX Z3.D, P0/M, Z3.D, Z11.D
    WORD $0x65C68184              // FMAX Z4.D, P0/M, Z4.D, Z12.D
    WORD $0x65C681A5              // FMAX Z5.D, P0/M, Z5.D, Z13.D
    WORD $0x65C681C6              // FMAX Z6.D, P0/M, Z6.D, Z14.D
    WORD $0x65C681E7              // FMAX Z7.D, P0/M, Z7.D, Z15.D
    
    LSL $3, R3, R5
    ADD R5, R0, R0
    SUBS R3, R1, R1
    CMP R3, R1
    BGE sve_fmax_loop8

sve_fmax_tail:
    MOVD ZR, R4
    CBZ R1, sve_fmax_reduce
    
sve_fmax_tail_loop:
    WORD $0x25E11C81              // WHILELO P1.D, X4, X1
    BEQ sve_fmax_reduce
    
    WORD $0xA5E0A408              // LD1D {Z8.D}, P1/Z, [R0, #0, MUL VL]
    WORD $0x65C68500              // FMAX Z0.D, P1/M, Z0.D, Z8.D
    
    WORD $0x04F0E3E4              // INCD X4
    LSL $3, R2, R5
    ADD R5, R0, R0
    B sve_fmax_tail_loop

sve_fmax_reduce:
    WORD $0x65C68080              // FMAX Z0.D, P0/M, Z0.D, Z4.D
    WORD $0x65C680A1              // FMAX Z1.D, P0/M, Z1.D, Z5.D
    WORD $0x65C680C2              // FMAX Z2.D, P0/M, Z2.D, Z6.D
    WORD $0x65C680E3              // FMAX Z3.D, P0/M, Z3.D, Z7.D
    WORD $0x65C68040              // FMAX Z0.D, P0/M, Z0.D, Z2.D
    WORD $0x65C68061              // FMAX Z1.D, P0/M, Z1.D, Z3.D
    WORD $0x65C68020              // FMAX Z0.D, P0/M, Z0.D, Z1.D
    
    // FMAXV - Horizontal max across all lanes
    WORD $0x65C62000              // FMAXV D0, P0, Z0.D
    
    FMOVD F0, ret+24(FP)
    RET

sve_fmax_empty:
    FMOVD $0.0, F0
    FMOVD F0, ret+24(FP)
    RET

// ┌──────────────────────────────────────────────────────────────────────────────┐
// │ func dotProductFloat64SVE(a, b []float64) float64                            │
// │                                                                              │
// │ Strategy: Use FMLA (fused multiply-add) for accumulation                     │
// │           acc += a[i] * b[i]                                                 │
// │                                                                              │
// │ FMLA Zda.D, Pg/M, Zn.D, Zm.D:                                                │
// │   Zda = Zda + (Zn × Zm)   (fused, single rounding)                           │
// │                                                                              │
// │   ┌──────────────────────────────────────────────────────────────────┐       │
// │   │  a:   [ a0 | a1 | a2 | a3 ]                                      │       │
// │   │          ×    ×    ×    ×                                        │       │
// │   │  b:   [ b0 | b1 | b2 | b3 ]                                      │       │
// │   │          │    │    │    │                                        │       │
// │   │  acc: [ s0 | s1 | s2 | s3 ]                                      │       │
// │   │          +    +    +    +                                        │       │
// │   │  acc' = [ s0+a0b0 | s1+a1b1 | s2+a2b2 | s3+a3b3 ]                │       │
// │   └──────────────────────────────────────────────────────────────────┘       │
// └──────────────────────────────────────────────────────────────────────────────┘
TEXT ·dotProductFloat64SVE(SB), NOSPLIT, $0-56
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
    
    WORD $0x04E0E3E3              // CNTD X3 (vector length)
    
    LSL $3, R3, R4                // R4 = elements per iteration = R3 * 8
    
    CMP R4, R1
    BLT sve_dot_tail

sve_dot_loop8:
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
    
    // FMLA: Z0-Z7 += Z8-Z15 * Z16-Z23
    WORD $0x65F00100              // FMLA Z0.D, P0/M, Z8.D, Z16.D
    WORD $0x65F10121              // FMLA Z1.D, P0/M, Z9.D, Z17.D
    WORD $0x65F20142              // FMLA Z2.D, P0/M, Z10.D, Z18.D
    WORD $0x65F30163              // FMLA Z3.D, P0/M, Z11.D, Z19.D
    WORD $0x65F40184              // FMLA Z4.D, P0/M, Z12.D, Z20.D
    WORD $0x65F501A5              // FMLA Z5.D, P0/M, Z13.D, Z21.D
    WORD $0x65F601C6              // FMLA Z6.D, P0/M, Z14.D, Z22.D
    WORD $0x65F701E7              // FMLA Z7.D, P0/M, Z15.D, Z23.D
    
    // Advance both pointers
    LSL $3, R4, R5
    ADD R5, R0, R0
    ADD R5, R2, R2
    SUBS R4, R1, R1
    CMP R4, R1
    BGE sve_dot_loop8

sve_dot_tail:
    MOVD ZR, R6                   // Index for WHILELO
    CBZ R1, sve_dot_reduce
    
sve_dot_tail_loop:
    WORD $0x25E11CC1              // WHILELO P1.D, X6, X1
    BEQ sve_dot_reduce
    
    WORD $0xA5E0A408              // LD1D {Z8.D}, P1/Z, [R0, #0, MUL VL]
    WORD $0xA5E0A450              // LD1D {Z16.D}, P1/Z, [R2, #0, MUL VL]
    
    WORD $0x65F00500              // FMLA Z0.D, P1/M, Z8.D, Z16.D
    
    WORD $0x04F0E3E6              // INCD X6
    LSL $3, R3, R5
    ADD R5, R0, R0
    ADD R5, R2, R2
    B sve_dot_tail_loop

sve_dot_reduce:
    // Combine accumulators: 8 → 4 → 2 → 1
    WORD $0x65C08080              // FADD Z0.D, P0/M, Z0.D, Z4.D
    WORD $0x65C080A1              // FADD Z1.D, P0/M, Z1.D, Z5.D
    WORD $0x65C080C2              // FADD Z2.D, P0/M, Z2.D, Z6.D
    WORD $0x65C080E3              // FADD Z3.D, P0/M, Z3.D, Z7.D
    WORD $0x65C08040              // FADD Z0.D, P0/M, Z0.D, Z2.D
    WORD $0x65C08061              // FADD Z1.D, P0/M, Z1.D, Z3.D
    WORD $0x65C08020              // FADD Z0.D, P0/M, Z0.D, Z1.D
    
    // FADDA for horizontal sum
    FMOVD $0.0, F16
    WORD $0x65D82010              // FADDA D16, P0, D16, Z0.D
    
    FMOVD F16, ret+48(FP)
    RET
