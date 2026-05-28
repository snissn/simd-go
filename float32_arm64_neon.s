//go:build arm64

#include "textflag.h"

#define FMLA4S(d, n, m) WORD $(0x4E20CC00 + ((m) << 16) + ((n) << 5) + (d))
#define FADD4S(d, n, m) WORD $(0x4E20D400 + ((m) << 16) + ((n) << 5) + (d))
#define FADDP4S(d, n, m) WORD $(0x6E20D400 + ((m) << 16) + ((n) << 5) + (d))
#define FADDP2S(d, n) WORD $(0x7E30D800 + ((n) << 5) + (d))

// ╔══════════════════════════════════════════════════════════════════════════════╗
// ║                         NEON Float32 SIMD Operations                         ║
// ║                                                                              ║
// ║  NEON processes 4 x float32 per vector register (128-bit vectors)           ║
// ║                                                                              ║
// ║  Vector register layout:                                                     ║
// ║  ┌─────────────────────────────────────────────────────────────────────┐     ║
// ║  │  V0.4S = [ lane0 | lane1 | lane2 | lane3 ]  (128 bits total)       │     ║
// ║  └─────────────────────────────────────────────────────────────────────┘     ║
// ╚══════════════════════════════════════════════════════════════════════════════╝

// ┌──────────────────────────────────────────────────────────────────────────────┐
// │ func sumFloat32NEON(vals []float32) float32                                   │
// │                                                                              │
// │ Strategy: 16 parallel accumulators to hide memory latency                    │
// │ Processes 64 elements per iteration (16 vectors × 4 lanes)                   │
// └──────────────────────────────────────────────────────────────────────────────┘
TEXT ·sumFloat32NEON(SB), NOSPLIT, $0-28
    MOVD vals_base+0(FP), R0      // R0 = pointer to vals[0]
    MOVD vals_len+8(FP), R1       // R1 = len(vals)
    
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
    
    CMP $64, R1
    BLT f32sum_tail32
    
// Main loop: Process 64 elements per iteration (16 vectors × 4 floats)
f32sum_loop64:
    // Load 64 elements (256 bytes total) into V16-V31
    VLD1.P 64(R0), [V16.S4, V17.S4, V18.S4, V19.S4]
    VLD1.P 64(R0), [V20.S4, V21.S4, V22.S4, V23.S4]
    VLD1.P 64(R0), [V24.S4, V25.S4, V26.S4, V27.S4]
    VLD1.P 64(R0), [V28.S4, V29.S4, V30.S4, V31.S4]
    
    // FADD V.4S: accumulate
    WORD $0x4E30D400              // FADD V0.4S, V0.4S, V16.4S
    WORD $0x4E31D421              // FADD V1.4S, V1.4S, V17.4S
    WORD $0x4E32D442              // FADD V2.4S, V2.4S, V18.4S
    WORD $0x4E33D463              // FADD V3.4S, V3.4S, V19.4S
    WORD $0x4E34D484              // FADD V4.4S, V4.4S, V20.4S
    WORD $0x4E35D4A5              // FADD V5.4S, V5.4S, V21.4S
    WORD $0x4E36D4C6              // FADD V6.4S, V6.4S, V22.4S
    WORD $0x4E37D4E7              // FADD V7.4S, V7.4S, V23.4S
    WORD $0x4E38D508              // FADD V8.4S, V8.4S, V24.4S
    WORD $0x4E39D529              // FADD V9.4S, V9.4S, V25.4S
    WORD $0x4E3AD54A              // FADD V10.4S, V10.4S, V26.4S
    WORD $0x4E3BD56B              // FADD V11.4S, V11.4S, V27.4S
    WORD $0x4E3CD58C              // FADD V12.4S, V12.4S, V28.4S
    WORD $0x4E3DD5AD              // FADD V13.4S, V13.4S, V29.4S
    WORD $0x4E3ED5CE              // FADD V14.4S, V14.4S, V30.4S
    WORD $0x4E3FD5EF              // FADD V15.4S, V15.4S, V31.4S
    
    SUB $64, R1
    CMP $64, R1
    BGE f32sum_loop64
    
    // Tree reduction: 16 → 8 → 4 → 2 → 1
    WORD $0x4E28D400              // FADD V0.4S, V0.4S, V8.4S
    WORD $0x4E29D421              // FADD V1.4S, V1.4S, V9.4S
    WORD $0x4E2AD442              // FADD V2.4S, V2.4S, V10.4S
    WORD $0x4E2BD463              // FADD V3.4S, V3.4S, V11.4S
    WORD $0x4E2CD484              // FADD V4.4S, V4.4S, V12.4S
    WORD $0x4E2DD4A5              // FADD V5.4S, V5.4S, V13.4S
    WORD $0x4E2ED4C6              // FADD V6.4S, V6.4S, V14.4S
    WORD $0x4E2FD4E7              // FADD V7.4S, V7.4S, V15.4S
    WORD $0x4E24D400              // FADD V0.4S, V0.4S, V4.4S
    WORD $0x4E25D421              // FADD V1.4S, V1.4S, V5.4S
    WORD $0x4E26D442              // FADD V2.4S, V2.4S, V6.4S
    WORD $0x4E27D463              // FADD V3.4S, V3.4S, V7.4S
    WORD $0x4E22D400              // FADD V0.4S, V0.4S, V2.4S
    WORD $0x4E23D421              // FADD V1.4S, V1.4S, V3.4S
    WORD $0x4E21D400              // FADD V0.4S, V0.4S, V1.4S
    
f32sum_tail32:
    CMP $32, R1
    BLT f32sum_tail16
    
    VLD1.P 64(R0), [V16.S4, V17.S4, V18.S4, V19.S4]
    VLD1.P 64(R0), [V20.S4, V21.S4, V22.S4, V23.S4]
    WORD $0x4E30D400              // FADD V0.4S, V0.4S, V16.4S
    WORD $0x4E31D400              // FADD V0.4S, V0.4S, V17.4S
    WORD $0x4E32D400              // FADD V0.4S, V0.4S, V18.4S
    WORD $0x4E33D400              // FADD V0.4S, V0.4S, V19.4S
    WORD $0x4E34D400              // FADD V0.4S, V0.4S, V20.4S
    WORD $0x4E35D400              // FADD V0.4S, V0.4S, V21.4S
    WORD $0x4E36D400              // FADD V0.4S, V0.4S, V22.4S
    WORD $0x4E37D400              // FADD V0.4S, V0.4S, V23.4S
    SUB $32, R1

f32sum_tail16:
    CMP $16, R1
    BLT f32sum_tail8
    
    VLD1.P 64(R0), [V16.S4, V17.S4, V18.S4, V19.S4]
    WORD $0x4E30D400              // FADD V0.4S, V0.4S, V16.4S
    WORD $0x4E31D400              // FADD V0.4S, V0.4S, V17.4S
    WORD $0x4E32D400              // FADD V0.4S, V0.4S, V18.4S
    WORD $0x4E33D400              // FADD V0.4S, V0.4S, V19.4S
    SUB $16, R1

f32sum_tail8:
    CMP $8, R1
    BLT f32sum_tail4
    
    VLD1.P 32(R0), [V16.S4, V17.S4]
    WORD $0x4E30D400              // FADD V0.4S, V0.4S, V16.4S
    WORD $0x4E31D400              // FADD V0.4S, V0.4S, V17.4S
    SUB $8, R1
    
f32sum_tail4:
    CMP $4, R1
    BLT f32sum_reduce
    
    VLD1.P 16(R0), [V16.S4]
    WORD $0x4E30D400              // FADD V0.4S, V0.4S, V16.4S
    SUB $4, R1
    
f32sum_reduce:
    // Horizontal sum using FADDP pairwise adds
    // FADDP V0.4S, V0.4S, V0.4S: 6E20D400 -> [a+b, c+d, a+b, c+d]
    WORD $0x6E20D400              // FADDP V0.4S, V0.4S, V0.4S
    // Now V0 = [a+b, c+d, a+b, c+d], need to add lanes 0 and 1
    // FADDP S0, V0.2S: 7E30D800 -> lane0 + lane1
    WORD $0x7E30D800              // FADDP S0, V0.2S
    
    // Handle remaining 1-3 elements
f32sum_scalar_loop:
    CBZ R1, f32sum_done
    FMOVS (R0), F1
    FADDS F0, F1, F0
    ADD $4, R0
    SUB $1, R1
    B f32sum_scalar_loop
    
f32sum_done:
    FMOVS F0, ret+24(FP)
    RET

// ┌──────────────────────────────────────────────────────────────────────────────┐
// │ func minFloat32NEON(vals []float32) float32                                   │
// └──────────────────────────────────────────────────────────────────────────────┘
TEXT ·minFloat32NEON(SB), NOSPLIT, $0-28
    MOVD vals_base+0(FP), R0
    MOVD vals_len+8(FP), R1
    
    // Broadcast first element to all accumulators
    FMOVS (R0), F0
    VDUP V0.S[0], V0.S4
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
    
    ADD $4, R0
    SUB $1, R1
    
    CMP $64, R1
    BLT f32min_tail32
    
f32min_loop64:
    VLD1.P 64(R0), [V16.S4, V17.S4, V18.S4, V19.S4]
    VLD1.P 64(R0), [V20.S4, V21.S4, V22.S4, V23.S4]
    VLD1.P 64(R0), [V24.S4, V25.S4, V26.S4, V27.S4]
    VLD1.P 64(R0), [V28.S4, V29.S4, V30.S4, V31.S4]
    
    // FMIN V.4S
    WORD $0x4EB0F400              // FMIN V0.4S, V0.4S, V16.4S
    WORD $0x4EB1F421              // FMIN V1.4S, V1.4S, V17.4S
    WORD $0x4EB2F442              // FMIN V2.4S, V2.4S, V18.4S
    WORD $0x4EB3F463              // FMIN V3.4S, V3.4S, V19.4S
    WORD $0x4EB4F484              // FMIN V4.4S, V4.4S, V20.4S
    WORD $0x4EB5F4A5              // FMIN V5.4S, V5.4S, V21.4S
    WORD $0x4EB6F4C6              // FMIN V6.4S, V6.4S, V22.4S
    WORD $0x4EB7F4E7              // FMIN V7.4S, V7.4S, V23.4S
    WORD $0x4EB8F508              // FMIN V8.4S, V8.4S, V24.4S
    WORD $0x4EB9F529              // FMIN V9.4S, V9.4S, V25.4S
    WORD $0x4EBAF54A              // FMIN V10.4S, V10.4S, V26.4S
    WORD $0x4EBBF56B              // FMIN V11.4S, V11.4S, V27.4S
    WORD $0x4EBCF58C              // FMIN V12.4S, V12.4S, V28.4S
    WORD $0x4EBDF5AD              // FMIN V13.4S, V13.4S, V29.4S
    WORD $0x4EBEF5CE              // FMIN V14.4S, V14.4S, V30.4S
    WORD $0x4EBFF5EF              // FMIN V15.4S, V15.4S, V31.4S
    
    SUB $64, R1
    CMP $64, R1
    BGE f32min_loop64
    
    // Tree reduction
    WORD $0x4EA8F400              // FMIN V0.4S, V0.4S, V8.4S
    WORD $0x4EA9F421              // FMIN V1.4S, V1.4S, V9.4S
    WORD $0x4EAAF442              // FMIN V2.4S, V2.4S, V10.4S
    WORD $0x4EABF463              // FMIN V3.4S, V3.4S, V11.4S
    WORD $0x4EACF484              // FMIN V4.4S, V4.4S, V12.4S
    WORD $0x4EADF4A5              // FMIN V5.4S, V5.4S, V13.4S
    WORD $0x4EAEF4C6              // FMIN V6.4S, V6.4S, V14.4S
    WORD $0x4EAFF4E7              // FMIN V7.4S, V7.4S, V15.4S
    WORD $0x4EA4F400              // FMIN V0.4S, V0.4S, V4.4S
    WORD $0x4EA5F421              // FMIN V1.4S, V1.4S, V5.4S
    WORD $0x4EA6F442              // FMIN V2.4S, V2.4S, V6.4S
    WORD $0x4EA7F463              // FMIN V3.4S, V3.4S, V7.4S
    WORD $0x4EA2F400              // FMIN V0.4S, V0.4S, V2.4S
    WORD $0x4EA3F421              // FMIN V1.4S, V1.4S, V3.4S
    WORD $0x4EA1F400              // FMIN V0.4S, V0.4S, V1.4S

f32min_tail32:
    CMP $32, R1
    BLT f32min_tail16
    
    VLD1.P 64(R0), [V16.S4, V17.S4, V18.S4, V19.S4]
    VLD1.P 64(R0), [V20.S4, V21.S4, V22.S4, V23.S4]
    WORD $0x4EB0F400              // FMIN V0.4S, V0.4S, V16.4S
    WORD $0x4EB1F400              // FMIN V0.4S, V0.4S, V17.4S
    WORD $0x4EB2F400              // FMIN V0.4S, V0.4S, V18.4S
    WORD $0x4EB3F400              // FMIN V0.4S, V0.4S, V19.4S
    WORD $0x4EB4F400              // FMIN V0.4S, V0.4S, V20.4S
    WORD $0x4EB5F400              // FMIN V0.4S, V0.4S, V21.4S
    WORD $0x4EB6F400              // FMIN V0.4S, V0.4S, V22.4S
    WORD $0x4EB7F400              // FMIN V0.4S, V0.4S, V23.4S
    SUB $32, R1

f32min_tail16:
    CMP $16, R1
    BLT f32min_tail8
    
    VLD1.P 64(R0), [V16.S4, V17.S4, V18.S4, V19.S4]
    WORD $0x4EB0F400              // FMIN V0.4S, V0.4S, V16.4S
    WORD $0x4EB1F400              // FMIN V0.4S, V0.4S, V17.4S
    WORD $0x4EB2F400              // FMIN V0.4S, V0.4S, V18.4S
    WORD $0x4EB3F400              // FMIN V0.4S, V0.4S, V19.4S
    SUB $16, R1

f32min_tail8:
    CMP $8, R1
    BLT f32min_tail4
    
    VLD1.P 32(R0), [V16.S4, V17.S4]
    WORD $0x4EB0F400              // FMIN V0.4S, V0.4S, V16.4S
    WORD $0x4EB1F400              // FMIN V0.4S, V0.4S, V17.4S
    SUB $8, R1
    
f32min_tail4:
    CMP $4, R1
    BLT f32min_reduce
    
    VLD1.P 16(R0), [V16.S4]
    WORD $0x4EB0F400              // FMIN V0.4S, V0.4S, V16.4S
    SUB $4, R1
    
f32min_reduce:
    // FMINV S0, V0.4S - horizontal min
    WORD $0x6EB0F800              // FMINV S0, V0.4S
    
f32min_scalar_loop:
    CBZ R1, f32min_done
    FMOVS (R0), F1
    FMINS F0, F1, F0
    ADD $4, R0
    SUB $1, R1
    B f32min_scalar_loop
    
f32min_done:
    FMOVS F0, ret+24(FP)
    RET

// ┌──────────────────────────────────────────────────────────────────────────────┐
// │ func maxFloat32NEON(vals []float32) float32                                   │
// └──────────────────────────────────────────────────────────────────────────────┘
TEXT ·maxFloat32NEON(SB), NOSPLIT, $0-28
    MOVD vals_base+0(FP), R0
    MOVD vals_len+8(FP), R1
    
    // Broadcast first element to all accumulators
    FMOVS (R0), F0
    VDUP V0.S[0], V0.S4
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
    
    ADD $4, R0
    SUB $1, R1
    
    CMP $64, R1
    BLT f32max_tail32
    
f32max_loop64:
    VLD1.P 64(R0), [V16.S4, V17.S4, V18.S4, V19.S4]
    VLD1.P 64(R0), [V20.S4, V21.S4, V22.S4, V23.S4]
    VLD1.P 64(R0), [V24.S4, V25.S4, V26.S4, V27.S4]
    VLD1.P 64(R0), [V28.S4, V29.S4, V30.S4, V31.S4]
    
    // FMAX V.4S
    WORD $0x4E30F400              // FMAX V0.4S, V0.4S, V16.4S
    WORD $0x4E31F421              // FMAX V1.4S, V1.4S, V17.4S
    WORD $0x4E32F442              // FMAX V2.4S, V2.4S, V18.4S
    WORD $0x4E33F463              // FMAX V3.4S, V3.4S, V19.4S
    WORD $0x4E34F484              // FMAX V4.4S, V4.4S, V20.4S
    WORD $0x4E35F4A5              // FMAX V5.4S, V5.4S, V21.4S
    WORD $0x4E36F4C6              // FMAX V6.4S, V6.4S, V22.4S
    WORD $0x4E37F4E7              // FMAX V7.4S, V7.4S, V23.4S
    WORD $0x4E38F508              // FMAX V8.4S, V8.4S, V24.4S
    WORD $0x4E39F529              // FMAX V9.4S, V9.4S, V25.4S
    WORD $0x4E3AF54A              // FMAX V10.4S, V10.4S, V26.4S
    WORD $0x4E3BF56B              // FMAX V11.4S, V11.4S, V27.4S
    WORD $0x4E3CF58C              // FMAX V12.4S, V12.4S, V28.4S
    WORD $0x4E3DF5AD              // FMAX V13.4S, V13.4S, V29.4S
    WORD $0x4E3EF5CE              // FMAX V14.4S, V14.4S, V30.4S
    WORD $0x4E3FF5EF              // FMAX V15.4S, V15.4S, V31.4S
    
    SUB $64, R1
    CMP $64, R1
    BGE f32max_loop64
    
    // Tree reduction
    WORD $0x4E28F400              // FMAX V0.4S, V0.4S, V8.4S
    WORD $0x4E29F421              // FMAX V1.4S, V1.4S, V9.4S
    WORD $0x4E2AF442              // FMAX V2.4S, V2.4S, V10.4S
    WORD $0x4E2BF463              // FMAX V3.4S, V3.4S, V11.4S
    WORD $0x4E2CF484              // FMAX V4.4S, V4.4S, V12.4S
    WORD $0x4E2DF4A5              // FMAX V5.4S, V5.4S, V13.4S
    WORD $0x4E2EF4C6              // FMAX V6.4S, V6.4S, V14.4S
    WORD $0x4E2FF4E7              // FMAX V7.4S, V7.4S, V15.4S
    WORD $0x4E24F400              // FMAX V0.4S, V0.4S, V4.4S
    WORD $0x4E25F421              // FMAX V1.4S, V1.4S, V5.4S
    WORD $0x4E26F442              // FMAX V2.4S, V2.4S, V6.4S
    WORD $0x4E27F463              // FMAX V3.4S, V3.4S, V7.4S
    WORD $0x4E22F400              // FMAX V0.4S, V0.4S, V2.4S
    WORD $0x4E23F421              // FMAX V1.4S, V1.4S, V3.4S
    WORD $0x4E21F400              // FMAX V0.4S, V0.4S, V1.4S

f32max_tail32:
    CMP $32, R1
    BLT f32max_tail16
    
    VLD1.P 64(R0), [V16.S4, V17.S4, V18.S4, V19.S4]
    VLD1.P 64(R0), [V20.S4, V21.S4, V22.S4, V23.S4]
    WORD $0x4E30F400              // FMAX V0.4S, V0.4S, V16.4S
    WORD $0x4E31F400              // FMAX V0.4S, V0.4S, V17.4S
    WORD $0x4E32F400              // FMAX V0.4S, V0.4S, V18.4S
    WORD $0x4E33F400              // FMAX V0.4S, V0.4S, V19.4S
    WORD $0x4E34F400              // FMAX V0.4S, V0.4S, V20.4S
    WORD $0x4E35F400              // FMAX V0.4S, V0.4S, V21.4S
    WORD $0x4E36F400              // FMAX V0.4S, V0.4S, V22.4S
    WORD $0x4E37F400              // FMAX V0.4S, V0.4S, V23.4S
    SUB $32, R1

f32max_tail16:
    CMP $16, R1
    BLT f32max_tail8
    
    VLD1.P 64(R0), [V16.S4, V17.S4, V18.S4, V19.S4]
    WORD $0x4E30F400              // FMAX V0.4S, V0.4S, V16.4S
    WORD $0x4E31F400              // FMAX V0.4S, V0.4S, V17.4S
    WORD $0x4E32F400              // FMAX V0.4S, V0.4S, V18.4S
    WORD $0x4E33F400              // FMAX V0.4S, V0.4S, V19.4S
    SUB $16, R1

f32max_tail8:
    CMP $8, R1
    BLT f32max_tail4
    
    VLD1.P 32(R0), [V16.S4, V17.S4]
    WORD $0x4E30F400              // FMAX V0.4S, V0.4S, V16.4S
    WORD $0x4E31F400              // FMAX V0.4S, V0.4S, V17.4S
    SUB $8, R1
    
f32max_tail4:
    CMP $4, R1
    BLT f32max_reduce
    
    VLD1.P 16(R0), [V16.S4]
    WORD $0x4E30F400              // FMAX V0.4S, V0.4S, V16.4S
    SUB $4, R1
    
f32max_reduce:
    // FMAXV S0, V0.4S - horizontal max
    WORD $0x6E30F800              // FMAXV S0, V0.4S
    
f32max_scalar_loop:
    CBZ R1, f32max_done
    FMOVS (R0), F1
    FMAXS F0, F1, F0
    ADD $4, R0
    SUB $1, R1
    B f32max_scalar_loop
    
f32max_done:
    FMOVS F0, ret+24(FP)
    RET

// ┌──────────────────────────────────────────────────────────────────────────────┐
// │ func dotProductFloat32NEON(a, b []float32) float32                            │
// │                                                                              │
// │ Strategy: Use FMLA (fused multiply-add) for better accuracy                  │
// └──────────────────────────────────────────────────────────────────────────────┘
TEXT ·dotProductFloat32NEON(SB), NOSPLIT, $0-52
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
    
    CMP $64, R1
    BLT f32dot_tail32
    
f32dot_loop64:
    // Load 32 elements from a into V16-V23
    VLD1.P 64(R0), [V16.S4, V17.S4, V18.S4, V19.S4]
    // Load corresponding 32 elements from b into V24-V27
    VLD1.P 64(R2), [V24.S4, V25.S4, V26.S4, V27.S4]
    VLD1.P 64(R0), [V20.S4, V21.S4, V22.S4, V23.S4]
    VLD1.P 64(R2), [V28.S4, V29.S4, V30.S4, V31.S4]
    
    // FMLA V.4S: acc += a * b
    WORD $0x4E38CE00              // FMLA V0.4S, V16.4S, V24.4S
    WORD $0x4E39CE21              // FMLA V1.4S, V17.4S, V25.4S
    WORD $0x4E3ACE42              // FMLA V2.4S, V18.4S, V26.4S
    WORD $0x4E3BCE63              // FMLA V3.4S, V19.4S, V27.4S
    WORD $0x4E3CCE84              // FMLA V4.4S, V20.4S, V28.4S
    WORD $0x4E3DCEA5              // FMLA V5.4S, V21.4S, V29.4S
    WORD $0x4E3ECEC6              // FMLA V6.4S, V22.4S, V30.4S
    WORD $0x4E3FCEE7              // FMLA V7.4S, V23.4S, V31.4S
    
    // Load next 32 elements and accumulate into V8-V15
    VLD1.P 64(R0), [V16.S4, V17.S4, V18.S4, V19.S4]
    VLD1.P 64(R2), [V24.S4, V25.S4, V26.S4, V27.S4]
    VLD1.P 64(R0), [V20.S4, V21.S4, V22.S4, V23.S4]
    VLD1.P 64(R2), [V28.S4, V29.S4, V30.S4, V31.S4]
    
    WORD $0x4E38CE08              // FMLA V8.4S, V16.4S, V24.4S
    WORD $0x4E39CE29              // FMLA V9.4S, V17.4S, V25.4S
    WORD $0x4E3ACE4A              // FMLA V10.4S, V18.4S, V26.4S
    WORD $0x4E3BCE6B              // FMLA V11.4S, V19.4S, V27.4S
    WORD $0x4E3CCE8C              // FMLA V12.4S, V20.4S, V28.4S
    WORD $0x4E3DCEAD              // FMLA V13.4S, V21.4S, V29.4S
    WORD $0x4E3ECECE              // FMLA V14.4S, V22.4S, V30.4S
    WORD $0x4E3FCEEF              // FMLA V15.4S, V23.4S, V31.4S
    
    SUB $64, R1
    CMP $64, R1
    BGE f32dot_loop64
    
    // Tree reduction
    WORD $0x4E28D400              // FADD V0.4S, V0.4S, V8.4S
    WORD $0x4E29D421              // FADD V1.4S, V1.4S, V9.4S
    WORD $0x4E2AD442              // FADD V2.4S, V2.4S, V10.4S
    WORD $0x4E2BD463              // FADD V3.4S, V3.4S, V11.4S
    WORD $0x4E2CD484              // FADD V4.4S, V4.4S, V12.4S
    WORD $0x4E2DD4A5              // FADD V5.4S, V5.4S, V13.4S
    WORD $0x4E2ED4C6              // FADD V6.4S, V6.4S, V14.4S
    WORD $0x4E2FD4E7              // FADD V7.4S, V7.4S, V15.4S
    WORD $0x4E24D400              // FADD V0.4S, V0.4S, V4.4S
    WORD $0x4E25D421              // FADD V1.4S, V1.4S, V5.4S
    WORD $0x4E26D442              // FADD V2.4S, V2.4S, V6.4S
    WORD $0x4E27D463              // FADD V3.4S, V3.4S, V7.4S
    WORD $0x4E22D400              // FADD V0.4S, V0.4S, V2.4S
    WORD $0x4E23D421              // FADD V1.4S, V1.4S, V3.4S
    WORD $0x4E21D400              // FADD V0.4S, V0.4S, V1.4S

f32dot_tail32:
    CMP $32, R1
    BLT f32dot_tail16
    
    VLD1.P 64(R0), [V16.S4, V17.S4, V18.S4, V19.S4]
    VLD1.P 64(R2), [V24.S4, V25.S4, V26.S4, V27.S4]
    VLD1.P 64(R0), [V20.S4, V21.S4, V22.S4, V23.S4]
    VLD1.P 64(R2), [V28.S4, V29.S4, V30.S4, V31.S4]
    WORD $0x4E38CE00              // FMLA V0.4S, V16.4S, V24.4S
    WORD $0x4E39CE20              // FMLA V0.4S, V17.4S, V25.4S
    WORD $0x4E3ACE40              // FMLA V0.4S, V18.4S, V26.4S
    WORD $0x4E3BCE60              // FMLA V0.4S, V19.4S, V27.4S
    WORD $0x4E3CCE80              // FMLA V0.4S, V20.4S, V28.4S
    WORD $0x4E3DCEA0              // FMLA V0.4S, V21.4S, V29.4S
    WORD $0x4E3ECEC0              // FMLA V0.4S, V22.4S, V30.4S
    WORD $0x4E3FCEE0              // FMLA V0.4S, V23.4S, V31.4S
    SUB $32, R1

f32dot_tail16:
    CMP $16, R1
    BLT f32dot_tail8
    
    VLD1.P 64(R0), [V16.S4, V17.S4, V18.S4, V19.S4]
    VLD1.P 64(R2), [V24.S4, V25.S4, V26.S4, V27.S4]
    WORD $0x4E38CE00              // FMLA V0.4S, V16.4S, V24.4S
    WORD $0x4E39CE20              // FMLA V0.4S, V17.4S, V25.4S
    WORD $0x4E3ACE40              // FMLA V0.4S, V18.4S, V26.4S
    WORD $0x4E3BCE60              // FMLA V0.4S, V19.4S, V27.4S
    SUB $16, R1

f32dot_tail8:
    CMP $8, R1
    BLT f32dot_tail4
    
    VLD1.P 32(R0), [V16.S4, V17.S4]
    VLD1.P 32(R2), [V24.S4, V25.S4]
    WORD $0x4E38CE00              // FMLA V0.4S, V16.4S, V24.4S
    WORD $0x4E39CE20              // FMLA V0.4S, V17.4S, V25.4S
    SUB $8, R1
    
f32dot_tail4:
    CMP $4, R1
    BLT f32dot_reduce
    
    VLD1.P 16(R0), [V16.S4]
    VLD1.P 16(R2), [V24.S4]
    WORD $0x4E38CE00              // FMLA V0.4S, V16.4S, V24.4S
    SUB $4, R1
    
f32dot_reduce:
    // Horizontal sum
    WORD $0x6E20D400              // FADDP V0.4S, V0.4S, V0.4S
    WORD $0x7E30D800              // FADDP S0, V0.2S
    
f32dot_scalar_loop:
    CBZ R1, f32dot_done
    FMOVS (R0), F1
    FMOVS (R2), F2
    FMULS F1, F2, F1
    FADDS F0, F1, F0
    ADD $4, R0
    ADD $4, R2
    SUB $1, R1
    B f32dot_scalar_loop
    
f32dot_done:
    FMOVS F0, ret+48(FP)
    RET

// ┌──────────────────────────────────────────────────────────────────────────────┐
// │ func dotProductFloat32IndexedNEON(dst, base, query []float32,                │
// │     rowIDs []uint32, rowCount, dims int)                                     │
// │                                                                              │
// │ Batch4 row-major indexed FP32 dot products. query is loaded once per vector  │
// │ chunk and multiplied against four independently addressed base rows.         │
// └──────────────────────────────────────────────────────────────────────────────┘
TEXT ·dotProductFloat32IndexedNEON(SB), NOSPLIT, $0-112
    MOVD dst_base+0(FP), R0       // R0 = &dst[0]
    MOVD base_base+24(FP), R1     // R1 = &base[0]
    MOVD query_base+48(FP), R2    // R2 = &query[0]
    MOVD rowIDs_base+72(FP), R3   // R3 = &rowIDs[0]
    MOVD rowCount+96(FP), R4      // multiple of 4
    MOVD dims+104(FP), R5
    LSL $2, R5, R6                // R6 = dims in bytes

f32batch_indexed_batch_loop:
    // Resolve four row pointers from uint32 row IDs.
    MOVWU.P 4(R3), R7
    MOVWU.P 4(R3), R8
    MOVWU.P 4(R3), R9
    MOVWU.P 4(R3), R10
    MUL R6, R7, R7
    MUL R6, R8, R8
    MUL R6, R9, R9
    MUL R6, R10, R10
    ADD R1, R7, R7
    ADD R1, R8, R8
    ADD R1, R9, R9
    ADD R1, R10, R10

    MOVD R2, R11                  // query cursor
    MOVD R5, R12                  // remaining dims

    VEOR V0.B16, V0.B16, V0.B16    // row 0 accumulators
    VEOR V1.B16, V1.B16, V1.B16
    VEOR V2.B16, V2.B16, V2.B16
    VEOR V3.B16, V3.B16, V3.B16
    VEOR V4.B16, V4.B16, V4.B16    // row 1 accumulators
    VEOR V5.B16, V5.B16, V5.B16
    VEOR V6.B16, V6.B16, V6.B16
    VEOR V7.B16, V7.B16, V7.B16
    VEOR V8.B16, V8.B16, V8.B16    // row 2 accumulators
    VEOR V9.B16, V9.B16, V9.B16
    VEOR V10.B16, V10.B16, V10.B16
    VEOR V11.B16, V11.B16, V11.B16
    VEOR V12.B16, V12.B16, V12.B16 // row 3 accumulators
    VEOR V13.B16, V13.B16, V13.B16
    VEOR V14.B16, V14.B16, V14.B16
    VEOR V15.B16, V15.B16, V15.B16

    CMP $16, R12
    BLT f32batch_indexed_tail8

f32batch_indexed_loop16:
    VLD1.P 64(R11), [V16.S4, V17.S4, V18.S4, V19.S4]

    VLD1.P 64(R7), [V20.S4, V21.S4, V22.S4, V23.S4]
    FMLA4S(0, 20, 16)
    FMLA4S(1, 21, 17)
    FMLA4S(2, 22, 18)
    FMLA4S(3, 23, 19)

    VLD1.P 64(R8), [V20.S4, V21.S4, V22.S4, V23.S4]
    FMLA4S(4, 20, 16)
    FMLA4S(5, 21, 17)
    FMLA4S(6, 22, 18)
    FMLA4S(7, 23, 19)

    VLD1.P 64(R9), [V20.S4, V21.S4, V22.S4, V23.S4]
    FMLA4S(8, 20, 16)
    FMLA4S(9, 21, 17)
    FMLA4S(10, 22, 18)
    FMLA4S(11, 23, 19)

    VLD1.P 64(R10), [V20.S4, V21.S4, V22.S4, V23.S4]
    FMLA4S(12, 20, 16)
    FMLA4S(13, 21, 17)
    FMLA4S(14, 22, 18)
    FMLA4S(15, 23, 19)

    SUB $16, R12
    CMP $16, R12
    BGE f32batch_indexed_loop16

f32batch_indexed_tail8:
    CMP $8, R12
    BLT f32batch_indexed_tail4

    VLD1.P 32(R11), [V16.S4, V17.S4]
    VLD1.P 32(R7), [V20.S4, V21.S4]
    FMLA4S(0, 20, 16)
    FMLA4S(1, 21, 17)
    VLD1.P 32(R8), [V20.S4, V21.S4]
    FMLA4S(4, 20, 16)
    FMLA4S(5, 21, 17)
    VLD1.P 32(R9), [V20.S4, V21.S4]
    FMLA4S(8, 20, 16)
    FMLA4S(9, 21, 17)
    VLD1.P 32(R10), [V20.S4, V21.S4]
    FMLA4S(12, 20, 16)
    FMLA4S(13, 21, 17)
    SUB $8, R12

f32batch_indexed_tail4:
    CMP $4, R12
    BLT f32batch_indexed_reduce

    VLD1.P 16(R11), [V16.S4]
    VLD1.P 16(R7), [V20.S4]
    FMLA4S(0, 20, 16)
    VLD1.P 16(R8), [V20.S4]
    FMLA4S(4, 20, 16)
    VLD1.P 16(R9), [V20.S4]
    FMLA4S(8, 20, 16)
    VLD1.P 16(R10), [V20.S4]
    FMLA4S(12, 20, 16)
    SUB $4, R12

f32batch_indexed_reduce:
    FADD4S(0, 0, 1)
    FADD4S(2, 2, 3)
    FADD4S(0, 0, 2)
    FADDP4S(0, 0, 0)
    FADDP2S(0, 0)

    FADD4S(4, 4, 5)
    FADD4S(6, 6, 7)
    FADD4S(4, 4, 6)
    FADDP4S(4, 4, 4)
    FADDP2S(4, 4)

    FADD4S(8, 8, 9)
    FADD4S(10, 10, 11)
    FADD4S(8, 8, 10)
    FADDP4S(8, 8, 8)
    FADDP2S(8, 8)

    FADD4S(12, 12, 13)
    FADD4S(14, 14, 15)
    FADD4S(12, 12, 14)
    FADDP4S(12, 12, 12)
    FADDP2S(12, 12)

f32batch_indexed_scalar_loop:
    CBZ R12, f32batch_indexed_store
    FMOVS (R11), F17
    FMOVS (R7), F16
    FMULS F16, F17, F16
    FADDS F0, F16, F0
    FMOVS (R8), F16
    FMULS F16, F17, F16
    FADDS F4, F16, F4
    FMOVS (R9), F16
    FMULS F16, F17, F16
    FADDS F8, F16, F8
    FMOVS (R10), F16
    FMULS F16, F17, F16
    FADDS F12, F16, F12
    ADD $4, R11
    ADD $4, R7
    ADD $4, R8
    ADD $4, R9
    ADD $4, R10
    SUB $1, R12
    B f32batch_indexed_scalar_loop

f32batch_indexed_store:
    FMOVS F0, 0(R0)
    FMOVS F4, 4(R0)
    FMOVS F8, 8(R0)
    FMOVS F12, 12(R0)
    ADD $16, R0
    SUB $4, R4
    CBNZ R4, f32batch_indexed_batch_loop
    RET

// ┌──────────────────────────────────────────────────────────────────────────────┐
// │ func dotProductFloat32StridedNEON(dst, base, query []float32,                │
// │     rowCount, dims, stride int)                                             │
// │                                                                              │
// │ Batch4 row-major strided FP32 dot products. stride is in float32 elements.   │
// └──────────────────────────────────────────────────────────────────────────────┘
TEXT ·dotProductFloat32StridedNEON(SB), NOSPLIT, $0-96
    MOVD dst_base+0(FP), R0       // R0 = &dst[0]
    MOVD base_base+24(FP), R1     // R1 = current row0 base pointer
    MOVD query_base+48(FP), R2    // R2 = &query[0]
    MOVD rowCount+72(FP), R3      // multiple of 4
    MOVD dims+80(FP), R4
    MOVD stride+88(FP), R5
    LSL $2, R5, R6                // R6 = stride in bytes
    LSL $2, R6, R13               // R13 = four-row advance in bytes

f32batch_strided_batch_loop:
    MOVD R1, R7
    ADD R6, R7, R8
    ADD R6, R8, R9
    ADD R6, R9, R10
    MOVD R2, R11                  // query cursor
    MOVD R4, R12                  // remaining dims

    VEOR V0.B16, V0.B16, V0.B16    // row 0 accumulators
    VEOR V1.B16, V1.B16, V1.B16
    VEOR V2.B16, V2.B16, V2.B16
    VEOR V3.B16, V3.B16, V3.B16
    VEOR V4.B16, V4.B16, V4.B16    // row 1 accumulators
    VEOR V5.B16, V5.B16, V5.B16
    VEOR V6.B16, V6.B16, V6.B16
    VEOR V7.B16, V7.B16, V7.B16
    VEOR V8.B16, V8.B16, V8.B16    // row 2 accumulators
    VEOR V9.B16, V9.B16, V9.B16
    VEOR V10.B16, V10.B16, V10.B16
    VEOR V11.B16, V11.B16, V11.B16
    VEOR V12.B16, V12.B16, V12.B16 // row 3 accumulators
    VEOR V13.B16, V13.B16, V13.B16
    VEOR V14.B16, V14.B16, V14.B16
    VEOR V15.B16, V15.B16, V15.B16

    CMP $16, R12
    BLT f32batch_strided_tail8

f32batch_strided_loop16:
    VLD1.P 64(R11), [V16.S4, V17.S4, V18.S4, V19.S4]

    VLD1.P 64(R7), [V20.S4, V21.S4, V22.S4, V23.S4]
    FMLA4S(0, 20, 16)
    FMLA4S(1, 21, 17)
    FMLA4S(2, 22, 18)
    FMLA4S(3, 23, 19)

    VLD1.P 64(R8), [V20.S4, V21.S4, V22.S4, V23.S4]
    FMLA4S(4, 20, 16)
    FMLA4S(5, 21, 17)
    FMLA4S(6, 22, 18)
    FMLA4S(7, 23, 19)

    VLD1.P 64(R9), [V20.S4, V21.S4, V22.S4, V23.S4]
    FMLA4S(8, 20, 16)
    FMLA4S(9, 21, 17)
    FMLA4S(10, 22, 18)
    FMLA4S(11, 23, 19)

    VLD1.P 64(R10), [V20.S4, V21.S4, V22.S4, V23.S4]
    FMLA4S(12, 20, 16)
    FMLA4S(13, 21, 17)
    FMLA4S(14, 22, 18)
    FMLA4S(15, 23, 19)

    SUB $16, R12
    CMP $16, R12
    BGE f32batch_strided_loop16

f32batch_strided_tail8:
    CMP $8, R12
    BLT f32batch_strided_tail4

    VLD1.P 32(R11), [V16.S4, V17.S4]
    VLD1.P 32(R7), [V20.S4, V21.S4]
    FMLA4S(0, 20, 16)
    FMLA4S(1, 21, 17)
    VLD1.P 32(R8), [V20.S4, V21.S4]
    FMLA4S(4, 20, 16)
    FMLA4S(5, 21, 17)
    VLD1.P 32(R9), [V20.S4, V21.S4]
    FMLA4S(8, 20, 16)
    FMLA4S(9, 21, 17)
    VLD1.P 32(R10), [V20.S4, V21.S4]
    FMLA4S(12, 20, 16)
    FMLA4S(13, 21, 17)
    SUB $8, R12

f32batch_strided_tail4:
    CMP $4, R12
    BLT f32batch_strided_reduce

    VLD1.P 16(R11), [V16.S4]
    VLD1.P 16(R7), [V20.S4]
    FMLA4S(0, 20, 16)
    VLD1.P 16(R8), [V20.S4]
    FMLA4S(4, 20, 16)
    VLD1.P 16(R9), [V20.S4]
    FMLA4S(8, 20, 16)
    VLD1.P 16(R10), [V20.S4]
    FMLA4S(12, 20, 16)
    SUB $4, R12

f32batch_strided_reduce:
    FADD4S(0, 0, 1)
    FADD4S(2, 2, 3)
    FADD4S(0, 0, 2)
    FADDP4S(0, 0, 0)
    FADDP2S(0, 0)

    FADD4S(4, 4, 5)
    FADD4S(6, 6, 7)
    FADD4S(4, 4, 6)
    FADDP4S(4, 4, 4)
    FADDP2S(4, 4)

    FADD4S(8, 8, 9)
    FADD4S(10, 10, 11)
    FADD4S(8, 8, 10)
    FADDP4S(8, 8, 8)
    FADDP2S(8, 8)

    FADD4S(12, 12, 13)
    FADD4S(14, 14, 15)
    FADD4S(12, 12, 14)
    FADDP4S(12, 12, 12)
    FADDP2S(12, 12)

f32batch_strided_scalar_loop:
    CBZ R12, f32batch_strided_store
    FMOVS (R11), F17
    FMOVS (R7), F16
    FMULS F16, F17, F16
    FADDS F0, F16, F0
    FMOVS (R8), F16
    FMULS F16, F17, F16
    FADDS F4, F16, F4
    FMOVS (R9), F16
    FMULS F16, F17, F16
    FADDS F8, F16, F8
    FMOVS (R10), F16
    FMULS F16, F17, F16
    FADDS F12, F16, F12
    ADD $4, R11
    ADD $4, R7
    ADD $4, R8
    ADD $4, R9
    ADD $4, R10
    SUB $1, R12
    B f32batch_strided_scalar_loop

f32batch_strided_store:
    FMOVS F0, 0(R0)
    FMOVS F4, 4(R0)
    FMOVS F8, 8(R0)
    FMOVS F12, 12(R0)
    ADD $16, R0
    ADD R13, R1, R1
    SUB $4, R3
    CBNZ R3, f32batch_strided_batch_loop
    RET
