# SIMD vs Scalar Threshold Analysis

Empirically measured crossover points where SIMD becomes faster than scalar.

## Summary Table

| Function           | Graviton4 NEON | Graviton4 SVE | Graviton3 NEON | Graviton3 SVE |
|--------------------|----------------|---------------|----------------|---------------|
| SumFloat64         | 32             | 8             | 32             | 48            |
| MinFloat64         | 8              | 8             | 8              | 48            |
| MaxFloat64         | 8              | 8             | 8              | 48            |
| DotProductFloat64  | 32             | 16            | 32             | 32            |
| SumInt64           | 8              | 16            | 8              | 32            |
| MinInt64           | 8              | 32            | 8              | 48            |
| MaxInt64           | 8              | 32            | 8              | 48            |
| DotProductInt64    | N/A            | 16            | N/A            | 32            |
| SumSqInt64         | 8              | 32            | 8              | 32            |

**Recommended Conservative Thresholds (n >=):**

| Function           | NEON | SVE  |
|--------------------|------|------|
| SumFloat64         | 32   | 32   |
| MinFloat64         | 8    | 32   |
| MaxFloat64         | 8    | 32   |
| DotProductFloat64  | 32   | 32   |
| SumInt64           | 8    | 32   |
| MinInt64           | 8    | 48   |
| MaxInt64           | 8    | 48   |
| DotProductInt64    | N/A  | 32   |
| SumSqInt64         | 8    | 32   |

## Key Findings

### Graviton4 (256-bit SVE)
- **SVE startup cost is low** (~3-5ns), making it competitive at smaller sizes
- SVE beats NEON for most operations at n >= 48-64
- NEON wins at n=8 for Min/Max operations due to lower function call overhead

### Graviton3 (256-bit SVE)
- **Higher SVE overhead** (~5-7ns startup), requiring larger arrays to amortize
- NEON is generally faster than SVE until n >= 48-64
- For SumFloat64/SumInt64, SVE catches up around n=48

## Detailed Analysis

### Float64 Operations

**SumFloat64:**
- G4: NEON wins at n=32 (7.1ns vs 14.0ns scalar), SVE wins at n=8 (4.8ns vs 5.1ns)
- G3: NEON wins at n=32 (7.1ns vs 14.5ns scalar), SVE wins at n=48 (8.3ns vs 21.2ns)

**MinFloat64:**  
- G4: NEON wins at n=8 (4.5ns vs 5.1ns scalar), SVE at n=8 (5.1ns vs 5.1ns)
- G3: NEON wins at n=8 (4.6ns vs 5.5ns scalar), SVE at n=48 (10.1ns vs 55.4ns)

**MaxFloat64:**
- G4: NEON wins at n=8 (4.3ns vs 4.7ns scalar), SVE at n=8 (4.8ns vs 4.7ns)
- G3: NEON wins at n=8 (4.7ns vs 4.7ns scalar), SVE at n=48 (9.7ns vs 21.7ns)

### Int64 Operations

**SumInt64:**
- G4: NEON wins at n=8 (4.1ns vs 4.7ns scalar), SVE at n=16 (4.8ns vs 7.6ns)
- G3: NEON wins at n=8 (4.0ns vs 4.7ns scalar), SVE at n=32 (7.0ns vs 14.0ns)

**MinInt64/MaxInt64:**
- G4: NEON wins at n=8 (4.3ns vs 4.9ns scalar), SVE at n=32 (8.3ns vs 16.5ns)
- G3: NEON wins at n=8 (4.6ns vs 4.7ns scalar), SVE at n=48 (9.7ns vs 31.2ns)

**DotProductInt64:** (SVE-only, no NEON implementation)
- G4: SVE wins at n=16 (6.8ns vs 7.6ns scalar)
- G3: SVE wins at n=32 (10.3ns vs 14.0ns scalar)

**SumSqInt64:**
- G4: NEON wins at n=8 (3.4ns vs 4.7ns scalar), SVE at n=32 (8.2ns vs 13.7ns)
- G3: NEON wins at n=8 (3.9ns vs 4.7ns scalar), SVE at n=32 (10.0ns vs 16.1ns)

## Recommendations

1. **Use NEON for n < 32** - NEON has lower startup overhead and is universally faster for small arrays

2. **Use SVE for n >= 32** on Graviton4 - SVE catches up and provides better scaling

3. **Use SVE for n >= 48** on Graviton3 - Higher SVE overhead means larger threshold needed

4. **Practical unified thresholds:**
   - NEON: n >= 8 (always use for small arrays on ARM64)
   - SVE: n >= 32 (conservative, works on both G3 and G4)

5. **Never use SIMD for n < 8** - Function call overhead dominates
