#ifndef __FIX_SQRT_H__
#define __FIX_SQRT_H__
#ifdef __cplusplus
extern "C" {
#endif
#include <stdint.h>

#define FIXSQRT_LUT_SIZE 513

typedef struct {
    int32_t K_int;
    int     shift;
    const int16_t *lut;
} FixedSqrt;

void fixed_sqrt_init(FixedSqrt *ctx, float s_i, float o_i);

/*
 * Fixed-point sqrt with scale conversion.
 *
 * Given:
 *   real_in  = in[i]  * s_i      (int32 input with scale s_i)
 *   real_out = out[i] * o_i      (int16 output with scale o_i)
 *
 * Computes:
 *   real_out = sign(real_in) * sqrt(abs(real_in))
 *
 * i.e. out[i] = sign(in[i]) * sqrt(abs(in[i]) * s_i) / o_i
 *             = sign(in[i]) * sqrt(s_i) / o_i * sqrt(abs(in[i]))
 *             = sign(in[i]) * K * sqrt(abs(in[i]))
 *
 * where K = sqrt(s_i) / o_i, encoded as K_int = round(K * 2^shift).
 * Output saturates to [-32767, 32767].
 */
void fixed_sqrt_compute(
        const FixedSqrt *ctx,
        int16_t *out,
        const int32_t *in,
        int len);

#ifdef __cplusplus
}
#endif
#endif /* __FIX_SQRT_H__ */
