#include <stdint.h>
#include <math.h>
#include "fixsqrt.h"
#include "ambiq_nnsp_debug.h"
#include "minmax.h"
#if ARM_OPTIMIZED == 3
#include <arm_mve.h>
#endif

/*
 * 513-entry LUT + linear interpolation integer sqrt.
 *
 * lut[i] = round(sqrt(i * 65536)) for i in [0..512].
 * Normalize input to ~17 bits via even shift (CLZ),
 * look up + interpolate, then shift back.
 */

static const int16_t sqrt_lut_default[FIXSQRT_LUT_SIZE] = {
        0,   256,   362,   443,   512,   572,   627,   677,   724,   768,
      810,   849,   887,   923,   958,   991,  1024,  1056,  1086,  1116,
     1145,  1173,  1201,  1228,  1254,  1280,  1305,  1330,  1355,  1379,
     1402,  1425,  1448,  1471,  1493,  1515,  1536,  1557,  1578,  1599,
     1619,  1639,  1659,  1679,  1698,  1717,  1736,  1755,  1774,  1792,
     1810,  1828,  1846,  1864,  1881,  1899,  1916,  1933,  1950,  1966,
     1983,  1999,  2016,  2032,  2048,  2064,  2080,  2095,  2111,  2126,
     2142,  2157,  2172,  2187,  2202,  2217,  2232,  2246,  2261,  2275,
     2290,  2304,  2318,  2332,  2346,  2360,  2374,  2388,  2401,  2415,
     2429,  2442,  2455,  2469,  2482,  2495,  2508,  2521,  2534,  2547,
     2560,  2573,  2585,  2598,  2611,  2623,  2636,  2648,  2660,  2673,
     2685,  2697,  2709,  2721,  2733,  2745,  2757,  2769,  2781,  2793,
     2804,  2816,  2828,  2839,  2851,  2862,  2874,  2885,  2896,  2908,
     2919,  2930,  2941,  2952,  2963,  2974,  2985,  2996,  3007,  3018,
     3029,  3040,  3051,  3061,  3072,  3083,  3093,  3104,  3114,  3125,
     3135,  3146,  3156,  3167,  3177,  3187,  3197,  3208,  3218,  3228,
     3238,  3248,  3258,  3268,  3278,  3288,  3298,  3308,  3318,  3328,
     3338,  3348,  3357,  3367,  3377,  3387,  3396,  3406,  3415,  3425,
     3435,  3444,  3454,  3463,  3473,  3482,  3491,  3501,  3510,  3519,
     3529,  3538,  3547,  3556,  3566,  3575,  3584,  3593,  3602,  3611,
     3620,  3629,  3638,  3647,  3656,  3665,  3674,  3683,  3692,  3701,
     3710,  3719,  3727,  3736,  3745,  3754,  3762,  3771,  3780,  3788,
     3797,  3806,  3814,  3823,  3831,  3840,  3849,  3857,  3866,  3874,
     3882,  3891,  3899,  3908,  3916,  3924,  3933,  3941,  3949,  3958,
     3966,  3974,  3982,  3991,  3999,  4007,  4015,  4023,  4031,  4040,
     4048,  4056,  4064,  4072,  4080,  4088,  4096,  4104,  4112,  4120,
     4128,  4136,  4144,  4152,  4160,  4167,  4175,  4183,  4191,  4199,
     4207,  4214,  4222,  4230,  4238,  4245,  4253,  4261,  4268,  4276,
     4284,  4291,  4299,  4307,  4314,  4322,  4329,  4337,  4344,  4352,
     4360,  4367,  4375,  4382,  4389,  4397,  4404,  4412,  4419,  4427,
     4434,  4441,  4449,  4456,  4464,  4471,  4478,  4485,  4493,  4500,
     4507,  4515,  4522,  4529,  4536,  4544,  4551,  4558,  4565,  4572,
     4579,  4587,  4594,  4601,  4608,  4615,  4622,  4629,  4636,  4643,
     4650,  4658,  4665,  4672,  4679,  4686,  4693,  4700,  4707,  4713,
     4720,  4727,  4734,  4741,  4748,  4755,  4762,  4769,  4776,  4782,
     4789,  4796,  4803,  4810,  4817,  4823,  4830,  4837,  4844,  4851,
     4857,  4864,  4871,  4877,  4884,  4891,  4898,  4904,  4911,  4918,
     4924,  4931,  4938,  4944,  4951,  4957,  4964,  4971,  4977,  4984,
     4990,  4997,  5003,  5010,  5017,  5023,  5030,  5036,  5043,  5049,
     5056,  5062,  5069,  5075,  5081,  5088,  5094,  5101,  5107,  5114,
     5120,  5126,  5133,  5139,  5146,  5152,  5158,  5165,  5171,  5177,
     5184,  5190,  5196,  5203,  5209,  5215,  5221,  5228,  5234,  5240,
     5246,  5253,  5259,  5265,  5271,  5278,  5284,  5290,  5296,  5302,
     5309,  5315,  5321,  5327,  5333,  5339,  5345,  5352,  5358,  5364,
     5370,  5376,  5382,  5388,  5394,  5400,  5406,  5412,  5418,  5425,
     5431,  5437,  5443,  5449,  5455,  5461,  5467,  5473,  5479,  5485,
     5491,  5497,  5503,  5508,  5514,  5520,  5526,  5532,  5538,  5544,
     5550,  5556,  5562,  5568,  5574,  5579,  5585,  5591,  5597,  5603,
     5609,  5615,  5620,  5626,  5632,  5638,  5644,  5649,  5655,  5661,
     5667,  5673,  5678,  5684,  5690,  5696,  5701,  5707,  5713,  5719,
     5724,  5730,  5736,  5741,  5747,  5753,  5759,  5764,  5770,  5776,
     5781,  5787,  5793
};

/*
 * Initialize fixed-point sqrt context.
 *
 * K_int encodes the input/output scale ratio:
 *   K     = sqrt(s_i) / o_i
 *   K_int = round(K * 2^FIXSQRT_SHIFT)
 *
 * where s_i = input scale  (real = int32 * s_i),
 *       o_i = output scale (real = int16 * o_i).
 *
 * FIXSQRT_SHIFT must satisfy:
 *   K_int * 46341 < 2^31   (no int32 overflow)
 *   i.e. FIXSQRT_SHIFT <= floor(log2(2^31 / (K * 46341)))
 *
 * Example 1: s_i = 1/32768, o_i = 0.000229
 *   K = sqrt(1/32768) / 0.000229 = 24.12
 *   shift = 10, K_int = round(24.12 * 1024) = 24699
 *
 * Example 2: s_i = o_i = 1/32768
 *   K = sqrt(32768) = 181.02
 *   shift = 7, K_int = round(181.02 * 128) = 23170
 */
void fixed_sqrt_init(FixedSqrt *ctx, float s_i, float o_i) {
    float K = sqrtf(s_i) / o_i;
    int shift;
    int32_t K_int;

    /* Find max shift so K_int * 46341 < 2^31 */
    for (shift = 15; shift >= 0; shift--) {
        K_int = (int32_t)(K * (float)(1 << shift) + 0.5f);
        if ((int64_t)K_int * 46341 < 2147483648LL)
            break;
    }

    ctx->K_int = K_int;
    ctx->shift = shift;
    ctx->lut = sqrt_lut_default;
}

#if ARM_OPTIMIZED != 3
static uint32_t isqrt32_lut(const int16_t *lut, uint32_t x) {
    int msb, sr, rs;
    uint32_t norm, idx, frac;
    int32_t y0, y1, interp;
    uint32_t result;

    if (x == 0)
        return 0;

    msb = 31 - __builtin_clz(x);

    /* Even shift so that norm has ~17 significant bits (9 idx + 8 frac).
       sr > 0: shift right (large x).  sr < 0: shift left (small x). */
    sr = msb - 16;
    if (sr & 1) sr++;   /* round up to even */

    if (sr >= 0)
        norm = x >> sr;
    else
        norm = x << (-sr);

    idx  = norm >> 8;
    frac = norm & 0xFF;
    if (idx >= 512) { idx = 511; frac = 255; }

    /* Linear interpolation: interp ~ sqrt(norm) * 16 */
    y0 = lut[idx];
    y1 = lut[idx + 1];
    interp = y0 + (((y1 - y0) * (int32_t)frac) >> 8);

    /* sqrt(x) ~= interp * 2^(sr/2 - 4) */
    rs = sr / 2 - 4;
    if (rs >= 0)
        result = (uint32_t)interp << rs;
    else
        result = (uint32_t)interp >> (-rs);

    return result;
}

void fixed_sqrt_compute(
        const FixedSqrt *ctx,
        int16_t *out,
        const int32_t *in,
        int len) {
    int i;
    const int32_t K = ctx->K_int;
    const int sh = ctx->shift;

    for (i = 0; i < len; i++) {
        int32_t v = in[i];
        uint32_t ux;
        if (v == 0) {
            out[i] = 0;
        } else {
            /* abs(v), saturate -2^31 to 2^31-1 */
            if (v < 0) v = (v == (int32_t)0x80000000) ? 0x7FFFFFFF : -v;
            ux = (uint32_t)v;
            uint32_t s = isqrt32_lut(ctx->lut, ux);
            int32_t y = (K * (int32_t)s + (1 << (sh - 1))) >> sh;
            if (y > 32767) y = 32767;
            out[i] = (in[i] < 0) ? (int16_t)(-y) : (int16_t)y;
        }
    }
}
#endif /* ARM_OPTIMIZED != 3 */

#if ARM_OPTIMIZED == 3
/*
 * MVE (Helium) vectorized fixed-point sqrt.
 * Processes 4 x int32 lanes at a time.
 */
void fixed_sqrt_compute(
        const FixedSqrt *ctx,
        int16_t *out,
        const int32_t *in,
        int len) {
    int i;
    const int32_t K = ctx->K_int;
    const int sh = ctx->shift;
    const int16_t *lut = ctx->lut;
    const int32x4_t vZero = vdupq_n_s32(0);
    const int32_t rnd = 1 << (sh - 1);
    const int32x4_t vNegSh = vdupq_n_s32(-sh);
    int num = len >> 2;
    int rem = len & 0x3;

    for (i = 0; i < num; i++) {
        int32x4_t xin = vldrwq_s32((const int32_t *)&in[i * 4]);

        /* abs(xin), saturating: -2^31 → 2^31-1 */
        uint32x4_t x = vreinterpretq_u32_s32(vqabsq_s32(xin));
        /* Zero where input == 0 */
        mve_pred16_t nz_mask = vcmpneq_s32(xin, vZero);

        /* CLZ → msb = 31 - clz */
        uint32x4_t clz = vclzq_u32(x);
        int32x4_t msb = vsubq_n_s32(
            vreinterpretq_s32_u32(vdupq_n_u32(31)),
            vreinterpretq_s32_u32(clz));

        /* sr = msb - 16; if odd, sr++ (round up to even) */
        int32x4_t sr = vsubq_n_s32(msb, 16);
        int32x4_t odd = vandq_s32(sr, vdupq_n_s32(1));
        sr = vaddq_s32(sr, odd);

        /* norm = x >> sr (vshl with negative shift = right shift) */
        int32x4_t neg_sr = vnegq_s32(sr);
        uint32x4_t norm = vshlq_u32(x, neg_sr);

        /* idx = norm >> 8, frac = norm & 0xFF */
        uint32x4_t idx  = vshrq_n_u32(norm, 8);
        uint32x4_t frac = vandq_u32(norm, vdupq_n_u32(0xFF));
        idx = vminq_u32(idx, vdupq_n_u32(511));

        /* Gather load from int16 LUT (byte offsets = idx * 2) */
        uint32x4_t byte_off0 = vshlq_n_u32(idx, 1);
        uint32x4_t byte_off1 = vshlq_n_u32(vaddq_n_u32(idx, 1), 1);
        int32x4_t y0 = vldrhq_gather_offset_s32(lut, byte_off0);
        int32x4_t y1 = vldrhq_gather_offset_s32(lut, byte_off1);

        /* Linear interpolation: interp = y0 + (y1-y0)*frac >> 8 */
        int32x4_t diff = vsubq_s32(y1, y0);
        int32x4_t interp = vaddq_s32(y0,
            vshrq_n_s32(vmulq_s32(diff, vreinterpretq_s32_u32(frac)), 8));

        /* Shift back: rs = sr/2 - 4 */
        int32x4_t rs = vsubq_n_s32(vshrq_n_s32(sr, 1), 4);
        uint32x4_t sqrtval = vshlq_u32(vreinterpretq_u32_s32(interp), rs);

        /* Zero out where input == 0 */
        sqrtval = vpselq_u32(sqrtval, vdupq_n_u32(0), nz_mask);

        /* K_int * sqrt — fully vectorized, fits int32 */
        int32x4_t scaled = vmulq_s32(vdupq_n_s32(K),
                                      vreinterpretq_s32_u32(sqrtval));
        scaled = vaddq_n_s32(scaled, rnd);
        scaled = vshlq_s32(scaled, vNegSh);
        scaled = vminq_s32(scaled, vdupq_n_s32(32767));

        /* Restore sign: negate where input was negative */
        mve_pred16_t neg_mask = vcmpltq_s32(xin, vZero);
        scaled = vnegq_m_s32(scaled, scaled, neg_mask);

        /* Narrow int32 to int16 and store */
        vstrhq_s32(&out[i * 4], scaled);
    }

    /* Remainder: 1-3 elements with tail predication */
    if (rem) {
        mve_pred16_t p = vctp32q((uint32_t)rem);
        int32x4_t xin = vldrwq_z_s32((const int32_t *)&in[num * 4], p);

        /* abs(xin), saturating: -2^31 → 2^31-1 */
        uint32x4_t x = vreinterpretq_u32_s32(vqabsq_s32(xin));
        /* Zero where input == 0 */
        mve_pred16_t nz_mask = vcmpneq_s32(xin, vZero);

        uint32x4_t clz = vclzq_u32(x);
        int32x4_t msb = vsubq_n_s32(
            vreinterpretq_s32_u32(vdupq_n_u32(31)),
            vreinterpretq_s32_u32(clz));

        int32x4_t sr = vsubq_n_s32(msb, 16);
        int32x4_t odd = vandq_s32(sr, vdupq_n_s32(1));
        sr = vaddq_s32(sr, odd);

        int32x4_t neg_sr = vnegq_s32(sr);
        uint32x4_t norm = vshlq_u32(x, neg_sr);

        uint32x4_t idx  = vshrq_n_u32(norm, 8);
        uint32x4_t frac = vandq_u32(norm, vdupq_n_u32(0xFF));
        idx = vminq_u32(idx, vdupq_n_u32(511));

        uint32x4_t byte_off0 = vshlq_n_u32(idx, 1);
        uint32x4_t byte_off1 = vshlq_n_u32(vaddq_n_u32(idx, 1), 1);
        int32x4_t y0 = vldrhq_gather_offset_s32(lut, byte_off0);
        int32x4_t y1 = vldrhq_gather_offset_s32(lut, byte_off1);

        int32x4_t diff = vsubq_s32(y1, y0);
        int32x4_t interp = vaddq_s32(y0,
            vshrq_n_s32(vmulq_s32(diff, vreinterpretq_s32_u32(frac)), 8));

        int32x4_t rs = vsubq_n_s32(vshrq_n_s32(sr, 1), 4);
        uint32x4_t sqrtval = vshlq_u32(vreinterpretq_u32_s32(interp), rs);
        sqrtval = vpselq_u32(sqrtval, vdupq_n_u32(0), nz_mask);

        /* K_int scaling — vectorized */
        int32x4_t scaled = vmulq_s32(vdupq_n_s32(K),
                                      vreinterpretq_s32_u32(sqrtval));
        scaled = vaddq_n_s32(scaled, rnd);
        scaled = vshlq_s32(scaled, vNegSh);
        scaled = vminq_s32(scaled, vdupq_n_s32(32767));

        /* Restore sign: negate where input was negative */
        mve_pred16_t neg_mask = vcmpltq_s32(xin, vZero);
        scaled = vnegq_m_s32(scaled, scaled, neg_mask);

        /* Narrow and store with predication */
        vstrhq_p_s32(&out[num * 4], scaled, p);
    }
}
#endif /* ARM_OPTIMIZED == 3 */
