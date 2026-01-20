#include "ambiq_stdint.h"
#include "spectrogram_module.h"
#include "feature_module.h"
#include "minmax.h"
#include "ambiq_nnsp_const.h"
#include "melSpecProc.h"
#include "fixlog10.h"
#include "ambiq_nnsp_debug.h"
#include "ns_ambiqsuite_harness.h"
#include "nn_speech.h"

#if ARM_OPTIMIZED == 3
#include "basic_mve.h"
#endif
#if AMBIQ_NNSP_DEBUG == 1
    #include "debug_files.h"
#endif
#define LOG10_2POW_N15_Q15 (-147963)
__attribute__((aligned(16))) int32_t GLOBAL_PSPEC[514];

extern const int16_t mfltrBank_coeff_nfilt72_fftsize512[];
extern const int16_t mfltrBank_coeff_nfilt40_fftsize512[];
extern const int16_t mfltrBank_coeff_nfilt22_fftsize256[];

void FeatureClass_construct(
        FeatureClass *ps, 
        const int32_t *norm_mean, 
        const int32_t *norm_stdR, 
        int8_t qbit_output,
        int16_t num_mfltrBank, 
        int16_t winsize, 
        int16_t hopsize, 
        int16_t fftsize,
        int16_t num_frames_infer,
        const int16_t *pt_stft_win_coeff,
        const int16_t *p_melBanks,
        FEATURE_TYPE_E feature_type) 
    {

        stftModule_construct(&ps->state_stftModule, winsize, hopsize, fftsize, pt_stft_win_coeff);
        ps->pt_norm_mean = norm_mean;
        ps->pt_norm_stdR = norm_stdR;
        ps->num_context = num_frames_infer;
        if (feature_type == feat_spec)
            ps->dim_feat = 2 * (1 + (fftsize >> 1));
        else if (feature_type == feat_logpspec)
            ps->dim_feat = 1 + (fftsize >> 1);
        else if (feature_type == feat_spec_erb)
        {
            ps->dim_feat = num_mfltrBank * 2;
        }
        else // feat_mel
            ps->dim_feat = num_mfltrBank;
        ps->qbit_output = qbit_output;
        ps->num_mfltrBank = num_mfltrBank;
        ps->p_melBanks = p_melBanks;
        ps->feature_type = feature_type;
    // if ((ps->num_mfltrBank == 72) && (fftsize == 512))
    //     ps->p_melBanks = mfltrBank_coeff_nfilt72_fftsize512;
    // else if ((ps->num_mfltrBank == 40) && (fftsize == 512))
    //     ps->p_melBanks = mfltrBank_coeff_nfilt40_fftsize512;
    // else if ((ps->num_mfltrBank == 22) && (fftsize == 256))
    //     ps->p_melBanks = mfltrBank_coeff_nfilt22_fftsize256;
}

void FeatureClass_setDefault(FeatureClass *ps) {
    int i, j;
    int64_t tmp64;
    int16_t tmp;

    stftModule_setDefault(&ps->state_stftModule);

    if (ps->feature_type == feat_spec_erb)
    {
        for (i = 0; i < ps->dim_feat; i++) {
            ps->normFeatContext[i] = 0;
        }
    }
    else
    {
        for (i = 0; i < ps->dim_feat; i++) {
            tmp64 = (int64_t)((int32_t)LOG10_2POW_N15_Q15 - ps->pt_norm_mean[i]);
            tmp64 = (tmp64 * (int64_t)ps->pt_norm_stdR[i]) >> (30 - ps->qbit_output);
            tmp64 = MIN(MAX(tmp64, (int64_t)MIN_INT16_T), (int64_t)MAX_INT16_T);
            tmp = (int16_t)tmp64;

            for (j = 0; j < (ps->num_context - 1); j++) {
                ps->normFeatContext[i + j * ps->dim_feat] = tmp;
            }
        }
    }
}

void FeatureClass_execute(FeatureClass *ps, int16_t *input) {
    int16_t qbit_out;
    int32_t *pspec = GLOBAL_PSPEC;
    int32_t *spec = ps->state_stftModule.spec;
    int shift = (ps->num_context - 1) * ps->dim_feat;
    int i;
    int64_t tmp;
    static int32_t spec_blk[257];
    static int32_t feats_no_interleave[129*2];
    if (ps->num_context > 1)
    {
        for (i = 0; i < shift; i++) {
            ps->normFeatContext[i] = ps->normFeatContext[i + ps->dim_feat];
        }

    }

    if (ps->feature_type != feat_time)
    {
        
    #if ARM_FFT == 0
        stftModule_analyze(&ps->state_stftModule, input, spec);
        #if AMBIQ_NNSP_DEBUG == 1
        for (i = 0; i < 1 + (LEN_FFT_NNSP >> 1); i++) {
            fprintf(file_spec_c, "%d %d ", spec[2 * i], spec[2 * i + 1]);
        }
        fprintf(file_spec_c, "\n");
        #endif
        spec2pspec(pspec, spec, 1 + (LEN_FFT_NNSP >> 1));
    #else
        stftModule_analyze_arm(
            (void *)&ps->state_stftModule,
            input, // q15
            spec,  // q21
            ps->state_stftModule.len_fft, &qbit_out);
        
    #endif
    }



#if AMBIQ_NNSP_DEBUG == 1
    for (i = 0; i < 1 + (LEN_FFT_NNSP >> 1); i++) {
        fprintf(file_pspec_c, "%d ", pspec[i]);
    }
    fprintf(file_pspec_c, "\n");
#endif

    if (ps->feature_type == feat_spec)
    {
        for (i = 0; i < 257; i++) {
            ps->feature[i] = spec[2*i]; // qbit_out
            ps->feature[i + 257] = spec[2*i+1]; // qbit_out
        }
    }
    else if (ps->feature_type == feat_time)
    {
        for (i = 0; i < ps->dim_feat; i++) {
            ps->feature[i] = (int32_t) input[i]; // q15
        }
        qbit_out = 15;
    }
    else if (ps->feature_type == feat_logpspec)
    {
        spec2pspec_arm(
            pspec,
            spec,
            1 + (ps->state_stftModule.len_fft >> 1),
            qbit_out);
        log10_vec(ps->feature, pspec, ps->dim_feat, 15);
        qbit_out = 15;
    }
    else if ((ps->feature_type == feat_mel) || (ps->feature_type == feat_hybrid) || (ps->feature_type == feat_erb_logpspec))
    {
        spec2pspec_arm(pspec, spec, 1 + (ps->state_stftModule.len_fft >> 1), qbit_out); // q15
        melSpecProc(pspec, ps->feature, ps->p_melBanks, ps->num_mfltrBank);
        log10_vec(ps->feature, ps->feature, ps->dim_feat, 15);
        qbit_out = 15;
    }
    else if (ps->feature_type == feat_erb_mag)
    {
        spec2pspec_arm(
            pspec,  // q15
            spec, // q21 , qbit_out
            1 + (ps->state_stftModule.len_fft >> 1),
            qbit_out);
        
        // sqrt of pspec
        for (i = 0; i < 1 + (ps->state_stftModule.len_fft >> 1); i++) {
            int32_t result;
            // x = sqrt(pspec[i])
            // x = x1 * 2**16, where x1 is in Q31
            // sqrt(x) = sqrt(x1) * 2**8, where sqrt(x1) is in Q31
            // so sqrt(x) in q23 (= 31 - 8)
            arm_sqrt_q31(
                pspec[i], // q15
                &result); // q23
            pspec[i] = result >> (23 - 15); // q15
            
        }

        melSpecProc(
            pspec, //q15
            ps->feature, // q15
            ps->p_melBanks,
            ps->num_mfltrBank);

        qbit_out = 15;
    }
    else if (ps->feature_type == feat_spec_erb) // No logarithm
    {
        // ns_printf("Spec real:\n");
        for (i = 0; i < 257; i++) {
            spec_blk[i] = spec[2*i]; // qbit_out
            // ns_printf("%d, ", spec_blk[i]);
        }
        // ns_printf("\n");
        melSpecProc( // real part
            spec_blk, // qbit_out
            feats_no_interleave, // qbit_out
            ps->p_melBanks,
            ps->num_mfltrBank);
        
        // ns_printf("Spec imag:\n");
        for (i = 0; i < 257; i++) {
            spec_blk[i] = spec[2*i+1];
            // ns_printf("%d, ", spec_blk[i]);
        }
        // ns_printf("\n");

        melSpecProc( // imaginary part
            spec_blk, // qbit_out
            feats_no_interleave + ps->num_mfltrBank, // qbit_out
            ps->p_melBanks,
            ps->num_mfltrBank);
        
        for (i = 0; i < ps->num_mfltrBank; i++) {
            ps->feature[2*i] = feats_no_interleave[i]; // real part
            ps->feature[2*i + 1] = feats_no_interleave[i + ps->num_mfltrBank]; // imaginary part
        }
        
        // ns_printf("ERB Spec:\n");
        // for (i = 0; i < ps->dim_feat; i++) {
        //     ns_printf("%d, ", ps->feature[i]);
        // }
        // ns_printf("\n");

    }

#if AMBIQ_NNSP_DEBUG == 1
    for (i = 0; i < ps->dim_feat; i++) {
        fprintf(file_feat_c, "%d ", ps->feature[i]);
    }
    fprintf(file_feat_c, "\n");
#endif
    if (ps->pt_norm_mean == NULL)
    {
        for (i = 0; i < ps->dim_feat; i++) {
            tmp = ps->feature[i];
            ps->normFeatContext[i + shift] = (int32_t)tmp;
        }
    }
    else
    {
        for (i = 0; i < ps->dim_feat; i++) {
            tmp = (int64_t)ps->feature[i] - (int64_t)ps->pt_norm_mean[i];
            tmp = (tmp * ((int64_t)ps->pt_norm_stdR[i])) >>
                (30 - ps->qbit_output); // Bit_frac_out = 30-22 = 8
            // tmp = MIN(MAX(tmp, (int64_t)MIN_INT16_T), (int64_t)MAX_INT16_T);
            ps->normFeatContext[i + shift] = (int32_t)tmp;
        }
    }
    
}
