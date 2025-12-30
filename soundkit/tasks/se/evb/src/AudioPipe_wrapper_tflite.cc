#include "arm_mve.h"
#include "def_nnse_params.h"
#include "mut_model_metadata.h"
#include "melSpecProc.h"
#include "mut_model_data.h"
#include "def_nn3_se.h"
#include "tflm_ns_model.h"
#include <stdint.h>
#include "AudioPipe_wrapper.h"

#include "feature_module.h"
#include "ns_ambiqsuite_harness.h"
#include "nn_speech.h"
#include "iir.h"
#include "third_party/ns_cmsis_nn/Include/arm_nnsupportfunctions.h"
extern int tflm_validator_model_init(ns_model_state_t *ms);
// Feature class instance
FeatureClass FEAT_INST;
IIR_CLASS dcrm_inst;

// TFLM Config
static ns_model_state_t tflm;

// TF Tensor Arena

#if (TFLM_MODEL_LOCATION == NS_AD_PSRAM)
    unsigned char *mut_model;
#endif

#if (TFLM_ARENA_LOCATION == NS_AD_PSRAM)
    static uint8_t *tensor_arena;
    static constexpr int kTensorArenaSize = 1024 * 1024 * 10; // 10MB
#else
    static constexpr int kTensorArenaSize = 1024 * TFLM_VALIDATOR_ARENA_SIZE;
    // #ifdef AM_PART_APOLLO3
    //     // Apollo3 doesn't have AM_SHARED_RW
    //     alignas(16) static uint8_t tensor_arena[kTensorArenaSize];
    // #else // not AM_PART_APOLLO3
        #if (TFLM_ARENA_LOCATION == NS_AD_SRAM)
            #ifdef keil6
            // Align to 16 bytes
            AM_SHARED_RW __attribute__((aligned(16))) static uint8_t tensor_arena[kTensorArenaSize];
            #else
            AM_SHARED_RW alignas(16) static uint8_t tensor_arena[kTensorArenaSize];
            #endif
        #else
            NS_PUT_IN_TCM alignas(16) static uint8_t tensor_arena[kTensorArenaSize];
        #endif
    // #endif
#endif

// Resource Variable Arena - always in TCM for now
static constexpr int kVarArenaSize = 4096;
    // 4 * (TFLM_VALIDATOR_MAX_RESOURCE_VARIABLES + 1) * sizeof(tflite::MicroResourceVariables);
alignas(16) static uint8_t var_arena[kVarArenaSize];
// Validator Stuff

volatile int example_status = 0; // Prevent the compiler from optimizing out while loops

extern const int16_t stft_win_coeff_w480_h160[];

int8_t num_lookeahead = NUM_LOOKAHEAD;
int32_t spec_buffer[514 * 4];
int16_t nn_input_dim;
int16_t nn_output_dim;
int AudioPipe_wrapper_init(void)
{ 
    FeatureClass_construct(
        &FEAT_INST,
        (const int32_t*) feature_mean_se,
        (const int32_t*) feature_stdR_se,
        FEATURE_QBIT,
        params_nn3_se.num_mfltrBank, // FEATURE_NUM_MFC
        params_nn3_se.winsize_stft, // FEATURE_WINSIZE,
        params_nn3_se.hopsize_stft, // FEATURE_HOPSIZE,
        params_nn3_se.fftsize, // FEATURE_FFTSIZE,
        params_nn3_se.pt_stft_win_coeff,
        params_nn3_se.p_melBanks,
        params_nn3_se.feature_type);

    IIR_CLASS_init(&dcrm_inst);
    
    // Initialize the model, get handle if successful
    
    tflm.runtime = TFLM;
    tflm.model_array = mut_model;
    tflm.arena = tensor_arena;
    tflm.arena_size = kTensorArenaSize;
    tflm.rv_arena = var_arena;
    tflm.rv_arena_size = kVarArenaSize;
    tflm.rv_count = TFLM_VALIDATOR_MAX_RESOURCE_VARIABLES;
    tflm.numInputTensors = 1;
    tflm.numOutputTensors = 1;

    int status = tflm_validator_model_init(&tflm); // model init with minimal defaults

    if (status == NS_STATUS_FAILURE) {
        while (1)
            example_status = NS_STATUS_INIT_FAILED; // hang
    }

    // Get data about input and output tensors
        // Get data about input and output tensors

    ns_model_state_t *pt_tflm = &tflm;
    int numInputs = pt_tflm->interpreter->inputs_size();
    int numOutputs = pt_tflm->interpreter->outputs_size();
    
    ns_lp_printf("Model has %d inputs and %d outputs\n", numInputs, numOutputs);
        
    for (int m = 0; m < numInputs; m++) {

        ns_lp_printf("Input tensor %d has %d bytes\n", m, pt_tflm->interpreter->input(m)->bytes);

        ns_lp_printf("input scale=%f\n", pt_tflm->interpreter->input(m)->params.scale);
        ns_lp_printf("input zero_point=%d\n", pt_tflm->interpreter->input(m)->params.zero_point);

        ns_lp_printf("input dims=%d\n", pt_tflm->interpreter->input(m)->dims->size);
        nn_input_dim = 1;
        for (int i = 0; i < pt_tflm->interpreter->input(m)->dims->size; i++) {
            nn_input_dim *= pt_tflm->interpreter->input(m)->dims->data[i];
            ns_lp_printf("input dim[%d]=%d\n", i, pt_tflm->interpreter->input(m)->dims->data[i]);
        }
    }
    
    for (int m = 0; m < numOutputs; m++) {
        ns_lp_printf("Output tensor %d has %d bytes\n", m, pt_tflm->interpreter->output(m)->bytes);
        nn_output_dim = 1;
        for (int i = 0; i < pt_tflm->interpreter->output(m)->dims->size; i++) {
            nn_output_dim *= pt_tflm->interpreter->output(m)->dims->data[i];
            ns_lp_printf("output dim[%d]=%d\n", i, pt_tflm->interpreter->output(m)->dims->data[i]);
        }
        // self->nn_dim_out = output_dim; // Set the number of output dimensions for the model
    }
    pt_tflm->interpreter->Reset();

    ns_lp_printf("Model initialized\n");
    return 0;
}

int AudioPipe_wrapper_reset(void)
{
    int32_t *pt_spec_buffer = spec_buffer;
    if (num_lookeahead > 0)
    {
        for (int i = 0; i < 514 * num_lookeahead; i++)
        {
            pt_spec_buffer[i] = 0;
        }
    }
    FeatureClass_setDefault(&FEAT_INST);
    IIR_CLASS_reset(&dcrm_inst);
    ns_model_state_t *pt_tflm = &tflm;
    pt_tflm->interpreter->Reset();
    return 0;
}
extern int16_t filter_banks_inv[];
int AudioPipe_wrapper_frameProc(
        int16_t *pcm_input,
        int16_t *pcm_output)
{
    /* feature extraction
    1. iir for dc remove
    2. melspectrogram
    */
    int fft_bins = (params_nn3_se.fftsize >> 1) + 1;
    int32_t *pt_spec = FEAT_INST.state_stftModule.spec;
    int32_t *pt_spec_buffer = spec_buffer;
    static int32_t tmp_v32[514];
    static int32_t mask_est_blk[514];
    
    int16_t *pt_tmp16 = (int16_t*) tmp_v32;
    // static int16_t tmp_16s[514]; // max output dim
    float scalar_norm;
    if (params_nn3_se.feature_type == feat_spec_erb)
        scalar_norm = 1.0 / (float) (1 << 21); // Q21, because erb feature is in Q21 x
    else
        scalar_norm = 1.0 / (float) (1 << FEATURE_QBIT);
    ns_model_state_t *pt_tflm = &tflm;
    int32_t gain= (int32_t) params_nn3_se.pre_gain_q1;
    for (int i = 0; i < params_nn3_se.hopsize_stft; i++)
    {
        int32_t tmp = (int32_t) pcm_input[i] * gain;
        pcm_input[i] = (int16_t) MIN(MAX((tmp >> 1), -32768), 32767); // Q1
    }

    IIR_CLASS_exec(&dcrm_inst, pt_tmp16, pcm_input, params_nn3_se.hopsize_stft);
    FeatureClass_execute(&FEAT_INST, pt_tmp16);

    int fft_bins_double = fft_bins * 2;
    // take out the oldest spec & cyclic insert the new one  at the end
    if (num_lookeahead > 0)
    {
        arm_memcpy_s8(
            (int8_t*) tmp_v32,
            (int8_t*) pt_spec_buffer,
            fft_bins_double * sizeof(int32_t));

        arm_memcpy_s8(
            (int8_t*) pt_spec_buffer,
            (int8_t*) (pt_spec_buffer + fft_bins_double),
            fft_bins_double * (num_lookeahead-1) * sizeof(int32_t));
        
        arm_memcpy_s8(
            (int8_t*) (pt_spec_buffer + fft_bins_double * (num_lookeahead-1)),
            (int8_t*) pt_spec,
            fft_bins_double * sizeof(int32_t));

        arm_memcpy_s8(
            (int8_t*) pt_spec,
            (int8_t*) tmp_v32,
            fft_bins_double * sizeof(int32_t));
    }
    int32_t *ptfeat = FEAT_INST.normFeatContext + params_nn3_se.num_mfltrBank * (FEATURE_CONTEXT-1);

    int input_idx=0;
    float32_t input_scale = pt_tflm->interpreter->input(input_idx)->params.scale;
    int input_zero_point = pt_tflm->interpreter->input(input_idx)->params.zero_point;

    for (int i =0; i < nn_input_dim; i++)
    {
        float32_t val = ((float32_t) ptfeat[i] ) * scalar_norm;
        int32_t input32 = (int32_t) ((float32_t) val / (float32_t) input_scale + (float32_t) input_zero_point);
        int16_t input = (int16_t) MAX(MIN(input32, 32767), -32768);
        pt_tflm->interpreter->input(input_idx)->data.i16[i] =  input;
    }

    TfLiteStatus invoke_status = tflm.interpreter->Invoke(); 
    
    if (invoke_status != kTfLiteOk) {
        while (1)
        {
            example_status = NS_STATUS_FAILURE; // invoke failed, so hang
        }
    }
    float32_t output_scale = tflm.model_output[0]->params.scale;
    int output_zero_point = tflm.model_output[0]->params.zero_point;
    
    for (int i = 0; i < nn_output_dim; i++) {
        float32_t out; 
        out = (float32_t) (tflm.model_output[0]->data.i16[i] - output_zero_point);
        out = out * output_scale;
        tmp_v32[i] = (int32_t)(out * 32768.0f); // q15
        // tmp_16s[i] = (int16_t) MAX(MIN(out_32s, 32767), -32768); // clamp to 16-bit range
    }


    int32_t *pt_out;
    static int32_t inputs_blk[129];
    if (params_nn3_se.feature_type == feat_spec_erb)
    {
        
        // real part
        for (int i = 0; i < params_nn3_se.num_mfltrBank; i++)
            inputs_blk[i] = tmp_v32[2*i];
        
        melSpecProc(
            inputs_blk, // input
            mask_est_blk, // output
            filter_banks_inv,
            fft_bins // num_fft_bins
        );
        for (int i = 0; i < 257; i++)
            ns_printf("%d, ", mask_est_blk[i]);
        ns_printf("\n");
        // imag part
        for (int i = 0; i < params_nn3_se.num_mfltrBank; i++)
            inputs_blk[i] = tmp_v32[2*i+1];
        
        melSpecProc(
            inputs_blk, // input
            mask_est_blk + fft_bins, // output
            filter_banks_inv,
            fft_bins // num_fft_bins
        );

        pt_out = mask_est_blk;
    }
    else
    {
        pt_out = tmp_v32;
    }
    // // ns_lp_printf("\n");
    // // // get the tf mask
    if (params_nn3_se.feature_type == feat_spec_erb)
    {
        se_post_proc_cmplx(
            &FEAT_INST,
            pt_out,
            pcm_output,
            0,
            NN_DIM_OUT);     
    }
    else
    {
        se_post_proc_real(
            &FEAT_INST,
            pt_out,
            pcm_output,
            0,
            NN_DIM_OUT);     
    }
    
    return 0;
}

