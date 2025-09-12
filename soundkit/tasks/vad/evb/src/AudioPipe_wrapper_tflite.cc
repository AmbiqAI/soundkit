#include "arm_mve.h"
#include "def_nnvad_params.h"
#include "mut_model_metadata.h"
#include "mut_model_data.h"
#include "def_nn1_nnvad.h"
#include "tflm_ns_model.h"
#include <stdint.h>
#include "AudioPipe_wrapper.h"

#include "feature_module.h"
#include "ns_ambiqsuite_harness.h"
#include "nn_speech.h"
#include "iir.h"
#include "third_party/ns_cmsis_nn/Include/arm_nnsupportfunctions.h"
#include <math.h>
extern int tflm_validator_model_init(ns_model_state_t *ms);
#define FEATURE_QBIT 8 // Q-bit for feature extraction
#define NN_DIM_OUT 2 // Number of output classes for the NN model
// Feature class instance
FeatureClass FEAT_INST;
IIR_CLASS dcrm_inst;
PARAMS_NNSP *pt_param = &params_nn1_nnvad;
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

int8_t num_lookeahead = NUM_LOOKAHEAD;
int32_t spec_buffer[514 * 4];

int16_t vad_trigger_counts=0;

int16_t nn_input_dim;
int16_t nn_output_dim;
int AudioPipe_wrapper_init(void)
{ 
    ns_model_state_t *pt_tflm = &tflm;
    
    FeatureClass_construct(
        &FEAT_INST,
        (const int32_t*) feature_mean_vad,
        (const int32_t*) feature_stdR_vad,
        FEATURE_QBIT,
        pt_param->num_mfltrBank, // FEATURE_NUM_MFC
        pt_param->winsize_stft, // FEATURE_WINSIZE,
        pt_param->hopsize_stft, // FEATURE_HOPSIZE,
        pt_param->fftsize, // FEATURE_FFTSIZE,
        pt_param->pt_stft_win_coeff,
        pt_param->p_melBanks);

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
    return 0;
}

int AudioPipe_wrapper_reset(void)
{
    int32_t *pt_spec_buffer = spec_buffer;
    ns_model_state_t *pt_tflm = &tflm;
    if (num_lookeahead > 0)
    {
        for (int i = 0; i < (pt_param->fftsize+2) * num_lookeahead; i++)
        {
            pt_spec_buffer[i] = 0;
        }
    }
    // tflm.interpreter->Reset();
    FeatureClass_setDefault(&FEAT_INST);
    IIR_CLASS_reset(&dcrm_inst);
    pt_tflm->interpreter->Reset();
    vad_trigger_counts = 0;
    return 0;
}

int AudioPipe_wrapper_frameProc(
        int16_t *pcm_input,
        int16_t *pcm_output)
{
    /* feature extraction
    1. iir for dc remove
    2. melspectrogram
    */
    ns_model_state_t *pt_tflm = &tflm;
    float32_t outputs[NN_DIM_OUT];
    int32_t *pt_spec = FEAT_INST.state_stftModule.spec;
    int32_t *pt_spec_buffer = spec_buffer;
    int32_t tmp_spec[514];

    static int16_t tmp_16s[300];
    static float scalar_norm = 1.0 / (float) (1 << FEATURE_QBIT);

    int32_t gain= (int32_t) pt_param->pre_gain_q1;
    
    if (gain != 2)
    {
        for (int i = 0; i < pt_param->hopsize_stft; i++)
        {
            int32_t tmp = (int32_t) pcm_input[i] * gain;
            pcm_input[i] = (int16_t) MIN(MAX((tmp >> 1), -32768), 32767);
        }

    }

    if (pt_param->is_dcrm)
    {
        IIR_CLASS_exec(&dcrm_inst, tmp_16s, pcm_input, pt_param->hopsize_stft);
        FeatureClass_execute(&FEAT_INST, tmp_16s);
    }
    else
    {
        FeatureClass_execute(&FEAT_INST, pcm_input);
    }
    
    // move pt_spec to pt_spec_buffer
    if (num_lookeahead > 0)
    {
        arm_memcpy_s8(
            (int8_t*) tmp_spec,
            (int8_t*) pt_spec,
             (pt_param->fftsize+2)  * sizeof(int32_t));

        arm_memcpy_s8(
            (int8_t*) pt_spec_buffer,
            (int8_t*) (pt_spec_buffer + (pt_param->fftsize+2) ),
            (pt_param->fftsize+2)  * (num_lookeahead-1) * sizeof(int32_t));
        
        arm_memcpy_s8(
            (int8_t*) (pt_spec_buffer + (pt_param->fftsize+2)  * (num_lookeahead-1)),
            (int8_t*) pt_spec,
            (pt_param->fftsize+2)  * sizeof(int32_t));

        arm_memcpy_s8(
            (int8_t*) pt_spec,
            (int8_t*) tmp_spec,
            (pt_param->fftsize+2)  * sizeof(int32_t));
    }
    int16_t *ptfeat = FEAT_INST.normFeatContext;


    int input_idx=0;
    float32_t input_scale = pt_tflm->interpreter->input(input_idx)->params.scale;
    int input_zero_point = pt_tflm->interpreter->input(input_idx)->params.zero_point;

    for (int i =0; i < nn_input_dim; i++)
    {
        float32_t val = ((float32_t) ptfeat[i] ) * scalar_norm;
        int16_t input = (int16_t) ((float32_t) val / (float32_t) input_scale + (float32_t) input_zero_point);
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

    float32_t den=0.0f;
    for (int i = 0; i < nn_output_dim; i++) {
        float32_t out; 
        out = (float32_t) (tflm.model_output[0]->data.i16[i] - output_zero_point);
        out = out * output_scale;
        outputs[i] = exp(out);
        den += outputs[i];
    }

    int16_t trigger = 0;
    outputs[1] /= den;
    if (outputs[1] >= 0.5f) {
        trigger = ((int16_t) 32767) >> 1; // trigger
        vad_trigger_counts++;
    }
    else {
        trigger = 0; // no trigger
        vad_trigger_counts=0;
    }

    if (vad_trigger_counts == 180) {
        // AudioPipe_wrapper_reset(); // Reset the vad trigger counts
        // trigger=-32768 >> 1;
        vad_trigger_counts=0; // Reset the vad trigger counts
    }

    
    for (int i = 0; i < 160; i++) {
        pcm_output[i] = (int16_t) trigger;
    }

    return 0;
}

