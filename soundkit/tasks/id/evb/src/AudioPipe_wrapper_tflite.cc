#include "arm_mve.h"
// #include "def_nnvad_params.h"
// #include "mut_model_metadata.h"
// #include "mut_model_data.h"
// #include "def_nn1_nnvad.h"
#include "tflm_ns_model.h"
#include <stdint.h>
#include "AudioPipe_wrapper.h"

#include "feature_module.h"
#include "ns_ambiqsuite_harness.h"
#include "nn_speech.h"
#include "iir.h"
#include "third_party/ns_cmsis_nn/Include/arm_nnsupportfunctions.h"
#include <math.h>
// extern int tflm_validator_model_init(ns_model_state_t *ms);
#define FEATURE_QBIT 8 // Q-bit for feature extraction
#define MAX_NN_DIM_OUT 1024 // Number of output classes for the NN model
// Feature class instance


volatile int example_status = 0; // Prevent the compiler from optimizing out while loops

// int8_t num_lookeahead = NUM_LOOKAHEAD;
// int32_t spec_buffer[514 * 4];

int nn_input_dim;
int nn_output_dim;

int AudioPipe_wrapper_init(AudioTaskClass *self)
{ 
    
    FeatureClass* pt_feat = (FeatureClass*) self->pt_feat_inst;
    IIR_CLASS* pt_dcrm = (IIR_CLASS*) self->pt_dcrm_inst;
    PARAMS_NNSP* pt_param = (PARAMS_NNSP*) self->pt_param;
    ns_model_state_t *pt_tflm = (ns_model_state_t *) self->pt_tflm;
    FeatureClass_construct(
        pt_feat,
        (const int32_t*) self->feature_mean,
        (const int32_t*) self->feature_stdR,
        FEATURE_QBIT,
        pt_param->num_mfltrBank, // FEATURE_NUM_MFC
        pt_param->winsize_stft, // FEATURE_WINSIZE,
        pt_param->hopsize_stft, // FEATURE_HOPSIZE,
        pt_param->fftsize, // FEATURE_FFTSIZE,
        pt_param->pt_stft_win_coeff,
        pt_param->p_melBanks);

    IIR_CLASS_init(pt_dcrm);
    
    // Initialize the model, get handle if successful

    pt_tflm->runtime = TFLM;
    
    pt_tflm->model_array = self->mut_model;
    
    // model arena
    pt_tflm->arena = self->tensor_arena;
    pt_tflm->arena_size = self->tensor_arena_size;
    
    // resource variables
    pt_tflm->rv_arena = self->var_arena;
    pt_tflm->rv_arena_size = self->var_arena_size;
    pt_tflm->rv_count = self->max_resource_variables;

    pt_tflm->numInputTensors = 1;
    pt_tflm->numOutputTensors = 1;

    int status = ((int (*)(ns_model_state_t *)) self->model_init)(pt_tflm); // model init with minimal defaults

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
        self->nn_dim_out = nn_output_dim; // Set the number of output dimensions for the model
    }

    ns_lp_printf("Model initialized\n");
    return 0;
}

int AudioPipe_wrapper_reset(AudioTaskClass *self)
{
    ns_model_state_t *pt_tflm = (ns_model_state_t *) self->pt_tflm;

    FeatureClass* pt_feat = (FeatureClass*) self->pt_feat_inst;
    IIR_CLASS* pt_dcrm = (IIR_CLASS*) self->pt_dcrm_inst;
    PARAMS_NNSP* pt_param = (PARAMS_NNSP*) self->pt_param;
    FeatureClass_setDefault(pt_feat);
    IIR_CLASS_reset(pt_dcrm);

    pt_tflm->interpreter->Reset(); // Reset all tensors to default values
    return 0;
}

int AudioPipe_wrapper_frameProc(
    AudioTaskClass *self,
        int16_t *pcm_input,
        void *output_)
{
    /* feature extraction
    1. iir for dc remove
    2. melspectrogram
    */
    float16_t *output = (float16_t *) output_;
    FeatureClass* pt_feat = (FeatureClass*) self->pt_feat_inst;
    IIR_CLASS* pt_dcrm = (IIR_CLASS*) self->pt_dcrm_inst;
    PARAMS_NNSP* pt_param = (PARAMS_NNSP*) self->pt_param;
    ns_model_state_t *pt_tflm = (ns_model_state_t *) self->pt_tflm;

    int32_t *pt_spec = pt_feat->state_stftModule.spec;
    // int32_t *pt_spec_buffer = spec_buffer;
    // int32_t tmp_spec[514];

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
        IIR_CLASS_exec(pt_dcrm, tmp_16s, pcm_input, pt_param->hopsize_stft);
        FeatureClass_execute(pt_feat, tmp_16s);
    }
    else
    {
        FeatureClass_execute(pt_feat, pcm_input);
    }
    

    int16_t *ptfeat = pt_feat->normFeatContext;

    int input_idx=0;
    float32_t input_scale = pt_tflm->interpreter->input(input_idx)->params.scale;
    int input_zero_point = pt_tflm->interpreter->input(input_idx)->params.zero_point;

    for (int i =0; i < pt_param->num_mfltrBank; i++)
    {
        float32_t val = ((float32_t) ptfeat[i] ) * scalar_norm;
        int16_t input = (int16_t) ((float32_t) val / (float32_t) input_scale + (float32_t) input_zero_point);
        pt_tflm->interpreter->input(input_idx)->data.i16[i] =  input;
    }
    TfLiteStatus invoke_status = pt_tflm->interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
        while (1)
        {
            example_status = NS_STATUS_FAILURE; // invoke failed, so hang
        }
    }
    float16_t output_scale = pt_tflm->interpreter->output(0)->params.scale;
    int output_zero_point = pt_tflm->interpreter->output(0)->params.zero_point;

    for (int i = 0; i < self->nn_dim_out; i++) {
        float16_t out; 
        out = (float16_t) (pt_tflm->interpreter->output(0)->data.i16[i] - output_zero_point);
        out = out * output_scale;
        output[i] = (float16_t) out;
    }

    return 0;
}

