#ifndef __AUDIO_PIPE_WRAPPER_H__
#define __AUDIO_PIPE_WRAPPER_H__
#ifdef __cplusplus
extern "C" {
#endif
// #include "arm_mve.h"
#include <stdint.h>

// #include "arm_math_types.h"
typedef struct {
    void *pt_feat_inst; // FeatureClass
    void *pt_dcrm_inst; // IIR_CLASS
    void *pt_param; // PARAMS_NNSP
    const int32_t *feature_mean; // Mean for feature normalization
    const int32_t *feature_stdR; // Standard deviation for feature normalization


    void *pt_tflm; // TFLM model state
    int num_lookeahead; // Number of lookahead frames
    unsigned char *mut_model;

    int32_t tflm_validator_arena_size; // Size of the TFLM validator arena
    
    // model arena
    uint8_t *tensor_arena; // Pointer to the tensor arena
    int32_t tensor_arena_size; // Size of the tensor arena
    

    // resource variable arena
    uint8_t *var_arena; // Pointer to the resource variable arena
    int32_t var_arena_size; // Size of the resource variable arena


    int32_t max_resource_variables;
    
    int16_t nn_dim_out;

    int (*model_init)(void *ms);
    int (*model_reset)(void *ms);

    float reset_nn; // Continuous inference flag
    
} AudioTaskClass;


int AudioPipe_wrapper_init(AudioTaskClass *self);
int AudioPipe_wrapper_reset(AudioTaskClass *self);

int AudioPipe_wrapper_frameProc(
        AudioTaskClass *self,
        int16_t *pcm_input,
        void *pcm_output);
#ifdef __cplusplus
}
#endif
#endif
