
#include "arm_mve.h"
#include "ns_ambiqsuite_harness.h"
#include "vad_init.h"
#include "feature_module.h"

#include "iir.h"
#include "def_nn1_nnvad.h"
#include "tflm_ns_model.h"
#include "mut_model_data_vad.h"
#include "mut_model_metadata_vad.h"


FeatureClass FEAT_INST_VAD;
IIR_CLASS dcrm_inst_vad;
PARAMS_NNSP *pt_param_vad = &params_nn1_nnvad;
static ns_model_state_t tflm_vad;

#if (TFLM_MODEL_LOCATION == NS_AD_PSRAM)
    unsigned char *mut_model;
#endif

#if (TFLM_ARENA_LOCATION == NS_AD_PSRAM)
    static uint8_t *tensor_arena_vad;
    static constexpr int kTensorArenaSize_vad = 1024 * 1024 * 10; // 10MB
#else
    static constexpr int kTensorArenaSize_vad = 1024 * TFLM_VALIDATOR_ARENA_SIZE_VAD;
    #if (TFLM_ARENA_LOCATION == NS_AD_SRAM)
        #ifdef keil6
            AM_SHARED_RW __attribute__((aligned(16))) static uint8_t tensor_arena_vad[kTensorArenaSize_vad];
        #else
            AM_SHARED_RW alignas(16) static uint8_t tensor_arena_vad[kTensorArenaSize_vad];
        #endif
    #else
        NS_PUT_IN_TCM alignas(16) static uint8_t tensor_arena_vad[kTensorArenaSize_vad];
    #endif
#endif

static constexpr int kVarArenaSize_vad =
    4 * (TFLM_VALIDATOR_MAX_RESOURCE_VARIABLES_VAD + 1) * sizeof(tflite::MicroResourceVariables);

alignas(16) static uint8_t var_arena_vad[kVarArenaSize_vad];

// AudioTaskClass audioTask_vad;
extern int tflm_vad_validator_model_init(ns_model_state_t *ms);
void vad_init(AudioTaskClass *self) {
    self->pt_feat_inst = &FEAT_INST_VAD;
    self->pt_dcrm_inst = &dcrm_inst_vad;
    self->pt_param = pt_param_vad;
    self->feature_mean = feature_mean_vad;
    self->feature_stdR = feature_stdR_vad;
    self->pt_tflm = (void *)&tflm_vad;
    self->num_lookeahead = NUM_LOOKAHEAD_VAD;
    self->mut_model = mut_model_vad;
    self->tflm_validator_arena_size = TFLM_VALIDATOR_ARENA_SIZE_VAD;
    self->tensor_arena = tensor_arena_vad;
    self->tensor_arena_size = kTensorArenaSize_vad;
    self->var_arena = var_arena_vad;
    self->var_arena_size = kVarArenaSize_vad;
    self->max_resource_variables = TFLM_VALIDATOR_MAX_RESOURCE_VARIABLES_VAD;
    self->nn_dim_out = 2;
    self->model_init = (int (*)(void *)) tflm_vad_validator_model_init;
}
