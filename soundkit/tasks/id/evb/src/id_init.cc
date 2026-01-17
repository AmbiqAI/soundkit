
#include "arm_mve.h"
#include "ns_ambiqsuite_harness.h"
#include "id_init.h"
#include "feature_module.h"

#include "iir.h"
#include "def_nn2_nnid.h"
#include "tflm_ns_model.h"
#include "mut_model_data.h"
#include "mut_model_metadata.h"


FeatureClass FEAT_INST_ID;
IIR_CLASS dcrm_inst_id;
PARAMS_NNSP *pt_param_id = &params_def_nn2_nnid;
static ns_model_state_t tflm_id;

#if (TFLM_MODEL_LOCATION == NS_AD_PSRAM)
    unsigned char *mut_model;
#endif

#if (TFLM_ARENA_LOCATION == NS_AD_PSRAM)
    static uint8_t *tensor_arena_id;
    static constexpr int kTensorArenaSize_id = 1024 * 1024 * 10; // 10MB
#else
    static constexpr int kTensorArenaSize_id = 1024 * TFLM_VALIDATOR_ARENA_SIZE;
    #if (TFLM_ARENA_LOCATION == NS_AD_SRAM)
        #ifdef keil6
            AM_SHARED_RW __attribute__((aligned(16))) static uint8_t tensor_arena_id[kTensorArenaSize_id];
        #else
            AM_SHARED_RW alignas(16) static uint8_t tensor_arena_id[kTensorArenaSize_id];
        #endif
    #else
        NS_PUT_IN_TCM alignas(16) static uint8_t tensor_arena_id[kTensorArenaSize_id];
    #endif
#endif

static constexpr int kVarArenaSize_id =
    4 * (TFLM_VALIDATOR_MAX_RESOURCE_VARIABLES + 1) * sizeof(tflite::MicroResourceVariables);

alignas(16) static uint8_t var_arena_id[kVarArenaSize_id];

// AudioTaskClass audioTask_id;
extern int tflm_validator_model_init(ns_model_state_t *ms);
void id_init(AudioTaskClass *self) {
    self->pt_feat_inst = &FEAT_INST_ID;
    self->pt_dcrm_inst = &dcrm_inst_id;
    self->pt_param = pt_param_id;
    self->feature_mean = feature_mean_id;
    self->feature_stdR = feature_stdR_id;
    self->pt_tflm = (void *)&tflm_id;
    self->num_lookeahead = NUM_LOOKAHEAD;
    self->mut_model = mut_model;
    self->tflm_validator_arena_size = TFLM_VALIDATOR_ARENA_SIZE;
    self->tensor_arena = tensor_arena_id;
    self->tensor_arena_size = kTensorArenaSize_id;
    self->var_arena = var_arena_id;
    self->var_arena_size = kVarArenaSize_id;
    self->max_resource_variables = TFLM_VALIDATOR_MAX_RESOURCE_VARIABLES;
    self->nn_dim_out = 64;
    self->model_init = (int (*)(void *)) tflm_validator_model_init;
}
