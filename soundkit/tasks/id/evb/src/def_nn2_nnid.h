#ifndef __DEF_NN3_ID__
#define __DEF_NN3_ID__

#include <stdint.h>
#include "neural_nets.h"
#include "nn_speech.h"

extern const int32_t feature_mean_id[];
extern const int32_t feature_stdR_id[];

#define NUM_LOOKAHEAD 0
#define FEATURE_EXTRACTION 1
#define FEATURE_QBIT 8
extern PARAMS_NNSP params_def_nn2_nnid;

#endif  // __DEF_NN3_SE__
