#ifndef __DEF_NN3_KWS__
#define __DEF_NN3_KWS__

#include <stdint.h>
#include "neural_nets.h"
#include "nn_speech.h"

extern const int32_t feature_mean_kws[];
extern const int32_t feature_stdR_kws[];

#define NUM_LOOKAHEAD 0
#define FEATURE_EXTRACTION 1
#define FEATURE_QBIT 8
extern PARAMS_NNSP params_nn_kws;

#endif  // __DEF_NN3_SE__
