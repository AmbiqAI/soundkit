#ifndef __DEF_NN3_VAD__
#define __DEF_NN3_VAD__

#include <stdint.h>
#include "neural_nets.h"
#include "nn_speech.h"

extern const int32_t feature_mean_vad[];
extern const int32_t feature_stdR_vad[];

#define NUM_LOOKAHEAD 0
extern PARAMS_NNSP params_nn1_nnvad;

#endif  // __DEF_NN3_SE__
