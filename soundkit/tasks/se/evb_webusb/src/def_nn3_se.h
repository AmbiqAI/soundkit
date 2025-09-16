#ifndef __DEF_NN3_SE__
#define __DEF_NN3_SE__

#include <stdint.h>
#include "neural_nets.h"
#include "nn_speech.h"

extern const int32_t feature_mean_se[];
extern const int32_t feature_stdR_se[];

#define NUM_LOOKAHEAD 1
#define FEATURE_EXTRACTION 1
#define FEATURE_QBIT 8
extern PARAMS_NNSP params_nn3_se;

#endif  // __DEF_NN3_SE__
