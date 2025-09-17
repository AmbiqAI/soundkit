#ifndef VAD_INIT_H
#define VAD_INIT_H

// #include "ns_model.h"


#ifdef __cplusplus
extern "C" {
#endif
#include "AudioPipe_wrapper.h"

void vad_init(AudioTaskClass *self);

#ifdef __cplusplus
}
#endif

#endif // VAD_INIT_H
