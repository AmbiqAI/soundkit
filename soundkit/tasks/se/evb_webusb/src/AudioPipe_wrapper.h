#ifndef __AUDIO_PIPE_WRAPPER_H__
#define __AUDIO_PIPE_WRAPPER_H__
#ifdef __cplusplus
extern "C" {
#endif
#include <stdint.h>
int AudioPipe_wrapper_init(void);
int AudioPipe_wrapper_reset(void);
int AudioPipe_wrapper_frameProc(
        int16_t *pcm_input,
        int16_t *pcm_output);
uint32_t AudioPipe_wrapper_get_feature_exec_cycles(void);
float AudioPipe_wrapper_get_feature_exec_mcps(void);
float AudioPipe_wrapper_get_feature_exec_avg_mcps(void);
#ifdef __cplusplus
}
#endif
#endif
