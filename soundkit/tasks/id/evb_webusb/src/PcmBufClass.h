#ifndef __PCM_BUF_CLASS_H__
#define __PCM_BUF_CLASS_H__
#ifdef __cplusplus
extern "C"
{
#endif
#include <stdint.h>

// a block of pcm buffer
typedef struct 
{
    int16_t *pcm_buffer;
    int16_t acc_frs_set;
    int16_t id_start_fr;
    int16_t num_frs;
    int16_t smpls_per_fr;
}PcmBufClass;

void PcmBufClass_init(PcmBufClass *pt_inst);

void PcmBufClass_reset(PcmBufClass *pt_inst);

void PcmBufClass_write(PcmBufClass *pt_inst, int16_t *pcm_input);

void PcmBufClass_readFrame_init(PcmBufClass *pt_inst, int16_t num_frs_lookback);

void PcmBufClass_readFrame(
        PcmBufClass *pt_inst, 
        int16_t *pcm_output);
#ifdef __cplusplus
}
#endif
#endif