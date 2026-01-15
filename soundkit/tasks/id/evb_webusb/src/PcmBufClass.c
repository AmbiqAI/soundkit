#include <stdint.h>
#include "PcmBufClass.h"
#include "minmax.h"
//#include "nnCntrlClass.h"
#define NUM_FRS_VBUF  200
#define SAMPLES_FRM_NNCNTRL_CLASS 160
int16_t PCM_BUFFER[SAMPLES_FRM_NNCNTRL_CLASS * NUM_FRS_VBUF];

void PcmBufClass_init(PcmBufClass *pt_inst) {
    /*
    Initialize PcmBuffer class
    */
    pt_inst->pcm_buffer = PCM_BUFFER;
    pt_inst->num_frs = NUM_FRS_VBUF;
    pt_inst->smpls_per_fr = SAMPLES_FRM_NNCNTRL_CLASS;
}

void PcmBufClass_reset(PcmBufClass *pt_inst) {
    /*
    Reset PcmBuffer class
    */
    int i;
    for (i = 0; i < (pt_inst->num_frs * pt_inst->smpls_per_fr); i++)
        pt_inst->pcm_buffer[i] = 0;
    pt_inst->acc_frs_set = 0;
}

void PcmBufClass_write( 
        PcmBufClass *pt_inst, 
        int16_t *pcm_input) {
    /*
    Put data from microphone to pcm buffer
    
    Inputs:
            pt_inst   : instance pointer
            pcm_input : data from microphone
    */
    int i;
    int16_t *pt_dst;
    pt_dst = pt_inst->pcm_buffer + pt_inst->acc_frs_set * pt_inst->smpls_per_fr;
    for (i = 0; i < pt_inst->smpls_per_fr; i++)
    {
        *pt_dst++ = pcm_input[i];
    }
    pt_inst->acc_frs_set = (pt_inst->acc_frs_set + 1) % pt_inst->num_frs;
}

void PcmBufClass_readFrame_init(
        PcmBufClass *pt_inst, 
        int16_t num_frs_lookback) {
    /*
    Initialize read pointer to the pcm buffer
    Inputs:
            pt_inst          : instance pointer
            num_frs_lookback : number of frames to look back
    */
    pt_inst->id_start_fr = pt_inst->acc_frs_set - num_frs_lookback;
    if (pt_inst->id_start_fr < 0)
        pt_inst->id_start_fr += pt_inst->num_frs;
    
}

void PcmBufClass_readFrame(
        PcmBufClass *pt_inst, 
        int16_t *pcm_output) {
    /*
    Read data from pcm buffer
    Inputs:
            pt_inst     : instance pointer
            pcm_output  : output buffer
    */
    int16_t *pt_src = pt_inst->pcm_buffer + pt_inst->id_start_fr * pt_inst->smpls_per_fr;
    pt_inst->id_start_fr = (pt_inst->id_start_fr + 1) % pt_inst->num_frs;
    for (int i = 0; i < pt_inst->smpls_per_fr; i++)
    {
        pcm_output[i] = *pt_src++;
    }
}