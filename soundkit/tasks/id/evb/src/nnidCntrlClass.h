#ifndef __NNID_CNTRL_CLASS__
#define __NNID_CNTRL_CLASS__
#ifdef __cplusplus
extern "C"
{
#endif
#include <stdint.h>
#include <arm_math.h>
typedef enum
{
	enroll_phase 	= 0,
	test_phase		= 1,
}enroll_state_T;

typedef struct
{
	void* pt_vad;
	void* pt_id;
	void *pt_pcmBuf; // PcmBufClass instance

	// vad
	int16_t thresh_num_vad_trigger; // parameter to trigger VAD

	int16_t count_vad_trigger;

	// id
	int8_t required_utterances_enroll; // total number of utterances to enroll a person
	int8_t acc_utterances_enroll;

	enroll_state_T enroll_state;
	float16_t *pt_embds;

	int8_t id_enroll_ppl;
	int16_t total_enroll_ppls; // total number of people to enroll

}nnidCntrlClass;

void nnidCntrlClass_reset(nnidCntrlClass* pt_inst);
void nnidCntrlClass_resetPcmBufClass(nnidCntrlClass* pt_inst);
void nnidCntrlClass_init(nnidCntrlClass *pt_inst);

int16_t nnidCntrlClass_exec(
	nnidCntrlClass* pt_inst,
	int16_t* rawPCM,
	float16_t* pt_corr,
	volatile bool *g_audioRecording);

void norm_then_ave(
	float16_t *outputs,
	float16_t *inputs,
	int num_sents,
	int len_vec);

float16_t inner_product(
	float16_t *x,
	float16_t *y,
	int len_vec);

float16_t cosineSimilarity(
	float16_t *x,
	float16_t *y,
	float16_t eps,
	int len_vec);

#ifdef __cplusplus
}
#endif
#endif