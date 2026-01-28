#include <stdint.h>
#include "ns_ambiqsuite_harness.h"
#include "feature_module.h"
#include "ambiq_nnsp_debug.h"
#include "nnsp_identification.h"
#include "nnid_class.h"
#include "PcmBufClass.h"
#include "nnidCntrlClass.h"
#include <math.h>
#include "ns_timer.h"
#include "AudioPipe_wrapper.h"

#include "vad_init.h"
#include "id_init.h"

#define MAX_ENROLL_PPLS 5
#define DIM_EMBD 64
#define REQUIRED_UTTERANCES_ENROLL 4
#define THRESH_NUM_VAD_TRIGGER 180


PcmBufClass pcmBuf_inst;
int16_t glob_th_prob = 0x7fff >> 1;
int16_t glob_count_trigger = 1;
// Model Stuff

AudioTaskClass audioTask_vad;
AudioTaskClass audioTask_id;

void nnidCntrlClass_resetPcmBufClass(nnidCntrlClass* pt_inst)
{
	PcmBufClass_reset(&pcmBuf_inst);
}

void nnidCntrlClass_init(nnidCntrlClass* pt_inst)
{
	// embds for all registered people
	static float16_t embds[MAX_ENROLL_PPLS* 64];
	pt_inst->pt_embds = embds;

	pt_inst->pt_vad = (void*) &audioTask_vad;
	pt_inst->pt_id 	= (void*) &audioTask_id;

	vad_init(pt_inst->pt_vad);
	AudioPipe_wrapper_init(pt_inst->pt_vad);

	id_init(pt_inst->pt_id);
	AudioPipe_wrapper_init(pt_inst->pt_id);
	

	// vad
	pt_inst->thresh_num_vad_trigger = THRESH_NUM_VAD_TRIGGER;
	
	// id
	pt_inst->required_utterances_enroll = REQUIRED_UTTERANCES_ENROLL;
	pt_inst->total_enroll_ppls = 0;
	// PCM_BUF init, reset
	pt_inst->pt_pcmBuf = (void*) &pcmBuf_inst;
	PcmBufClass_init(pt_inst->pt_pcmBuf);

}
void nnidCntrlClass_reset(nnidCntrlClass* pt_inst)
{
	// vad
	AudioPipe_wrapper_reset(pt_inst->pt_vad);
	pt_inst->count_vad_trigger = 0;

	// id
	AudioPipe_wrapper_reset(pt_inst->pt_id);
	pt_inst->acc_utterances_enroll = 0;

	// PCM_BUF reset
	PcmBufClass_reset(pt_inst->pt_pcmBuf);

	pt_inst->vad_onhold_frs = 20; // VAD on-hold time in frames
}

float16_t inner_product(
	float16_t *x,
	float16_t *y,
	int len_vec)
{
	float16_t acc = 0;
	for (int i = 0; i < len_vec; i++)
	{
		acc += x[i] * y[i];
	}
	return acc;
}


float16_t cosineSimilarity(
	float16_t *x,
	float16_t *y,
	float16_t eps,
	int len_vec)
{
	float16_t norm_x = inner_product(x, x, len_vec);
	float16_t norm_y = inner_product(y, y, len_vec);
	// ns_printf("norm_x=%f, norm_y=%f\n", norm_x, norm_y);
	// for (int i = 0; i < len_vec; i++)
	// {
	// 	ns_printf("x[%d]=%f, y[%d]=%f\n", i, x[i], i, y[i]);
	// }
	return inner_product(x, y, len_vec) / (sqrtf(norm_x) * sqrtf(norm_y) + eps);
}

void l2_normalization(
	float16_t *inputs,
	float16_t eps,
	int len_vec)
{
	float16_t norm = inner_product(inputs, inputs, len_vec);
	norm = 1.0f / (sqrtf(norm) + eps);
	for (int i = 0; i < len_vec; i++)
	{
		inputs[i] = norm * inputs[i];
	}
}

void norm_then_ave(
	float16_t *outputs,
	float16_t *inputs,
	int num_sents,
	int len_vec)
{
	float16_t acc;
	float16_t norm[REQUIRED_UTTERANCES_ENROLL];
	
	for (int i =  0; i < num_sents; i++)
	{
		norm[i] = 0;
		for (int j = 0; j < len_vec; j++)
		{
			acc = inputs[i*len_vec + j];
			norm[i] += acc * acc; 
		}
		norm[i] = 1.0f / sqrtf(norm[i]);
	}

	for (int i =  0; i < len_vec; i++)
	{
		acc = 0;
		for (int j = 0; j < num_sents; j++)
		{
			acc += norm[j] * ((float) inputs[j*len_vec + i]);
		}
		acc *= 0.25;
		outputs[i] = acc;
	}
}
extern volatile bool g_audioRecording;
int16_t nnidCntrlClass_exec(
			nnidCntrlClass* pt_inst,
			int16_t *rawPCM,
			float16_t *pt_corr)
{
	static float16_t embd_sentences[REQUIRED_UTTERANCES_ENROLL * DIM_EMBD];
	int16_t vad_detected = 0;
	int16_t is_detected = 0;
	int16_t *pt_outputs=rawPCM + 160;
	AudioTaskClass *pt_vad = (AudioTaskClass*) pt_inst->pt_vad;
	AudioTaskClass *pt_id = (AudioTaskClass*) pt_inst->pt_id;
	static float16_t nnout[DIM_EMBD];
	

	PcmBufClass_write(&pcmBuf_inst, rawPCM);
	AudioPipe_wrapper_frameProc(pt_vad, rawPCM, nnout);
	
	if (nnout[0] > nnout[1])
	{
		
		pt_inst->vad_onhold_frs -= 1;
		if (pt_inst->vad_onhold_frs < 0)
		{
			pt_inst->vad_onhold_frs = 0;
			vad_detected = 0; // detected
		}
		else
		{
			vad_detected = 1; // not detected
		}
	}
	else
	{
		pt_inst->vad_onhold_frs = 20;
		vad_detected = 1; // not detected
	}

	if (vad_detected)
	{
		for (int i = 0; i < 160 ; i++)
		{
			pt_outputs[i] = 32767;
		}
	}
	else
	{
		for (int i = 0; i < 160 ; i++)
		{
			pt_outputs[i] = 0;
		}
	}

	pt_inst->count_vad_trigger = (vad_detected) ? pt_inst->count_vad_trigger + 1 : 0;
	
	if (pt_inst->count_vad_trigger == pt_inst->thresh_num_vad_trigger)
	{
		g_audioRecording = false;
		ns_printf("VAD detected!\n");
		ns_printf("pt_inst->enroll_state=%d, pt_inst->id_enroll_ppl=%d, pt_inst->total_enroll_ppls=%d\n",
			pt_inst->enroll_state,
			pt_inst->id_enroll_ppl,
			pt_inst->total_enroll_ppls);
		

		if (pt_inst->enroll_state == enroll_phase)
		{
			ns_printf("Enrollment phase!\n");
		}
		else
		{
			ns_printf("Identification phase!\n");
		}

		PcmBufClass_readFrame_init(
			&pcmBuf_inst, 
			pt_inst->thresh_num_vad_trigger);
		
		for (int f = 0; f < pt_inst->thresh_num_vad_trigger; f++)
		{
			PcmBufClass_readFrame(
				&pcmBuf_inst,
				rawPCM);
			AudioPipe_wrapper_frameProc(pt_id, rawPCM, nnout);
		}

		if (pt_inst->enroll_state == enroll_phase)
		{
			for (int i = 0; i < DIM_EMBD; i++)
			{
				embd_sentences[pt_inst->acc_utterances_enroll * DIM_EMBD + i] = nnout[i];
			}
			pt_inst->acc_utterances_enroll += 1;
			ns_printf("acc_utterances_enroll=%d\n", pt_inst->acc_utterances_enroll);
			if (pt_inst->acc_utterances_enroll == pt_inst->required_utterances_enroll)
			{
				norm_then_ave(
					pt_inst->pt_embds + pt_inst->id_enroll_ppl * DIM_EMBD,
					embd_sentences,
					pt_inst->required_utterances_enroll,
					DIM_EMBD);

			}
			is_detected = -(0x7fff >> 1);
		}
		else
		{
			for (int i = 0; i < pt_inst->total_enroll_ppls; i++)
			{
				float16_t *pt_center= pt_inst->pt_embds + i * DIM_EMBD;
				pt_corr[i] = cosineSimilarity(
					pt_center,
					(float16_t *) nnout,
					1e-5f,
					DIM_EMBD);
				ns_printf("corr[%d]=%f\n", i, pt_corr[i]);
			}
			is_detected = 0x7fff >> 1;

		}

		g_audioRecording = true;
			
		// reset
		pt_inst->vad_onhold_frs = 20;
		AudioPipe_wrapper_reset(pt_vad);
		AudioPipe_wrapper_reset(pt_id);
		// reset VAD trigger counter
		pt_inst->count_vad_trigger = 0;

		// PCM_BUF reset
		PcmBufClass_reset(pt_inst->pt_pcmBuf);
		

	}
	return is_detected;
}