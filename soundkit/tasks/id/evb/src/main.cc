/**
 * @file rpc_client.cc
 * @author Carlos Morales
 * @brief Generic Data Operations RPC Example
 * @version 0.1
 * @date 2022-08-26
 *
 * This example showcases the NS Generic Data RPC interface, which
 * is useful for sending data back and forth across the EVB's second
 * USB interface via remote procedure calls.
 *
 * Both remote procedure call directions (EVB to PC and PC to EVB) are
 * demonstrated, using audio as an example data source.
 *
 * @copyright Copyright (c) 2022
 *
 */

// #include "nnse_model.h"
// // #include "ns_model.h"

#include "ns_perf_profile.h"
#include "ns_ambiqsuite_harness.h"

#include "ns_core.h"
#include "ns_peripherals_button.h"
#include "ns_peripherals_power.h"
#include "ns_rpc_generic_data.h"
#include "def_nn1_nnvad.h"
#include "AudioPipe_wrapper.h"
#include "def_AudioSystem.h"
#include "third_party/ns_cmsis_nn/Include/arm_nnsupportfunctions.h"
#include "downsample.h"

#include "nn_speech.h"
#include "iir.h"
#include "feature_module.h"

#include "nnidCntrlClass.h"
#include "ns_energy_monitor.h"
#define RECORD_10S 1
#define AUDIO_ON 1
#define PERF_TEST 1

static uint32_t elapsedTime = 0;
ns_timer_config_t tickTimer = {
    .api = &ns_timer_V1_0_0,
    .timer = NS_TIMER_COUNTER,
    .enableInterrupt = false,
};
 
void tic() {
    uint32_t oldTime = elapsedTime;
    elapsedTime = ns_us_ticker_read(&tickTimer);
    if (elapsedTime == oldTime) {
        // We've saturated the timer, reset it
        ns_timer_clear(&tickTimer);
    }
}
uint32_t toc() { return ns_us_ticker_read(&tickTimer) - elapsedTime; }

extern const int16_t data_wav[];
volatile bool static g_audioReady = false;
volatile bool static g_audioRecording = false;

#if NUM_CHANNELS == 1
int16_t static audioDataBuffer[SAMPLES_IN_FRAME * NUM_CHANNELS * 2+20]; // incoming PCM audio data
#else
int32_t static audioDataBuffer[SAMPLES_IN_FRAME * NUM_CHANNELS * 2+1];
#endif

alignas(16) uint32_t static dmaBuffer[SAMPLES_IN_FRAME * NUM_CHANNELS * 2];     // DMA target
am_hal_audadc_sample_t static sLGSampleBuffer[SAMPLES_IN_FRAME * NUM_CHANNELS * 2]; // working buffer
                                                                                // used by AUDADC

#ifndef NS_AMBIQSUITE_VERSION_R4_1_0
am_hal_offset_cal_coeffs_array_t sOffsetCalib;
#endif

void audio_frame_callback(ns_audio_config_t *config, uint16_t bytesCollected) {
    if (g_audioRecording) 
    {
        if (g_audioReady) // g_audioReady should be false here
        {
            ns_lp_printf("Overflow!\n");
        }
        ns_audio_getPCM_v2(config, audioDataBuffer);
        g_audioReady = true;
    }
}

ns_audio_config_t audioConfig = {
    .api = &ns_audio_V2_1_0,
    .eAudioApiMode = NS_AUDIO_API_CALLBACK,
    .callback = audio_frame_callback,
    .audioBuffer = (void *) &audioDataBuffer,
    .eAudioSource = AUDIO_SOURCE,
    .sampleBuffer = dmaBuffer,
    .workingBuffer = sLGSampleBuffer,
    .numChannels = NUM_CHANNELS,
    .numSamples = SAMPLES_IN_FRAME,
    .sampleRate = SAMPLE_RATE,
    .audioSystemHandle = NULL, // filled in by init
    .bufferHandle = NULL,      // only for ringbuffer mode
#ifndef NS_AMBIQSUITE_VERSION_R4_1_0
    .sOffsetCalib = &sOffsetCalib,
#endif
};
// -- Audio Stuff Ends ----------------------

// RPC Stuff
#if (configAPPLICATION_ALLOCATED_HEAP == 1)
    // RPC uses malloc internally, so we need to declare it here
    #define RPCC_HEAP_SIZE (NS_RPC_MALLOC_SIZE_IN_K * 4 * 1024)
size_t ucHeapSize = RPCC_HEAP_SIZE;
uint8_t ucHeap[RPCC_HEAP_SIZE] __attribute__((aligned(4)));
#endif

#define MY_USB_RX_BUFSIZE 2048
#define MY_USB_TX_BUFSIZE 2048
static uint8_t my_cdc_rx_ff_buf[MY_USB_RX_BUFSIZE];
static uint8_t my_cdc_tx_ff_buf[MY_USB_TX_BUFSIZE];
// End RPC Stuff

int16_t buf_downsample[160];
DOWNSAMPLE_CLASS downsample_inst;
nnidCntrlClass nnidControl_inst;

typedef enum
{
	button_stop     = 0,
	button_start	= 1,
}button_state_T;


int main(void) {
    ns_core_config_t ns_core_cfg = {.api = &ns_core_V1_0_0};
    int16_t *pt_acc_utterances_enroll = audioDataBuffer + (LEN_STFT_HOP << 1);
    int16_t *pt_displayID = pt_acc_utterances_enroll + 1;
    int16_t *pt_corr_i16 = pt_displayID + 1;
    // ----- Non-RPC Init ------
    // -- These are needed for the demo, not directly related to RPC
    NS_TRY(ns_core_init(&ns_core_cfg), "Core init failed.\b");
    NS_TRY(ns_power_config(&ns_development_default), "Power Init Failed\n");
    // NS_TRY(ns_power_config(&ns_audio_default), "Power Init Failed\n");
    NS_TRY(ns_set_performance_mode(NS_MAXIMUM_PERF), "Set CPU Perf mode failed.");
    // ---

    ns_itm_printf_enable();
    ns_interrupt_master_enable();

    // -- Init the button handler, used in the example, not needed by RPC
    volatile int g_intButtonPressed = 0;
    ns_button_config_t button_config = {
        .api = &ns_button_V1_0_0,
        .button_0_enable = true,
        .button_1_enable = false,
        .button_0_flag = &g_intButtonPressed,
        .button_1_flag = NULL};
    NS_TRY(ns_peripheral_button_init(&button_config), "Button init failed\n");
#if AUDIO_ON==1 // only activate audio if streaming is enabled
    // -- Audio init
    NS_TRY(ns_audio_init(&audioConfig), "Audio Initialization Failed.\n");
    NS_TRY(ns_audio_set_gain(AM_HAL_PDM_GAIN_P195DB, AM_HAL_PDM_GAIN_P195DB), "Gain set failed.\n");
    // NS_TRY(ns_audio_set_gain(24, 24), "Audio gain set failed.\n"); // AUDADC gain
    NS_TRY(ns_start_audio(&audioConfig), "Audio Start Failed.\n");
#endif
    // Vars and init the RPC system - note this also inits the USB interface
    status stat;
    binary_t binaryBlock = {
        .data = (uint8_t *)audioDataBuffer, // point this to audio buffer
        .dataLength = sizeof(audioDataBuffer)};

#if NUM_CHANNELS == 1
    char msg_store[30] = "Audio16bPCM_to_WAV";
#else
    char msg_store[30] = "Audio32bPCM_to_WAV";
#endif

    char msg_compute[30] = "CalculateMFCC_Please";

    // Block sent to PC
    dataBlock outBlock = {
        .length = sizeof(audioDataBuffer),
        .dType = uint8_e,
        .description = msg_store,
        .cmd = write_cmd,
        .buffer = binaryBlock};

    // Block sent to PC for computation
    dataBlock computeBlock = {
        .length =  sizeof(audioDataBuffer),
        .dType = uint8_e,
        .description = msg_compute,
        .cmd = extract_cmd,
        .buffer = binaryBlock};
    dataBlock IsRecordBlock;
#ifdef NS_USB_PRESENT
    ns_printf("RPC over USB\n");
#endif

#ifndef NS_USB_PRESENT
    ns_uart_config_t rpcGenericUARTHandle = {
        .api = &ns_uart_V0_0_1,
        .uart_config = NULL,
        .rx_cb = NULL,
        .tx_cb = NULL,
        .tx_blocking = true,
        .rx_blocking = false};
#endif
    ns_rpc_config_t rpcConfig = {
        .api = &ns_rpc_gdo_V1_1_0,
        .mode = NS_RPC_GENERICDATA_CLIENT,
        .rx_buf = my_cdc_rx_ff_buf,
        .rx_bufLength = MY_USB_RX_BUFSIZE,
        .tx_buf = my_cdc_tx_ff_buf,
        .tx_bufLength = MY_USB_TX_BUFSIZE,
#ifndef NS_USB_PRESENT
        .uartHandle = (ns_uart_handle_t)&rpcGenericUARTHandle, // we temporarily set the uartHandle here to allow the user to set the blocking/nonblocking send/receive in the uart transport layer
#endif        
        .sendBlockToEVB_cb = NULL,
        .fetchBlockFromEVB_cb = NULL,
        .computeOnEVB_cb = NULL,
#ifdef NS_USB_PRESENT
        .transport = NS_RPC_TRANSPORT_USB};
#else
        .transport = NS_RPC_TRANSPORT_UART};
#endif
    // Result of computation
    dataBlock resultBlock;
    NS_TRY(ns_rpc_genericDataOperations_init(&rpcConfig), "RPC Init Failed\n"); // init RPC and USB


#if PERF_TEST==1
    ns_lp_printf("\n|------ MCPS Measurement: run 100 times of inference ------|\n");
    

    nnidCntrlClass_init(&nnidControl_inst);
    nnidCntrlClass_reset(&nnidControl_inst);
    if (params_nn1_nnvad.samplingRate == 8000)
    {
        // Initialize the downsample instance
        DOWNSAMPLE_CLASS_init(&downsample_inst);
    }
    
    int16_t *pcm_input = (int16_t*) audioDataBuffer;
    int16_t *pcm_output = audioDataBuffer + SAMPLES_IN_FRAME;
    
    static ns_perf_counters_t pp;
    ns_init_perf_profiler();
    ns_reset_perf_counters();
    ns_start_perf_profiler();
    float16_t corr[4];
    for (int i=0; i < 100; i++)
    {
        nnidCntrlClass_exec(&nnidControl_inst, pcm_input, corr, &g_audioRecording);
    }
    ns_printf("\n");
    ns_stop_perf_profiler();
    ns_capture_perf_profiler(&pp);
    ns_print_perf_profile(&pp);
    ns_lp_printf("cpu freq=%d\n", g_ui32TrimVer);
    ns_lp_printf("Model execution completed\n");
    ns_lp_printf("Output after execution\n");
    ns_lp_printf("|------End MCPS Measurement ------|\n\n");
    
    ns_printf("\nElapsedtime measurement\n");
    NS_TRY(ns_timer_init(&tickTimer), "Timer Init Failed\n");
    tic();
    for (int i=0; i < 100; i++)
    {
        nnidCntrlClass_exec(&nnidControl_inst, pcm_input, corr, &g_audioRecording);
    }
    elapsedTime = toc();
    ns_printf("Elapsed time: %d us\n", elapsedTime);
#endif // PERF_TEST

    // There is a chicken-and-egg thing involved in getting the RPC
    // started. The PC-side server cant start until the USB TTY interface
    // shows up as a device, and that doesn't happen until we start servicing
    // TinyUSB, and even then it doesn't happen immediately.
    //
    // To address this, we loop waiting for a button press, servicing
    // USB. This gives the user a chance to start the server then
    // pressing the button to let the EVB it is ready to start RPCing.

    ns_printf("Type $tools/python -m record_evb --tty <your tty>\n");

    ns_printf("Start the PC-side server, then press Button 0 to get started\n");
    while (g_intButtonPressed == 0) {
        ns_delay_us(1000);
    }

    // g_intButtonPressed is retired after this point.
    ns_printf("Starting remote procedure call demo.\n");

    // In the app loop we service USB and the RPC server while
    // we collect data and send it over the various RPC
    // interfaces. Any incoming RPC calls will result in calls to the
    // RPC handler functions defined above.
    int16_t detected = 0;
    while (1) 
    {
        g_audioRecording = false;
        g_intButtonPressed = 0;
        // ns_deep_sleep();
        // This infinite loop is checking whether the start button in PC side is pressed. 
        while (1)
        {
            // EVB received IsRecordBlock sent form PC
            ns_rpc_data_computeOnPC(&computeBlock, &IsRecordBlock); 
            if (IsRecordBlock.buffer.data[0]==button_start)
            {
                ns_printf("Start button pressed!   \n");
                nnidControl_inst.id_enroll_ppl=IsRecordBlock.buffer.data[1];
                nnidControl_inst.total_enroll_ppls=IsRecordBlock.buffer.data[2];
                nnidControl_inst.enroll_state = (enroll_state_T) IsRecordBlock.buffer.data[3];
                ns_printf("enroll_state=%d, id_enroll_ppl=%d, total_enroll_ppls=%d\n",
                    nnidControl_inst.enroll_state,
                    nnidControl_inst.id_enroll_ppl,
                    nnidControl_inst.total_enroll_ppls);
                
                // if button pressed, break the loop
                g_intButtonPressed = 1;
                ns_rpc_data_clientDoneWithBlockFromPC(&IsRecordBlock);
                nnidCntrlClass_reset(&nnidControl_inst);
                break;
            }
            ns_rpc_data_clientDoneWithBlockFromPC(&IsRecordBlock);
            am_hal_delay_us(20000); 
        }

        if ( (g_intButtonPressed == 1) && (!g_audioRecording) ) 
        {
            ns_lp_printf("\nYou'd pressed the button. Program start!\n");
            g_intButtonPressed = 0;
            g_audioRecording = true;
            am_hal_delay_us(10);   
            while (1)
            {   
                // ns_set_power_monitor_state(NS_DATA_COLLECTION);
                // ns_deep_sleep();
                if (g_audioReady) 
                {
                    // ns_printf("enroll_state=%d, id_enroll_ppl=%d, total_enroll_ppls=%d\n",
                    // nnidControl_inst.enroll_state,
                    // nnidControl_inst.id_enroll_ppl,
                    // nnidControl_inst.total_enroll_ppls);
                    // execution of each time frame data
                    detected = nnidCntrlClass_exec(
                        &nnidControl_inst,
                        pcm_input,
                        corr,
                        &g_audioRecording);
                    
                    // prepare information and send to PC side
                    *pt_acc_utterances_enroll = (int16_t) nnidControl_inst.acc_utterances_enroll;
                    *pt_displayID = 0;
                    if (nnidControl_inst.enroll_state == test_phase)
                    {
                        if (detected)
                        {
                            *pt_displayID = 1;
                            for (int i=0; i < nnidControl_inst.total_enroll_ppls; i++)
                                pt_corr_i16[i] = (int16_t) (corr[i] * 32768.0f);
                        }
                    }

                    int16_t *pt_debug = pcm_output;
                    for (int i = 0; i < LEN_STFT_HOP; i++)
                        *pt_debug++ = detected;

                    ns_rpc_data_sendBlockToPC(&outBlock); // send data to PC sice
                    ns_rpc_data_computeOnPC(&computeBlock, &IsRecordBlock); // receive data from PC side
                    // check if PC side asks for stopping the program
                    if (IsRecordBlock.buffer.data[0] == button_stop)
                    {
                        g_audioRecording = false;
                        g_audioReady = false;
                        g_intButtonPressed = 0;
                        ns_printf("Stop button pressed!   \n");
                        ns_rpc_data_clientDoneWithBlockFromPC(&IsRecordBlock);
                        break;
                    }
                    ns_rpc_data_clientDoneWithBlockFromPC(&IsRecordBlock);

                    g_audioReady = false;
                }
                
            }  // while(1)
            ns_lp_printf("\nPress button to start!\n");
        }
    } // while(1)

}
