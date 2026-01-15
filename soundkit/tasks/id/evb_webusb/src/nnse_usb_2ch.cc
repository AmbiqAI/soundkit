#include <stdint.h>

// 1. Include ARM MVE and Math headers FIRST and strictly OUTSIDE extern "C"
//    These headers use C++ overloading (polymorphism) which breaks inside extern "C".
#include <arm_mve.h> 
#include <arm_math.h> // Move this up here to let it see arm_mve.h is already included

// 2. Standard & SDK Includes
#include "am_util_stdio.h"
#include "ns_peripherals_button.h"
#include "ns_peripherals_power.h"
#include "ns_ambiqsuite_harness.h"
#include "ns_audio.h"
#include "ambiq_nnsp_const.h"
#include "ns_timer.h"
#include "ns_energy_monitor.h"
#include "nn_speech.h"
#include "nnidCntrlClass.h"
#include "ns_perf_profile.h"
#include "ns_usb.h"
#include "ae_api.h"
#include "FreeRTOS.h"
#include "task.h"
#include "AudioPipe_wrapper.h"
#include "def_AudioSystem.h"
#include "tusb.h" 

// --- CONFIGURATION ---
#define NUM_CHANNELS 1
#define MAX_ENCODED_LEN 1300 
#define XBUFSIZE 2560 

// --- GLOBAL VARIABLES ---
alignas(16) int16_t static g_in16AudioDataBuffer[LEN_STFT_HOP << 1];
alignas(16) int16_t static tempSEBuffer[LEN_STFT_HOP]; 
alignas(16) uint32_t static audadcSampleBuffer[(LEN_STFT_HOP << 1) + 3];
alignas(16) int16_t static xmitBuffer[XBUFSIZE];

volatile uint16_t xmitWritePtr = 0;
volatile uint16_t xmitReadPtr = 0;
volatile uint16_t xmitAvailable = 0;

int volatile g_intButtonPressed = 0;
// bool volatile static g_audioRecording = false;
// bool volatile static g_audioReady = false;
// In your .cpp file
#include <stdbool.h>

extern "C" {
    volatile bool g_audioRecording = false;
    volatile bool g_audioReady = false;
}
// *** NEW: COMMAND FLAGS ***
volatile bool g_newCommand = false;
volatile uint8_t g_cmdMode = 0;      
volatile uint8_t g_cmdID = 0;        
volatile uint8_t g_totalEnrolled = 0;
volatile uint8_t g_id_enroll_ppl = 0;
volatile uint8_t g_is_recording = 0;

// --- USB DATA STRUCTURE ---
typedef struct __attribute__((packed)) usb_data { 
    uint8_t type;
    uint8_t length;
    uint8_t platform;
    uint8_t padding;
    uint8_t data[MAX_ENCODED_LEN];
} usb_data_t;

alignas(16) usb_data_t data;

// --- TASKS & HANDLES ---
TaskHandle_t audio_task_handle;  
TaskHandle_t encode_task_handle; 
TaskHandle_t my_xSetupTask;      

// --- BUTTON CONFIG ---
ns_button_config_t button_config_nnsp = {
    .api = &ns_button_V1_0_0,
    .button_0_enable = true,
    .button_1_enable = false,
    .button_0_flag = &g_intButtonPressed,
    .button_1_flag = NULL
};

// --- USB BUFFERS ---
#define MY_RX_BUFSIZE 4096
#define MY_TX_BUFSIZE 4096
static uint8_t my_rx_ff_buf[MY_RX_BUFSIZE] __attribute__((aligned(16)));
static uint8_t my_tx_ff_buf[MY_TX_BUFSIZE] __attribute__((aligned(16)));

static ns_tusb_desc_webusb_url_t webusb_url;
static ns_usb_config_t webUsbConfig = {
    .api = &ns_usb_V1_0_0,
    .deviceType = NS_USB_VENDOR_DEVICE,
    .desc_url = &webusb_url 
};

const ns_power_config_t ns_power_usb = {
    .api = &ns_power_V1_0_0,
    .eAIPowerMode = NS_MAXIMUM_PERF,
    .bNeedAudAdc = false,
    .bNeedSharedSRAM = true,
    .bNeedCrypto = false,
    .bNeedBluetooth = false,
    .bNeedUSB = true,
    .bNeedIOM = false,
    .bNeedAlternativeUART = false,
    .b128kTCM = false,
    .bEnableTempCo = false,
    .bNeedITM = true,
    .bNeedXtal = true
};

// --- AUDIO CALLBACK ---
void audio_frame_callback(ns_audio_config_t *config, uint16_t bytesCollected) {
    if (g_audioRecording) {
        ns_audio_getPCM_v2(config, g_in16AudioDataBuffer);
        g_audioReady = true;
    }
}

ns_audio_config_t audio_config = {
    .api = &ns_audio_V2_1_0,
    .eAudioApiMode = NS_AUDIO_API_CALLBACK,
    .callback = audio_frame_callback,
    .audioBuffer = (void *)&g_in16AudioDataBuffer,
    .eAudioSource = NS_AUDIO_SOURCE_PDM,
    .sampleBuffer = audadcSampleBuffer,
    .workingBuffer = NULL,
    .numChannels = NUM_CHANNELS,
    .numSamples = LEN_STFT_HOP,
    .sampleRate = SAMPLING_RATE,
    .audioSystemHandle = NULL, 
    .bufferHandle = NULL
};

typedef enum {
    enroll_mode_pc = 1,
    test_mode_pc   = 2,
} command_mode_T;


nnidCntrlClass nnidControl_inst;

// Structure to hold metadata
typedef struct {
    int8_t acc_utterances_enroll;
    int8_t displayID;
    int8_t corr[5];
} packet_meta_t;

volatile packet_meta_t g_packetMeta = {0};


void audioTask(void *pvParameters) {
    static int16_t rawPCM[LEN_STFT_HOP];
    float16_t corr[5];
    
    while (1) {
        // 1. Always process incoming commands from the Hijacker/GUI
        if (g_newCommand) {
            g_newCommand = false;
            
            if (g_is_recording) {
                ns_lp_printf("GUI Command: START Recording (Mode: %d)\n", g_cmdMode);
                g_audioRecording = true;
                // Reset internal algos if needed
                nnidCntrlClass_reset(&nnidControl_inst); 
            } else {
                ns_lp_printf("GUI Command: STOP Recording\n");
                g_audioRecording = false;
                g_audioReady = false; // Flush pending frames
            }
        }

        // 2. Only fill the transmission buffer if the GUI has enabled recording
        if (g_audioReady && g_audioRecording) { 
            // Optional: Run your AI/DSP here
            // detected = nnidCntrlClass_exec(&nnidControl_inst, g_in16AudioDataBuffer, corr);

            taskENTER_CRITICAL();
            for (int i = 0; i < LEN_STFT_HOP; i++) {
                rawPCM[i] = g_in16AudioDataBuffer[i];
            }
            taskEXIT_CRITICAL();
            int16_t detected = nnidCntrlClass_exec(
                &nnidControl_inst,
                rawPCM, corr);
            
            g_packetMeta.acc_utterances_enroll = (int8_t) nnidControl_inst.acc_utterances_enroll;

            if (nnidControl_inst.enroll_state == test_phase) {
                if (detected) {
                    g_packetMeta.displayID = 1; 
                    for (int i = 0; i < nnidControl_inst.total_enroll_ppls; i++)
                        g_packetMeta.corr[i] = (int8_t) (corr[i] * 128);
                }
            }
  
            taskENTER_CRITICAL();
            for(int i=0; i<LEN_STFT_HOP; i++){
                xmitBuffer[xmitWritePtr] = g_in16AudioDataBuffer[i];
                xmitWritePtr = (xmitWritePtr + 1) % XBUFSIZE;
                
                // Use a default value or AI output for the second channel
                xmitBuffer[xmitWritePtr] = detected; 
                xmitWritePtr = (xmitWritePtr + 1) % XBUFSIZE;
            }
            xmitAvailable += (LEN_STFT_HOP * 2); 
            taskEXIT_CRITICAL();
            
            g_audioReady = false;
        } else if (g_audioReady && !g_audioRecording) {
            // Drop frames if GUI hasn't pressed 'start'
            g_audioReady = false;
        }

        vTaskDelay(pdMS_TO_TICKS(1)); 
    }
}

void encodeAndXferTask(void *pvParameters) {
    const int SAMPLES_TO_SEND = 640; 
    const int BYTES_TO_SEND = SAMPLES_TO_SEND * sizeof(int16_t) + 20;

    while (1) {
        if (0) {
            vTaskDelay(pdMS_TO_TICKS(100));
        }
        else
        {
            if (xmitAvailable >= SAMPLES_TO_SEND) {
                int16_t* pDest = (int16_t*) data.data;
                for(int i=0; i<SAMPLES_TO_SEND; i++){
                    pDest[i] = xmitBuffer[xmitReadPtr];
                    xmitReadPtr = (xmitReadPtr + 1) % XBUFSIZE;
                }

                taskENTER_CRITICAL();
                xmitAvailable -= SAMPLES_TO_SEND;
                
                int8_t *pMeta = (int8_t *) data.data; 
                pMeta[1280] = g_packetMeta.acc_utterances_enroll;
                pMeta[1281] = g_packetMeta.displayID;

                for(int i=0; i<nnidControl_inst.total_enroll_ppls; i++) { // 1282,..., 1286
                    pMeta[1282 + i] = g_packetMeta.corr[i];
                }
                for (int i = nnidControl_inst.total_enroll_ppls; i < 5; i++) {
                    pMeta[1282 + i] = 0; 
                }

                g_packetMeta.displayID = 0;
                taskEXIT_CRITICAL();

                data.type = 0x04;
                data.length = 0; 
                data.padding = 0;
                webusb_send_data((uint8_t *)&data, 4 + BYTES_TO_SEND);

            }
            vTaskDelay(pdMS_TO_TICKS(1));
        }

        
    }
}

// *** THE HIJACKERS (EXTERN C) ***
extern "C" {
    
    void process_hijack_buffer(uint8_t* buf, uint32_t count) {
        if (count >= 7) {
            g_cmdMode = buf[4];
            
            g_is_recording = 1;
            if (g_cmdMode == 0) {
                g_is_recording = 0;
            }
            
            if (g_cmdMode == enroll_mode_pc)
                nnidControl_inst.enroll_state = enroll_phase;
            else
                nnidControl_inst.enroll_state = test_phase;
            if (g_cmdMode==enroll_mode_pc)
            {
                ns_printf("id = %d: \n", buf[5]);
                g_cmdID   = buf[5]; // 0-indexed
            }
            nnidControl_inst.id_enroll_ppl = g_cmdID;

            g_totalEnrolled = buf[6];
            nnidControl_inst.total_enroll_ppls = g_totalEnrolled;
            
            g_newCommand = true;
        }

        for (int i = 0; i < count; i++)
            ns_lp_printf("%02x ", buf[i]);
        ns_lp_printf("\n");
    }

    void tud_cdc_rx_cb(uint8_t itf) {
        if (tud_cdc_n_available(itf)) {
            uint8_t buf[64];
            uint32_t count = tud_cdc_n_read(itf, buf, sizeof(buf));
            process_hijack_buffer(buf, count);
        }
    }

    void tud_vendor_rx_cb(uint8_t itf) {
        if (tud_vendor_n_available(itf)) {
            uint8_t buf[64];
            uint32_t count = tud_vendor_n_read(itf, buf, sizeof(buf));
            ns_lp_printf("RX: %d bytes (Mode:%d, ID:%d, Total enrolled:%d)\n", count, buf[4], buf[5], buf[6]);
            process_hijack_buffer(buf, count);
        }
    }
}

void setup_task(void *pvParameters) {
    ns_lp_printf("Setting up Tasks...\n");
    xTaskCreate(audioTask, "AudioTask", 8912, 0, 3, &audio_task_handle);
    xTaskCreate(encodeAndXferTask, "SenderTask", 2048, 0, 3, &encode_task_handle);
    vTaskSuspend(NULL);
    while (1);
}

int main(void) {
    usb_handle_t usb_handle = NULL;
    ns_core_config_t ns_core_cfg = {.api = &ns_core_V1_0_0};
    NS_TRY(ns_core_init(&ns_core_cfg), "Core Init Failed");
    ns_power_config(&ns_power_usb);
    ns_itm_printf_enable();
    ns_interrupt_master_enable();
    
    NS_TRY(ns_audio_init(&audio_config), "Audio Init Failed");
    NS_TRY(ns_audio_set_gain(AM_HAL_PDM_GAIN_P195DB, AM_HAL_PDM_GAIN_P195DB), "Gain Failed");
    NS_TRY(ns_start_audio(&audio_config), "Audio Start Failed");
    ns_peripheral_button_init(&button_config_nnsp);

    strcpy(webusb_url.url, "ambiqai.github.io/web-ble-dashboards/nnse-usb/");
    webusb_url.bDescriptorType = 3;
    webusb_url.bScheme = 1;
    webusb_url.bLength = 3 + sizeof(webusb_url.url) - 1;

    webUsbConfig.rx_buffer = my_rx_ff_buf;
    webUsbConfig.rx_bufferLength = MY_RX_BUFSIZE;
    webUsbConfig.tx_buffer = my_tx_ff_buf;
    webUsbConfig.tx_bufferLength = MY_TX_BUFSIZE;
    webUsbConfig.rx_cb = NULL; 

    NS_TRY(ns_usb_init(&webUsbConfig, &usb_handle), "USB Init Failed");
    ns_lp_printf("USB Ready. Hijackers Armed.\n");

    nnidCntrlClass_init(&nnidControl_inst);
    nnidCntrlClass_reset(&nnidControl_inst);

    xTaskCreate(setup_task, "Setup", 512, 0, 1, &my_xSetupTask);

    vTaskStartScheduler();
    while (1) {};
}