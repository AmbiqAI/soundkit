// #include "tflite.h"
#include <stdint.h>

#include "am_util_stdio.h"
#include "ns_peripherals_button.h"
#include "ns_peripherals_power.h"
#include "ns_ambiqsuite_harness.h"
#include "ns_audio.h"
#include "ambiq_nnsp_const.h"
#include "ns_timer.h"
#include "ns_energy_monitor.h"
#include "nn_speech.h"
#include "ns_perf_profile.h"
#include "ns_usb.h"

#include "ae_api.h"
#include "FreeRTOS.h"
#include "task.h"
#include "arm_math.h"

#include "AudioPipe_wrapper.h"
#include "def_AudioSystem.h"

#define RECORD_10S 1
#define AUDIO_ON 1
#define PERF_TEST 1

// --- CHANGED: Increased to hold 20ms of 16kHz Stereo PCM (640 samples * 2 bytes = 1280 bytes) ---
#define MAX_ENCODED_LEN 1280 

alignas(16) unsigned char static encodedDataBuffer[MAX_ENCODED_LEN]; 

bool enableSE = true; // Default to sending both, flag used for processing
uint32_t seLatency = 0;
uint32_t opusLatency = 0;

typedef enum {
    SET_SE_MODE = 0x1,
    SE_LATENCY = 0x2,
    OPUS_LATENCY = 0x3,
    AUDIO_DATA = 0x4,
} usb_data_descriptor_e;

typedef struct __attribute__((packed)) usb_data { // Added packed to be safe
    uint8_t type; // CHANGED from enum to uint8_t (Forces 1 byte)
    uint8_t length;
    uint8_t platform; 
    uint8_t data[MAX_ENCODED_LEN]; 
} usb_data_t;

#if (configAPPLICATION_ALLOCATED_HEAP == 1)
    #define NNSE_HEAP_SIZE (40 * 1024)
size_t ucHeapSize = NNSE_HEAP_SIZE;
uint8_t ucHeap[NNSE_HEAP_SIZE] __attribute__((aligned(4)));
#endif

#define NUM_CHANNELS 1

// #define USE_AUDADC // Uncomment this to use the AUDADC instead of the PDM

/// Button Peripheral Config
int volatile g_intButtonPressed = 0;
ns_button_config_t button_config_nnsp = {
    .api = &ns_button_V1_0_0,
    .button_0_enable = true,
    .button_1_enable = false,
    .button_0_flag = &g_intButtonPressed,
    .button_1_flag = NULL};

/// Audio Config
bool volatile static g_audioRecording = false;
bool volatile static g_audioReady = false;
alignas(16) int16_t static g_in16AudioDataBuffer[LEN_STFT_HOP << 1];

// --- ADDED: Temp buffer for SE output before interleaving ---
alignas(16) int16_t static tempSEBuffer[LEN_STFT_HOP]; 

alignas(16) uint32_t static audadcSampleBuffer[(LEN_STFT_HOP << 1) + 3];
#ifdef USE_AUDADC
alignas(16) am_hal_audadc_sample_t static workingBuffer[SAMPLES_IN_FRAME * NUM_CHANNELS]; 
#endif
#if !defined(NS_AMBIQSUITE_VERSION_R4_1_0) && defined(NS_AUDADC_PRESENT)
am_hal_offset_cal_coeffs_array_t sOffsetCalib;
#endif

// Use this to generate a sinwave for debugging instead
// of using the microphone
alignas(16) int16_t static sinWave[320];

// WebUSB Configuration and Datatypes
#define MY_RX_BUFSIZE 4096
#define MY_TX_BUFSIZE 4096

static uint8_t my_rx_ff_buf[MY_RX_BUFSIZE] __attribute__((aligned(16)));
static uint8_t my_tx_ff_buf[MY_TX_BUFSIZE] __attribute__((aligned(16)));
// WebUSB URL
static ns_tusb_desc_webusb_url_t webusb_url;
static ns_usb_config_t webUsbConfig = {
    .api = &ns_usb_V1_0_0,
    .deviceType = NS_USB_VENDOR_DEVICE,
    .rx_buffer = NULL,
    .rx_bufferLength = 0,
    .tx_buffer = NULL,
    .tx_bufferLength = 0,
    .rx_cb = NULL,
    .tx_cb = NULL,
    .service_cb = NULL,
    .desc_url = &webusb_url // Filled in at runtime
};

alignas(16) uint32_t static dmaBuffer[SAMPLES_IN_FRAME * NUM_CHANNELS * 2];     // DMA target
am_hal_audadc_sample_t static sLGSampleBuffer[SAMPLES_IN_FRAME * NUM_CHANNELS * 2]; // working buffer

#ifndef NS_AMBIQSUITE_VERSION_R4_1_0
am_hal_offset_cal_coeffs_array_t sOffsetCalib;
#endif


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
#ifdef USE_AUDADC
    .eAudioSource = NS_AUDIO_SOURCE_AUDADC,
#else
    .eAudioSource = NS_AUDIO_SOURCE_PDM,
#endif
    .sampleBuffer = audadcSampleBuffer,
#ifdef USE_AUDADC
    .workingBuffer = workingBuffer,
#else
    .workingBuffer = NULL,
#endif
    .numChannels = NUM_CHANNELS,
    .numSamples = LEN_STFT_HOP,
    .sampleRate = SAMPLING_RATE,
    .audioSystemHandle = NULL, 
    .bufferHandle = NULL,      
#if !defined(NS_AMBIQSUITE_VERSION_R4_1_0) && defined(NS_AUDADC_PRESENT)
    .sOffsetCalib = &sOffsetCalib,
#endif
};

// Custom power mode for USB+Audio
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
    .bNeedXtal = true};

// --- CHANGED: Increased XBUFSIZE to handle Stereo (2x data) ---
// We need at least 640 samples (20ms stereo). Let's use 2560 to stay safe (4 frames).
#define XBUFSIZE 2560
alignas(16) int16_t static xmitBuffer[XBUFSIZE];
uint16_t static xmitWritePtr = 0;
uint16_t static xmitReadPtr = 0;
uint16_t static xmitAvailable = 0;

// FreeRTOS Tasks
TaskHandle_t audio_task_handle;  
TaskHandle_t radio_task_handle;  
TaskHandle_t encode_task_handle; 
TaskHandle_t my_xSetupTask;      

ns_timer_config_t basic_tickTimer = {
    .api = &ns_timer_V1_0_0,
    .timer = NS_TIMER_COUNTER,
    .enableInterrupt = false,
};

usb_data_t data;

// Audio Task
uint32_t seStart, seEnd;
uint32_t seLatencyCapturePeriod = 10; 
uint32_t currentSESample = 0;

// USB Senders
void set_se_mode(bool enable) {
    ns_lp_printf("SE Mode %d\n", enable);
    data.type = SET_SE_MODE;
    data.length = 1;
    data.data[0] = enable ? 1 : 0;
    webusb_send_data((uint8_t *) &data, 5);
}

void send_se_latency(uint32_t latency) {
    data.type = SE_LATENCY;
    data.length = sizeof(latency);
    memcpy(data.data, &latency, sizeof(latency));
    webusb_send_data((uint8_t *)&data, 7);
}

// --- MODIFIED: Audio Task to Interleave Stereo ---
void audioTask(void *pvParameters) {
    while (1) {
        if (g_intButtonPressed) {
            // Button can be used to toggle other things if needed
            ns_lp_printf("Button Pressed\n");
            g_intButtonPressed = 0;
        }

        if (g_audioReady) { // 160 samples, 10ms mono incoming
            NS_TRY(ns_set_performance_mode(NS_MAXIMUM_PERF), "Set CPU Perf mode failed. ");
            
            if (currentSESample == seLatencyCapturePeriod) {
                seStart = ns_us_ticker_read(&basic_tickTimer);
            }

            // 1. Run SE Model - Output to tempSEBuffer
            // We always run this to keep the model state valid
            AudioPipe_wrapper_frameProc(g_in16AudioDataBuffer, tempSEBuffer);

            if (currentSESample == seLatencyCapturePeriod) {
                seEnd = ns_us_ticker_read(&basic_tickTimer);
                seLatency = seEnd - seStart;
                send_se_latency(seLatency);
                currentSESample = 0;
            } else {
                currentSESample++;
            }

            // 2. Interleave Raw and Enhanced into xmitBuffer
            // Input: 160 samples. Output: 320 samples (160 Left, 160 Right)
            for(int i=0; i<LEN_STFT_HOP; i++){
                // LEFT Channel: Raw Audio
                xmitBuffer[xmitWritePtr] = g_in16AudioDataBuffer[i];
                xmitWritePtr = (xmitWritePtr + 1) % XBUFSIZE;

                // RIGHT Channel: Enhanced Audio
                xmitBuffer[xmitWritePtr] = tempSEBuffer[i];
                xmitWritePtr = (xmitWritePtr + 1) % XBUFSIZE;
            }

            NS_TRY(ns_set_performance_mode(NS_MINIMUM_PERF), "Set CPU Perf mode failed. ");

            // We added 160 * 2 = 320 samples to the buffer
            xmitAvailable += (LEN_STFT_HOP * 2); 
            g_audioReady = false;
        }
        vTaskDelay(pdMS_TO_TICKS(1)); 
    }
}

// --- MODIFIED: Encode Task to Send Raw PCM ---
// Renamed logic inside, task name kept same for compatibility with setup
void encodeAndXferTask(void *pvParameters) {
    
    // We want to send 20ms frames to keep USB traffic manageable
    // 20ms @ 16kHz Stereo = 640 samples
    const int SAMPLES_TO_SEND = 640; 
    const int BYTES_TO_SEND = SAMPLES_TO_SEND * sizeof(int16_t); // 1280 bytes

    while (1) {
        if (xmitAvailable >= SAMPLES_TO_SEND) {
            
            // 1. Copy data from Ring Buffer to Linear Buffer (data.data)
            // We do this manually to handle the ring buffer wrap-around
            int16_t* pDest = (int16_t*)data.data;
            
            for(int i=0; i<SAMPLES_TO_SEND; i++){
                pDest[i] = xmitBuffer[xmitReadPtr];
                xmitReadPtr = (xmitReadPtr + 1) % XBUFSIZE;
            }

            // 2. Prepare USB Packet
            data.type = AUDIO_DATA;
            // The length field is uint8_t (max 255), so 1280 won't fit.
            // We set it to 0. The receiver should calculate length based on the USB transfer size.
            data.length = 0; 
            
            // 3. Send via WebUSB
            // Header is 3 bytes (type, length, platform) + 1280 bytes audio
            webusb_send_data((uint8_t *)&data, 3 + BYTES_TO_SEND);
            
            // Decrement available count
            xmitAvailable -= SAMPLES_TO_SEND;
        }
        vTaskDelay(pdMS_TO_TICKS(1));
    }
}

void setup_task(void *pvParameters) {
    ns_lp_printf("Setting up USB FreeRTOS Tasks\n");
    xTaskCreate(audioTask, "AudioTask", 2048, 0, 3, &audio_task_handle);
    xTaskCreate(encodeAndXferTask, "encodeAndXferTask", 4096, 0, 3, &encode_task_handle);
    vTaskSuspend(NULL);
    while (1)
        ;
}

void msgReceived(const uint8_t *buffer, uint32_t length, void *args) {
    ns_lp_printf("Received %d bytes: %s\n", length, buffer);
}

int main(void) {
    usb_handle_t usb_handle = NULL;

    ns_core_config_t ns_core_cfg = {.api = &ns_core_V1_0_0};
    NS_TRY(ns_core_init(&ns_core_cfg), "Core init failed.\b");

    ns_power_config(&ns_power_usb);
    NS_TRY(ns_set_performance_mode(NS_MINIMUM_PERF), "Set CPU Perf mode failed. ");

    ns_itm_printf_enable();
    ns_interrupt_master_enable();
    
    // -- Init the audio system
    NS_TRY(ns_audio_init(&audio_config), "Audio Initialization Failed.\n");
    NS_TRY(ns_audio_set_gain(AM_HAL_PDM_GAIN_P195DB, AM_HAL_PDM_GAIN_P195DB), "Gain set failed.\n");
    NS_TRY(ns_start_audio(&audio_config), "Audio Start Failed.\n");
    
    ns_peripheral_button_init(&button_config_nnsp);
    ns_init_perf_profiler();
    ns_start_perf_profiler();

    // REMOVED: audio_enc_init(0); -> We are not using Opus anymore

    // Initialize the URL descriptor
    strcpy(webusb_url.url, "ambiqai.github.io/web-ble-dashboards/nnse-usb/");
    webusb_url.bDescriptorType = 3;
    webusb_url.bScheme = 1;
    webusb_url.bLength = 3 + sizeof(webusb_url.url) - 1;

    // WebUSB Setup
    webusb_register_raw_cb(msgReceived, NULL);
    webUsbConfig.rx_buffer = my_rx_ff_buf;
    webUsbConfig.rx_bufferLength = MY_RX_BUFSIZE;
    webUsbConfig.tx_buffer = my_tx_ff_buf;
    webUsbConfig.tx_bufferLength = MY_TX_BUFSIZE;

    NS_TRY(ns_usb_init(&webUsbConfig, &usb_handle), "USB Init Failed\n");
    ns_lp_printf("USB Init Success\n");

    NS_TRY(ns_timer_init(&basic_tickTimer), "Timer init failed.\n");
   
    // Generate a 400hz sin wave (for debugging)
    for (int i = 0; i < 320; i++) {
        sinWave[i] = (int16_t)(sin(2 * 3.14159 * 400 * i / SAMPLING_RATE) * 32767);
    }

    #if defined(AM_PART_APOLLO5B)
    data.platform = 2;
    #elif defined(AM_PART_APOLLO4P)
    data.platform = 1;
    #else
    data.platform = 0;
    #endif

    ns_printf("Starting Dual-Channel Audio Stream (Raw + Enhanced)\n");
    ns_printf("Connect via WebUSB\n");

    // Initialize NNSE2 model
    AudioPipe_wrapper_init();
    AudioPipe_wrapper_reset();
    
    g_audioRecording = true;
    xTaskCreate(setup_task, "Setup", 512, 0, 1, &my_xSetupTask);

    vTaskStartScheduler();

    while (1) {
    };
}