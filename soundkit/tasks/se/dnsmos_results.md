DNS MOS Comparision
| Net Type        | Sample 1 | Sample 2 | Sample 3 | **Wind Noise** | Average | # Params | Loss  | Sampling Rate | hop size | latency |
| --------------- | -------- | -------- | -------- | -------------- | ------- | -------- | ----- | ------------- | -------- | ------- |
| **CRNN-10ms (ambiq)**   | 2.60     | 2.91     | 2.24     | **1.96**       | 2.58    | 100k     | MSE   | 16k           | 10ms     | 20ms    |
| **UNet**        | 2.67     | 3.02     | 2.15     | —              | 2.61    | 163k     | MSE   | 16k           | 10ms     | 40ms    |
| **CTCRN**       | 2.89     | 3.06     | 2.56     | **1.94**       | 2.83    | 35k      | MIX   | 16k           | 10ms     | ?       |
| **UNet t-loss** | 3.14     | 3.19     | 2.86     | —              | 3.06    | 126k     | MAE-t | 16k           | 10ms     | 40ms    |
| **Demucs (TD)** | 3.14     | 3.21     | 2.88     | —              | 3.07    | 33.1M    | MRL+  | 16k           | 16ms     | 40ms    |
| **DF2**         | 3.23     | 3.09     | 2.94     | —              | 3.09    | 2.1M     | MRL   | 48k           | 10ms     | 50ms    |


- CTCRN: 
    1. MCPS should be high
    1. group-rnn. \
        Shape=(T,F,C) -> reshape (F,T,C). rnn along T. Run F time at each t \
        Shape=(T,F,C) -> reshape (T,F,C). bi-rnn along F at each t \
        So at each t, you run F + 2*F rnn on 
    