DNS MOS Comparision
| Net Type     | Sample 1 | Sample 2 | Sample 3 | Average | # Params | Loss | Sampling Rate  | hop size | latency |
|--------------|----------|----------|----------|---------|----------|------|----------------|----------|---------|
| CRNN-4ms     | 2.30     | 2.90     | 2.07     | 2.43    | 124k     | MSE  | 16k            |4ms       | 4ms       
| CRNN-10ms    | 2.60     | 2.91     | 2.24     | 2.58    | 100k     | MSE  | 16k            |10ms      | 20ms
| UNet         | 2.67     | 3.02     | 2.15     | 2.61    | 163k     | MSE  | 16k            |10ms      | 40ms
| CRNN-4ms (L) | 2.90     | 3.01     | 2.34     | 2.75    | 357k     | MRL  | 16k            |4ms       | 16ms
| Demucs (TD)  | 3.14     | 3.21     | 2.88     | 3.07    | 33.1M    | MRL+ | 16k            |16ms      | 16ms
| DF2          | 3.23     | 3.09     | 2.94     | 3.09    | 2.1M     | MRL  | 48k            |10ms      | 30ms