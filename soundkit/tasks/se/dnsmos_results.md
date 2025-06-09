DNS MOS Comparision
| Net Type     | Sample 1 | Sample 2 | Sample 3 | Average | # Params | Loss | Sampling Rate | frame size|
|--------------|----------|----------|----------|---------|----------|------|----------------|----------|
| CRNN         | 2.60     | 2.91     | 2.24     | 2.58    | 100k     | MSE  | 16k            |160       |
| UNet         | 2.67     | 3.02     | 2.15     | 2.61    | 163k     | MSE  | 16k            |160       |
| CRNN (Large) | 2.90     | 3.01     | 2.34     | 2.75    | 357k     | MRL  | 16k            |64        |
| ULC          | 2.85     | 2.73     | 2.26     | 2.61    | 910k     | MRL  | 48k            |?         |