# Wind Noise DNSMOS Comparison

| Model                     | Sample 1 | Sample 2 | Sample 3 | Average | Params | Sample Rate | MCPS | Latency |
|--------------------------|----------|----------|----------|---------|--------|-------------|------|---------|
| **CRNN-10ms (Ambiq)**    | 2.44     | 1.89     | 2.22     | 2.18    | 100k   | 16 kHz      | 25   | 20 ms   |
| **Competitor**           | 2.53     | 1.98     | 2.53     | 2.35    | ?      | 16 kHz      | ?    | 40 ms   |
| **UNet (Ambiq)**         | 2.61     | 2.01     | 2.60     | 2.41    | 116k   | 16 kHz      | 95   | 40 ms   |
| **Demucs (FB)**          | 2.67     | 2.12     | 2.75     | 2.51    | 33.5M  | 16 kHz      | ?    | ?       |
