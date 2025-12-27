| **Net Type**              | **Average MOS** | **# Params** | **Loss** | **Sampling Rate** | **Hop Size** | **Latency** | **MCPS** |
| ------------------------- | --------------- | ------------ | -------- | ----------------- | ------------ | ----------- | -------- |
| **CRNN-10ms – Ambiq**     | 2.58            | 100k         | MSE      | 16k               | 10ms         | 20ms        | **25**   |
| **UNet (small) – Ambiq**  | 2.81            | 149k         | t-loss   | 16k               | 10ms         | 40ms        | 100      |
| **CTCRN**                 | 2.83            | 35k          | MIX      | 16k               | 10ms         | 40ms        | >1000    |
| **UNet (medium) – Ambiq** | 2.90            | 183k         | t-loss   | 16k               | 10ms         | 40ms        | 240      |
| **UNet (large) – Ambiq**  | 3.01            | 711k         | t-loss   | 16k               | 10ms         | 40ms        | 500      |
| **Demucs (TD)**           | 3.07            | 33.1M        | MRL+     | 16k               | 16ms         | 40ms        | >1000    |
| **DF2**                   | 3.09            | 2.1M         | MRL      | 48k               | 10ms         | 50ms        | >1000    |
