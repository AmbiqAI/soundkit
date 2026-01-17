import os
import time
import requests
import numpy as np
import librosa
import onnxruntime as ort
import argparse
from tqdm import tqdm 

# ==============================================================================
# ✅ BATCH PROCESSING SCRIPT (Updated with BAK)
# ==============================================================================

class DNSMOS_Batch:
    MODEL_URL = "https://github.com/microsoft/DNS-Challenge/raw/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx"
    FILENAME = "sig_bak_ovr.onnx"
    SAMPLING_RATE = 16000
    INPUT_LENGTH = 144160
    HOP_SIZE = int(INPUT_LENGTH / 2)

    def __init__(self, use_gpu=True, batch_size=32):
        self.BATCH_SIZE = batch_size
        self._ensure_model_exists()
        self.session = self._init_session(use_gpu)
        self.input_name = self.session.get_inputs()[0].name

    def _ensure_model_exists(self):
        if not os.path.exists(self.FILENAME):
            print(f"⬇️ Downloading model...")
            r = requests.get(self.MODEL_URL)
            with open(self.FILENAME, 'wb') as f:
                f.write(r.content)

    def _init_session(self, use_gpu):
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_gpu else ['CPUExecutionProvider']
        try:
            sess = ort.InferenceSession(self.FILENAME, providers=providers)
            print(f"🚀 DNSMOS Running on: {sess.get_providers()[0]}")
            return sess
        except:
            return ort.InferenceSession(self.FILENAME, providers=['CPUExecutionProvider'])

    def run_folder(self, folder_path):
        t_start_total = time.time()
        
        files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.wav')]
        if not files:
            print("No wav files found.")
            return

        print(f"📂 Found {len(files)} files. Loading...")
        
        t_load_start = time.time()
        
        file_map = {} 
        all_chunks = []
        chunk_counter = 0
        total_audio_duration = 0 
        
        for fpath in tqdm(files, desc="Loading"):
            try:
                audio, _ = librosa.load(fpath, sr=self.SAMPLING_RATE)
                total_audio_duration += len(audio) / self.SAMPLING_RATE
            except:
                continue

            chunks_from_this_file = []
            if len(audio) < self.INPUT_LENGTH:
                padded = np.pad(audio, (0, self.INPUT_LENGTH - len(audio)), 'constant')
                chunks_from_this_file.append(padded)
            else:
                for start in range(0, len(audio) - self.INPUT_LENGTH + 1, self.HOP_SIZE):
                    chunks_from_this_file.append(audio[start : start + self.INPUT_LENGTH])
                if len(audio) % self.HOP_SIZE != 0:
                     chunks_from_this_file.append(audio[-self.INPUT_LENGTH:])

            all_chunks.extend(chunks_from_this_file)
            num_chunks = len(chunks_from_this_file)
            file_map[fpath] = (chunk_counter, chunk_counter + num_chunks)
            chunk_counter += num_chunks

        t_load_end = time.time()

        if not all_chunks: return

        print(f"⚡ Inference on {len(all_chunks)} chunks...")
        t_inf_start = time.time()
        
        all_scores = []
        all_chunks_np = np.stack(all_chunks).astype(np.float32)

        for i in tqdm(range(0, len(all_chunks), self.BATCH_SIZE), desc="GPU Inference"):
            batch = all_chunks_np[i : i + self.BATCH_SIZE]
            outputs = self.session.run(None, {self.input_name: batch})[0]
            all_scores.extend(outputs)

        t_inf_end = time.time()

        # ============================================================
        # 📝 UPDATED PRINTING LOGIC (With BAK)
        # ============================================================
        print("\n" + "="*65)
        print(f"{'FILENAME':<30} | {'OVR':<8} | {'SIG':<8} | {'BAK':<8}")
        print("-" * 65)

        for fpath, (start, end) in file_map.items():
            file_scores = np.array(all_scores[start:end])
            
            # Index 0 = SIG, Index 1 = BAK, Index 2 = OVR
            avg_sig = np.mean(file_scores[:, 0])
            avg_bak = np.mean(file_scores[:, 1])  # <--- Added BAK extraction
            avg_ovr = np.mean(file_scores[:, 2])
            
            name = os.path.basename(fpath)
            if len(name) > 28: name = name[:25] + "..."
            
            print(f"{name:<30} | {avg_ovr:.4f}   | {avg_sig:.4f}   | {avg_bak:.4f}")

        t_total = time.time() - t_start_total
        t_load = t_load_end - t_load_start
        t_inf = t_inf_end - t_inf_start
        
        print("=" * 65)
        print(f"⏱️  PERFORMANCE REPORT")
        print(f"   Total Audio:      {total_audio_duration:.2f}s")
        print(f"   Loading (CPU):    {t_load:.3f}s")
        print(f"   Inference (GPU):  {t_inf:.3f}s")
        print(f"   Wall Time:        {t_total:.3f}s")
        if t_total > 0:
            print(f"   Speed:            {total_audio_duration / t_total:.1f}x Real-time")
        print("=" * 65)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="Path to folder containing WAV files")
    parser.add_argument("--bs", type=int, default=32, help="Batch Size")
    args = parser.parse_args()

    runner = DNSMOS_Batch(use_gpu=True, batch_size=args.bs)
    runner.run_folder(args.folder)