import os
import soundfile as sf
from datasets import load_dataset

os.makedirs("data", exist_ok=True)
CLEAN_DIR = "data/clean_train"
NOISY_DIR = "data/noisy_train"
os.makedirs(CLEAN_DIR, exist_ok=True)
os.makedirs(NOISY_DIR, exist_ok=True)

print("Downloading VoiceBank-DEMAND-16k dataset")
try:
    ds = load_dataset("JacobLinCool/VoiceBank-DEMAND-16k", split="train")
    print(f"Loaded {len(ds)} audio pairs.")
except Exception as e:
    print(f"ERROR: Could not load dataset: {e}")
    exit()

print("Saving clean and noisy WAV files to disk")
for i, sample in enumerate(ds):
    clean_audio = sample["clean"]["array"]
    noisy_audio = sample["noisy"]["array"]

    clean_path = os.path.join(CLEAN_DIR, f"clean_{i:05d}.wav")
    noisy_path = os.path.join(NOISY_DIR, f"noisy_{i:05d}.wav")

    sf.write(clean_path, clean_audio, 16000)
    sf.write(noisy_path, noisy_audio, 16000)

    if i % 100 == 0:
        print(f"   Saved {i} samples...")

print("Done! WAV files saved to:")
print(f"   {CLEAN_DIR}")
print(f"   {NOISY_DIR}")
