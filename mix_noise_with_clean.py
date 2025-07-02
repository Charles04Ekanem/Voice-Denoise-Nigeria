import os
import random
import librosa
import soundfile as sf
import numpy as np
from glob import glob

def add_noise(clean, noise, snr_db):
    if len(noise) < len(clean):
        repeats = int(np.ceil(len(clean) / len(noise)))
        noise = np.tile(noise, repeats)
    noise = noise[:len(clean)]

    rms_clean = np.sqrt(np.mean(clean**2))
    rms_noise = np.sqrt(np.mean(noise**2))

    snr = 10 ** (snr_db / 20)
    noise_scaled = noise * (rms_clean / (snr * rms_noise))

    return clean + noise_scaled

clean_dir = "data/clean_train"
noise_dir = "data/noise_bank"
output_dir = "data/custom_noisy"
snr_db = 5  

os.makedirs(output_dir, exist_ok=True)

clean_files = glob(os.path.join(clean_dir, "*.wav"))
noise_files = glob(os.path.join(noise_dir, "*.wav"))

print(f"Mixing {len(clean_files)} clean files with {len(noise_files)} real noise recordings")

for idx, clean_path in enumerate(clean_files):
    try:
        clean_audio, sr = librosa.load(clean_path, sr=None)
        noise_path = random.choice(noise_files)
        noise_audio, _ = librosa.load(noise_path, sr=sr)

        mixed = add_noise(clean_audio, noise_audio, snr_db)

        filename = os.path.basename(clean_path)
        output_path = os.path.join(output_dir, filename)
        sf.write(output_path, mixed, sr)
        
        if idx % 500 == 0:
            print(f"  Mixed {idx}/{len(clean_files)}")
    except Exception as e:
        print(f"Error mixing {clean_path}: {e}")

print(f"Done. Noisy samples saved in: {output_dir}")
