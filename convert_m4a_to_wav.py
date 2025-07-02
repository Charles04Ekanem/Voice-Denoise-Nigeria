import os
import subprocess

input_folder = "data/noise_bank_raw"
output_folder = "data/noise_bank"
os.makedirs(output_folder, exist_ok=True)

files_found = 0

for file in os.listdir(input_folder):
    if file.endswith(".m4a"):
        files_found += 1
        input_path = os.path.join(input_folder, file)
        output_path = os.path.join(output_folder, os.path.splitext(file)[0] + ".wav")

        command = [r"C:\Users\CHARLES EKANEM\Downloads\ffmpeg-7.1.1-essentials_build\ffmpeg-7.1.1-essentials_build\bin\ffmpeg.exe", "-y", "-i", input_path, output_path]

        print(f"Converting {file}...")
        subprocess.run(command)

if files_found == 0:
    print("No .m4a files found in data/noise_bank_raw/")
else:
    print(f"{files_found} file(s) converted to WAV in data/noise_bank/")