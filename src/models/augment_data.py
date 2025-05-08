
import os
import torchaudio
from torchaudio import transforms as T

INPUT_DIR = os.environ.get("INPUT_DIR", "/gscratch/stf/abimaelh/Shiro_Corpus_Segments")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/gscratch/stf/abimaelh/Shiro_Corpus_Segments_Augmented")

SPEED_FACTORS = [0.9, 1.1]
PITCH_SHIFTS = [-100, 100]

os.makedirs(OUTPUT_DIR, exist_ok=True)

for subject in os.listdir(INPUT_DIR):
    subject_path = os.path.join(INPUT_DIR, subject)
    if not os.path.isdir(subject_path):
        continue

    out_subject_path = os.path.join(OUTPUT_DIR, subject)
    os.makedirs(out_subject_path, exist_ok=True)

    for file_name in os.listdir(subject_path):
        if not file_name.endswith(".wav"):
            continue

        in_file_path = os.path.join(subject_path, file_name)
        waveform, sample_rate = torchaudio.load(in_file_path)

        base_name = os.path.splitext(file_name)[0]

        torchaudio.save(os.path.join(out_subject_path, base_name + "_orig.wav"), waveform, sample_rate)

        for speed in SPEED_FACTORS:
            new_sample_rate = int(sample_rate * speed)
            resampled = T.Resample(orig_freq=sample_rate, new_freq=new_sample_rate)(waveform)

            speed_tag = f"speed{speed}"
            file_speed = f"{base_name}_{speed_tag}.wav"
            torchaudio.save(os.path.join(out_subject_path, file_speed), resampled, new_sample_rate)

            for pitch in PITCH_SHIFTS:
                pitch_str = f"pitch{pitch:+d}"
                effects = [["pitch", str(pitch)]]
                try:
                    shifted_waveform, _ = torchaudio.sox_effects.apply_effects_tensor(resampled, new_sample_rate, effects)
                    file_aug = f"{base_name}_{speed_tag}_{pitch_str}.wav"
                    out_path = os.path.join(out_subject_path, file_aug)
                    torchaudio.save(out_path, shifted_waveform, new_sample_rate)
                except Exception as e:
                    print(f"Error processing {file_name} with speed={speed}, pitch={pitch}: {e}")