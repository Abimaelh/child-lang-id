import os
import torchaudio
from speechbrain.inference.classifiers import EncoderClassifier
from collections import Counter

DATA_DIR = os.environ.get("DATA_DIR", "/gscratch/stf/abimaelh/Shiro_Corpus_Segments")
MODEL_DIR = os.environ.get("MODEL_DIR", "/gscratch/stf/abimaelh/speechbrain_models")

language_id = EncoderClassifier.from_hparams(
    source="speechbrain/lang-id-voxlingua107-ecapa",
    savedir=MODEL_DIR
)

IGNORE = {"Archive", "tmp", "model_path", "speechbrain_models"}

subject_predictions = {}

for subject in os.listdir(DATA_DIR):
    subject_path = os.path.join(DATA_DIR, subject)
    if subject in IGNORE or not os.path.isdir(subject_path):
        continue

    print(f"\nProcessing Subject: {subject}")
    subject_languages = []

    for file_name in os.listdir(subject_path):
        if not file_name.endswith(".wav"):
            continue

        file_path = os.path.join(subject_path, file_name)
        waveform = language_id.load_audio(file_path)
        prediction = language_id.classify_batch(waveform)
        predicted_language = prediction[3][0]
        subject_languages.append(predicted_language)
        print(f" File: {file_name}, Predicted: {predicted_language}")

    if subject_languages:
        most_common = Counter(subject_languages).most_common(1)[0][0]
        subject_predictions[subject] = most_common
        print(f" Majority language for {subject}: {most_common}")

output_path = os.path.join(DATA_DIR, "baseline_predictions.txt")
with open(output_path, "w") as f:
    for subject, lang in subject_predictions.items():
        f.write(f"{subject}\t{lang}\n")

print(f"\nSaved predictions to: {output_path}")
