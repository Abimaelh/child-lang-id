import os
import argparse
import torchaudio
import pandas as pd
from speechbrain.inference.classifiers import EncoderClassifier
from collections import defaultdict
from pathlib import Path
from glob import glob
from sklearn.metrics import precision_recall_fscore_support

# condor args
parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", required=True)
parser.add_argument("--data_dir", required=True)
parser.add_argument("--output_dir", required=True)
args = parser.parse_args()

# Ensure output directory exists
os.makedirs(args.output_dir, exist_ok=True)

MODEL_DIR = args.model_dir
DATA_DIR = args.data_dir
OUTPUT_DIR = args.output_dir

os.makedirs(OUTPUT_DIR, exist_ok=True)

# load voxlingua107
language_id = EncoderClassifier.from_hparams(
    source="speechbrain/lang-id-voxlingua107-ecapa",
    savedir=MODEL_DIR
)

# initialize result lists
predictions = []
subject_correct_counts = defaultdict(int)
subject_total_counts = defaultdict(int)

# find all .wav files in nested subdirectories
wav_files = glob(f"{DATA_DIR}/**/*.wav", recursive=True)
if not wav_files:
    raise ValueError("No .wav files found in DATA_DIR. Check the folder structure.")

# run inference
for file_path in wav_files:
    if "archive" in file_path.lower():
        continue
    try:
        signal, fs = torchaudio.load(file_path)
        prediction = language_id.classify_file(file_path)
        predicted_lang = prediction[3]
        if isinstance(predicted_lang, list):
            predicted_lang = predicted_lang[0]
    except Exception as e:
        print(f"Failed to process {file_path}: {e}")
        continue

    subject_id = Path(file_path).parts[-2]
    corpus = Path(file_path).parts[-4] if "cslu_segments" in file_path else "shiro"
    true_lang = "spanish" if corpus == "shiro" else "english"

    predictions.append({
        "filename": subject_id,
        "file_path": file_path,
        "predicted_lang": predicted_lang,
        "true_lang": true_lang,
        "corpus": corpus
    })

    subject_total_counts[(corpus, subject_id)] += 1
    predicted_lang_clean = predicted_lang.split(":", 1)[-1].strip().lower()
    if predicted_lang_clean == true_lang.lower():
        subject_correct_counts[(corpus, subject_id)] += 1

# save predictions
pred_df = pd.DataFrame(predictions)
if pred_df.empty:
    raise ValueError("Prediction dataframe is empty. No predictions were made.")

pred_df["filename"] = pred_df["filename"].str.lower()
pred_df.to_csv(os.path.join(OUTPUT_DIR, "predictions.csv"), index=False)

# corpus accuracy
for corpus_name in pred_df["corpus"].unique():
    corpus_df = pred_df[pred_df["corpus"] == corpus_name]
    print(f"\n=== Results for {corpus_name.upper()} ===")

    subject_correct_counts = defaultdict(int)
    subject_total_counts = defaultdict(int)

    for _, row in corpus_df.iterrows():
        subject_id = row["filename"]
        predicted_lang = row["predicted_lang"]
        true_lang = row["true_lang"]
        corpus = row["corpus"]

        subject_total_counts[(corpus, subject_id)] += 1
        predicted_lang_clean = predicted_lang.split(":", 1)[-1].strip().lower()
        if predicted_lang_clean == true_lang.lower():
            subject_correct_counts[(corpus, subject_id)] += 1

    subject_acc = []
    for subject in corpus_df["filename"].unique():
        key = (corpus_name, subject)
        correct = subject_correct_counts.get(key, 0)
        total = subject_total_counts.get(key, 0)
        acc = correct / total if total > 0 else 0
        subject_acc.append({"filename": subject, "accuracy": acc})

    subject_acc_df = pd.DataFrame(subject_acc)
    mean_acc = subject_acc_df["accuracy"].mean()
    std_acc = subject_acc_df["accuracy"].std()

    subject_acc_df.to_csv(os.path.join(OUTPUT_DIR, f"per_subject_accuracy_{corpus_name}.csv"), index=False)

    corpus_df["predicted_lang_clean"] = corpus_df["predicted_lang"].apply(
        lambda x: x.split(":", 1)[-1].strip().lower() if isinstance(x, str) else x)
    overall_acc = (corpus_df["predicted_lang_clean"] == corpus_df["true_lang"].str.lower()).mean()
    y_true = corpus_df["true_lang"].str.lower()
    y_pred = corpus_df["predicted_lang_clean"]
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)

    print(f"Overall Accuracy: {overall_acc:.2%}")
    print(f"Mean Per-Subject Accuracy: {mean_acc:.2%} ± {std_acc:.2%}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

pred_df["predicted_lang_clean"] = pred_df["predicted_lang"].str.extract(r":\s*(.*)").str.lower().str.strip()
mismatches = pred_df[pred_df["predicted_lang_clean"] != pred_df["true_lang"].str.lower()]
mismatches.to_csv(os.path.join(OUTPUT_DIR, "mismatched_predictions.csv"), index=False)
print(f"Saved predictions to: {os.path.join(args.output_dir, 'predictions.csv')}")