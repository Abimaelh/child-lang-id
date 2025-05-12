import os
import torchaudio
import pandas as pd
from speechbrain.inference.classifiers import EncoderClassifier
from collections import defaultdict, Counter
from pathlib import Path
from glob import glob
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
from sklearn.metrics import classification_report

MODEL_DIR = "/home2/abimaelh/child-lang-id/MODEL_DIR"
DATA_DIR = "/home2/abimaelh/child-lang-id/DATA_DIR"
OUTPUT_DIR = "/home2/abimaelh/child-lang-id/OUTPUT_DIR"
METADATA_PATH = "/home2/abimaelh/child-lang-id/METADATA_DIR/shiro_metadata.xlsx"

os.makedirs(OUTPUT_DIR, exist_ok=True)

language_id = EncoderClassifier.from_hparams(
    source="speechbrain/lang-id-voxlingua107-ecapa",
    savedir=MODEL_DIR
)

metadata_df = pd.read_excel(METADATA_PATH)
metadata_df.columns = metadata_df.columns.str.strip().str.lower() 
metadata_df["filename"] = metadata_df["filename"].astype(str).str.strip().str.lower()

predictions = []
subject_correct_counts = defaultdict(int)
subject_total_counts = defaultdict(int)

wav_files = glob(f"{DATA_DIR}/**/*.wav", recursive=True)

if not wav_files:
    raise ValueError("No .wav files found in DATA_DIR. Check the folder structure.")

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

    if corpus == "shiro":
        true_lang = "spanish"
    else:  # cslu_segments
        true_lang = "english"

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

pred_df = pd.DataFrame(predictions)

if pred_df.empty:
    raise ValueError("Prediction dataframe is empty. No predictions were made.")

pred_df["filename"] = pred_df["filename"].str.lower()
pred_df.to_csv(os.path.join(OUTPUT_DIR, "predictions.csv"), index=False)

print("Metadata columns:", metadata_df.columns.tolist())
merged_df = pred_df.merge(metadata_df, on="filename", how="left")

for corpus_name in pred_df["corpus"].unique():
    corpus_df = pred_df[pred_df["corpus"] == corpus_name]
    print(f"\n=== Results for {corpus_name.upper()} ===")

    subject_correct_counts = defaultdict(int)
    subject_total_counts = defaultdict(int)

    for index, row in corpus_df.iterrows():
        subject_id = row["filename"]
        corpus = row["corpus"]
        predicted_lang = row["predicted_lang"]
        true_lang = row["true_lang"]

        subject_total_counts[(corpus, subject_id)] += 1

        predicted_lang_clean = predicted_lang.split(":", 1)[-1].strip().lower()
        if predicted_lang_clean == true_lang.lower():
            subject_correct_counts[(corpus, subject_id)] += 1

    corpus_subjects = corpus_df["filename"].unique()
    subject_acc = []
    for subject in corpus_subjects:
        key = (corpus_name, subject)
        correct = subject_correct_counts.get(key, 0)
        total = subject_total_counts.get(key, 0)
        acc = correct / total if total > 0 else 0
        subject_acc.append({"filename": subject, "accuracy": acc})

    subject_acc_df = pd.DataFrame(subject_acc)
    mean_acc = subject_acc_df["accuracy"].mean()
    std_acc = subject_acc_df["accuracy"].std()

    subject_acc_df.to_csv(os.path.join(OUTPUT_DIR, f"per_subject_accuracy_{corpus_name}.csv"), index=False)

    corpus_df["predicted_lang_clean"] = corpus_df["predicted_lang"].apply(lambda x: x.split(":", 1)[-1].strip().lower() if isinstance(x, str) else x)
    overall_acc = (corpus_df["predicted_lang_clean"] == corpus_df["true_lang"].str.lower()).mean()
    y_true = corpus_df["true_lang"].str.lower()
    y_pred = corpus_df["predicted_lang_clean"]
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)

    print(f"Overall Accuracy: {overall_acc:.2%}")
    print(f"Mean Per-Subject Accuracy: {mean_acc:.2%} ± {std_acc:.2%}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

print("\nbaseline_inference.py completed successfully.")

# export data

df = pd.read_csv("/home2/abimaelh/child-lang-id/OUTPUT_DIR/predictions.csv")

df["predicted_lang_clean"] = df["predicted_lang"].str.extract(r":\s*(.*)")

y_true = df["true_lang"].str.lower().str.strip()
y_pred = df["predicted_lang_clean"].str.lower().str.strip()

accuracy = (y_true == y_pred).mean()
print(f"Accuracy: {accuracy:.4f}")

report = classification_report(y_true, y_pred, zero_division=0)
print(report)

with open("classification_report.txt", "w") as f:
    f.write(report)

mismatches = df[y_true != y_pred]
mismatches.to_csv("mismatched_predictions.csv", index=False)