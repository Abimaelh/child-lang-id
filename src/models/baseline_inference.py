import os
import torchaudio
import pandas as pd
from speechbrain.inference.classifiers import EncoderClassifier
from collections import defaultdict, Counter
from scipy.stats import chi2_contingency
from pathlib import Path
from glob import glob

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
    except Exception as e:
        print(f"Failed to process {file_path}: {e}")
        continue

    subject_id = Path(file_path).parts[-2]  
    corpus = Path(file_path).parts[-4] if "cslu_segments" in file_path else "shiro"
    gold_lang = "spanish"
    predictions.append({
        "filename": subject_id,
        "file_path": file_path,
        "predicted_lang": predicted_lang,
        "true_lang": gold_lang,
        "corpus": corpus
    })
    subject_total_counts[(corpus, subject_id)] += 1
    if predicted_lang.lower() == gold_lang.lower():
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

    if corpus_name == "shiro":
        corpus_merged = corpus_df.merge(metadata_df, on="filename", how="left")

        gender_counts = corpus_merged.groupby(["gender", "predicted_lang", "true_lang"]).size().unstack(fill_value=0)
        chi2_gender, p_gender, _, _ = chi2_contingency(gender_counts)

        median_age = metadata_df["age_in_months"].median()
        corpus_merged["age_group"] = corpus_merged["age_in_months"].apply(lambda x: "younger" if x <= median_age else "older")
        age_counts = corpus_merged.groupby(["age_group", "predicted_lang", "true_lang"]).size().unstack(fill_value=0)
        chi2_age, p_age, _, _ = chi2_contingency(age_counts)

        overall_acc = (corpus_df["predicted_lang"].str.lower() == corpus_df["true_lang"].str.lower()).mean()

        print(f"Overall Accuracy: {overall_acc:.2%}")
        print(f"Mean Per-Subject Accuracy: {mean_acc:.2%} ± {std_acc:.2%}")
        print(f"Chi-square Gender p-value: {p_gender:.4f}")
        print(f"Chi-square Age Group p-value: {p_age:.4f}")
    else:
        overall_acc = (corpus_df["predicted_lang"].str.lower() == corpus_df["true_lang"].str.lower()).mean()
        print(f"Overall Accuracy: {overall_acc:.2%}")
        print(f"Mean Per-Subject Accuracy: {mean_acc:.2%} ± {std_acc:.2%}")

print("\nbaseline_inference.py completed successfully.")
