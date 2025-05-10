import os
import torchaudio
import pandas as pd
from speechbrain.inference.classifiers import EncoderClassifier
from collections import defaultdict, Counter
from scipy.stats import chi2_contingency

MODEL_DIR = "/home2/abimaelh/child-lang-id/MODEL_DIR"
DATA_DIR = "/home2/abimaelh/child-lang-id/DATA_DIR"
OUTPUT_DIR = "/home2/abimaelh/child-lang-id/OUTPUT_DIR"
METADATA_PATH = "/home2/abimaelh/child-lang-id/metadata/shiro_metadata.xlsx"

os.makedirs(OUTPUT_DIR, exist_ok=True)

language_id = EncoderClassifier.from_hparams(
    source="speechbrain/lang-id-voxlingua107-ecapa",
    savedir=MODEL_DIR
)

metadata_df = pd.read_excel(METADATA_PATH)
metadata_df["filename"] = metadata_df["filename"].astype(str)

predictions = []
subject_correct_counts = defaultdict(int)
subject_total_counts = defaultdict(int)

for subject in os.listdir(DATA_DIR):
    subject_path = os.path.join(DATA_DIR, subject)
    if not os.path.isdir(subject_path):
        continue

    print(f"\nProcessing Subject: {subject}")

    for file_name in os.listdir(subject_path):
        if not file_name.endswith(".wav"):
            continue

        file_path = os.path.join(subject_path, file_name)
        try:
            signal, fs = torchaudio.load(file_path)
            prediction = language_id.classify_file(file_path)
            predicted_lang = prediction[3]
        except Exception as e:
            print(f"Failed to process {file_path}: {e}")
            continue

        base_name = subject
        predictions.append({
            "filename": base_name,
            "file_path": file_path,
            "predicted_lang": predicted_lang
        })
        subject_total_counts[base_name] += 1
        gold_lang = "Spanish"
        predictions[-1]["true_lang"] = gold_lang
        if predicted_lang == gold_lang:
            subject_correct_counts[base_name] += 1

pred_df = pd.DataFrame(predictions)
pred_df.to_csv(os.path.join(OUTPUT_DIR, "predictions.csv"), index=False)

merged_df = pred_df.merge(metadata_df, on="filename", how="left")

subject_acc = []
for subject, total in subject_total_counts.items():
    correct = subject_correct_counts[subject]
    acc = correct / total if total > 0 else 0
    subject_acc.append({"filename": subject, "accuracy": acc})

subject_acc_df = pd.DataFrame(subject_acc)
mean_acc = subject_acc_df["accuracy"].mean()
std_acc = subject_acc_df["accuracy"].std()

gender_counts = merged_df.groupby(["gender", "predicted_lang", "true_lang"]).size().unstack(fill_value=0)
chi2_gender, p_gender, _, _ = chi2_contingency(gender_counts)

median_age = metadata_df["age_in_months"].median()
merged_df["age_group"] = merged_df["age_in_months"].apply(lambda x: "younger" if x <= median_age else "older")
age_counts = merged_df.groupby(["age_group", "predicted_lang", "true_lang"]).size().unstack(fill_value=0)
chi2_age, p_age, _, _ = chi2_contingency(age_counts)

overall_acc = (pred_df["predicted_lang"] == pred_df["true_lang"]).mean()

print("\n=== Evaluation Summary ===")
print(f"Overall Accuracy: {overall_acc:.2%}")
print(f"Mean Per-Subject Accuracy: {mean_acc:.2%} ± {std_acc:.2%}")
print(f"Chi-square Gender p-value: {p_gender:.4f}")
print(f"Chi-square Age Group p-value: {p_age:.4f}")

subject_acc_df.to_csv(os.path.join(OUTPUT_DIR, "per_subject_accuracy.csv"), index=False)
print(f"Detailed results saved to: {OUTPUT_DIR}")