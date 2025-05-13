## ChildLangID: Language Identification in Children’s Speech

A fine-tuned ECAPA-TDNN model for language identification in children's speech, addressing the gap in pre-trained models that primarily focus on adult speech.

## Overview
The goal of this project is to improve language identification accuracy for child speech by fine-tuning existing models on manually extracted speech segments from 6-7-year-old children.
We use two pre-trained models for baseline evaluation:
- [**CommonLanguage**](https://huggingface.co/speechbrain/lang-id-commonlanguage_ecapa)
- [**VoxLingua107**](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa)

After obtaining baseline accuracy scores, we fine-tune these models using child speech data and compare the post-training performance.

## Methodology
1. Baseline evaluation: Test the pre-trained models on 6-7-year-old speech data.
2. Fine-Tuning: Train the models on manually extracted speech segments from the [**Shiro**](https://childes.talkbank.org/access/Spanish/Shiro.html) and [**OGI Kids' Speech**](https://catalog.ldc.upenn.edu/LDC2007S18) corpora.
3. Evaluation: Compare the fine-tuned models' accuracy to the baseline results. Test the fine-tuned model on the [**Madrid Corpus**](https://ilabs.uw.edu/sites/default/files/2020_ferjanramirez_kuhl_earlysecondlanguage.pdf).

## Current Status
Currently working on data augmentation and extracting segments for the OGI Kids' Speech Corpus.

Data Augmentation:
Implementing changes to the speed and pitch perturbation of child speech. Following Shahnawazuddin et al. (2020) has been shown to lead to improvements in WER in large vocabulary speech recognition tasks.

Hyperparameters:
Prasad et. al. (2024) added a linear layer with orthonormal constraints to ECAPA and used a learning rate of 3e-05 and a batch size of 8. These parameters apply to the present project because both studies work with limited datasets.

## Setup Instructions

### 1. Clone the Repository on Dryas (or your environment)
```bash
git clone https://github.com/your-username/child-lang-id.git
cd child-lang-id
```

### 2. Create the Conda Environment (Recommended)
```bash
conda env create -f environment.yml
conda activate langid
```

> If using pip instead of conda, see `requirements.txt`.

---

## Downloading the Pretrained Model

You must download the SpeechBrain VoxLingua107 ECAPA-TDNN model before running inference. You can do this by running the script below.

### Option 1: Run interactively
```bash
python
```
```python
from speechbrain.pretrained import EncoderClassifier
EncoderClassifier.from_hparams(
    source="speechbrain/lang-id-voxlingua107-ecapa",
    savedir="speechbrain_models"
)
```

### Option 2: Use a helper script
Create a file called `download_model.py` and paste:
```python
from speechbrain.pretrained import EncoderClassifier
EncoderClassifier.from_hparams(
    source="speechbrain/lang-id-voxlingua107-ecapa",
    savedir="speechbrain_models"
)
print("Model downloaded to ./speechbrain_models/")
```
Then run:
```bash
python download_model.py
```

---

## Preparing the Audio Data

The audio data (e.g. `cslu_segments`, `shiro_segments`) are available upon request. Once you obtain these files place them in a directory like this:

```
child-lang-id/
├── cslu_segments/
├── shiro_segments/
├── speechbrain_models/
```
---

## Running Inference

### Option 1: Run locally
```bash
bash baseline_inference.sh --model_dir speechbrain_models --data_dir cslu_segments --output_dir output_dir
```

### Option 2: Run on Condor
Edit the `baseline_inference.cmd` file:
```text
arguments = --model_dir speechbrain_models --data_dir cslu_segments --output_dir output_dir
transfer_input_files = baseline_inference.py,baseline_inference.sh,speechbrain_models/,cslu_segments/
```

Then submit:
```bash
condor_submit baseline_inference.cmd
```

---

## Updating Scripts for Your Environment

Make sure `baseline_inference.sh` uses the Python interpreter from your current environment. This is already handled with:

```bash
PYTHON_EXEC=$(which python)
```

So you don’t need to manually update any paths as long as you've activated the `langid` environment.

---

## Input/Output

- **Input**: Audio segments per subject (e.g., WAV files under `cslu_segments/subject/`)
- **Output**: `predictions.csv`, `per_subject_accuracy.csv`, logs in `logs/`

---

## Notes

- Pretrained model: `speechbrain/lang-id-voxlingua107-ecapa`
- Python version: `>=3.9` is supported (you can use 3.9.21)
- Uses `pandas`, `speechbrain`, `scikit-learn`, `torchaudio`
- Fine-tuning is handled in a separate script

---

## Citation

Coming soon :]
