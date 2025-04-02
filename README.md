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
