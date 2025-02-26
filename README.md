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
3. Evaluation: Compare the fine-tuned models' accuracy to the baseline results. Test the fine-tuned model on the [**Madrid Corpus**]().

## Current Status
Fine-tuning in progress