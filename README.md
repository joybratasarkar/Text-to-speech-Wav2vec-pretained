# Wav2Vec2 Fine-Tuning for Automatic Speech Recognition (ASR)

## Overview
This repository contains scripts for fine-tuning the Wav2Vec2 model on a speech dataset, augmenting audio data, and evaluating the model's performance using Word Error Rate (WER). It includes the following main components:

- **`train.py`** – Fine-tunes Wav2Vec2 on a speech dataset with augmentation.
- **`predict.py`** – Uses the trained model to transcribe audio files.
- **`wer.py`** – Evaluates the model's performance using WER.

---

## Features
- **Data Augmentation**: Adds noise, time stretching, pitch shifting, and silence insertion to improve model generalization.
- **Efficient Training**: Uses `Trainer` from Hugging Face Transformers to fine-tune the model.
- **Automatic Transcription**: Converts speech to text using Wav2Vec2.
- **WER Calculation**: Evaluates model accuracy by computing Word Error Rate.
- **Supports Local & Remote Datasets**: Uses Mozilla Common Voice dataset or a local dataset.

---

## Installation
Ensure you have the required dependencies installed:
```sh
pip install torch transformers datasets librosa evaluate pandas
```

---

## Training the Model
To fine-tune the Wav2Vec2 model, run:
```sh
python train.py
```
This script:
1. Loads and preprocesses the dataset.
2. Applies augmentations to training data.
3. Fine-tunes Wav2Vec2 for 5 epochs.
4. Saves the trained model and processor.

---

## Transcribing Audio
To transcribe an audio file, use:
```sh
python predict.py path/to/audio.wav
```

---

## Evaluating the Model
To calculate the WER on a dataset, run:
```sh
python wer.py
```
This script:
1. Loads a test dataset (local or Common Voice).
2. Transcribes each audio file.
3. Computes and prints the Word Error Rate.

---

## Dataset Preparation
1. Place audio files in the `./en/clips/` directory.
2. Ensure `validated.tsv` contains metadata with `path` and `sentence` columns.

---

## Model Details
- Pretrained model: `facebook/wav2vec2-large-960h`
- Fine-tuning method: CTC Loss Optimization
- Augmentation methods: Noise addition, time stretching, pitch shifting, silence insertion

---

## Acknowledgments
This project utilizes:
- Hugging Face `transformers` and `datasets`
- Mozilla Common Voice dataset
- Librosa for audio processing

For questions, open an issue in the repository.

