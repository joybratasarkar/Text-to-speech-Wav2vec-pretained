from datasets import load_dataset, Dataset
from evaluate import load  # For WER metric
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
import pandas as pd
import librosa
import os

# Load the fine-tuned model and processor
processor = Wav2Vec2Processor.from_pretrained("./wav2vec2-finetuned-commonvoice")
model = Wav2Vec2ForCTC.from_pretrained("./wav2vec2-finetuned-commonvoice")

# Load the WER metric
wer_metric = load("wer")

# Function to transcribe audio
def transcribe_audio(audio_path):
    # Load and resample audio to 16kHz
    audio, rate = librosa.load(audio_path, sr=16000)
    
    # Prepare input for model
    input_values = processor(audio, sampling_rate=16000, return_tensors="pt").input_values

    # Predict
    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)

    # Decode transcription
    transcription = processor.batch_decode(predicted_ids)[0]
    return transcription

# Function to evaluate using local dataset
def evaluate_local_dataset():
    dataset_dir = "./en"
    metadata_path = os.path.join(dataset_dir, "validated.tsv")
    audio_dir = os.path.join(dataset_dir, "clips")

    # Check if the local dataset exists
    if not os.path.exists(metadata_path):
        print("Local dataset not found. Falling back to the updated Common Voice dataset.")
        return None

    # Load metadata
    metadata = pd.read_csv(metadata_path, sep="\t")[["path", "sentence"]]
    metadata["audio"] = metadata["path"].apply(lambda x: os.path.join(audio_dir, x))
    metadata = metadata[metadata["audio"].apply(os.path.exists)]

    # Convert to Hugging Face Dataset
    hf_dataset = Dataset.from_pandas(metadata)
    test_dataset = hf_dataset.train_test_split(test_size=0.1)["test"]
    return test_dataset

# Function to evaluate using remote dataset
def evaluate_remote_dataset():
    print("Loading the updated Common Voice dataset...")
    return load_dataset("mozilla-foundation/common_voice_11_0", "en", split="test[:10]")

# Evaluation function
def evaluate_model(dataset):
    predictions = []
    references = []

    for sample in dataset:
        audio_path = sample["audio"]
        ground_truth = sample["sentence"]

        # Transcribe the audio
        prediction = transcribe_audio(audio_path)

        # Store predictions and references
        predictions.append(prediction.lower())
        references.append(ground_truth.lower())

        print(f"Audio: {audio_path}")
        print(f"Prediction: {prediction}")
        print(f"Reference: {ground_truth}")
        print()

    # Compute WER
    wer = wer_metric.compute(predictions=predictions, references=references)
    print(f"Word Error Rate (WER): {wer:.2%}")
    return wer

# Main function
if __name__ == "__main__":
    # Try to load the local dataset
    dataset = evaluate_local_dataset()

    # If local dataset is not found, fall back to the updated remote dataset
    if dataset is None:
        dataset = evaluate_remote_dataset()

    # Evaluate the model
    evaluate_model(dataset)
