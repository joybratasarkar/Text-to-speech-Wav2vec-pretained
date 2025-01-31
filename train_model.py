from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset
import os
import pandas as pd
import librosa
import numpy as np
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union


# Data Collator for CTC
@dataclass
class DataCollatorForCTC:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = "longest"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features, padding=self.padding, return_tensors="pt"
        )

        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features, padding=self.padding, return_tensors="pt"
            )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch["input_ids"] == self.processor.tokenizer.pad_token_id, -100
        )
        batch["labels"] = labels
        return batch


# Augmentation Functions
def add_noise(audio, noise_level=0.005):
    noise = np.random.randn(len(audio))
    return audio + noise_level * noise

def time_stretch(audio, rate=0.9):
    return librosa.effects.time_stretch(audio, rate)

def pitch_shift(audio, sr, n_steps=2):
    return librosa.effects.pitch_shift(audio, sr, n_steps=n_steps)

def insert_silence(audio, duration=0.5, sr=16000):
    silence = np.zeros(int(duration * sr))
    idx = np.random.randint(0, len(audio))
    return np.concatenate([audio[:idx], silence, audio[idx:]])

def augment_audio(audio, sr):
    # Apply a random set of augmentations
    if np.random.rand() < 0.5:
        audio = add_noise(audio)
    if np.random.rand() < 0.5:
        audio = time_stretch(audio, rate=np.random.uniform(0.8, 1.2))
    if np.random.rand() < 0.5:
        audio = pitch_shift(audio, sr, n_steps=np.random.randint(-2, 2))
    if np.random.rand() < 0.3:
        audio = insert_silence(audio, duration=np.random.uniform(0.2, 0.5), sr=sr)
    return audio


# Main Training Pipeline
def main():
    # Dataset Paths
    dataset_dir = "./en"
    metadata_path = os.path.join(dataset_dir, "validated.tsv")
    audio_dir = os.path.join(dataset_dir, "clips")

    # Load metadata
    metadata = pd.read_csv(metadata_path, sep="\t")[["path", "sentence"]]
    metadata["audio"] = metadata["path"].apply(lambda x: os.path.join(audio_dir, x))
    metadata = metadata[metadata["audio"].apply(os.path.exists)]

    # Convert to Hugging Face Dataset
    hf_dataset = Dataset.from_pandas(metadata)
    train_test_split = hf_dataset.train_test_split(test_size=0.1)
    train_dataset, test_dataset = train_test_split["train"], train_test_split["test"]

    # Load Wav2Vec2 Processor
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")

    # Preprocessing Function with Augmentation
    def prepare_batch(batch):
        audio, _ = librosa.load(batch["audio"], sr=16000)
        # Apply augmentation only to training data
        if "train" in batch["audio"]:
            audio = augment_audio(audio, sr=16000)
        batch["input_values"] = processor(audio, sampling_rate=16000).input_values[0]
        batch["labels"] = processor.tokenizer(
            batch["sentence"], truncation=True, max_length=128
        ).input_ids
        return batch

    # Apply preprocessing
    train_dataset = train_dataset.map(prepare_batch, remove_columns=["audio", "sentence"])
    test_dataset = test_dataset.map(prepare_batch, remove_columns=["audio", "sentence"])

    # Load Pretrained Wav2Vec2 Model
    model = Wav2Vec2ForCTC.from_pretrained(
        "facebook/wav2vec2-large-960h",
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
    )

    # Define Data Collator
    data_collator = DataCollatorForCTC(processor)

    # Training Arguments
    training_args = TrainingArguments(
        output_dir="./wav2vec2-finetuned-augmented",
        evaluation_strategy="steps",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        num_train_epochs=5,
        save_steps=500,
        eval_steps=500,
        logging_steps=500,
        learning_rate=1e-5,
        warmup_steps=1000,
        lr_scheduler_type="linear",
        fp16=True,
        logging_dir="./logs",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=processor,
        data_collator=data_collator,
    )

    # Train the Model
    trainer.train()

    # Save Model and Processor
    model.save_pretrained("./wav2vec2-finetuned-augmented")
    processor.save_pretrained("./wav2vec2-finetuned-augmented")


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
    main()
