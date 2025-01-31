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
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union

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


def main():
    dataset_dir = "./en"
    metadata_path = os.path.join(dataset_dir, "validated.tsv")
    audio_dir = os.path.join(dataset_dir, "clips")

    metadata = pd.read_csv(metadata_path, sep="\t")[["path", "sentence"]]
    metadata["audio"] = metadata["path"].apply(lambda x: os.path.join(audio_dir, x))
    metadata = metadata[metadata["audio"].apply(os.path.exists)]

    hf_dataset = Dataset.from_pandas(metadata)
    train_test_split = hf_dataset.train_test_split(test_size=0.1)
    train_dataset, test_dataset = train_test_split["train"], train_test_split["test"]

    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")

    def prepare_batch(batch):
        audio, _ = librosa.load(batch["audio"], sr=16000)
        batch["input_values"] = processor(audio, sampling_rate=16000).input_values[0]
        batch["labels"] = processor.tokenizer(
            batch["sentence"], truncation=True, max_length=128
        ).input_ids
        return batch

    train_dataset = train_dataset.map(prepare_batch, remove_columns=["audio", "sentence"])
    test_dataset = test_dataset.map(prepare_batch, remove_columns=["audio", "sentence"])

    model = Wav2Vec2ForCTC.from_pretrained(
        "facebook/wav2vec2-large-960h",
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
    )

    data_collator = DataCollatorForCTC(processor)

    training_args = TrainingArguments(
    
    output_dir="./wav2vec2-finetuned-optimized",
    evaluation_strategy="steps",
    per_device_train_batch_size=16,  # Increased batch size
    gradient_accumulation_steps=2,
    num_train_epochs=10,  # Increased epochs
    learning_rate=1e-5,
    warmup_steps=1000,
    lr_scheduler_type="cosine",  # Smoothed decay
    weight_decay=0.01,  # Regularization
    logging_steps=250,  # More frequent logging
    eval_steps=250,  # More frequent evaluation
    save_steps=500,
    save_total_limit=3,
    fp16=True,
    dataloader_num_workers=4,
    logging_dir="./logs",
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=processor,
        data_collator=data_collator,
    )

    trainer.train()
    model.save_pretrained("./wav2vec2-finetuned-commonvoice-v1")
    processor.save_pretrained("./wav2vec2-finetuned-commonvoice-v1")


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
    main()
