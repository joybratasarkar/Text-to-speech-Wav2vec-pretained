from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
import librosa

# Load the fine-tuned model and processor
processor = Wav2Vec2Processor.from_pretrained("./wav2vec2-finetuned-commonvoice-v1")
model = Wav2Vec2ForCTC.from_pretrained("./wav2vec2-finetuned-commonvoice-v1")

# Function to transcribe audio
def transcribe_audio(file_path):
    # Load audio file
    audio, rate = librosa.load(file_path, sr=16000)
    
    # Process audio
    input_values = processor(audio, sampling_rate=16000, return_tensors="pt").input_values
    
    # Predict
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    
    # Decode prediction
    transcription = processor.batch_decode(predicted_ids)[0]
    return transcription

# Example usage
if __name__ == "__main__":
    audio_file = "030750_maggie39s-diner-documentary-narration-71150.wav"
    transcription = transcribe_audio(audio_file)
    print("Transcription:", transcription)
