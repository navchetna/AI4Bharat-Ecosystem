import argparse
import time
import random
from typing import List
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModel
from jiwer import wer


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def calculate_rtf(audio_duration: float, processing_time: float) -> float:
    """Calculate Real-Time Factor (RTF)."""
    return processing_time / audio_duration if audio_duration > 0 else 0.0


def pad_audio(audio_list: List[np.ndarray], target_length: int) -> torch.Tensor:
    """Pad audio arrays to the same length and convert to tensor."""
    padded_audios = []
    for audio in audio_list:
        if len(audio) < target_length:
            padding = target_length - len(audio)
            padded_audio = np.pad(audio, (0, padding), mode='constant')
        else:
            padded_audio = audio[:target_length]
        padded_audios.append(padded_audio)
    return torch.from_numpy(np.stack(padded_audios)).float()


def main():
    parser = argparse.ArgumentParser(description="Benchmark Whisper model on audio dataset")
    parser.add_argument("--model", type=str, required=True, default="ai4bharat/indic-conformer-600m-multilingual", help="HuggingFace model name (e.g., openai/whisper-small)")
    parser.add_argument("--dataset", type=str, required=True, help="HuggingFace dataset name")
    parser.add_argument("--dataset_config", type=str, default=None, help="Dataset configuration name")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to use")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to process")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--language", type=str, default=None, help="Language for transcription")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"], help="Data type for model weights")
    parser.add_argument("--decoder", type=str, default="ctc", choices=["ctc", "rnnt"])
    args = parser.parse_args()

    # Set seed for reproducibility
    set_seed(args.seed)

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, args.dtype)
    print(f"Using device: {device}")

    # Load model and processor
    print(f"Loading model: {args.model}")
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True).to(dtype)
    model.eval()

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    if args.dataset_config:
        dataset = load_dataset(args.dataset, args.dataset_config, split=args.split, trust_remote_code=True)
    else:
        dataset = load_dataset(args.dataset, split=args.split, trust_remote_code=True)

    # Sample from dataset
    num_samples = min(args.num_samples, len(dataset))
    indices = random.sample(range(len(dataset)), num_samples)
    sampled_dataset = dataset.select(indices)

    print(f"Processing {num_samples} samples with batch size {args.batch_size}")

    # Prepare for metrics
    all_predictions = []
    all_references = []
    total_audio_duration = 0.0
    total_processing_time = 0.0

    # Process in batches
    for i in range(0, num_samples, args.batch_size):
        batch_end = min(i + args.batch_size, num_samples)
        batch = sampled_dataset.select(range(i, batch_end))

        # Extract audio and references
        audio_arrays = []
        references = []
        batch_audio_duration = 0.0

        for sample in batch:
            # Handle different dataset formats for audio
            if "audio" in sample:
                audio_data = sample["audio"]
                audio_array = audio_data["array"]
                sampling_rate = audio_data["sampling_rate"]
            elif "speech" in sample:
                audio_array = sample["speech"]
                sampling_rate = sample.get("sampling_rate", 16000)
            else:
                raise ValueError("Could not find audio data in sample")

            # Resample if necessary
            if sampling_rate != 16000:
                import librosa
                audio_array = librosa.resample(audio_array, orig_sr=sampling_rate, target_sr=16000)

            audio_arrays.append(audio_array)
            audio_duration = len(audio_array) / 16000
            batch_audio_duration += audio_duration

            # Handle different dataset formats for reference text
            if "text" in sample:
                references.append(sample["text"])
            elif "sentence" in sample:
                references.append(sample["sentence"])
            elif "transcription" in sample:
                references.append(sample["transcription"])
            else:
                references.append("")

        total_audio_duration += batch_audio_duration
        # Pad audio
        audio_arrays = pad_audio(audio_arrays, target_length=16000 * 30)
        audio_tensors = torch.tensor(audio_arrays)

        # Generate with timing
        start_time = time.perf_counter()
        with torch.no_grad():
            predictions = model(audio_tensors, args.language, args.decoder)
        
        end_time = time.perf_counter()
        batch_time = end_time - start_time
        total_processing_time += batch_time

        # Decode predictions
        all_predictions.extend(predictions)
        all_references.extend(references)

        print(f"Processed batch {i // args.batch_size + 1}/{(num_samples + args.batch_size - 1) // args.batch_size} "
              f"- Batch time: {batch_time:.2f}s")

    # Calculate metrics
    # Filter out empty references for WER calculation
    valid_pairs = [(p, r) for p, r in zip(all_predictions, all_references) if r.strip()]
    
    if valid_pairs:
        valid_predictions, valid_references = zip(*valid_pairs)
        word_error_rate = wer(list(valid_references), list(valid_predictions))
    else:
        word_error_rate = float('nan')

    rtf = calculate_rtf(total_audio_duration, total_processing_time)

    # Print results
    print("\n" + "=" * 50)
    print("BENCHMARK RESULTS")
    print("=" * 50)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Samples processed: {num_samples}")
    print(f"Batch size: {args.batch_size}")
    print(f"Seed: {args.seed}")
    print("-" * 50)
    print(f"Total audio duration: {total_audio_duration:.2f} seconds")
    print(f"Total processing time: {total_processing_time:.2f} seconds")
    print(f"Real-Time Factor (RTF): {rtf:.4f}")
    print(f"Word Error Rate (WER): {word_error_rate:.4f}" if not np.isnan(word_error_rate) else "WER: N/A (no valid references)")
    print("=" * 50)

    # Return results as dict for programmatic use
    return {
        "model": args.model,
        "dataset": args.dataset,
        "num_samples": num_samples,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "total_audio_duration": total_audio_duration,
        "total_processing_time": total_processing_time,
        "rtf": rtf,
        "wer": word_error_rate
    }


if __name__ == "__main__":
    main()