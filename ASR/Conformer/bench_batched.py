import argparse
import time
import random
from typing import List, Tuple
import numpy as np
import torch
from transformers import AutoModel


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def calculate_rtf(audio_duration: float, processing_time: float) -> float:
    """Calculate Real-Time Factor (RTF)."""
    return processing_time / audio_duration if audio_duration > 0 else 0.0


def parse_shape(shape_str: str) -> Tuple[int, ...]:
    """Parse shape string like '1,480000' into tuple of ints."""
    return tuple(int(x.strip()) for x in shape_str.split(','))


def main():
    parser = argparse.ArgumentParser(description="Benchmark ASR model with random tensor input")
    parser.add_argument("--model", type=str, required=True, default="ai4bharat/indic-conformer-600m-multilingual", help="HuggingFace model name")
    parser.add_argument("--shape", type=str, required=True, help="Tensor shape as comma-separated integers (e.g., 1,480000 for 2D)")
    parser.add_argument("--num_iterations", type=int, default=10, help="Number of inference iterations")
    parser.add_argument("--warmup", type=int, default=3, help="Number of warmup iterations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--language", type=str, default="hi", help="Language for transcription")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"], help="Data type for model weights")
    parser.add_argument("--decoder", type=str, default="rnnt", choices=["ctc", "rnnt"])
    parser.add_argument("--sample_rate", type=int, default=16000, help="Sample rate for audio duration calculation")
    args = parser.parse_args()

    # Set seed for reproducibility
    set_seed(args.seed)

    # Parse tensor shape
    tensor_shape = parse_shape(args.shape)
    print(f"Tensor shape: {tensor_shape}")

    # Calculate audio duration based on last dimension (assuming it's audio samples)
    audio_samples = tensor_shape[-1]
    audio_duration = audio_samples / args.sample_rate
    print(f"Audio duration per tensor: {audio_duration:.2f} seconds (at {args.sample_rate} Hz)")

    # Set device and dtype
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, args.dtype)
    print(f"Using device: {device}, dtype: {args.dtype}")

    # Load model
    print(f"Loading model: {args.model}")
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True).to(dtype)
    model.eval()
    model = model.to(device)

    # Create random tensor
    audio_tensor = torch.randn(tensor_shape, dtype=dtype)
    print(f"Created random tensor with shape: {audio_tensor.shape}")

    # Warmup iterations
    print(f"\nRunning {args.warmup} warmup iterations...")
    for i in range(args.warmup):
        with torch.no_grad():
            _ = model(audio_tensor, args.language, args.decoder)
        print(f"Warmup {i + 1}/{args.warmup} complete")

    # Benchmark iterations
    print(f"\nRunning {args.num_iterations} benchmark iterations...")
    iteration_times = []
    
    for i in range(args.num_iterations):
        start_time = time.perf_counter()
        with torch.no_grad():
            predictions = model(audio_tensor, args.language, args.decoder)
        end_time = time.perf_counter()
        
        iteration_time = end_time - start_time
        iteration_times.append(iteration_time)
        print(f"Iteration {i + 1}/{args.num_iterations} - Time: {iteration_time:.4f}s")

    # Calculate statistics
    total_processing_time = sum(iteration_times)
    avg_processing_time = np.mean(iteration_times)
    std_processing_time = np.std(iteration_times)
    min_processing_time = np.min(iteration_times)
    max_processing_time = np.max(iteration_times)
    
    total_audio_duration = audio_duration * args.num_iterations
    rtf = calculate_rtf(audio_duration, avg_processing_time)

    # Print results
    print("\n" + "=" * 50)
    print("BENCHMARK RESULTS")
    print("=" * 50)
    print(f"Model: {args.model}")
    print(f"Tensor shape: {tensor_shape}")
    print(f"Audio duration per iteration: {audio_duration:.2f} seconds")
    print(f"Number of iterations: {args.num_iterations}")
    print(f"Warmup iterations: {args.warmup}")
    print(f"Seed: {args.seed}")
    print(f"Dtype: {args.dtype}")
    print("-" * 50)
    print(f"Total audio duration: {total_audio_duration:.2f} seconds")
    print(f"Total processing time: {total_processing_time:.2f} seconds")
    print(f"Average processing time: {avg_processing_time:.4f} seconds")
    print(f"Std processing time: {std_processing_time:.4f} seconds")
    print(f"Min processing time: {min_processing_time:.4f} seconds")
    print(f"Max processing time: {max_processing_time:.4f} seconds")
    print(f"Real-Time Factor (RTF): {rtf:.4f}")
    print("=" * 50)

    # Return results as dict for programmatic use
    return {
        "model": args.model,
        "tensor_shape": tensor_shape,
        "audio_duration": audio_duration,
        "num_iterations": args.num_iterations,
        "warmup": args.warmup,
        "seed": args.seed,
        "total_audio_duration": total_audio_duration,
        "total_processing_time": total_processing_time,
        "avg_processing_time": avg_processing_time,
        "std_processing_time": std_processing_time,
        "min_processing_time": min_processing_time,
        "max_processing_time": max_processing_time,
        "rtf": rtf,
        "iteration_times": iteration_times
    }


if __name__ == "__main__":
    main()