import argparse
import torch
import numpy as np
from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler
from transformers import AutoModel


def load_audio(audio_path: str, target_sr: int = 16000) -> np.ndarray:
    """Load audio file and resample to target sample rate."""
    import librosa
    audio, sr = librosa.load(audio_path, sr=target_sr)
    return audio


def pad_audio(audio: np.ndarray, target_length: int) -> torch.Tensor:
    """Pad audio array to target length and convert to tensor."""
    if len(audio) < target_length:
        padding = target_length - len(audio)
        audio = np.pad(audio, (0, padding), mode='constant')
    else:
        audio = audio[:target_length]
    return torch.from_numpy(audio).float().unsqueeze(0)


def main():
    parser = argparse.ArgumentParser(description="Profile Conformer model with PyTorch Profiler")
    parser.add_argument("--audio", type=str, required=True, help="Path to audio file for profiling")
    parser.add_argument("--model", type=str, default="ai4bharat/indic-conformer-600m-multilingual", 
                        help="HuggingFace model name")
    parser.add_argument("--warmup", type=int, default=3, help="Number of warmup iterations")
    parser.add_argument("--active", type=int, default=5, help="Number of active profiling iterations")
    parser.add_argument("--repeat", type=int, default=1, help="Number of times to repeat the cycle (total trials)")
    parser.add_argument("--output", type=str, default="./profile_output", 
                        help="Directory name to store profile output (for TensorBoard)")
    parser.add_argument("--language", type=str, default="hi", help="Language for transcription")
    parser.add_argument("--decoder", type=str, default="ctc", choices=["ctc", "rnnt"], 
                        help="Decoder type")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"], 
                        help="Data type for model weights")
    parser.add_argument("--export_chrome_trace", type=str, default=None, 
                        help="Path to export Chrome trace JSON file")
    parser.add_argument("--export_stacks", type=str, default=None,
                        help="Path to export stack traces for flame graph")
    parser.add_argument("--max_duration", type=float, default=30.0,
                        help="Maximum audio duration in seconds (audio will be padded/truncated)")
    args = parser.parse_args()

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, args.dtype)
    print(f"Using device: {device}, dtype: {dtype}")

    # Load model
    print(f"Loading model: {args.model}")
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True).to(dtype)
    model.eval()

    # Load and prepare audio
    print(f"Loading audio: {args.audio}")
    audio = load_audio(args.audio)
    target_length = int(16000 * args.max_duration)
    audio_tensor = pad_audio(audio, target_length)
    
    audio_duration = len(audio) / 16000
    print(f"Audio duration: {audio_duration:.2f}s (padded/truncated to {args.max_duration}s)")

    # Define the inference function
    def run_inference():
        with torch.no_grad():
            _ = model(audio_tensor, args.language, args.decoder)

    # Calculate total iterations
    total_iterations = (args.warmup + args.active) * args.repeat
    print(f"\nProfiling configuration:")
    print(f"  Warmup iterations: {args.warmup}")
    print(f"  Active iterations: {args.active}")
    print(f"  Repeat cycles: {args.repeat}")
    print(f"  Total iterations: {total_iterations}")
    print(f"  Output directory: {args.output}")

    # Setup profiler schedule
    profiler_schedule = schedule(
        wait=0,
        warmup=args.warmup,
        active=args.active,
        repeat=args.repeat
    )

    # Profile activities
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    print(f"\nStarting profiling...")
    
    with profile(
        activities=activities,
        schedule=profiler_schedule,
        on_trace_ready=tensorboard_trace_handler(args.output),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
        with_modules=True
    ) as prof:
        for i in range(total_iterations):
            run_inference()
            prof.step()

    print(f"\nProfiling complete!")
    print(f"TensorBoard traces saved to: {args.output}")

    # Print summary table
    print("\n" + "=" * 80)
    print("PROFILER SUMMARY (CPU time)")
    print("=" * 80)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))

    # Print self CPU time summary
    print("\n" + "=" * 80)
    print("PROFILER SUMMARY (Self CPU time)")
    print("=" * 80)
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=20))

    # Export Chrome trace if requested
    if args.export_chrome_trace:
        prof.export_chrome_trace(args.export_chrome_trace)
        print(f"\nChrome trace exported to: {args.export_chrome_trace}")

    # Export stacks for flame graph if requested
    if args.export_stacks:
        prof.export_stacks(args.export_stacks, "self_cpu_time_total")
        print(f"Stack traces exported to: {args.export_stacks}")
        print(f"Generate flame graph with: ./flamegraph.pl --title 'Conformer CPU' {args.export_stacks} > flamegraph.svg")

    print("\nTo view TensorBoard traces, run:")
    print(f"  tensorboard --logdir={args.output}")


if __name__ == "__main__":
    main()
