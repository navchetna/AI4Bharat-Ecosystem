import os
import numpy as np
import soundfile as sf


MAX_VALUE = 32768.0  # For int16 scaling

def align_by_correlation(a1, a2):
    # Normalize to zero mean to focus on shape, not DC offset
    a1_norm = a1 - np.mean(a1)
    a2_norm = a2 - np.mean(a2)
    
    corr = np.correlate(a1_norm, a2_norm, mode="full")
    shift = np.argmax(corr) - (len(a2) - 1)

    # Apply shift
    if shift > 0:
        # a1 is "ahead" of a2, crop start of a1
        a1 = a1[shift:]
        a2 = a2[:len(a1)]
    elif shift < 0:
        # a2 is "ahead", crop start of a2
        a2 = a2[-shift:]
        a1 = a1[:len(a2)]

    # Crop ends to match length
    min_len = min(len(a1), len(a2))
    return a1[:min_len], a2[:min_len], shift


def evaluate_audios(audios_dir_1, audios_dir_2):
    files1 = sorted(os.listdir(audios_dir_1))
    files2 = sorted(os.listdir(audios_dir_2))

    total_rmse = 0.0
    total_snr = 0.0
    count = 0

    for f1, f2 in zip(files1, files2):
        if f1 != f2:
            print(f"Skipping mismatched files: {f1}, {f2}")
            continue

        p1 = os.path.join(audios_dir_1, f1)
        p2 = os.path.join(audios_dir_2, f2)

        if f1.endswith(".npy"):
            # Normalizing the value to float32 range by dividing by the int16 range
            a1 = np.load(p1) / MAX_VALUE        # Numpy already stored in int16
            a2 = np.load(p2) / MAX_VALUE        # Numpy already stored in int16
        else:
            # Normalizing the value to float32 range by dividing by the int16 range
            a1, sr1 = sf.read(p1, dtype="int16") 
            a2, sr2 = sf.read(p2, dtype="int16") 
            a1, a2 = a1 / MAX_VALUE, a2 / MAX_VALUE
            assert sr1 == sr2, "Sample rate mismatch!"

        a1, a2, shift = align_by_correlation(a1, a2)

        # RMSE
        rmse = np.sqrt(np.mean((a1 - a2) ** 2))

        # SNR
        signal_power = np.mean(a1 ** 2)
        noise_power = np.mean((a1 - a2) ** 2)
        snr = 10 * np.log10(signal_power / (noise_power + 1e-12))

        total_rmse += rmse
        total_snr += snr
        count += 1

    return total_rmse / count, total_snr / count


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate rmse between two directories of audio files. Pass the path to the audio directories for comparison.")
    parser.add_argument(
        "--audios_dir_1", type=str, required=True, help="Path to first directory of audios"
    )
    parser.add_argument(
        "--audios_dir_2", type=str, required=True, help="Path to second directory of audios"
    )
    
    args = parser.parse_args()

    avg_rmse, avg_snr = evaluate_audios(args.audios_dir_1, args.audios_dir_2)
    print(f"Average rmse between audio files: {avg_rmse}")
    print(f"Average SNR between audio files: {avg_snr}")

    # NOTE 
    # 1. The RMSE without alignment was very high pointing for alignement requirement since the audio sounded almost the same
    # 2. The minute difference in numerical calculation results in different audio duration by the druation predictor leading to different length audio files.
    # 3. A 1 sample shift can completely flips the phase, increase numerical error significantly. Hence alignment is very important before calculating such metrics.
    # 4. For audio scaling, dont not use downscaling -> audio.astype(np.int16) instead use proper scaling -> audio * 32768 and then convert to int16 for proper RMSE calculation.





