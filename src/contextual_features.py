"""
Contextual Feature Extraction ("Beyond Column" Enhancement).

Implements temporal context windows and non-linear feature expansions
to improve RBF-based Q estimation beyond simple per-frame features.

Based on ChatGPT requirements for enhanced feature engineering.
"""

import numpy as np
from scipy.fftpack import dct


def add_temporal_context(features, context_frames=5):
    """
    Add temporal context by concatenating neighboring frames.

    Args:
        features: Per-frame features [N, d]
        context_frames: Number of frames to include before/after (±k)

    Returns:
        contextual_features: [N, d*(2*context_frames+1)]
            Each row contains [frame_{n-k}, ..., frame_n, ..., frame_{n+k}]
    """
    features = np.atleast_2d(features)
    N, d = features.shape

    # Pad edges with replication
    padded = np.pad(features, ((context_frames, context_frames), (0, 0)), mode='edge')

    # Collect context windows
    context_list = []

    for i in range(N):
        # Window from i to i+2*context_frames+1 in padded array
        window = padded[i:i + 2*context_frames + 1, :]  # [2k+1, d]

        # Flatten to 1D
        context_vector = window.flatten()  # [(2k+1)*d]

        context_list.append(context_vector)

    contextual_features = np.array(context_list)
    
    # Handle NaN and inf values - replace with 0
    contextual_features = np.nan_to_num(contextual_features, nan=0.0, posinf=0.0, neginf=0.0)

    return contextual_features


def expand_nonlinear(features):
    """
    Add non-linear expansions of features.

    Expansions:
    1. Squared terms: x²
    2. Pairwise products: x_i * x_j for i < j
    3. Log transforms: log(1 + |x|)

    Args:
        features: Input features [N, d]

    Returns:
        expanded_features: [N, d_expanded]
            d_expanded = d + d + (d choose 2) + d
    """
    features = np.atleast_2d(features)
    N, d = features.shape

    expansions = [features]  # Original

    # 1. Squared terms
    squared = features ** 2
    expansions.append(squared)

    # 2. Pairwise products (upper triangle only)
    if d > 1:
        pairwise = []
        for i in range(d):
            for j in range(i+1, d):
                pairwise.append(features[:, i] * features[:, j])

        if pairwise:
            pairwise = np.column_stack(pairwise)
            expansions.append(pairwise)

    # 3. Log transforms
    log_features = np.log(1 + np.abs(features))
    expansions.append(log_features)

    # Concatenate all
    expanded_features = np.hstack(expansions)
    
    # Handle NaN and inf values - replace with 0
    expanded_features = np.nan_to_num(expanded_features, nan=0.0, posinf=0.0, neginf=0.0)

    return expanded_features


def extract_mfcc_features(signal, sr=16000, n_mfcc=13, frame_len=512, hop_len=256):
    """
    Extract MFCC features from signal.

    MFCCs are more robust spectral features than raw FFT magnitudes.

    Args:
        signal: Input signal
        sr: Sample rate
        n_mfcc: Number of MFCC coefficients
        frame_len: Frame length for FFT
        hop_len: Hop length between frames

    Returns:
        mfcc: MFCC features [N_frames, n_mfcc]
    """
    from scipy.signal import get_window
    from scipy.fftpack import fft

    signal = np.asarray(signal).flatten()

    # Number of frames
    n_frames = 1 + (len(signal) - frame_len) // hop_len

    # Window
    window = get_window('hann', frame_len)

    # Mel filterbank (simplified - use triangular filters)
    n_fft = frame_len
    n_mels = 40  # Standard

    # Mel scale
    def hz_to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    def mel_to_hz(mel):
        return 700 * (10**(mel / 2595) - 1)

    # Create mel filterbank
    mel_low = 0
    mel_high = hz_to_mel(sr / 2)
    mel_points = np.linspace(mel_low, mel_high, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    filterbank = np.zeros((n_mels, n_fft // 2 + 1))

    for m in range(1, n_mels + 1):
        f_left = bin_points[m - 1]
        f_center = bin_points[m]
        f_right = bin_points[m + 1]

        # Rising slope
        for k in range(f_left, f_center):
            if f_center > f_left:
                filterbank[m - 1, k] = (k - f_left) / (f_center - f_left)

        # Falling slope
        for k in range(f_center, f_right):
            if f_right > f_center:
                filterbank[m - 1, k] = (f_right - k) / (f_right - f_center)

    # Process frames
    mfcc_list = []

    for i in range(n_frames):
        start = i * hop_len
        end = start + frame_len

        if end > len(signal):
            break

        frame = signal[start:end] * window

        # FFT
        spectrum = np.abs(fft(frame, n=n_fft))
        power_spectrum = (spectrum[:n_fft // 2 + 1] ** 2) / n_fft

        # Apply mel filterbank
        mel_spectrum = filterbank @ power_spectrum

        # Log
        log_mel = np.log(mel_spectrum + 1e-10)

        # DCT to get MFCCs
        mfcc_frame = dct(log_mel, type=2, norm='ortho')[:n_mfcc]

        mfcc_list.append(mfcc_frame)

    mfcc = np.array(mfcc_list)

    return mfcc


def extract_advanced_frame_features(signal, sr=16000, frame_len=320, hop_len=160):
    """
    Extract comprehensive per-frame features for RBF-Q estimation.

    Features extracted:
    1. RMS energy
    2. Zero-crossing rate
    3. Spectral centroid
    4. Spectral flatness
    5. Local SNR estimate
    6. MFCCs (first 5 coefficients)

    Args:
        signal: Input signal
        sr: Sample rate
        frame_len: Frame length in samples
        hop_len: Hop length in samples

    Returns:
        features: [N_frames, d] where d is total number of features
    """
    from scipy.signal import get_window
    from scipy.fftpack import fft

    signal = np.asarray(signal).flatten()

    # Number of frames
    n_frames = 1 + (len(signal) - frame_len) // hop_len

    window = get_window('hann', frame_len)

    feature_list = []

    for i in range(n_frames):
        start = i * hop_len
        end = start + frame_len

        if end > len(signal):
            break

        frame = signal[start:end]
        frame_windowed = frame * window

        # 1. RMS energy
        rms = np.sqrt(np.mean(frame ** 2))

        # 2. Zero-crossing rate
        zcr = np.sum(np.abs(np.diff(np.sign(frame)))) / (2 * frame_len)

        # 3. Spectral centroid
        spectrum = np.abs(fft(frame_windowed, n=frame_len))
        freqs = np.fft.fftfreq(frame_len, 1/sr)[:frame_len // 2]
        magnitude = spectrum[:frame_len // 2]

        if np.sum(magnitude) > 0:
            centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
        else:
            centroid = 0

        # 4. Spectral flatness
        if np.sum(magnitude) > 0:
            geometric_mean = np.exp(np.mean(np.log(magnitude + 1e-10)))
            arithmetic_mean = np.mean(magnitude)
            flatness = geometric_mean / (arithmetic_mean + 1e-10)
        else:
            flatness = 0

        # 5. Local SNR estimate (simple: signal variance / min variance in window)
        if len(signal) > end + frame_len:
            local_var = np.var(frame)
            noise_est = np.min([np.var(signal[max(0, start-frame_len):start]),
                               np.var(signal[end:min(len(signal), end+frame_len)])])
            local_snr = 10 * np.log10((local_var + 1e-10) / (noise_est + 1e-10))
        else:
            local_snr = 0

        # Combine features
        frame_features = [rms, zcr, centroid, flatness, local_snr]

        feature_list.append(frame_features)

    features = np.array(feature_list)
    
    # Handle NaN and inf values - replace with 0
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    return features


if __name__ == "__main__":
    # Test contextual features
    print("Testing contextual features module...")

    # Generate synthetic signal
    sr = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    signal = np.sin(2 * np.pi * 440 * t) + 0.1 * np.random.randn(len(t))

    # Extract base features
    print("\n1. Extracting per-frame features...")
    features = extract_advanced_frame_features(signal, sr, frame_len=320, hop_len=160)
    print(f"   Base features shape: {features.shape}")

    # Add temporal context
    print("\n2. Adding temporal context (±5 frames)...")
    context_features = add_temporal_context(features, context_frames=5)
    print(f"   Contextual features shape: {context_features.shape}")
    print(f"   Expansion factor: {context_features.shape[1] / features.shape[1]:.1f}x")

    # Non-linear expansion
    print("\n3. Adding non-linear expansions...")
    expanded_features = expand_nonlinear(features)
    print(f"   Expanded features shape: {expanded_features.shape}")
    print(f"   Expansion factor: {expanded_features.shape[1] / features.shape[1]:.1f}x")

    # MFCCs
    print("\n4. Extracting MFCCs...")
    mfcc = extract_mfcc_features(signal, sr, n_mfcc=13)
    print(f"   MFCC shape: {mfcc.shape}")

    # Combined: context + nonlinear
    print("\n5. Combining context + nonlinear...")
    combined = add_temporal_context(expanded_features, context_frames=3)
    print(f"   Combined features shape: {combined.shape}")
    print(f"   Total expansion: {combined.shape[1] / features.shape[1]:.1f}x")

    print("\n✓ Contextual features module test complete")
