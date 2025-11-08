"""
Audio utilities for loading and preprocessing speech signals.
"""

import numpy as np
import soundfile as sf
from scipy.io import wavfile
from scipy import signal as sp_signal
import os


class AudioProcessor:
    """
    Process and prepare audio signals.
    """

    def __init__(self, sample_rate=44100):
        """
        Initialize audio processor.

        Args:
            sample_rate: Sampling rate in Hz
        """
        self.sample_rate = sample_rate

    def load_audio(self, filename):
        """
        Load audio from file.

        Args:
            filename: Path to audio file

        Returns:
            audio: Audio signal
            sr: Sample rate
        """
        try:
            audio, sr = sf.read(filename)

            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)

            return audio, sr
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            return None, None

    def normalize(self, audio):
        """
        Normalize audio signal to [-1, 1] range.

        Args:
            audio: Input audio signal

        Returns:
            normalized_audio: Normalized signal
        """
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            normalized_audio = audio / max_val
        else:
            normalized_audio = audio

        return normalized_audio

    def save_audio(self, audio, filename, sample_rate=None):
        """
        Save audio signal to file.

        Args:
            audio: Audio signal to save
            filename: Output file path
            sample_rate: Sample rate (uses self.sample_rate if not provided)

        Returns:
            success: True if saved successfully
        """
        if sample_rate is None:
            sample_rate = self.sample_rate
        
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            # Normalize to prevent clipping
            audio_normalized = self.normalize(audio)
            
            # Save using soundfile
            sf.write(filename, audio_normalized, sample_rate)
            return True
        except Exception as e:
            print(f"Error saving {filename}: {e}")
            return False

    def add_noise(self, audio, snr_db=10, noise_type='white'):
        """
        Add noise to audio signal.

        Args:
            audio: Clean audio signal
            snr_db: Signal-to-Noise Ratio in dB
            noise_type: Type of noise ('white', 'fan', 'street', 'ambient')

        Returns:
            noisy_audio: Audio with added noise
            noise: The noise that was added
        """
        # Calculate signal power
        signal_power = np.mean(audio ** 2)

        # Calculate noise power from SNR
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear

        # Generate noise based on type
        if noise_type == 'white':
            noise = self._generate_white_noise(len(audio), noise_power)
        elif noise_type == 'fan':
            noise = self._generate_fan_noise(len(audio), noise_power)
        elif noise_type == 'street':
            noise = self._generate_street_noise(len(audio), noise_power)
        elif noise_type == 'ambient':
            noise = self._generate_ambient_noise(len(audio), noise_power)
        else:
            raise ValueError(f"Unknown noise type: {noise_type}. Use 'white', 'fan', 'street', or 'ambient'")

        # Add noise to signal
        noisy_audio = audio + noise

        return noisy_audio, noise

    def _generate_white_noise(self, length, power):
        """
        Generate white Gaussian noise.

        Args:
            length: Number of samples
            power: Noise power

        Returns:
            noise: White noise signal
        """
        noise = np.sqrt(power) * np.random.randn(length)
        return noise

    def _generate_fan_noise(self, length, power):
        """
        Generate fan noise (low-frequency tonal + broadband).

        Fan noise typically has:
        - Strong low-frequency tonal components (fundamental + harmonics)
        - Moderate broadband noise

        Args:
            length: Number of samples
            power: Target noise power

        Returns:
            noise: Synthesized fan noise
        """
        t = np.arange(length) / self.sample_rate

        # Fundamental frequency (50-120 Hz for typical fans)
        f0 = 60.0 + 20.0 * np.random.rand()

        # Tonal components (fundamental + harmonics)
        tonal = np.zeros(length)
        num_harmonics = 5
        for h in range(1, num_harmonics + 1):
            amplitude = 1.0 / h  # Decreasing amplitude for harmonics
            phase = 2 * np.pi * np.random.rand()
            tonal += amplitude * np.sin(2 * np.pi * f0 * h * t + phase)

        # Add slight frequency modulation (blade passing variations)
        modulation_freq = 2.0  # Hz
        modulation_depth = 0.1
        freq_mod = 1.0 + modulation_depth * np.sin(2 * np.pi * modulation_freq * t)
        tonal *= freq_mod

        # Broadband component (filtered white noise)
        broadband = np.random.randn(length)

        # Low-pass filter for broadband (fan noise is mostly low frequency)
        nyquist = self.sample_rate / 2
        cutoff = 2000  # Hz
        b, a = sp_signal.butter(4, cutoff / nyquist, btype='low')
        broadband = sp_signal.filtfilt(b, a, broadband)

        # Mix tonal (70%) and broadband (30%)
        noise = 0.7 * tonal + 0.3 * broadband

        # Normalize to target power
        current_power = np.mean(noise ** 2)
        noise = noise * np.sqrt(power / current_power)

        return noise

    def _generate_street_noise(self, length, power):
        """
        Generate street noise (traffic, wind, mixed frequencies).

        Street noise typically has:
        - Broadband component (traffic rumble)
        - Mid-frequency emphasis (car engines, horns)
        - Occasional transients (honks, brakes)
        - Wind-like low-frequency variations

        Args:
            length: Number of samples
            power: Target noise power

        Returns:
            noise: Synthesized street noise
        """
        # Broadband base (traffic rumble)
        broadband = np.random.randn(length)

        # Band-pass filter to emphasize mid frequencies (200-4000 Hz)
        nyquist = self.sample_rate / 2
        low_cutoff = 200 / nyquist
        high_cutoff = 4000 / nyquist
        b, a = sp_signal.butter(3, [low_cutoff, high_cutoff], btype='band')
        broadband = sp_signal.filtfilt(b, a, broadband)

        # Add low-frequency rumble (heavy vehicles)
        t = np.arange(length) / self.sample_rate
        rumble = 0.5 * np.sin(2 * np.pi * 40 * t + 2 * np.pi * np.random.rand())
        rumble += 0.3 * np.sin(2 * np.pi * 80 * t + 2 * np.pi * np.random.rand())

        # Add occasional transients (honks, brakes)
        num_transients = int(length / self.sample_rate * 2)  # ~2 per second
        transients = np.zeros(length)
        for _ in range(num_transients):
            pos = np.random.randint(0, length - int(0.2 * self.sample_rate))
            duration = int(0.1 * self.sample_rate)  # 100ms transients
            transient_freq = 400 + 800 * np.random.rand()
            t_local = np.arange(duration) / self.sample_rate
            transient_signal = np.sin(2 * np.pi * transient_freq * t_local)
            # Envelope
            envelope = np.exp(-10 * t_local)
            transients[pos:pos+duration] += 0.3 * transient_signal * envelope

        # Combine components
        noise = 0.6 * broadband + 0.3 * rumble + 0.1 * transients

        # Normalize to target power
        current_power = np.mean(noise ** 2)
        noise = noise * np.sqrt(power / current_power)

        return noise

    def _generate_ambient_noise(self, length, power):
        """
        Generate ambient noise (office, room tone, HVAC).

        Ambient noise typically has:
        - Pink noise character (1/f spectrum)
        - Low-level constant background
        - Occasional clicks/rustles

        Args:
            length: Number of samples
            power: Target noise power

        Returns:
            noise: Synthesized ambient noise
        """
        # Generate pink noise (1/f spectrum)
        # Using Voss-McCartney algorithm
        white = np.random.randn(length)

        # Apply 1/f filter using multiple octave bands
        pink = np.zeros(length)
        num_octaves = 10

        for i in range(num_octaves):
            # Generate white noise for this octave
            octave_noise = np.random.randn(length)

            # Low-pass filter with decreasing cutoff
            nyquist = self.sample_rate / 2
            cutoff = nyquist / (2 ** (i + 1))
            b, a = sp_signal.butter(1, cutoff / nyquist, btype='low')
            octave_filtered = sp_signal.filtfilt(b, a, octave_noise)

            # Add to pink noise
            pink += octave_filtered / (i + 1)

        # Normalize pink noise
        pink = pink / np.std(pink)

        # Add HVAC hum (low-frequency component)
        t = np.arange(length) / self.sample_rate
        hvac_hum = 0.2 * np.sin(2 * np.pi * 60 * t)  # 60 Hz electrical hum
        hvac_hum += 0.1 * np.sin(2 * np.pi * 120 * t)  # 120 Hz harmonic

        # Add occasional clicks/rustles
        num_clicks = int(length / self.sample_rate * 0.5)  # 0.5 per second
        clicks = np.zeros(length)
        for _ in range(num_clicks):
            pos = np.random.randint(0, length - int(0.01 * self.sample_rate))
            click_duration = int(0.005 * self.sample_rate)  # 5ms clicks
            clicks[pos:pos+click_duration] = 0.5 * np.random.randn(click_duration)

        # Combine components
        noise = 0.8 * pink + 0.15 * hvac_hum + 0.05 * clicks

        # Normalize to target power
        current_power = np.mean(noise ** 2)
        noise = noise * np.sqrt(power / current_power)

        return noise

    def resample(self, audio, original_sr, target_sr):
        """
        Resample audio to target sample rate.

        Args:
            audio: Input audio signal
            original_sr: Original sample rate
            target_sr: Target sample rate

        Returns:
            resampled_audio: Resampled signal
        """
        if original_sr == target_sr:
            return audio

        # Calculate resampling ratio
        num_samples = int(len(audio) * target_sr / original_sr)

        # Resample using scipy
        resampled_audio = sp_signal.resample(audio, num_samples)

        return resampled_audio

    def pre_emphasize(self, audio, coeff=0.97):
        """
        Apply pre-emphasis filter to enhance high frequencies.

        Args:
            audio: Input audio signal
            coeff: Pre-emphasis coefficient

        Returns:
            emphasized_audio: Pre-emphasized signal
        """
        emphasized_audio = np.append(audio[0], audio[1:] - coeff * audio[:-1])
        return emphasized_audio

    def compute_fft(self, audio):
        """
        Compute FFT of audio signal.

        Args:
            audio: Input audio signal

        Returns:
            frequencies: Frequency bins
            magnitude: Magnitude spectrum
        """
        # Compute FFT
        fft_result = np.fft.fft(audio)

        # Get magnitude spectrum (single-sided)
        n = len(audio)
        magnitude = np.abs(fft_result[:n // 2])

        # Frequency bins
        frequencies = np.fft.fftfreq(n, 1 / self.sample_rate)[:n // 2]

        return frequencies, magnitude


def create_dataset_structure(base_dir="data"):
    """
    Create directory structure for storing audio dataset.

    Args:
        base_dir: Base directory for dataset

    Returns:
        paths: Dictionary of directory paths
    """
    paths = {
        'base': base_dir,
        'raw': os.path.join(base_dir, 'raw'),
        'processed': os.path.join(base_dir, 'processed'),
        'database': os.path.join(base_dir, 'database'),
        'test': os.path.join(base_dir, 'test'),
        'results': os.path.join(base_dir, 'results')
    }

    # Create directories
    for path in paths.values():
        os.makedirs(path, exist_ok=True)

    return paths

