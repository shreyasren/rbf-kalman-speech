"""
Visualization utilities for plotting signals and results.
Recreates figures from the paper.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


class SpeechVisualizer:
    """
    Visualization tools for speech signals and enhancement results.
    """

    def __init__(self, figsize=(12, 8)):
        """
        Initialize visualizer.

        Args:
            figsize: Default figure size
        """
        self.figsize = figsize
        plt.style.use('seaborn-v0_8-darkgrid')

    def plot_time_domain(self, signals, titles, sample_rate=44100, save_path=None):
        """
        Plot signals in time domain (recreates Figures 1 and 3 from paper).

        Args:
            signals: List of signals to plot
            titles: List of titles for each signal
            sample_rate: Sampling rate
            save_path: Path to save figure
        """
        n_signals = len(signals)
        fig, axes = plt.subplots(n_signals, 1, figsize=(10, 3 * n_signals))

        if n_signals == 1:
            axes = [axes]

        for idx, (signal, title) in enumerate(zip(signals, titles)):
            time = np.arange(len(signal)) / sample_rate

            axes[idx].plot(time, signal, 'b-', linewidth=0.5)
            axes[idx].set_xlabel('Time (Sec)')
            axes[idx].set_ylabel('Amplitude')
            axes[idx].set_title(title)
            axes[idx].grid(True, alpha=0.3)
            axes[idx].set_ylim([-1, 1])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")

        return fig

    def plot_frequency_domain(self, signals, titles, sample_rate=44100, save_path=None):
        """
        Plot signals in frequency domain (recreates Figures 2 and 4 from paper).

        Args:
            signals: List of signals to plot
            titles: List of titles for each signal
            sample_rate: Sampling rate
            save_path: Path to save figure
        """
        n_signals = len(signals)
        fig, axes = plt.subplots(n_signals, 1, figsize=(10, 3 * n_signals))

        if n_signals == 1:
            axes = [axes]

        for idx, (signal, title) in enumerate(zip(signals, titles)):
            # Compute FFT
            n = len(signal)
            fft_result = np.fft.fft(signal)
            magnitude = np.abs(fft_result[:n // 2])
            frequencies = np.fft.fftfreq(n, 1 / sample_rate)[:n // 2]

            axes[idx].plot(frequencies, magnitude, 'b-', linewidth=0.5)
            axes[idx].set_xlabel('Frequency (Hz)')
            axes[idx].set_ylabel('Magnitude')
            axes[idx].set_title(f'Single-sided Magnitude spectrum (Hertz) of {title}')
            axes[idx].grid(True, alpha=0.3)
            axes[idx].set_xlim([0, sample_rate / 2])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")

        return fig

    def plot_envelope_detection(self, signal, envelope, voiced_mask, sample_rate=44100, save_path=None):
        """
        Plot envelope detection results (recreates Figure 5 from paper).

        Args:
            signal: Original signal
            envelope: Detected envelope
            voiced_mask: Boolean mask for voiced regions
            sample_rate: Sampling rate
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        time = np.arange(len(signal)) / sample_rate

        # Plot original signal
        ax.plot(time, signal, 'b-', linewidth=0.5, label='Audio signal', alpha=0.7)

        # Plot envelope
        ax.plot(time, envelope, 'r-', linewidth=2, label='Envelope')

        ax.set_xlabel('Time (Sec)')
        ax.set_ylabel('Amplitude')
        ax.set_title('Original Signal + Envelope of Mario')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")

        return fig

    def plot_voiced_signal(self, voiced_signal, sample_rate=44100, save_path=None):
        """
        Plot extracted voiced signal (recreates Figure 6 from paper).

        Args:
            voiced_signal: Voiced-only signal
            sample_rate: Sampling rate
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))

        samples = np.arange(len(voiced_signal))

        ax.plot(samples, voiced_signal, 'b-', linewidth=0.5)
        ax.set_xlabel('Samples')
        ax.set_ylabel('Amplitude')
        ax.set_title('Edited Signal')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")

        return fig

    def plot_enhancement_results(self, original, noisy, enhanced, sample_rate=44100, save_path=None):
        """
        Plot speech enhancement results (recreates Figure 7 from paper).

        Args:
            original: Original clean signal
            noisy: Noisy signal
            enhanced: Enhanced signal
            sample_rate: Sampling rate
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(3, 1, figsize=(10, 10))

        time_orig = np.arange(len(original)) / sample_rate
        time_noisy = np.arange(len(noisy)) / sample_rate
        time_enhanced = np.arange(len(enhanced)) / sample_rate

        # Original signal
        axes[0].plot(time_orig, original, 'g-', linewidth=0.5)
        axes[0].set_ylabel('Amplitude')
        axes[0].set_title('ORIGINAL SIGNAL')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim([-2, 2])

        # Noisy speech signal
        axes[1].plot(time_noisy, noisy, 'b-', linewidth=0.5)
        axes[1].set_ylabel('Amplitude')
        axes[1].set_title('Noisy Speech Signal')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([-2, 2])

        # Enhanced signal
        axes[2].plot(time_enhanced, enhanced, 'g-', linewidth=0.5)
        axes[2].set_xlabel('Time (Sec)')
        axes[2].set_ylabel('Amplitude')
        axes[2].set_title('ESTIMATED SIGNAL')
        axes[2].grid(True, alpha=0.3)
        axes[2].set_ylim([-2, 2])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")

        return fig

    def plot_comparison_grid(self, signals_dict, sample_rate=44100, save_path=None):
        """
        Plot multiple signals in a grid layout.

        Args:
            signals_dict: Dictionary of {label: signal}
            sample_rate: Sampling rate
            save_path: Path to save figure
        """
        n_signals = len(signals_dict)
        n_cols = 2
        n_rows = (n_signals + 1) // 2

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
        axes = axes.flatten()

        for idx, (label, signal) in enumerate(signals_dict.items()):
            time = np.arange(len(signal)) / sample_rate

            axes[idx].plot(time, signal, 'b-', linewidth=0.5)
            axes[idx].set_xlabel('Time (Sec)')
            axes[idx].set_ylabel('Amplitude')
            axes[idx].set_title(label)
            axes[idx].grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(n_signals, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")

        return fig

    def plot_recognition_results(self, test_signal, recognized_word, confidence, all_scores,
                                 sample_rate=44100, save_path=None):
        """
        Plot speech recognition results.

        Args:
            test_signal: Test signal
            recognized_word: Recognized word
            confidence: Recognition confidence
            all_scores: Scores for all words
            sample_rate: Sampling rate
            save_path: Path to save figure
        """
        fig = plt.figure(figsize=(12, 8))
        gs = GridSpec(2, 2, figure=fig)

        # Plot test signal
        ax1 = fig.add_subplot(gs[0, :])
        time = np.arange(len(test_signal)) / sample_rate
        ax1.plot(time, test_signal, 'b-', linewidth=0.5)
        ax1.set_xlabel('Time (Sec)')
        ax1.set_ylabel('Amplitude')
        ax1.set_title(f'Test Signal - Recognized: "{recognized_word}" (Confidence: {confidence:.3f})')
        ax1.grid(True, alpha=0.3)

        # Plot FFT
        ax2 = fig.add_subplot(gs[1, 0])
        n = len(test_signal)
        fft_result = np.fft.fft(test_signal)
        magnitude = np.abs(fft_result[:n // 2])
        frequencies = np.fft.fftfreq(n, 1 / sample_rate)[:n // 2]
        ax2.plot(frequencies, magnitude, 'b-', linewidth=0.5)
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Magnitude')
        ax2.set_title('Frequency Spectrum')
        ax2.grid(True, alpha=0.3)

        # Plot recognition scores
        ax3 = fig.add_subplot(gs[1, 1])
        words = list(all_scores.keys())
        scores = list(all_scores.values())
        colors = ['green' if w == recognized_word else 'blue' for w in words]

        ax3.barh(words, scores, color=colors, alpha=0.7)
        ax3.set_xlabel('Correlation Score')
        ax3.set_title('Recognition Scores')
        ax3.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")

        return fig

    def plot_snr_improvement_heatmap(self, results, save_path=None):
        """
        Create a heatmap showing SNR improvement across noise types and SNR levels.
        Generic version that works with any noise sources.
        
        Args:
            results: Dictionary with structure {noise_type: {condition: metrics}}
            save_path: Path to save figure
        """
        # Dynamically extract available noise types and SNR levels
        noise_types = sorted(list(results.keys()))
        snr_levels = [5, 10, 15, 20]  # Standard test levels
        methods = ['baseline', 'contextual']
        
        fig, axes = plt.subplots(1, 2, figsize=(14, max(5, len(noise_types) * 0.8)))
        
        for method_idx, method in enumerate(methods):
            data = np.zeros((len(noise_types), len(snr_levels)))
            
            for i, noise_type in enumerate(noise_types):
                for j, snr in enumerate(snr_levels):
                    key = f"snr{snr}_{method}"
                    if noise_type in results and key in results[noise_type]:
                        data[i, j] = results[noise_type][key]['snr']
            
            im = axes[method_idx].imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=25)
            axes[method_idx].set_xticks(range(len(snr_levels)))
            axes[method_idx].set_xticklabels([f'{snr} dB' for snr in snr_levels])
            axes[method_idx].set_yticks(range(len(noise_types)))
            axes[method_idx].set_yticklabels([n.capitalize() for n in noise_types])
            axes[method_idx].set_xlabel('Input SNR')
            axes[method_idx].set_ylabel('Noise Type')
            axes[method_idx].set_title(f'SNR Improvement - {method.capitalize()} Method')
            
            # Add text annotations
            for i in range(len(noise_types)):
                for j in range(len(snr_levels)):
                    text = axes[method_idx].text(j, i, f'{data[i, j]:.1f}',
                                                ha="center", va="center", color="black", fontsize=10)
        
        plt.colorbar(im, ax=axes, label='Output SNR (dB)', fraction=0.046, pad=0.04)
        plt.suptitle('SNR Improvement Across Noise Types and Conditions', fontsize=14, y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig

    def plot_improvement_comparison(self, results, save_path=None):
        """
        Bar plot comparing baseline vs contextual features improvement.
        Generic version that works with any noise sources.
        
        Args:
            results: Dictionary with structure {noise_type: {condition: metrics}}
            save_path: Path to save figure
        """
        # Dynamically extract available noise types
        noise_types = sorted(list(results.keys()))
        snr_levels = [5, 10, 15, 20]
        
        fig, axes = plt.subplots(2, 2, figsize=(max(14, len(noise_types) * 2), 10))
        axes = axes.flatten()
        
        for idx, snr in enumerate(snr_levels):
            baseline_improvements = []
            contextual_improvements = []
            
            for noise_type in noise_types:
                base_key = f"snr{snr}_baseline"
                ctx_key = f"snr{snr}_contextual"
                
                if noise_type in results:
                    baseline_snr = results[noise_type].get(base_key, {}).get('snr', 0)
                    contextual_snr = results[noise_type].get(ctx_key, {}).get('snr', 0)
                    
                    # Calculate improvement from input SNR
                    baseline_improvements.append(baseline_snr - snr)
                    contextual_improvements.append(contextual_snr - snr)
            
            x = np.arange(len(noise_types))
            width = 0.35
            
            axes[idx].bar(x - width/2, baseline_improvements, width, label='Baseline', 
                         color='steelblue', alpha=0.8)
            axes[idx].bar(x + width/2, contextual_improvements, width, label='Contextual Features',
                         color='coral', alpha=0.8)
            
            axes[idx].set_xlabel('Noise Type')
            axes[idx].set_ylabel('SNR Improvement (dB)')
            axes[idx].set_title(f'Input SNR: {snr} dB')
            axes[idx].set_xticks(x)
            axes[idx].set_xticklabels([n.capitalize() for n in noise_types])
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3, axis='y')
            axes[idx].axhline(y=0, color='k', linestyle='--', linewidth=0.5)
        
        plt.suptitle('SNR Improvement: Baseline vs Contextual Features', fontsize=14, y=1.00)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig

    def plot_spectrogram_comparison(self, clean, noisy, enhanced_baseline, enhanced_contextual,
                                    sample_rate=16000, save_path=None):
        """
        Create spectrogram comparison showing clean, noisy, and both enhanced versions.
        
        Args:
            clean: Clean signal
            noisy: Noisy signal
            enhanced_baseline: Enhanced signal (baseline method)
            enhanced_contextual: Enhanced signal (contextual features method)
            sample_rate: Sampling rate
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(4, 1, figsize=(12, 12))
        
        signals = [clean, noisy, enhanced_baseline, enhanced_contextual]
        titles = ['Clean Signal', 'Noisy Signal', 
                 'Enhanced (Baseline)', 'Enhanced (Contextual Features)']
        
        for ax, signal, title in zip(axes, signals, titles):
            # Compute spectrogram using scipy
            from scipy import signal as sp_signal
            f, t, Sxx = sp_signal.spectrogram(signal, fs=sample_rate, nperseg=512,
                                             noverlap=256, window='hann')
            
            im = ax.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud',
                              cmap='viridis', vmin=-80, vmax=0)
            ax.set_ylabel('Frequency (Hz)')
            ax.set_title(title)
            ax.set_ylim([0, 8000])
            plt.colorbar(im, ax=ax, label='Power (dB)')
        
        axes[-1].set_xlabel('Time (s)')
        plt.suptitle('Spectrogram Comparison', fontsize=14, y=0.995)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig

    def plot_waveform_comparison(self, clean, noisy, enhanced_baseline, enhanced_contextual,
                                sample_rate=16000, zoom_start=0.5, zoom_duration=0.2,
                                save_path=None):
        """
        Detailed waveform comparison with zoomed view.
        
        Args:
            clean: Clean signal
            noisy: Noisy signal
            enhanced_baseline: Enhanced signal (baseline)
            enhanced_contextual: Enhanced signal (contextual)
            sample_rate: Sampling rate
            zoom_start: Start time for zoom (seconds)
            zoom_duration: Duration of zoom window (seconds)
            save_path: Path to save figure
        """
        fig = plt.figure(figsize=(14, 10))
        gs = GridSpec(4, 2, figure=fig)
        
        # Full waveforms (left column)
        signals = [clean, noisy, enhanced_baseline, enhanced_contextual]
        titles = ['Clean', 'Noisy', 'Enhanced (Baseline)', 'Enhanced (Contextual)']
        colors = ['green', 'red', 'blue', 'purple']
        
        for i, (signal, title, color) in enumerate(zip(signals, titles, colors)):
            ax = fig.add_subplot(gs[i, 0])
            time = np.arange(len(signal)) / sample_rate
            ax.plot(time, signal, color=color, linewidth=0.5, alpha=0.7)
            ax.set_ylabel('Amplitude')
            ax.set_title(f'{title} - Full Signal')
            ax.grid(True, alpha=0.3)
            ax.set_xlim([0, len(signal) / sample_rate])
            
            # Highlight zoom region
            ax.axvspan(zoom_start, zoom_start + zoom_duration, alpha=0.2, color='yellow')
        
        axes[3].set_xlabel('Time (s)')
        
        # Zoomed waveforms (right column)
        zoom_start_idx = int(zoom_start * sample_rate)
        zoom_end_idx = int((zoom_start + zoom_duration) * sample_rate)
        
        for i, (signal, title, color) in enumerate(zip(signals, titles, colors)):
            ax = fig.add_subplot(gs[i, 1])
            zoomed = signal[zoom_start_idx:zoom_end_idx]
            time_zoom = np.arange(len(zoomed)) / sample_rate + zoom_start
            ax.plot(time_zoom, zoomed, color=color, linewidth=1)
            ax.set_ylabel('Amplitude')
            ax.set_title(f'{title} - Zoomed View')
            ax.grid(True, alpha=0.3)
            ax.set_xlim([zoom_start, zoom_start + zoom_duration])
        
        axes[3].set_xlabel('Time (s)')
        
        plt.suptitle('Waveform Comparison: Full vs Zoomed View', fontsize=14, y=0.995)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig

    def plot_metrics_radar(self, metrics_baseline, metrics_contextual, save_path=None):
        """
        Radar plot comparing multiple metrics between baseline and contextual methods.
        
        Args:
            metrics_baseline: Metrics dictionary for baseline
            metrics_contextual: Metrics dictionary for contextual
            save_path: Path to save figure
        """
        # Select key metrics (normalized to 0-1 scale)
        metric_names = ['SNR\nImprovement', 'SDR', 'Seg SNR', 'Low MSE\n(inverted)']
        
        # Normalize metrics to 0-1 scale for visualization
        baseline_values = [
            min(metrics_baseline.get('snr', 0) / 25, 1.0),  # SNR (max 25 dB)
            min(metrics_baseline.get('sdr', 0) / 25, 1.0),  # SDR (max 25 dB)
            min(metrics_baseline.get('seg_snr', 0) / 20, 1.0),  # Seg SNR (max 20 dB)
            1.0 - min(metrics_baseline.get('mse', 0.1) / 0.1, 1.0)  # MSE inverted
        ]
        
        contextual_values = [
            min(metrics_contextual.get('snr', 0) / 25, 1.0),
            min(metrics_contextual.get('sdr', 0) / 25, 1.0),
            min(metrics_contextual.get('seg_snr', 0) / 20, 1.0),
            1.0 - min(metrics_contextual.get('mse', 0.1) / 0.1, 1.0)
        ]
        
        # Number of variables
        num_vars = len(metric_names)
        
        # Compute angle for each axis
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        baseline_values += baseline_values[:1]
        contextual_values += contextual_values[:1]
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        
        ax.plot(angles, baseline_values, 'o-', linewidth=2, label='Baseline', color='steelblue')
        ax.fill(angles, baseline_values, alpha=0.25, color='steelblue')
        
        ax.plot(angles, contextual_values, 'o-', linewidth=2, label='Contextual Features', color='coral')
        ax.fill(angles, contextual_values, alpha=0.25, color='coral')
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_names, size=10)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=8)
        ax.grid(True)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        
        plt.title('Performance Metrics Comparison\n(Normalized)', size=14, pad=20)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig

    @staticmethod
    def show():
        """Show all plots."""
        plt.show()

    @staticmethod
    def close_all():
        """Close all plots."""
        plt.close('all')
