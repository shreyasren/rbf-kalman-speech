"""
Advanced Experiments with Integrated Features

This script tests:
1. Multiple noise types (white, fan, street, ambient)
2. RBF with contextual features ("Beyond Column")
3. Multi-layer RBF classifier for recognition
4. Comparison of all methods

Usage:
    python3 run_advanced_experiments.py
"""

import numpy as np
import os
import sys
import soundfile as sf
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from audio_utils import AudioProcessor, create_dataset_structure
from rbf import create_rbf_for_kalman
from kalman_filter import enhance_speech_rbf_kalman
from envelope_detection import remove_silence_from_signal
from rbf_classifier import MultiLayerRBFClassifier
from contextual_features import extract_advanced_frame_features
from metrics import compute_all_metrics, print_metrics
from speech_recognition import CorrelationSpeechRecognizer


def run_noise_type_comparison():
    """
    Compare enhancement performance across different noise types.
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 1: NOISE TYPE COMPARISON")
    print("=" * 80)

    processor = AudioProcessor(sample_rate=16000)
    paths = create_dataset_structure()

    # Load a test signal
    raw_files = [f for f in os.listdir(paths['raw']) if f.endswith('.wav')]
    if not raw_files:
        print("No audio files found. Please run generate_synthetic_data.py first.")
        return

    test_file = os.path.join(paths['raw'], raw_files[0])
    clean_signal, sr = processor.load_audio(test_file)

    if clean_signal is None:
        print(f"Error loading {test_file}")
        return

    # Resample if needed
    if sr != 16000:
        clean_signal = processor.resample(clean_signal, sr, 16000)

    clean_signal = processor.normalize(clean_signal)
    
    # Save clean reference
    clean_ref_path = os.path.join(paths['test'], 'clean_reference.wav')
    processor.save_audio(clean_signal, clean_ref_path, sample_rate=16000)

    # Test parameters
    noise_types = ['white', 'fan', 'street', 'ambient']
    snr_levels = [5, 10, 15, 20]
    use_contextual = [False, True]

    results = {}
    
    # Store signals for visualization
    noisy_signals = {}
    enhanced_signals = {}

    for noise_type in noise_types:
        print(f"\n{'-' * 80}")
        print(f"Testing Noise Type: {noise_type.upper()}")
        print(f"{'-' * 80}")

        results[noise_type] = {}

        for snr_db in snr_levels:
            print(f"\n  SNR: {snr_db} dB")

            # Add noise
            noisy_signal, _ = processor.add_noise(clean_signal, snr_db, noise_type=noise_type)
            
            # Store for visualization
            noisy_signals[(noise_type, snr_db)] = noisy_signal.copy()
            
            # Save noisy sample (once per SNR/noise combo)
            noisy_filename = f"noisy_{noise_type}_{snr_db}dB.wav"
            noisy_path = os.path.join(paths['test'], noisy_filename)
            processor.save_audio(noisy_signal, noisy_path, sample_rate=16000)

            for use_ctx in use_contextual:
                method_name = "With Contextual Features" if use_ctx else "Baseline"
                print(f"    Method: {method_name}")

                # Estimate Q using RBF
                rbf, Q = create_rbf_for_kalman(
                    noisy_signal,
                    gamma=1.0,
                    use_contextual_features=use_ctx,
                    sample_rate=16000
                )

                # Enhance
                enhanced_signal = enhance_speech_rbf_kalman(
                    noisy_signal,
                    Q_rbf=Q,
                    R=0.01,
                    order=12
                )
                
                # Store for visualization
                method_key = 'contextual' if use_ctx else 'baseline'
                enhanced_signals[(noise_type, snr_db, method_key)] = enhanced_signal.copy()
                
                # Save enhanced sample
                method_suffix = "contextual" if use_ctx else "baseline"
                enhanced_filename = f"enhanced_{method_suffix}_{noise_type}_{snr_db}dB.wav"
                enhanced_path = os.path.join(paths['processed'], enhanced_filename)
                processor.save_audio(enhanced_signal, enhanced_path, sample_rate=16000)

                # Compute metrics
                metrics = compute_all_metrics(clean_signal, enhanced_signal, sr=16000)

                # Store results
                key = f"snr{snr_db}_{'ctx' if use_ctx else 'base'}"
                results[noise_type][key] = metrics

                print(f"      SNR Improvement: {metrics['snr']:.2f} dB")
                print(f"      MSE: {metrics['mse']:.6f}")

    # Save results summary with visualizations
    save_noise_comparison_results(
        results, 
        paths['results'],
        clean_signal=clean_signal,
        noisy_signals=noisy_signals,
        enhanced_signals=enhanced_signals
    )
    
    # Print audio save summary
    num_noisy = len(noise_types) * len(snr_levels)
    num_enhanced = num_noisy * 2  # baseline + contextual for each
    print(f"\n✓ Saved {num_noisy} noisy samples to: data/test/")
    print(f"✓ Saved {num_enhanced} enhanced samples to: data/processed/")
    print(f"✓ Saved 1 clean reference to: data/test/")

    return results


def run_rbf_classifier_experiment():
    """
    Train and test multi-layer RBF classifier for speech recognition.
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: RBF CLASSIFIER EVALUATION")
    print("=" * 80)

    processor = AudioProcessor(sample_rate=16000)
    paths = create_dataset_structure()

    # Load all available samples
    raw_files = [f for f in os.listdir(paths['raw']) if f.endswith('.wav')]

    if not raw_files:
        print("No audio files found. Please run generate_synthetic_data.py first.")
        return

    print(f"\nFound {len(raw_files)} audio files")

    # Organize by word
    word_signals = {}

    for filename in raw_files:
        filepath = os.path.join(paths['raw'], filename)
        signal, sr = processor.load_audio(filepath)

        if signal is None:
            continue

        # Resample if needed
        if sr != 16000:
            signal = processor.resample(signal, sr, 16000)

        signal = processor.normalize(signal)

        # Extract word label
        word = filename.split('_')[0]

        if word not in word_signals:
            word_signals[word] = []

        word_signals[word].append(signal)

    print(f"\nWords found: {list(word_signals.keys())}")
    for word, signals in word_signals.items():
        print(f"  {word}: {len(signals)} samples")

    # Prepare training and test sets
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    word_to_idx = {}

    for word, signals in word_signals.items():
        if word not in word_to_idx:
            word_to_idx[word] = len(word_to_idx)

        # Use first 2/3 for training, last 1/3 for testing
        n_train = max(1, int(len(signals) * 0.67))

        for i, signal in enumerate(signals):
            # Remove silence
            voiced_signal = remove_silence_from_signal(signal)

            # Extract features
            features = extract_advanced_frame_features(
                voiced_signal,
                sr=16000,
                frame_len=320,
                hop_len=160
            )

            if len(features) == 0:
                continue

            mean_features = np.mean(features, axis=0)

            if i < n_train:
                X_train.append(mean_features)
                y_train.append(word_to_idx[word])
            else:
                X_test.append(mean_features)
                y_test.append(word_to_idx[word])

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test) if X_test else np.array([]).reshape(0, X_train.shape[1])
    y_test = np.array(y_test)
    
    # Safety check: Replace any remaining NaN or inf values
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    if len(X_test) > 0:
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")

    # Train RBF classifier
    print("\nTraining Multi-Layer RBF Classifier...")
    classifier = MultiLayerRBFClassifier(
        k1=min(40, len(X_train)),
        k2=min(20, len(X_train) // 2),
        gamma1=0.01,
        gamma2=0.05,
        n_classes=len(word_to_idx),
        random_state=42
    )

    classifier.fit(X_train, y_train, max_iter=100, learning_rate=0.01, verbose=True)

    # Evaluate
    train_acc = classifier.score(X_train, y_train)
    print(f"\n✓ Training Accuracy: {train_acc:.4f}")

    if len(X_test) > 0:
        test_acc = classifier.score(X_test, y_test)
        print(f"✓ Test Accuracy: {test_acc:.4f}")

        # Show predictions
        y_pred = classifier.predict(X_test)
        idx_to_word = {idx: word for word, idx in word_to_idx.items()}

        print("\nTest Predictions:")
        for i in range(min(10, len(X_test))):
            true_word = idx_to_word[y_test[i]]
            pred_word = idx_to_word[y_pred[i]]
            status = "✓" if true_word == pred_word else "✗"
            print(f"  {status} True: {true_word}, Predicted: {pred_word}")
    else:
        print("\nNo test samples available (all used for training)")

    return classifier, word_to_idx


def run_method_comparison():
    """
    Compare Correlation, DTW, and RBF Classifier recognition methods.
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 3: RECOGNITION METHOD COMPARISON")
    print("=" * 80)

    processor = AudioProcessor(sample_rate=16000)
    paths = create_dataset_structure()

    # This would require a more extensive test set
    # For now, just report that the infrastructure is in place
    print("\nRecognition methods now available:")
    print("  1. Correlation-based template matching (baseline)")
    print("  2. DTW (Dynamic Time Warping)")
    print("  3. Multi-Layer RBF Classifier (NEW - Neural Network)")
    print("\nTo fully test, run main.py with enhanced pipeline.")


def save_noise_comparison_results(results, results_dir, clean_signal=None, noisy_signals=None, 
                                 enhanced_signals=None):
    """
    Save noise comparison results as figures and text.
    
    Args:
        results: Dictionary of results
        results_dir: Directory to save results
        clean_signal: Clean reference signal (optional, for additional plots)
        noisy_signals: Dictionary of noisy signals (optional)
        enhanced_signals: Dictionary of enhanced signals (optional)
    """
    print(f"\nSaving results to {results_dir}...")
    
    from src.visualization import SpeechVisualizer
    visualizer = SpeechVisualizer()

    # Create comparison plots
    noise_types = list(results.keys())
    snr_levels = [5, 10, 15, 20]

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Noise Type Comparison: Enhancement Performance', fontsize=16, fontweight='bold')

    for idx, noise_type in enumerate(noise_types):
        ax = axes[idx // 2, idx % 2]

        baseline_snrs = []
        contextual_snrs = []

        for snr in snr_levels:
            base_key = f"snr{snr}_base"
            ctx_key = f"snr{snr}_ctx"

            if base_key in results[noise_type]:
                baseline_snrs.append(results[noise_type][base_key]['snr'])
            if ctx_key in results[noise_type]:
                contextual_snrs.append(results[noise_type][ctx_key]['snr'])

        x = np.arange(len(snr_levels))
        width = 0.35

        ax.bar(x - width/2, baseline_snrs, width, label='Baseline RBF', alpha=0.8)
        ax.bar(x + width/2, contextual_snrs, width, label='With Contextual Features', alpha=0.8)

        ax.set_xlabel('Input SNR (dB)', fontweight='bold')
        ax.set_ylabel('SNR Improvement (dB)', fontweight='bold')
        ax.set_title(f'{noise_type.capitalize()} Noise', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(snr_levels)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'advanced_noise_comparison.png'), dpi=150)
    print(f"  ✓ Saved: advanced_noise_comparison.png")

    # Save text summary
    summary_file = os.path.join(results_dir, 'advanced_results_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("ADVANCED EXPERIMENTS SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        for noise_type in noise_types:
            f.write(f"\n{noise_type.upper()} NOISE\n")
            f.write("-" * 80 + "\n")

            for snr in snr_levels:
                base_key = f"snr{snr}_base"
                ctx_key = f"snr{snr}_ctx"

                f.write(f"\n  SNR {snr} dB:\n")

                if base_key in results[noise_type]:
                    metrics = results[noise_type][base_key]
                    f.write(f"    Baseline:    SNR={metrics['snr']:.2f} dB, MSE={metrics['mse']:.6f}\n")

                if ctx_key in results[noise_type]:
                    metrics = results[noise_type][ctx_key]
                    f.write(f"    Contextual:  SNR={metrics['snr']:.2f} dB, MSE={metrics['mse']:.6f}\n")

                    # Calculate improvement
                    if base_key in results[noise_type]:
                        base_snr = results[noise_type][base_key]['snr']
                        ctx_snr = metrics['snr']
                        improvement = ctx_snr - base_snr
                        f.write(f"    Improvement: {improvement:+.2f} dB\n")

    print(f"  ✓ Saved: advanced_results_summary.txt")
    
    # Generate additional visualizations
    print(f"\nGenerating additional visualizations...")
    
    # SNR Improvement Heatmap
    try:
        visualizer.plot_snr_improvement_heatmap(
            results, 
            save_path=os.path.join(results_dir, 'snr_improvement_heatmap.png')
        )
    except Exception as e:
        print(f"  ⚠ Could not generate heatmap: {e}")
    
    # Improvement Comparison Bar Charts
    try:
        visualizer.plot_improvement_comparison(
            results,
            save_path=os.path.join(results_dir, 'improvement_comparison.png')
        )
    except Exception as e:
        print(f"  ⚠ Could not generate improvement comparison: {e}")
    
    # If audio signals provided, create spectrogram and waveform comparisons
    if clean_signal is not None and noisy_signals is not None and enhanced_signals is not None:
        # Pick one example: street noise at 10 dB
        if ('street', 10) in noisy_signals and ('street', 10, 'baseline') in enhanced_signals and ('street', 10, 'contextual') in enhanced_signals:
            try:
                visualizer.plot_spectrogram_comparison(
                    clean_signal,
                    noisy_signals[('street', 10)],
                    enhanced_signals[('street', 10, 'baseline')],
                    enhanced_signals[('street', 10, 'contextual')],
                    sample_rate=16000,
                    save_path=os.path.join(results_dir, 'spectrogram_comparison_street_10dB.png')
                )
            except Exception as e:
                print(f"  ⚠ Could not generate spectrogram: {e}")
            
            try:
                visualizer.plot_waveform_comparison(
                    clean_signal,
                    noisy_signals[('street', 10)],
                    enhanced_signals[('street', 10, 'baseline')],
                    enhanced_signals[('street', 10, 'contextual')],
                    sample_rate=16000,
                    save_path=os.path.join(results_dir, 'waveform_comparison_street_10dB.png')
                )
            except Exception as e:
                print(f"  ⚠ Could not generate waveform comparison: {e}")
        
        # Metrics radar plot for one condition
        if 'street' in results:
            base_metrics = results['street'].get('snr10_base', {})
            ctx_metrics = results['street'].get('snr10_ctx', {})
            if base_metrics and ctx_metrics:
                try:
                    visualizer.plot_metrics_radar(
                        base_metrics,
                        ctx_metrics,
                        save_path=os.path.join(results_dir, 'metrics_radar_street_10dB.png')
                    )
                except Exception as e:
                    print(f"  ⚠ Could not generate radar plot: {e}")


def main():
    """
    Run all advanced experiments.
    """
    print("=" * 80)
    print("ADVANCED EXPERIMENTS - INTEGRATED FEATURES")
    print("=" * 80)
    print("\nFeatures tested:")
    print("  1. Multiple noise types (white, fan, street, ambient)")
    print("  2. Contextual features for RBF Q estimation ('Beyond Column')")
    print("  3. Multi-layer RBF classifier for speech recognition")
    print("=" * 80)

    # Experiment 1: Noise type comparison
    noise_results = run_noise_type_comparison()

    # Experiment 2: RBF classifier
    classifier, word_mapping = run_rbf_classifier_experiment()

    # Experiment 3: Method comparison (informational)
    run_method_comparison()

    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 80)
    print("\nResults saved in: data/results/")
    print("  - advanced_noise_comparison.png")
    print("  - advanced_results_summary.txt")


if __name__ == "__main__":
    main()
