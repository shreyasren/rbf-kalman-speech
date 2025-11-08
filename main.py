"""
Main experiment pipeline for Speech Enhancement and Recognition using RBF-Kalman Filter.

This script recreates the experiments from the paper:
"Speech Enhancement and Recognition using Kalman Filter Modified via Radial Basis Function"
by Mario Barnard, Farag M. Lagnf, Amr S. Mahmoud, Mohamed Zohdy (2020)

This version works with pre-recorded audio files only (no microphone recording).
"""

import numpy as np
import os
import sys
import soundfile as sf

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from audio_utils import AudioProcessor, create_dataset_structure
from rbf import RadialBasisFunction, create_rbf_for_kalman
from kalman_filter import KalmanFilterSpeech, enhance_speech_rbf_kalman
from envelope_detection import EnvelopeDetector, detect_envelope, remove_silence_from_signal
from speech_recognition import CorrelationSpeechRecognizer, DTWSpeechRecognizer
from rbf_classifier import MultiLayerRBFClassifier
from contextual_features import extract_advanced_frame_features
from visualization import SpeechVisualizer


class SpeechEnhancementExperiment:
    """
    Complete experiment pipeline for speech enhancement and recognition.
    """

    def __init__(self, sample_rate=44100, duration=3.0, ar_order=10):
        """
        Initialize experiment.

        Args:
            sample_rate: Audio sampling rate (Hz)
            duration: Recording duration (seconds)
            ar_order: Autoregressive model order
        """
        self.sample_rate = sample_rate
        self.duration = duration
        self.ar_order = ar_order

        # Initialize components
        self.processor = AudioProcessor(sample_rate)
        self.envelope_detector = EnvelopeDetector()
        self.recognizer = CorrelationSpeechRecognizer()
        self.dtw_recognizer = DTWSpeechRecognizer()
        self.rbf_classifier = None  # Will be trained later
        self.visualizer = SpeechVisualizer()

        # Create directory structure
        self.paths = create_dataset_structure()

        # Database
        self.database = {}
        self.words = ['Hello', 'Estimation', 'Oakland']

    def load_database_from_files(self):
        """
        Load database from saved WAV files.
        """
        print("Loading database from files...")

        raw_path = self.paths['raw']
        files = [f for f in os.listdir(raw_path) if f.endswith('.wav')]

        for file in files:
            filepath = os.path.join(raw_path, file)
            audio, sr = self.processor.load_audio(filepath)

            if audio is not None:
                # Resample if necessary
                if sr != self.sample_rate:
                    audio = self.processor.resample(audio, sr, self.sample_rate)

                # Normalize
                audio = self.processor.normalize(audio)

                # Store in database
                key = file.replace('.wav', '')
                self.database[key] = audio

        print(f"Loaded {len(self.database)} samples from files")

    def enhance_database(self, snr_db=10, rbf_gamma=1.0, measurement_noise=0.01):
        """
        Enhance all database signals using RBF-Kalman filter.

        Args:
            snr_db: Signal-to-Noise Ratio for added noise (dB)
            rbf_gamma: RBF shape parameter
            measurement_noise: Measurement noise covariance R
        """
        print("\n" + "=" * 60)
        print("ENHANCING DATABASE SIGNALS")
        print("=" * 60)

        enhanced_database = {}

        for key, clean_signal in self.database.items():
            print(f"\nProcessing: {key}")

            # Add noise
            noisy_signal, noise = self.processor.add_noise(clean_signal, snr_db)

            # Estimate Q using RBF
            rbf, Q = create_rbf_for_kalman(noisy_signal, gamma=rbf_gamma)
            print(f"  Estimated Q (process noise): {Q:.6f}")

            # Apply Kalman filter
            enhanced_signal = enhance_speech_rbf_kalman(
                noisy_signal,
                Q_rbf=Q,
                R=measurement_noise,
                order=self.ar_order
            )

            # Normalize enhanced signal
            enhanced_signal = self.processor.normalize(enhanced_signal)

            # Save enhanced signal
            filename = os.path.join(self.paths['database'], f"{key}_enhanced.wav")
            sf.write(filename, enhanced_signal, self.sample_rate)

            enhanced_database[key] = {
                'clean': clean_signal,
                'noisy': noisy_signal,
                'enhanced': enhanced_signal,
                'noise': noise,
                'Q': Q
            }

        print(f"\nEnhanced {len(enhanced_database)} signals")
        return enhanced_database

    def apply_envelope_detection(self, enhanced_database):
        """
        Apply envelope detection to remove silent portions.

        Args:
            enhanced_database: Dictionary of enhanced signals
        """
        print("\n" + "=" * 60)
        print("APPLYING ENVELOPE DETECTION")
        print("=" * 60)

        voiced_database = {}

        for key, data in enhanced_database.items():
            print(f"\nProcessing: {key}")

            signal = data['enhanced']

            # Detect envelope
            envelope, voiced_mask = detect_envelope(signal)

            # Extract voiced signal
            voiced_signal = remove_silence_from_signal(signal)

            print(f"  Original samples: {len(signal)}")
            print(f"  Voiced samples: {len(voiced_signal)}")
            print(f"  Reduction: {(1 - len(voiced_signal) / len(signal)) * 100:.1f}%")

            voiced_database[key] = {
                'signal': signal,
                'envelope': envelope,
                'voiced_mask': voiced_mask,
                'voiced_signal': voiced_signal
            }

            # Save voiced signal
            filename = os.path.join(self.paths['processed'], f"{key}_voiced.wav")
            sf.write(filename, voiced_signal, self.sample_rate)

        return voiced_database

    def build_recognition_database(self, voiced_database):
        """
        Build database for speech recognition.

        Args:
            voiced_database: Dictionary of voiced signals
        """
        print("\n" + "=" * 60)
        print("BUILDING RECOGNITION DATABASE")
        print("=" * 60)

        # For traditional recognizers
        for key, data in voiced_database.items():
            # Extract word label
            word = key.split('_')[0]

            # Add to recognizers
            voiced_signal = data['voiced_signal']
            self.recognizer.add_to_database(word, voiced_signal)
            self.dtw_recognizer.add_to_database(word, voiced_signal)

        print(f"Database contains {len(self.recognizer.labels)} words")
        for word in self.recognizer.labels:
            count = len(self.recognizer.database[word])
            print(f"  {word}: {count} samples")

        # Train RBF classifier
        self._train_rbf_classifier(voiced_database)

    def _train_rbf_classifier(self, voiced_database):
        """
        Train the multi-layer RBF classifier for speech recognition.

        Args:
            voiced_database: Dictionary of voiced signals
        """
        print("\n" + "=" * 60)
        print("TRAINING RBF CLASSIFIER (Multi-Layer Neural Network)")
        print("=" * 60)

        # Prepare training data
        X_train = []
        y_train = []
        word_to_idx = {}

        for key, data in voiced_database.items():
            word = key.split('_')[0]

            # Assign class index
            if word not in word_to_idx:
                word_to_idx[word] = len(word_to_idx)

            # Extract features from voiced signal
            voiced_signal = data['voiced_signal']
            features = extract_advanced_frame_features(
                voiced_signal,
                sr=self.sample_rate,
                frame_len=320,
                hop_len=160
            )

            # Use mean features across frames
            if len(features) > 0:
                mean_features = np.mean(features, axis=0)
                X_train.append(mean_features)
                y_train.append(word_to_idx[word])

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        print(f"Training data: {X_train.shape[0]} samples, {X_train.shape[1]} features")
        print(f"Classes: {len(word_to_idx)} words - {list(word_to_idx.keys())}")

        # Create and train classifier
        self.rbf_classifier = MultiLayerRBFClassifier(
            k1=min(40, X_train.shape[0]),  # Adjust to data size
            k2=min(20, X_train.shape[0] // 2),
            gamma1=0.01,
            gamma2=0.05,
            n_classes=len(word_to_idx),
            random_state=42
        )

        self.rbf_classifier.fit(X_train, y_train, max_iter=100, learning_rate=0.01, verbose=True)

        # Store word mapping
        self.rbf_word_mapping = {idx: word for word, idx in word_to_idx.items()}

        # Evaluate on training data
        train_acc = self.rbf_classifier.score(X_train, y_train)
        print(f"\n✓ RBF Classifier training complete. Training accuracy: {train_acc:.4f}")

    def test_recognition(self, test_word, test_file=None, snr_db=10):
        """
        Test speech recognition with a test file.

        Args:
            test_word: Word to test
            test_file: Path to test audio file (if None, will look in data/test/)
            snr_db: SNR for added noise
        """
        print("\n" + "=" * 60)
        print(f"TESTING RECOGNITION: {test_word}")
        print("=" * 60)

        # Load test signal
        if test_file is None:
            test_file = os.path.join(self.paths['test'], f"{test_word}_test.wav")
        
        if not os.path.exists(test_file):
            print(f"Error: Test file not found: {test_file}")
            print(f"Please provide a valid test file path.")
            return None

        print(f"\nLoading test signal from: {test_file}")
        test_signal, sr = self.processor.load_audio(test_file)
        
        if test_signal is None:
            print("Error loading test file")
            return None

        # Resample if needed
        if sr != self.sample_rate:
            test_signal = self.processor.resample(test_signal, sr, self.sample_rate)

        # Normalize
        test_signal = self.processor.normalize(test_signal)

        # Normalize
        test_signal = self.processor.normalize(test_signal)

        # Add noise
        noisy_test, _ = self.processor.add_noise(test_signal, snr_db)

        # Enhance using RBF-Kalman
        rbf, Q = create_rbf_for_kalman(noisy_test)
        enhanced_test = enhance_speech_rbf_kalman(noisy_test, Q_rbf=Q, R=0.01, order=self.ar_order)

        # Remove silence
        voiced_test = remove_silence_from_signal(enhanced_test)

        # Recognize using correlation
        recognized_word, confidence, all_scores = self.recognizer.recognize(voiced_test)

        print(f"\nCorrelation-based Recognition:")
        print(f"  Recognized: {recognized_word}")
        print(f"  Confidence: {confidence:.3f}")
        print(f"  All scores: {all_scores}")

        # Recognize using DTW
        dtw_word, dtw_distance, all_distances = self.dtw_recognizer.recognize(voiced_test)

        print(f"\nDTW-based Recognition:")
        print(f"  Recognized: {dtw_word}")
        print(f"  Distance: {dtw_distance:.3f}")
        print(f"  All distances: {all_distances}")

        # Recognize using RBF Classifier
        rbf_word = None
        rbf_confidence = 0.0
        rbf_probs = None

        if self.rbf_classifier is not None:
            # Extract features from test signal
            test_features = extract_advanced_frame_features(
                voiced_test,
                sr=self.sample_rate,
                frame_len=320,
                hop_len=160
            )

            if len(test_features) > 0:
                mean_test_features = np.mean(test_features, axis=0, keepdims=True)

                # Predict
                y_pred = self.rbf_classifier.predict(mean_test_features)[0]
                rbf_probs = self.rbf_classifier.predict_proba(mean_test_features)[0]
                rbf_word = self.rbf_word_mapping[y_pred]
                rbf_confidence = rbf_probs[y_pred]

                print(f"\nRBF Classifier Recognition (Multi-Layer Neural Network):")
                print(f"  Recognized: {rbf_word}")
                print(f"  Confidence: {rbf_confidence:.4f}")
                print(f"  Class probabilities:")
                for idx, prob in enumerate(rbf_probs):
                    word = self.rbf_word_mapping[idx]
                    print(f"    {word}: {prob:.4f}")

        # Save test signal
        test_key = f"test_{test_word}"
        filename = os.path.join(self.paths['test'], f"{test_key}.wav")
        sf.write(filename, enhanced_test, self.sample_rate)

        return {
            'test_signal': test_signal,
            'noisy': noisy_test,
            'enhanced': enhanced_test,
            'voiced': voiced_test,
            'recognized_corr': recognized_word,
            'confidence_corr': confidence,
            'scores_corr': all_scores,
            'recognized_dtw': dtw_word,
            'distance_dtw': dtw_distance,
            'distances_dtw': all_distances,
            'recognized_rbf': rbf_word,
            'confidence_rbf': rbf_confidence,
            'probs_rbf': rbf_probs
        }

    def visualize_results(self, enhanced_database, voiced_database):
        """
        Generate visualizations matching paper figures.

        Args:
            enhanced_database: Enhanced signals
            voiced_database: Voiced signals
        """
        print("\n" + "=" * 60)
        print("GENERATING VISUALIZATIONS")
        print("=" * 60)

        results_path = self.paths['results']

        # Figure 1: Time domain of "Estimation" from 3 people
        estimation_signals = []
        estimation_titles = []
        for i in range(1, 4):
            key = f"Estimation_person{i}"
            if key in self.database:
                estimation_signals.append(self.database[key])
                estimation_titles.append(f'{key.split("_")[1].capitalize()} Says "Estimation"')

        if estimation_signals:
            self.visualizer.plot_time_domain(
                estimation_signals,
                estimation_titles,
                self.sample_rate,
                save_path=os.path.join(results_path, 'figure1_time_domain_estimation.png')
            )

        # Figure 2: Frequency domain of "Estimation" from 3 people
        if estimation_signals:
            self.visualizer.plot_frequency_domain(
                estimation_signals,
                estimation_titles,
                self.sample_rate,
                save_path=os.path.join(results_path, 'figure2_frequency_domain_estimation.png')
            )

        # Figure 7: Enhancement results
        for key in list(enhanced_database.keys())[:3]:  # Show first 3 examples
            data = enhanced_database[key]
            self.visualizer.plot_enhancement_results(
                data['clean'],
                data['noisy'],
                data['enhanced'],
                self.sample_rate,
                save_path=os.path.join(results_path, f'figure7_enhancement_{key}.png')
            )

        # Envelope detection visualization
        for key in list(voiced_database.keys())[:3]:
            data = voiced_database[key]
            self.visualizer.plot_envelope_detection(
                data['signal'],
                data['envelope'],
                data['voiced_mask'],
                self.sample_rate,
                save_path=os.path.join(results_path, f'figure5_envelope_{key}.png')
            )

            self.visualizer.plot_voiced_signal(
                data['voiced_signal'],
                self.sample_rate,
                save_path=os.path.join(results_path, f'figure6_voiced_{key}.png')
            )

        print(f"\nVisualizations saved to: {results_path}")


def main():
    """
    Main execution function.
    """
    print("=" * 60)
    print("SPEECH ENHANCEMENT AND RECOGNITION USING RBF-KALMAN FILTER")
    print("=" * 60)

    # Initialize experiment
    exp = SpeechEnhancementExperiment(
        sample_rate=44100,
        duration=3.0,
        ar_order=10
    )

    # Load existing database from files
    print("\nLoading audio database from data/raw/...")
    exp.load_database_from_files()

    if not exp.database:
        print("\n" + "=" * 60)
        print("ERROR: No audio files found in data/raw/")
        print("=" * 60)
        print("\nPlease ensure audio files exist in data/raw/")
        print("You can run 'generate_synthetic_data.py' to create sample data:")
        print("  python3 generate_synthetic_data.py")
        return

    # Enhance database
    enhanced_db = exp.enhance_database(snr_db=10, rbf_gamma=1.0, measurement_noise=0.01)

    # Apply envelope detection
    voiced_db = exp.apply_envelope_detection(enhanced_db)

    # Build recognition database
    exp.build_recognition_database(voiced_db)

    # Visualize results
    exp.visualize_results(enhanced_db, voiced_db)

    # Test recognition (if test files exist)
    print("\n" + "=" * 60)
    print("SPEECH RECOGNITION TEST")
    print("=" * 60)
    
    test_available = False
    for word in exp.words:
        test_file = os.path.join(exp.paths['test'], f"{word}_test.wav")
        if os.path.exists(test_file):
            test_available = True
            break
    
    if test_available:
        for word in exp.words:
            test_file = os.path.join(exp.paths['test'], f"{word}_test.wav")
            if os.path.exists(test_file):
                print(f"\nTesting word: {word}")
                result = exp.test_recognition(word, test_file=test_file, snr_db=10)
                
                if result is not None:
                    # Visualize recognition result
                    exp.visualizer.plot_recognition_results(
                        result['voiced'],
                        result['recognized_corr'],
                        result['confidence_corr'],
                        result['scores_corr'],
                        exp.sample_rate,
                        save_path=os.path.join(exp.paths['results'], f'recognition_test_{word}.png')
                    )
    else:
        print("\nNo test files found in data/test/")
        print("Skipping recognition testing.")

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"Results saved in: {exp.paths['results']}")


if __name__ == "__main__":
    main()
