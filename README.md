# Machine-Learning-Prototype-for-Heart-Rate-and-Heart-Rate-Variability-Estimation-from-Audio-Signals

Machine Learning Prototype for Heart Rate and Heart Rate Variability Estimation from Audio Signals
Project Overview
This project explores the feasibility of using machine learning, specifically Convolutional Neural Networks (CNNs), to estimate Heart Rate (HR) and Heart Rate Variability (HRV) from synthetic audio-like signals. The goal is to develop a non-invasive and potentially more accessible method for monitoring cardiovascular health, aligning with the mission of transforming sound into health insights.

The prototype involves generating synthetic heartbeat signals, preprocessing them into spectrograms, building and training a CNN model, and evaluating its performance. This README provides a guide to the project, including setup, execution, and a summary of the findings and future directions.

Table of Contents
Project Setup
Data Simulation
Data Preprocessing
Prepare Data for Training
Model Architecture
Model Training
Model Evaluation
Results and Discussion
Future Work
License
Project Setup
To run this notebook, you will need a Python environment with the necessary libraries installed. We recommend using Google Colab for ease of setup and access to necessary resources.

If running locally, you will need:

Python 3.7+
TensorFlow
Keras
NumPy
Matplotlib
SciPy
Scikit-learn
You can install the required libraries using pip:

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

# 2. Define parameters for enhanced signal generation
sampling_rate = 1000  # Hz
duration = 5  # seconds
num_samples = sampling_rate * duration
min_hr_bpm = 50
max_hr_bpm = 120
min_hrv_std = 0.005  # in seconds, minimum standard deviation of beat intervals
max_hrv_std = 0.05  # in seconds, maximum standard deviation of beat intervals
num_signals = 500  # Increased number of synthetic signals to generate

# Parameters for waveform variation
min_peak_duration = 0.03 # seconds
max_peak_duration = 0.07 # seconds
min_amplitude_scale = 0.8
max_amplitude_scale = 1.2

# Parameters for noise
min_snr_db = 5 # Minimum Signal-to-Noise Ratio in dB
max_snr_db = 25 # Maximum Signal-to-Noise Ratio in dB

# 3. Create an enhanced function that generates a single synthetic heartbeat signal
                                      
                                       def generate_heartbeat_signal_enhanced(hr_bpm, hrv_std, sampling_rate, duration,
                                       min_peak_duration, max_peak_duration,
                                       min_amplitude_scale, max_amplitude_scale,
                                       min_snr_db, max_snr_db):
    """Generates an enhanced synthetic heartbeat-like signal with specified HR, HRV, waveform variations, and noise."""
    hr_interval_sec_mean = 60 / hr_bpm
    num_beats_approx = int(duration / hr_interval_sec_mean)

    # Generate beat intervals with realistic HRV (normal distribution around the mean interval)
    # Ensure beat intervals are positive
    beat_intervals = np.maximum(np.random.normal(hr_interval_sec_mean, hrv_std, num_beats_approx), 0.1)


    beat_times = [0]
    current_time = 0
    for interval in beat_intervals:
        next_beat_time = current_time + interval
        if next_beat_time < duration:
            beat_times.append(next_beat_time)
            current_time = next_beat_time
        else:
            break

    signal = np.zeros(int(duration * sampling_rate))
    for beat_time in beat_times:
        beat_sample = int(beat_time * sampling_rate)
        if beat_sample < len(signal):
            # Simulate a pulse waveform with varying duration and amplitude
            peak_duration_sec = np.random.uniform(min_peak_duration, max_peak_duration)
            peak_duration_samples = int(peak_duration_sec * sampling_rate)
            amplitude_scale = np.random.uniform(min_amplitude_scale, max_amplitude_scale)

            for i in range(peak_duration_samples):
                if beat_sample + i < len(signal):
                    # Use a more complex waveform, e.g., a skewed Gaussian or a combination
                    # Here we'll use a simple scaled Gaussian for demonstration
                    signal[beat_sample + i] += amplitude_scale * np.exp(-(i / (sampling_rate * (peak_duration_sec / 6)))**2)

    # Add noise (e.g., Gaussian noise with varying SNR)
    snr_db = np.random.uniform(min_snr_db, max_snr_db)
    signal_power = np.mean(signal**2)
    if signal_power > 1e-10: # Avoid division by zero or very small numbers
        noise_power = signal_power / (10**(snr_db / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), len(signal))
        signal += noise

    # Calculate true HR based on the actual number of beats
    true_hr = len(beat_times) / duration * 60

    # Calculate true HRV based on the standard deviation of the beat-to-beat intervals
    if len(beat_times) > 1:
        true_rr_intervals = np.diff(beat_times)
        true_hrv = np.std(true_rr_intervals)
    else:
        true_hrv = 0 # Or some indicator for very low HR

    return signal, true_hr, true_hrv
    # 4. Generate a dataset of enhanced synthetic signals
    synthetic_signals_enhanced = []
    true_hr_values_enhanced = []
    true_hrv_values_enhanced = []
    
    for _ in range(num_signals):
    hr_bpm = np.random.uniform(min_hr_bpm, max_hr_bpm)
    hrv_std = np.random.uniform(min_hrv_std, max_hrv_std)
    signal, true_hr, true_hrv = generate_heartbeat_signal_enhanced(
        hr_bpm, hrv_std, sampling_rate, duration,
        min_peak_duration, max_peak_duration,
        min_amplitude_scale, max_amplitude_scale,
        min_snr_db, max_snr_db
    )
    synthetic_signals_enhanced.append(signal)
    true_hr_values_enhanced.append(true_hr)
    true_hrv_values_enhanced.append(true_hrv)
    
    # Convert to NumPy arrays
    synthetic_signals_enhanced = np.array(synthetic_signals_enhanced)
    true_hr_values_enhanced = np.array(true_hr_values_enhanced)
    true_hrv_values_enhanced = np.array(true_hrv_values_enhanced)
    
    print(f"Generated {len(synthetic_signals_enhanced)} enhanced synthetic signals.")
    print(f"Shape of synthetic_signals_enhanced: {synthetic_signals_enhanced.shape}")
    print(f"Shape of true_hr_values_enhanced: {true_hr_values_enhanced.shape}")
    print(f"Shape of true_hrv_values_enhanced: {true_hrv_values_enhanced.shape}")
    
    # Display a sample enhanced signal
    plt.figure(figsize=(12, 4))
    plt.plot(np.linspace(0, duration, num_samples), synthetic_signals_enhanced[0])
    plt.title(f"Sample Enhanced Synthetic Heartbeat Signal (HR: {true_hr_values_enhanced[0]:.2f} bpm, HRV: {true_hrv_values_enhanced[0]:.4f} sec)")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()
    
    # Iterate through each enhanced synthetic signal and compute its spectrogram
    spectrograms_enhanced = []
    # Use the same spectrogram parameters as before
    nperseg = 256
    noverlap = nperseg // 2
    nfft = 256
    
    for signal in synthetic_signals_enhanced:
    # Compute the spectrogram
    f, t, Sxx = spectrogram(signal, fs=sampling_rate, nperseg=nperseg, noverlap=noverlap, nfft=nfft)
    # Store the magnitude spectrogram (Sxx)
    spectrograms_enhanced.append(Sxx)
    
    # Ensure consistent dimensions and Convert to NumPy array
    # The spectrogram function should produce consistent dimensions given fixed parameters and signal length.
    # Let's check the shape of the first enhanced spectrogram
    print(f"Shape of the first enhanced spectrogram: {spectrograms_enhanced[0].shape}")
    
    # Convert the list of spectrograms to a NumPy array
    spectrograms_enhanced = np.array(spectrograms_enhanced)
    
    print(f"Shape of the enhanced spectrograms array: {spectrograms_enhanced.shape}")
    
    # Display a sample enhanced spectrogram
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(t, f, 10 * np.log10(spectrograms_enhanced[0]), shading='gouraud')
    plt.title('Spectrogram of a Sample Enhanced Synthetic Signal')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.colorbar(label='Intensity [dB]')
    plt.show()
    
    from sklearn.model_selection import train_test_split
    
    # Split the enhanced data into training and testing sets
    X_train_enhanced, X_test_enhanced, y_hr_train_enhanced, y_hr_test_enhanced, y_hrv_train_enhanced, y_hrv_test_enhanced = train_test_split(
    spectrograms_enhanced, true_hr_values_enhanced, true_hrv_values_enhanced, test_size=0.2, random_state=42
    )
    
    # Reshape enhanced spectrogram data for CNN input (add a channel dimension)
    # Current shape: (number_of_samples, number_of_frequency_bins, number_of_time_segments)
    # Desired shape: (number_of_samples, number_of_frequency_bins, number_of_time_segments, 1)
    X_train_enhanced = np.expand_dims(X_train_enhanced, axis=-1)
    X_test_enhanced = np.expand_dims(X_test_enhanced, axis=-1)
    
    # Print the shapes of the resulting sets
    print(f"Shape of X_train_enhanced: {X_train_enhanced.shape}")
    print(f"Shape of X_test_enhanced: {X_test_enhanced.shape}")
    print(f"Shape of y_hr_train_enhanced: {y_hr_train_enhanced.shape}")
    print(f"Shape of y_hr_test_enhanced: {y_hr_test_enhanced.shape}")
    print(f"Shape of y_hrv_train_enhanced: {y_hrv_train_enhanced.shape}")
    print(f"Shape of y_hrv_test_enhanced: {y_hrv_test_enhanced.shape}")
    
    from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
    from tensorflow.keras import Model
    
    # Define the input shape based on the shape of the preprocessed spectrograms
    input_shape_enhanced = X_train_enhanced.shape[1:] # (number_of_frequency_bins, number_of_time_segments, 1)
    print(f"Input shape for the modified CNN: {input_shape_enhanced}")
    
    # Create the Keras Input layer
    input_layer_modified = Input(shape=input_shape_enhanced)
    
    # Add convolutional and pooling layers (same as the previous enhanced model)
    x = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer_modified)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Flatten the output of the convolutional layers
    x = Flatten()(x)
    
    # Add dense layers with Dropout for regularization
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x) # Add dropout
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x) # Add dropout
    
    # Create two separate output layers: one for HR (linear) and one for HRV (with ReLU activation for non-negativity)
    output_hr_modified = Dense(1, name='hr_output')(x) # Linear activation for HR regression
    output_hrv_modified = Dense(1, activation='relu', name='hrv_output')(x) # ReLU activation for non-negative HRV regression
    
    # Create the modified Keras Model
    model_modified = Model(inputs=input_layer_modified, outputs=[output_hr_modified, output_hrv_modified])
    
    # Compile the modified model
    model_modified.compile(optimizer='adam',
                       loss={'hr_output': 'mse', 'hrv_output': 'mse'}, # Mean Squared Error for regression
                       metrics={'hr_output': 'mae', 'hrv_output': 'mae'}) # Mean Absolute Error as a metric
    # Print a summary of the modified model architecture
    model_modified.summary()
    
    # Train the modified model
    history_modified = model_modified.fit(
    X_train_enhanced,
    [y_hr_train_enhanced, y_hrv_train_enhanced],
    epochs=100, # Increased number of epochs
    batch_size=32,
    validation_data=(X_test_enhanced, [y_hr_test_enhanced, y_hrv_test_enhanced])
    
    # Evaluate the trained modified model on the enhanced test data
    print("\nEvaluating the modified model on the enhanced test data:")
    results_modified = model_modified.evaluate(X_test_enhanced, [y_hr_test_enhanced, y_hrv_test_enhanced])
    
    # Print the evaluation results
    print("\nEvaluation Results (Modified Model):")
    print(f"Test Loss: {results_modified[0]:.4f}")
    print(f"Test HR MAE: {results_modified[1]:.4f}")
    print(f"Test HRV MAE: {results_modified[2]:.4f}")
    
    # Make predictions on the enhanced test data for detailed comparison
    predictions_modified = model_modified.predict(X_test_enhanced)
    
    # The predictions are a list containing two arrays: [predicted_hr, predicted_hrv]
    predicted_hr_modified = predictions_modified[0]
    predicted_hrv_modified = predictions_modified[1]
    
    # Select a few samples to display (e.g., the first 10)
    num_samples_to_display = 10
    
    print("\nComparison of True vs. Predicted HR and HRV for Enhanced Test Samples (after retraining modified model):")
    print("-" * 80)
    
    for i in range(num_samples_to_display):
    true_hr = y_hr_test_enhanced[i]
    pred_hr = predicted_hr_modified[i][0] # predictions are arrays of shape (n_samples, 1)
    true_hrv = y_hrv_test_enhanced[i]
    pred_hrv = predicted_hrv_modified[i][0] # predictions are arrays of shape (n_samples, 1)

    print(f"Sample {i+1}:")
    print(f"  True HR: {true_hr:.2f} bpm, Predicted HR: {pred_hr:.2f} bpm")
    print(f"  True HRV: {true_hrv:.4f} sec, Predicted HRV: {pred_hrv:.4f} sec")
    print("-" * 20)
