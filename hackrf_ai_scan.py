# -----------------------------------------------------------------------------
# Copyright (c) 2025 3rdEyeOperation
# All rights reserved.
#
# This software and associated documentation files (the "Software") is provided
# under the following conditions:
#
# Permission is granted to use, copy, modify, and/or distribute this software
# for non-commercial research and educational purposes only, provided that this
# copyright notice and this permission notice appear in all copies of the
# Software, derivative works, or associated documentation.
#
# COMMERCIAL USE IS PROHIBITED WITHOUT EXPRESS WRITTEN PERMISSION FROM 3rdEyeOperation.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
# -----------------------------------------------------------------------------

import numpy as np
import subprocess
import tensorflow as tf
import time
import os

# Parameters
CENTER_FREQ = 2_440_000_000  # 2.44 GHz (center of 2.4 GHz ISM)
SAMPLE_RATE = 10_000_000     # 10 MHz
CAPTURE_SECONDS = 1
OUTPUT_FILE = "capture.iq"
MODEL_PATH = "rf_classifier_model.h5"
LABELS = ["WiFi", "Bluetooth Classic", "BLE", "Other"]

def capture_hackrf(freq, rate, seconds, outfile):
    # Uses hackrf_transfer via subprocess (cross-platform, no pyhackrf dependency)
    cmd = [
        "hackrf_transfer",
        "-f", str(freq),
        "-s", str(rate),
        "-r", outfile,
        "-n", str(rate * seconds)
    ]
    print(f"Capturing {seconds}s of spectrum at {freq/1e6}MHz ...")
    subprocess.run(cmd, check=True)
    print("Capture complete.")

def load_iq_samples(filename, sample_rate, seconds):
    # HackRF outputs signed 8-bit IQ interleaved
    num_samples = sample_rate * seconds
    raw = np.fromfile(filename, dtype=np.int8)
    iq = raw[::2] + 1j * raw[1::2]
    print(f"Loaded {len(iq)} IQ samples.")
    return iq

def preprocess_iq(iq, sample_rate):
    # Example feature extraction: spectrogram
    from scipy.signal import spectrogram
    f, t, Sxx = spectrogram(np.real(iq), fs=sample_rate, nperseg=1024, noverlap=512)
    Sxx_log = 10 * np.log10(Sxx + 1e-12)
    # Resize or crop/pad as needed for your AI model input shape
    Sxx_resized = Sxx_log[:128, :128]  # Example: 128x128
    Sxx_resized = np.expand_dims(Sxx_resized, axis=(0, -1))  # Add batch and channel dims
    return Sxx_resized

def classify_signal(model, features):
    predictions = model.predict(features)
    class_id = np.argmax(predictions)
    confidence = np.max(predictions)
    return LABELS[class_id], confidence

def main():
    # 1. Capture IQ samples
    capture_hackrf(CENTER_FREQ, SAMPLE_RATE, CAPTURE_SECONDS, OUTPUT_FILE)

    # 2. Load signal
    iq = load_iq_samples(OUTPUT_FILE, SAMPLE_RATE, CAPTURE_SECONDS)

    # 3. Feature extraction
    features = preprocess_iq(iq, SAMPLE_RATE)

    # 4. Load TensorFlow model
    model = tf.keras.models.load_model(MODEL_PATH)

    # 5. Classify
    label, conf = classify_signal(model, features)
    print(f"Detected signal: {label} (Confidence: {conf:.2f})")

if __name__ == "__main__":
    main()
