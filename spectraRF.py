import SoapySDR
from SoapySDR import *  # SOAPY_SDR_ constants
import numpy as np
import time
import datetime
import math
import logging
import json
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# --- Configuration Parameters ---
CONFIG = {
    "sample_rate_msps": 20,  # Mega-samples per second
    "center_freq_mhz": 2440,  # MHz, will sweep around this
    "freq_sweep_range_mhz": 80,  # Total sweep range around center freq (e.g., 2400-2480 MHz)
    "freq_step_mhz": 1,  # MHz per step
    "dwell_time_ms": 200,  # Milliseconds per frequency step
    "gain_lna_db": 32,  # HackRF LNA gain
    "gain_vga_db": 32,  # HackRF VGA gain
    "fft_size": 2048,  # Number of bins for FFT
    "buffer_size_samples": 20 * 1024,  # Buffer for I/Q samples per read (20KB for 20MS/s, 1ms data)
    "detection_threshold_dbm": -80,  # dBm, Power threshold for initial signal detection
    "min_drone_bandwidth_mhz": 5,  # Minimum bandwidth for drone-like signal
    "max_drone_bandwidth_mhz": 40,  # Maximum bandwidth for drone-like signal
    "snr_threshold_db": 15,  # Minimum SNR for confident classification
    "confidence_threshold": 0.7,  # ML classification confidence
    "log_dir": "drone_detections_logs",
    "signature_db_path": "drone_signatures.json"  # Placeholder for a real DB
}

# Ensure log directory exists
os.makedirs(CONFIG["log_dir"], exist_ok=True)

# --- Dummy Drone Signature Database (for simulation) ---
DRONE_SIGNATURES = {
    "DJI Mavic 3": {
        "bandwidth_mhz": 10.2,
        "modulation": "FHSS/OFDM",
        "duty_cycle": 0.8,
        "spectral_shape": "comb-like",
        "freq_hop_pattern": "50ms"
    },
    "Autel EVO II": {
        "bandwidth_mhz": 20.0,
        "modulation": "DSSS/OFDM",
        "duty_cycle": 0.7,
        "spectral_shape": "broadband",
        "freq_hop_pattern": "none"
    },
    "FPV Racing (Analog)": {
        "bandwidth_mhz": 8.0,
        "modulation": "FM",
        "duty_cycle": 0.95,
        "spectral_shape": "continuous",
        "freq_hop_pattern": "none"
    },
    "Wi-Fi Interference": {
        "bandwidth_mhz": 20.0,
        "modulation": "OFDM",
        "duty_cycle": 0.5,
        "spectral_shape": "fixed-channel-bursts",
        "freq_hop_pattern": "none"
    },
    "Unknown / Noise": {
        "bandwidth_mhz": None,
        "modulation": "N/A",
        "duty_cycle": None,
        "spectral_shape": "flat/random",
        "freq_hop_pattern": "none"
    }
}

def initialize_soapysdr():
    try:
        args = "driver=hackrf"
        sdr = SoapySDR.Device(args)
        sdr.setSampleRate(SOAPY_SDR_RX, 0, CONFIG["sample_rate_msps"] * 1e6)
        sdr.setFrequency(SOAPY_SDR_RX, 0, CONFIG["center_freq_mhz"] * 1e6)
        sdr.setGain(SOAPY_SDR_RX, 0, CONFIG["gain_lna_db"])
        logging.info("HackRF (via SoapySDR) initialized successfully.")
        return sdr
    except Exception as e:
        logging.warning(f"Could not initialize HackRF via SoapySDR: {e}. Running in simulation mode.")
        return None

def read_samples_soapysdr(sdr, num_samples):
    # Setup stream
    rxStream = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
    sdr.activateStream(rxStream)
    buff = np.zeros(num_samples, np.complex64)
    sr = sdr.readStream(rxStream, [buff], num_samples)
    sdr.deactivateStream(rxStream)
    sdr.closeStream(rxStream)
    if sr.ret > 0:
        return buff[:sr.ret]
    else:
        logging.warning("No samples read from SDR.")
        return np.array([])

def calculate_fft(iq_samples):
    if len(iq_samples) == 0:
        return np.array([]), np.array([])
    window = np.blackman(len(iq_samples))
    windowed_samples = iq_samples * window
    fft_output = np.fft.fft(windowed_samples, n=CONFIG["fft_size"])
    fft_magnitude = np.abs(fft_output) ** 2
    fft_magnitude_shifted = np.fft.fftshift(fft_magnitude)
    freq_bins = np.fft.fftshift(
        np.fft.fftfreq(CONFIG["fft_size"], d=1/(CONFIG["sample_rate_msps"] * 1e6))
    )
    return freq_bins, 10 * np.log10(fft_magnitude_shifted + 1e-10)

def detect_signal_presence(power_spectrum_db):
    if len(power_spectrum_db) == 0:
        return False, 0
    max_power = np.max(power_spectrum_db)
    return max_power > CONFIG["detection_threshold_dbm"], max_power

def estimate_bandwidth(power_spectrum_db, freq_bins, peak_idx):
    peak_power = power_spectrum_db[peak_idx]
    threshold_power = peak_power - 6
    left_edge = peak_idx
    while left_edge > 0 and power_spectrum_db[left_edge] > threshold_power:
        left_edge -= 1
    right_edge = peak_idx
    while right_edge < len(power_spectrum_db) - 1 and power_spectrum_db[right_edge] > threshold_power:
        right_edge += 1
    if right_edge > left_edge:
        return abs(freq_bins[right_edge] - freq_bins[left_edge]) / 1e6
    return 0.0

def calculate_snr(power_spectrum_db, signal_peak_idx):
    if len(power_spectrum_db) == 0:
        return 0
    signal_power = power_spectrum_db[signal_peak_idx]
    noise_regions = power_spectrum_db[power_spectrum_db < (signal_power - 20)]
    if len(noise_regions) > 0:
        noise_floor_avg = np.mean(noise_regions)
        return signal_power - noise_floor_avg
    return 0

def classify_signal_ml(features, current_freq_mhz):
    detected_type = "Unknown / Noise"
    confidence = 0.5
    max_power = features.get('max_power_dbm', -100)
    bandwidth_mhz = features.get('bandwidth_mhz', 0)
    snr_db = features.get('snr_db', 0)
    if max_power > CONFIG["detection_threshold_dbm"] and snr_db > CONFIG["snr_threshold_db"]:
        if CONFIG["min_drone_bandwidth_mhz"] <= bandwidth_mhz <= CONFIG["max_drone_bandwidth_mhz"]:
            if 2400 <= current_freq_mhz <= 2483.5:
                if 8.0 <= bandwidth_mhz <= 12.0 and np.random.rand() > 0.3:
                    detected_type = "DJI Mavic 3"
                    confidence = 0.9 + (np.random.rand() * 0.05)
                elif 18.0 <= bandwidth_mhz <= 22.0 and np.random.rand() > 0.5:
                    detected_type = "Autel EVO II"
                    confidence = 0.85 + (np.random.rand() * 0.05)
                elif 5.0 <= bandwidth_mhz <= 10.0 and np.random.rand() > 0.7:
                    detected_type = "FPV Racing (Analog)"
                    confidence = 0.75 + (np.random.rand() * 0.05)
                else:
                    detected_type = "Wi-Fi Interference"
                    confidence = 0.6 + (np.random.rand() * 0.1)
            elif 5725 <= current_freq_mhz <= 5850:
                if 18.0 <= bandwidth_mhz <= 40.0 and np.random.rand() > 0.2:
                    detected_type = "DJI Mavic 3"
                    confidence = 0.92 + (np.random.rand() * 0.03)
                elif 5.0 <= bandwidth_mhz <= 15.0 and np.random.rand() > 0.6:
                    detected_type = "FPV Racing (Analog)"
                    confidence = 0.8 + (np.random.rand() * 0.1)
                else:
                    detected_type = "Wi-Fi Interference"
                    confidence = 0.65 + (np.random.rand() * 0.08)
            else:
                detected_type = "Unknown Drone / RF Activity"
                confidence = 0.7 + (np.random.rand() * 0.1)
    return detected_type, confidence

def log_detection(detection_info):
    timestamp = datetime.datetime.now().isoformat()
    log_entry = {
        "timestamp": timestamp,
        "detection_id": str(datetime.datetime.now().timestamp()),
        "frequency_mhz": detection_info.get("frequency_mhz"),
        "max_power_dbm": detection_info.get("max_power_dbm"),
        "bandwidth_mhz": detection_info.get("bandwidth_mhz"),
        "snr_db": detection_info.get("snr_db"),
        "classified_type": detection_info.get("classified_type"),
        "confidence": detection_info.get("confidence"),
        "message": detection_info.get("message", "Drone signal detected."),
        "location": "Simulated GPS: N34.0522, W118.2437"
    }
    log_filename = os.path.join(CONFIG["log_dir"], f"detection_log_{datetime.date.today()}.json")
    with open(log_filename, 'a') as f:
        json.dump(log_entry, f)
        f.write('\n')
    logging.info(
        f"Detection logged: {log_entry['classified_type']} at {log_entry['frequency_mhz']} MHz "
        f"with confidence {log_entry['confidence']:.2f}"
    )

def run_drone_detection_system():
    sdr = initialize_soapysdr()
    if sdr is None:
        logging.warning("Running in simulation mode (no SDR found).")
    start_freq_mhz = CONFIG["center_freq_mhz"] - (CONFIG["freq_sweep_range_mhz"] / 2)
    end_freq_mhz = CONFIG["center_freq_mhz"] + (CONFIG["freq_sweep_range_mhz"] / 2)
    freq_steps_mhz = np.arange(start_freq_mhz, end_freq_mhz + CONFIG["freq_step_mhz"], CONFIG["freq_step_mhz"])
    logging.info(f"Starting SDR drone detection. Sweeping from {start_freq_mhz} MHz to {end_freq_mhz} MHz.")

    recent_detections = []
    try:
        while True:
            for current_freq_mhz in freq_steps_mhz:
                iq_samples = None
                if sdr:
                    sdr.setFrequency(SOAPY_SDR_RX, 0, current_freq_mhz * 1e6)
                    num_samples_to_read = int((CONFIG["sample_rate_msps"] * 1e6) * (CONFIG["dwell_time_ms"] / 1000.0))
                    iq_samples = read_samples_soapysdr(sdr, num_samples_to_read)
                else:
                    # Simulation mode: Generate a random signal or simulated drone signal
                    if np.random.rand() < 0.2:
                        if current_freq_mhz in [2412, 2437, 2462, 5745, 5785, 5825]:
                            logging.info(f"Simulating signal at {current_freq_mhz} MHz.")
                            if current_freq_mhz == 2412:
                                iq_samples = np.random.randn(CONFIG["buffer_size_samples"]) + \
                                             1j * np.random.randn(CONFIG["buffer_size_samples"])
                                iq_samples += 0.5 * np.exp(1j * 2 * np.pi * 0.1 * np.arange(CONFIG["buffer_size_samples"]))
                                iq_samples *= 10 ** ((-70 + np.random.rand() * 10) / 20)
                            elif current_freq_mhz == 2437 or current_freq_mhz == 5785:
                                iq_samples = np.random.randn(CONFIG["buffer_size_samples"]) + \
                                             1j * np.random.randn(CONFIG["buffer_size_samples"])
                                iq_samples += 0.8 * np.exp(1j * 2 * np.pi * 0.05 * np.arange(CONFIG["buffer_size_samples"]))
                                iq_samples *= 10 ** ((-55 + np.random.rand() * 5) / 20)
                            else:
                                iq_samples = np.random.randn(CONFIG["buffer_size_samples"]) + \
                                             1j * np.random.randn(CONFIG["buffer_size_samples"])
                                iq_samples *= 10 ** ((-90 + np.random.rand() * 10) / 20)
                        else:
                            iq_samples = np.random.randn(CONFIG["buffer_size_samples"]) + \
                                         1j * np.random.randn(CONFIG["buffer_size_samples"])
                            iq_samples *= 10 ** ((-95 + np.random.rand() * 5) / 20)
                    else:
                        iq_samples = np.random.randn(CONFIG["buffer_size_samples"]) + \
                                     1j * np.random.randn(CONFIG["buffer_size_samples"])
                        iq_samples *= 10 ** ((-95 + np.random.rand() * 5) / 20)
                if iq_samples is None or len(iq_samples) == 0:
                    logging.warning(f"No samples read for {current_freq_mhz} MHz. Skipping analysis.")
                    continue
                freq_bins, power_spectrum_db = calculate_fft(iq_samples)
                signal_present, max_power_dbm = detect_signal_presence(power_spectrum_db)
                if signal_present:
                    peak_idx = np.argmax(power_spectrum_db)
                    bandwidth_mhz = estimate_bandwidth(power_spectrum_db, freq_bins, peak_idx)
                    snr_db = calculate_snr(power_spectrum_db, peak_idx)
                    features = {
                        "max_power_dbm": max_power_dbm,
                        "bandwidth_mhz": bandwidth_mhz,
                        "snr_db": snr_db,
                        "spectral_data": power_spectrum_db.tolist()
                    }
                    classified_type, confidence = classify_signal_ml(features, current_freq_mhz)
                    if confidence >= CONFIG["confidence_threshold"]:
                        detection_info = {
                            "frequency_mhz": current_freq_mhz,
                            "max_power_dbm": max_power_dbm,
                            "bandwidth_mhz": bandwidth_mhz,
                            "snr_db": snr_db,
                            "classified_type": classified_type,
                            "confidence": confidence,
                            "message": f"High confidence drone detection: {classified_type}."
                        }
                        log_detection(detection_info)
                        recent_detections.append(detection_info)
                        logging.info(
                            f"Confirmed {classified_type} at {current_freq_mhz} MHz "
                            f"(Power: {max_power_dbm:.2f}dBm, BW: {bandwidth_mhz:.2f}MHz, "
                            f"SNR: {snr_db:.2f}dB, Confidence: {confidence:.2f})"
                        )
                        if classified_type != "Wi-Fi Interference" and classified_type != "Unknown / Noise":
                            logging.info(f"Drone detected. Pausing sweep to monitor {current_freq_mhz} MHz.")
                            time.sleep(1)
                    else:
                        logging.info(
                            f"Potential signal at {current_freq_mhz} MHz "
                            f"(Power: {max_power_dbm:.2f}dBm, BW: {bandwidth_mhz:.2f}MHz, "
                            f"SNR: {snr_db:.2f}dB), Low Confidence ({confidence:.2f}) classified as: {classified_type}"
                        )
                else:
                    logging.debug(
                        f"No significant signal at {current_freq_mhz} MHz (Max Power: {max_power_dbm:.2f}dBm)"
                    )
                time_taken = (CONFIG["dwell_time_ms"] / 1000.0)
                if time_taken > 0:
                    time.sleep(time_taken)
    except KeyboardInterrupt:
        logging.info("Drone detection system stopped by user.")
    finally:
        if sdr:
            del sdr
        logging.info("SDR device closed.")

if __name__ == "__main__":
    logging.info("Starting drone detection system (SoapySDR)...")
    run_drone_detection_system()