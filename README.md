# Pandora

**Owner**: 3rd Eye Operation  
**Contact**: [facebook.com/3rdEyeOperation](https://facebook.com/3rdEyeOperation)  
**Copyright**: © 2025 3rd Eye Operation. All rights reserved.

## About 3rd Eye Operation

3rd Eye Operation is dedicated to developing cutting-edge research tools for educational and training purposes in the field of signal intelligence and geolocation analysis. Our projects focus on Software-Defined Radio (SDR) applications, signal processing, and triangulation techniques for academic and authorized operational use.

## Project Overview

Pandora is a comprehensive research toolkit developed by 3rd Eye Operation that provides two main capabilities:

1. **Drone Detection System** (`spectraRF.py`) - An SDR-based system for detecting and classifying drone signals
2. **Signal Triangulation** (`triangulator.py`) - A tool for estimating positions using GPS coordinates and azimuth bearings

## Tools

### SpectraRF - Drone Detection System

A Software-Defined Radio (SDR) based drone detection system using HackRF with SoapySDR. 

**Features:**
- Frequency sweeping over configurable bands
- Power spectrum analysis and bandwidth estimation
- Signal-to-noise ratio (SNR) calculation
- Basic signal classification and confidence scoring
- Logging of detections with timestamp and metadata
- Simulation mode when no SDR hardware is present

**Usage:**
```bash
python3 spectraRF.py
```

### Triangulator - Position Estimation

Given GPS lat/lon and azimuth degrees from multiple stations, estimates the intersection point using local equirectangular projection and least-squares methods.

**Usage:**
```bash
# Run with built-in example
python3 triangulator.py

# Provide station data via command line
python3 triangulator.py --station 13.736717 100.523186 45 \
                       --station 13.743000 100.534000 135 \
                       --station 13.730000 100.540000 -90

# Use CSV file input
python3 triangulator.py --csv stations.csv

# Show version information
python3 triangulator.py --version
```

## Requirements

- Python 3.x
- NumPy
- SoapySDR (for spectraRF.py)

## Disclaimer and Legal Notice

**⚠️ IMPORTANT - EDUCATIONAL USE ONLY ⚠️**

- This software is provided for **EDUCATIONAL and RESEARCH purposes only**
- Do **NOT** use these tools for unlawful surveillance, tracking of individuals, or any activity that violates privacy or local laws
- The authors and distributors (3rd Eye Operation) accept **NO LIABILITY** for misuse
- By using this software you agree to comply with applicable laws and ethical guidelines
- Intended for classroom/lab simulation and testing with synthetic data ONLY

## License

Proprietary — For research, training, and authorized operational use only.

Unauthorized copying, modification, or distribution of this software, via any medium, is strictly prohibited unless with express permission from 3rd Eye Operation.

## Contact

For inquiries, support, or authorization requests:

📘 **Facebook**: [facebook.com/3rdEyeOperation](https://facebook.com/3rdEyeOperation)

---
*Developed by 3rd Eye Operation (2025)*
