# audio-snr-tools
Python utilities for SNR-controlled audio preprocessing and analysis.

This project provides Python tools for audio signal processing, including spectrogram generation and signal-to-noise ratio (SNR) experimentation using mel-spectrograms and loudness measurements.

## Description

The script `audio_tools.py` contains two main functions:

- `generate_spectrogram(category=0)`: Generates grayscale spectrograms from WAV audio using the short-time Fourier transform (STFT).
- `experiment()`: Loads a clean signal and adds synthetic noise, adjusting the SNR and visualising the combined mel-spectrogram.

## Features

- Spectrogram generation using FFT and OpenCV
- SNR control with mel power and LUFS calculation
- Audio manipulation using `librosa`, `numpy`, and `pyloudnorm`
- Visualisation using `matplotlib`

## Requirements

Install the dependencies with pip: pip install numpy librosa matplotlib opencv-python pyloudnorm scipy


## Usage

1. Download or clone the repository.
2. Place your `.wav` files in the working directory. For example:
   - `gtzan-m-22.wav` (music)
   - `gtzan-s-22.wav` (speech)
   - `yaafe/lvrysis-o-22.wav` (other)
   - `other.wav` (for use in `experiment()`)

3. Open `audio_tools.py` and uncomment the function you want to run in the `if __name__ == "__main__":` block.

### To generate spectrograms:

generate_spectrogram(category=0)  # 0 = Music, 1 = Speech, 2 = Other

### Run SNR Experiment

experiment()




