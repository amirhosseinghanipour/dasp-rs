# DASP-RS: Digital Audio Signal Processing for Rust
[![Crates.io](https://img.shields.io/crates/v/dasp-rs.svg)](https://crates.io/crates/dasp-rs)  
[![Documentation](https://docs.rs/dasp-rs/badge.svg)](https://docs.rs/dasp-rs)  

`dasp-rs` is a crate for digital audio signal processing for developers, researchers, phoneticians, and students.  

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Modules](#modules)
- [Example](#example)
- [Performance](#performance)
- [Contributing](#contributing)
- [License](#license)

## Features
- **Audio I/O**: Load and export WAV files with flexible resampling and channel conversion.
- **Signal Processing**: Arithmetic operations, mixing, panning, and resampling for audio signals.
- **Feature Extraction**: Spectral, harmonic, rhythm, and manipulation features for advanced analysis.
- **Time-Frequency Analysis**: STFT, CQT, VQT, and more with phase recovery and reassignment.
- **Signal Generation**: Tones, chirps, and clicks for testing and synthesis.
- **Parallelization**: Multi-core processing with `rayon` for efficiency.
- **Error Handling**: Robust error types with detailed diagnostics using `thiserror`.
- **Testing**: Unit tests for core functionality to ensure reliability.

## Installation
Add `dasp-rs` to your `Cargo.toml`:
```toml
[dependencies]
dasp-rs = "0.1.1"
```

## Modules
| Module                  | Description                                                                 |
|-------------------------|-----------------------------------------------------------------------------|
| `core::io`              | WAV I/O with streaming and preprocessing (resampling, mono conversion).     |
| `core::ops`             | Sample-wise operations (mix, subtract, multiply, etc.).                     |
| `features::harmonics`   | Harmonic analysis and phase vocoding.                                       |
| `features::inverse`     | MFCC and mel spectrogram inversion to audio.                                |
| `features::manipulation`| Temporal context stacking and statistical features (e.g., kurtosis).        |
| `features::rhythm`     | Tempo estimation and tempogram computation.                                 |
| `features::spectral`    | Spectral features (chroma, MFCC, centroid, etc.).                           |
| `pitch::tuning`         | Pitch detection (YIN, pYIN) and tuning estimation.                          |
| `magnitude::scaling`    | Amplitude/power to dB conversion and perceptual weighting.                  |
| `signal_generation::generators`     | Signal generators (tones, chirps, clicks).                                  |
| `signal_processing::time_frequency`        | STFT, CQT, VQT, and other time-frequency transforms.                        |
| `signal_processing::mixing`                | Stereo, multi-channel, and dry/wet mixing.                                  |
| `signal_processing::mono`          | Mono signal utility to stereo.                                              |
| `signal_processing::time_domain`           | Time-domain processing (e.g., autocorrelation).                             |
| `signal_processing::resampling`            | Sample rate conversion utilities.                                           |
| `utils::frequency`       | Frequency-related utilities (e.g., FFT bin frequencies).                    |
| `utils::notation`        | Music notation conversions (e.g., Hz to MIDI).                              |
| `utils::time`            | Time-related utilities (e.g., frame-to-time mapping).                       |

Discover more on crates https://crates.io/crates/dasp-rs/ and docs https://docs.rs/dasp-rs/0.1.1.

## Example
```rust
use dasp_rs::core::io::load;
use dasp_rs::time_frequency::stft;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load audio file
    let audio = load("input.wav", Some(44100), Some(true), None, None)?;
    
    // Compute Short-Time Fourier Transform with a 2048-sample window
    let spectrogram = stft(&audio.samples, Some(2048), None, None)?;
    
    // Print the shape of the resulting spectrogram
    println!("Spectrogram shape: {} time frames, {} frequency bins", 
        spectrogram.nrows(), spectrogram.ncols());
    
    Ok(())
}
```

## Performance
- **Parallelization**: Uses `rayon` for multi-core efficiency.
- **Memory Efficiency**: Lazy streaming for large files.
- **Numerical Stability**: Built on `ndarray`.

## Contributing
Submit pull requests or issues to the GitHub repository. Follow Rust conventions and include tests.

Contributing steps:
1. Fork the repository.
2. Create a branch: `git checkout -b feature-name`.
3. Commit: `git commit -m "Add feature"`.
4. Push: `git push origin feature-name`.
5. Open a pull request.

## License
`dasp-rs` is licensed under the GPLv3 License. See `LICENSE` for details.
