//! # DASP-RS: Digital Audio Signal Processing in Rust
//!
//! DASP-RS provides a collection of tools and utilities for audio signal processing,
//! analysis, and generation. It includes functionality for handling audio input/output,
//! performing signal transformations, generating synthetic signals, extracting audio features,
//! working with magnitude spectra, and pitch-related operations. The library is designed
//! to be modular and extensible, leveraging Rust's performance and safety features.
//!
//! ## Key Features
//! - Audio I/O: Loading and saving audio files with flexible options.
//! - Signal Processing: Time-frequency transforms (e.g., STFT, CQT) and filtering.
//! - Signal Generation: Creating synthetic waveforms and noise.
//! - Feature Extraction: Computing audio features like tempo, pitch, and spectral properties.
//! - Magnitude Operations: Manipulating and analyzing magnitude spectra.
//! - Pitch Utilities: Converting between frequency, MIDI, and musical notations.
//! - Utilities: General-purpose functions for audio analysis and conversion.
//!
//! ## Usage
//! To use this library, add it to your `Cargo.toml` and import the desired modules or items:
//!
//! ```toml
//! [dependencies]
//! dasp-rs = "0.1.0"
//! ```
//!
//! ```rust
//! use dasp-rs::get_duration;
//! let audio = dasp-rs::audio_io::load("example.wav", None, None, None, None).unwrap();
//! let duration = get_duration(&audio);
//! println!("Duration: {} seconds", duration);
//! ```
//!
//! ## Modules
//! See the individual module documentation for detailed information on available functionality.

/// Audio input/output module.
///
/// Provides functions for loading and saving audio files, as well as handling audio data structures.
pub mod io;

/// Signal processing module.
///
/// Contains implementations of signal transformations such as STFT, CQT, and filtering operations.
pub mod signal_processing;

/// Signal generation module.
///
/// Offers tools for generating synthetic audio signals, including waveforms and noise.
pub mod signal_generation;

/// Feature extraction module.
///
/// Includes functions for extracting audio features like pitch, tempo, and spectral characteristics.
pub mod features;

/// Magnitude spectrum module.
///
/// Provides utilities for manipulating and analyzing magnitude spectra from audio signals.
pub mod magnitude;

/// Utility module.
///
/// General-purpose functions and helpers for audio processing and analysis.
pub mod utils;

/// Pitch processing module.
///
/// Tools for pitch detection, conversion between frequency/MIDI/notes, and musical notation systems.
pub mod pitch;

// Re-export all public items from the modules for convenient access at the crate root.
pub use io::*;
pub use signal_processing::*;
pub use signal_generation::*;
pub use features::*;
pub use magnitude::*;
pub use utils::*;
pub use pitch::*;