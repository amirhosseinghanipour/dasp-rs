use hound::{WavReader, WavWriter, WavSpec, SampleFormat};
use std::path::Path;
use thiserror::Error;
use crate::signal_processing::to_mono;
use ndarray::ShapeError;

/// Custom error types for audio processing operations.
///
/// This enum encompasses various error conditions that may occur during audio file handling,
/// processing, and analysis operations.
#[derive(Error, Debug)]
pub enum AudioError {
    /// Error when opening a WAV file.
    #[error("Failed to open WAV file: {0}")]
    OpenError(#[from] hound::Error),
    
    /// Error when WAV format is not supported.
    #[error("Unsupported WAV format")]
    UnsupportedFormat,
    
    /// Error when specified offset or duration is invalid.
    #[error("Invalid offset or duration")]
    InvalidRange,
    
    /// General I/O error during audio processing.
    #[error("Audio IO error: {0}")]
    IoError(#[from] std::io::Error),
    
    /// Specific error from the hound library.
    #[error("Hound error: {0}")]
    HoundError(hound::Error),
    
    /// Error during resampling operation.
    #[error("Resample error: {0}")]
    ResampleError(#[from] crate::signal_processing::resampling::ResampleError),
    
    /// Error during audio streaming.
    #[error("Streaming error")]
    StreamError,
    
    /// Error related to array shape mismatch.
    #[error("Array shape error: {0}")]
    ShapeError(#[from] ShapeError),
    
    /// Error when there's not enough data for operation.
    #[error("Insufficient data: {0}")]
    InsufficientData(String),
    
    /// Error when input parameters are invalid.
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    
    /// Error when computation fails.
    #[error("Computation failed: {0}")]
    ComputationFailed(String),
}

/// Represents audio data with samples and metadata.
///
/// This struct holds the core audio information including the sample data,
/// sample rate, and number of channels.
pub struct AudioData {
    /// Audio samples as 32-bit floats
    pub samples: Vec<f32>,
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Number of audio channels
    pub channels: u16,
}

/// Loads audio data from a WAV file with optional processing parameters.
///
/// # Arguments
/// * `path` - Path to the WAV file
/// * `sr` - Optional target sample rate for resampling
/// * `mono` - Optional flag to convert to mono (defaults to true if None)
/// * `offset` - Optional start time in seconds
/// * `duration` - Optional duration in seconds
///
/// # Returns
/// Returns `Result<AudioData, AudioError>` containing the processed audio data
/// or an error if loading fails.
///
/// # Examples
/// ```
/// use std::path::Path;
/// let audio = load(Path::new("audio.wav"), None, Some(true), None, None);
/// ```
pub fn load<P: AsRef<Path>>(
    path: P,
    sr: Option<u32>,
    mono: Option<bool>,
    offset: Option<f32>,
    duration: Option<f32>
) -> Result<AudioData, AudioError> {
    let mut reader = WavReader::open(path)?;
    let spec = reader.spec();

    let sample_rate = spec.sample_rate;
    let start_sample = offset.unwrap_or(0.0) * sample_rate as f32;
    let duration_samples = duration.map(|d| (d * sample_rate as f32) as usize);
    let start = start_sample as usize;

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().map(|s| s.unwrap()).collect(),
        hound::SampleFormat::Int => reader
            .samples::<i16>()
            .map(|s| s.unwrap() as f32 / i16::MAX as f32)
            .collect(),
    };

    let end = duration_samples
        .map(|d| std::cmp::min(start + d, samples.len()))
        .unwrap_or(samples.len());
    if start >= samples.len() || end > samples.len() {
        return Err(AudioError::InvalidRange);
    }
    let mut samples = samples[start..end].to_vec();

    let channels = spec.channels as usize;
    if channels > 1 && mono.unwrap_or(true) {
        samples = to_mono(&samples, channels);
    }

    let final_samples = if let Some(target_sr) = sr {
        if target_sr != sample_rate {
            resample(&samples, sample_rate, target_sr)?
        } else {
            samples
        }
    } else {
        samples
    };

    Ok(AudioData {
        samples: final_samples,
        sample_rate: sr.unwrap_or(sample_rate),
        channels: if mono.unwrap_or(true) { 1 } else { spec.channels },
    })
}

/// Exports audio data to a WAV file.
///
/// # Arguments
/// * `path` - Path to the output WAV file
/// * `audio_data` - Audio data to be exported
///
/// # Returns
/// Returns `Result<(), AudioError>` indicating success or failure.
///
/// # Examples
/// ```
/// let audio_data = AudioData {
///     samples: vec![0.0, 0.1, 0.2, 0.3],
///     sample_rate: 44100,
///     channels: 1,
/// };
/// export_to_wav("output.wav", &audio_data)?;
/// ```
pub fn export_to_wav<P: AsRef<Path>>(path: P, audio_data: &AudioData) -> Result<(), AudioError> {
    let spec = WavSpec {
        channels: audio_data.channels,
        sample_rate: audio_data.sample_rate,
        bits_per_sample: 32,
        sample_format: SampleFormat::Float,
    };

    let mut writer = WavWriter::create(path, spec)?;

    for sample in &audio_data.samples {
        writer.write_sample(*sample)?;
    }

    writer.finalize()?;
    Ok(())
}


/// Creates an iterator over audio blocks from a WAV file.
///
/// # Arguments
/// * `path` - Path to the WAV file
/// * `block_length` - Maximum number of blocks to return
/// * `frame_length` - Size of each frame in samples
/// * `hop_length` - Optional number of samples between frames (defaults to frame_length)
///
/// # Returns
/// Returns `Result<impl Iterator<Item = Vec<f32>>, AudioError>` containing
/// an iterator over audio blocks or an error if streaming fails.
///
/// # Examples
/// ```
/// use std::path::Path;
/// let blocks = stream(Path::new("audio.wav"), 10, 1024, None);
/// ```
pub fn stream<P: AsRef<Path>>(
    path: P,
    block_length: usize,
    frame_length: usize,
    hop_length: Option<usize>,
) -> Result<impl Iterator<Item = Vec<f32>>, AudioError> {
    let mut reader = WavReader::open(path)?;
    let spec = reader.spec();
    let hop = hop_length.unwrap_or(frame_length);

    // Read all samples into memory
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().map(|s| s.unwrap()).collect(),
        hound::SampleFormat::Int => reader
            .samples::<i16>()
            .map(|s| s.unwrap() as f32 / i16::MAX as f32)
            .collect(),
    };

    let mut blocks = Vec::new();
    let mut buffer = Vec::with_capacity(frame_length);

    for i in (0..samples.len()).step_by(hop) {
        buffer.clear();
        let start = i;
        let end = std::cmp::min(i + frame_length, samples.len());
        buffer.extend_from_slice(&samples[start..end]);
        buffer.resize(frame_length, 0.0);
        blocks.push(buffer.clone());
        if blocks.len() >= block_length {
            break;
        }
    }

    Ok(blocks.into_iter())
}

/// Resamples audio data to a target sample rate.
///
/// # Arguments
/// * `samples` - Input audio samples
/// * `orig_sr` - Original sample rate
/// * `target_sr` - Target sample rate
///
/// # Returns
/// Returns `Result<Vec<f32>, AudioError>` with resampled audio data
fn resample(samples: &[f32], orig_sr: u32, target_sr: u32) -> Result<Vec<f32>, AudioError> {
    crate::signal_processing::resample(samples, orig_sr, target_sr)
        .map_err(|e| AudioError::OpenError(hound::Error::IoError(std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))))
}

/// Gets the sample rate of a WAV file.
///
/// # Arguments
/// * `path` - Path to the WAV file
///
/// # Returns
/// Returns `Result<u32, AudioError>` with the sample rate in Hz
///
/// # Examples
/// ```
/// use std::path::Path;
/// let sr = get_samplerate(Path::new("audio.wav"));
/// ```
pub fn get_samplerate<P: AsRef<Path>>(path: P) -> Result<u32, AudioError> {
    let reader = WavReader::open(path)?;
    Ok(reader.spec().sample_rate)
}