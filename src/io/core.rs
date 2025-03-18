use hound::{WavReader, WavWriter, WavSpec, SampleFormat};
use std::path::Path;
use thiserror::Error;
use crate::signal_processing::{to_mono, resample};
use ndarray::ShapeError;
use rayon::prelude::*;
use std::sync::mpsc::{channel, Receiver};

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

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().map(|s| s.unwrap()).collect(),
        hound::SampleFormat::Int => reader
            .samples::<i16>()
            .map(|s| s.unwrap() as f32 / i16::MAX as f32)
            .collect(),
    };

    let indices: Vec<usize> = (0..samples.len())
        .step_by(hop)
        .take(block_length)
        .collect();

    let blocks: Vec<Vec<f32>> = indices
        .into_par_iter()
        .map(|i| {
            let start = i;
            let end = std::cmp::min(i + frame_length, samples.len());
            let mut buffer = Vec::with_capacity(frame_length);
            buffer.extend_from_slice(&samples[start..end]);
            buffer.resize(frame_length, 0.0);
            buffer
        })
        .collect();

    Ok(blocks.into_iter())
}

/// Lazily streams audio blocks from a WAV file, processing chunks in parallel.
///
/// This function streams blocks of audio data without loading the entire file into memory,
/// using parallel processing for efficiency. It returns a `Receiver` that yields blocks
/// as they become available.
///
/// # Arguments
/// * `path` - Path to the WAV file
/// * `block_length` - Maximum number of blocks to return
/// * `frame_length` - Size of each frame in samples
/// * `hop_length` - Optional number of samples between frames (defaults to frame_length)
///
/// # Returns
/// Returns `Result<Receiver<Vec<f32>>, AudioError>` containing a receiver that yields
/// audio blocks or an error if streaming fails.
///
/// # Examples
/// ```
/// use std::path::Path;
/// let rx = stream_lazy(Path::new("audio.wav"), 10, 1024, None)?;
/// for block in rx {
///     println!("Block length: {}", block.len());
/// }
/// ```
pub fn stream_lazy<P: AsRef<Path>>(
    path: P,
    block_length: usize,
    frame_length: usize,
    hop_length: Option<usize>,
) -> Result<Receiver<Vec<f32>>, AudioError> {
    let mut reader = WavReader::open(path)?;
    let spec = reader.spec();
    let hop = hop_length.unwrap_or(frame_length);

    let (tx, rx) = channel();

    std::thread::spawn(move || {
        let samples_iter: Box<dyn Iterator<Item = f32>> = match spec.sample_format {
            hound::SampleFormat::Float => Box::new(reader
                .samples::<f32>()
                .map(|s| s.unwrap_or(0.0))),
            hound::SampleFormat::Int => Box::new(reader
                .samples::<i16>()
                .map(|s| s.unwrap_or(0))
                .map(|s| s as f32 / i16::MAX as f32)),
        };

        let mut chunk = Vec::with_capacity(frame_length * block_length);
        let mut block_count = 0;

        for (i, sample) in samples_iter.enumerate() {
            chunk.push(sample);

            if (i + 1) % hop == 0 || chunk.len() >= frame_length {
                if chunk.len() >= frame_length {
                    let indices: Vec<usize> = (0..chunk.len())
                        .step_by(hop)
                        .take(block_length - block_count)
                        .collect();

                    let blocks: Vec<Vec<f32>> = indices
                        .par_iter()
                        .map(|&start| {
                            let end = std::cmp::min(start + frame_length, chunk.len());
                            let mut buffer = Vec::with_capacity(frame_length);
                            buffer.extend_from_slice(&chunk[start..end]);
                            buffer.resize(frame_length, 0.0);
                            buffer
                        })
                        .collect();

                    // Send blocks to the receiver
                    for block in blocks {
                        if tx.send(block).is_err() {
                            return;
                        }
                        block_count += 1;
                        if block_count >= block_length {
                            return;
                        }
                    }

                    let last_hop = (indices.last().unwrap_or(&0) + hop).min(chunk.len());
                    chunk.drain(..last_hop);
                }
            }
        }

        // Process any remaining samples
        if !chunk.is_empty() && block_count < block_length {
            let mut buffer = chunk;
            buffer.resize(frame_length, 0.0);
            let _ = tx.send(buffer);
        }
    });

    Ok(rx)
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
/// let sr = get_sr(Path::new("audio.wav"));
/// ```
pub fn get_sr<P: AsRef<Path>>(path: P) -> Result<u32, AudioError> {
    let reader = WavReader::open(path)?;
    Ok(reader.spec().sample_rate)
}