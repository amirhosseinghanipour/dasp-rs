use hound::{WavReader, WavWriter, WavSpec, SampleFormat};
use std::path::Path;
use thiserror::Error;
use crate::signal_processing::{to_mono, resample};
use ndarray::ShapeError;
use rayon::prelude::*;
use std::sync::mpsc::{channel, Receiver};
use std::io::Cursor;

/// Enumerates error conditions for WAV-based audio operations.
///
/// Variants encapsulate specific failure modes encountered during file I/O, format parsing,
/// or signal processing, with detailed diagnostics for DSP pipeline debugging.
#[derive(Error, Debug)]
pub enum AudioError {
    /// WAV file open failure, typically due to invalid path or corrupted header.
    #[error("WAV open failed: {0}")]
    OpenError(#[from] hound::Error),
    
    /// Unsupported WAV sample format (only PCM 16-bit int and 32-bit float are supported).
    #[error("Unsupported WAV sample format")]
    UnsupportedFormat,
    
    /// Offset or duration exceeds sample bounds.
    #[error("Offset or duration out of bounds")]
    InvalidRange,
    
    /// General I/O error outside `hound` operations (e.g., filesystem issues).
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),
    
    /// `hound`-specific error during sample read/write.
    #[error("Hound processing error: {0}")]
    HoundError(hound::Error),
    
    /// Resampling failure from `signal_processing::resampling`.
    #[error("Resampling error: {0}")]
    ResampleError(#[from] crate::signal_processing::resampling::ResampleError),
    
    /// Streaming operation failure (e.g., channel disconnect).
    #[error("Stream processing error")]
    StreamError,
    
    /// Array shape mismatch from `ndarray` operations.
    #[error("Shape mismatch: {0}")]
    ShapeError(#[from] ShapeError),
    
    /// Insufficient samples for requested operation.
    #[error("Insufficient sample count: {0}")]
    InsufficientData(String),
    
    /// Invalid parameter (e.g., negative offset).
    #[error("Invalid parameter: {0}")]
    InvalidInput(String),
    
    /// Numerical computation failure (e.g., overflow).
    #[error("Computation error: {0}")]
    ComputationFailed(String),
}

/// Core audio data container for WAV-based DSP workflows.
///
/// Stores interleaved 32-bit float samples with associated sample rate and channel count.
/// Optimized for in-memory processing and compatibility with `signal_processing` operations.
///
/// # Fields
/// - `samples`: Interleaved `f32` sample buffer.
/// - `sample_rate`: Samples per second (Hz).
/// - `channels`: Number of interleaved channels.
#[derive(Debug, Clone)]
pub struct AudioData {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub channels: u16,
}

impl AudioData {
    /// Constructs an `AudioData` instance from raw components.
    ///
    /// # Parameters
    /// - `samples`: Interleaved `f32` sample buffer.
    /// - `sample_rate`: Sample rate in Hz.
    /// - `channels`: Channel count.
    ///
    /// # Returns
    /// Initialized `AudioData` instance.
    pub fn new(samples: Vec<f32>, sample_rate: u32, channels: u16) -> Self {
        Self { samples, sample_rate, channels }
    }
}

/// Loads WAV file into `AudioData` with optional DSP transformations.
///
/// Reads WAV data in-memory via `Cursor`, supporting 16-bit PCM and 32-bit float formats.
/// Applies resampling, mono conversion, and sample trimming as specified.
///
/// # Parameters
/// - `path`: WAV file path (`AsRef<Path>`).
/// - `sr`: Target sample rate (Hz); `None` retains source rate.
/// - `mono`: Convert to mono if `Some(true)`; `None` defaults to `true`.
/// - `offset`: Start time (seconds); `None` defaults to 0.0.
/// - `duration`: Segment length (seconds); `None` takes full length.
///
/// # Returns
/// - `Ok(AudioData)`: Processed audio data.
/// - `Err(AudioError)`: Failure due to I/O, format, or parameter errors.
pub fn load<P: AsRef<Path>>(
    path: P,
    sr: Option<u32>,
    mono: Option<bool>,
    offset: Option<f32>,
    duration: Option<f32>,
) -> Result<AudioData, AudioError> {
    let wav_data = std::fs::read(&path)?;
    let mut reader = WavReader::new(Cursor::new(wav_data))?;
    let spec = reader.spec();
    let sample_rate = spec.sample_rate;

    let start = (offset.unwrap_or(0.0) * sample_rate as f32) as usize;
    let len = duration.map(|d| (d * sample_rate as f32) as usize);

    let samples: Vec<f32> = match spec.sample_format {
        SampleFormat::Float => reader.samples::<f32>()
            .skip(start)
            .take(len.unwrap_or(usize::MAX))
            .map(|s| s.unwrap())
            .collect(),
        SampleFormat::Int => reader.samples::<i16>()
            .skip(start)
            .take(len.unwrap_or(usize::MAX))
            .map(|s| s.unwrap() as f32 / i16::MAX as f32)
            .collect(),
    };

    if start >= samples.len() && !samples.is_empty() {
        return Err(AudioError::InvalidRange);
    }

    let mut samples = samples;
    let channels = spec.channels as usize;
    if channels > 1 && mono.unwrap_or(true) {
        samples = to_mono(&samples, channels);
    }

    let final_samples = if let Some(target_samplerate) = sr {
        if target_samplerate != sample_rate {
            resample(&samples, sample_rate, target_samplerate)?
        } else {
            samples
        }
    } else {
        samples
    };

    Ok(AudioData::new(final_samples, sr.unwrap_or(sample_rate), if mono.unwrap_or(true) { 1 } else { spec.channels }))
}

/// Exports `AudioData` to a WAV file using in-memory buffering.
///
/// Writes 32-bit float WAV data via `Cursor`, committing to disk in a single operation.
///
/// # Parameters
/// - `path`: Output WAV file path (`AsRef<Path>`).
/// - `audio_data`: Source `AudioData` reference.
///
/// # Returns
/// - `Ok(())`: Successful write.
/// - `Err(AudioError)`: I/O or format error.
pub fn export<P: AsRef<Path>>(path: P, audio_data: &AudioData) -> Result<(), AudioError> {
    let spec = WavSpec {
        channels: audio_data.channels,
        sample_rate: audio_data.sample_rate,
        bits_per_sample: 32,
        sample_format: SampleFormat::Float,
    };

    let mut buffer = Vec::with_capacity(audio_data.samples.len() * 4 + 44); // Rough WAV size estimate
    let mut writer = WavWriter::new(Cursor::new(&mut buffer), spec)?;
    for &sample in &audio_data.samples {
        writer.write_sample(sample)?;
    }
    writer.finalize()?;
    std::fs::write(path, buffer)?;
    Ok(())
}

/// Generates an iterator over WAV sample blocks with parallel processing.
///
/// Splits WAV data into fixed-size blocks, processed in parallel using `rayon`.
///
/// # Parameters
/// - `path`: WAV file path (`AsRef<Path>`).
/// - `block_length`: Maximum block count.
/// - `frame_length`: Samples per block.
/// - `hop_length`: Step size between blocks; `None` uses `frame_length`.
///
/// # Returns
/// - `Ok(impl Iterator<Item = Vec<f32>>)`: Block iterator.
/// - `Err(AudioError)`: I/O or format error.
pub fn stream<P: AsRef<Path>>(
    path: P,
    block_length: usize,
    frame_length: usize,
    hop_length: Option<usize>,
) -> Result<impl Iterator<Item = Vec<f32>>, AudioError> {
    let wav_data = std::fs::read(&path)?;
    let mut reader = WavReader::new(Cursor::new(wav_data))?;
    let spec = reader.spec();
    let hop = hop_length.unwrap_or(frame_length);

    let samples: Vec<f32> = match spec.sample_format {
        SampleFormat::Float => reader.samples::<f32>().map(|s| s.unwrap()).collect(),
        SampleFormat::Int => reader.samples::<i16>().map(|s| s.unwrap() as f32 / i16::MAX as f32).collect(),
    };

    let indices: Vec<usize> = (0..samples.len()).step_by(hop).take(block_length).collect();
    let blocks: Vec<Vec<f32>> = indices
        .into_par_iter()
        .map(|i| {
            let end = (i + frame_length).min(samples.len());
            let mut block = Vec::with_capacity(frame_length);
            block.extend_from_slice(&samples[i..end]);
            block.resize(frame_length, 0.0);
            block
        })
        .collect();

    Ok(blocks.into_iter())
}

/// Streams WAV sample blocks lazily with parallel chunk processing.
///
/// Processes WAV data incrementally in a separate thread, generating blocks in parallel
/// within chunks to minimize memory footprint.
///
/// # Parameters
/// - `path`: WAV file path (`AsRef<Path>`).
/// - `block_length`: Maximum block count.
/// - `frame_length`: Samples per block.
/// - `hop_length`: Step size between blocks; `None` uses `frame_length`.
///
/// # Returns
/// - `Ok(Receiver<Vec<f32>>)`: Channel receiver for blocks.
/// - `Err(AudioError)`: I/O or streaming error.
pub fn stream_lazy<P: AsRef<Path>>(
    path: P,
    block_length: usize,
    frame_length: usize,
    hop_length: Option<usize>,
) -> Result<Receiver<Vec<f32>>, AudioError> {
    let wav_data = std::fs::read(&path)?;
    let mut reader = WavReader::new(Cursor::new(wav_data))?;
    let spec = reader.spec();
    let hop = hop_length.unwrap_or(frame_length);

    let (tx, rx) = channel();
    std::thread::spawn(move || {
        let samples_iter: Box<dyn Iterator<Item = Result<f32, _>>> = match spec.sample_format {
            SampleFormat::Float => Box::new(reader.samples::<f32>()),
            SampleFormat::Int => Box::new(reader.samples::<i16>().map(|s| s.map(|v| v as f32 / i16::MAX as f32))),
        };

        let mut chunk = Vec::with_capacity(frame_length * block_length);
        let mut block_count = 0;

        for sample in samples_iter {
            let sample = sample.unwrap_or(0.0);
            chunk.push(sample);

            if chunk.len() >= frame_length && (chunk.len() % hop == 0 || chunk.len() >= frame_length * block_length) {
                let indices: Vec<usize> = (0..chunk.len())
                    .step_by(hop)
                    .take(block_length - block_count)
                    .collect();
                let drain_to = indices.last().map_or(0, |&i| (i + hop).min(chunk.len()));

                let blocks: Vec<Vec<f32>> = indices
                    .into_par_iter()
                    .map(|i| {
                        let end = (i + frame_length).min(chunk.len());
                        let mut block = Vec::with_capacity(frame_length);
                        block.extend_from_slice(&chunk[i..end]);
                        block.resize(frame_length, 0.0);
                        block
                    })
                    .collect();

                for block in blocks {
                    if tx.send(block).is_err() {
                        return;
                    }
                    block_count += 1;
                    if block_count >= block_length {
                        return;
                    }
                }
                chunk.drain(..drain_to);
            }
        }

        if !chunk.is_empty() && block_count < block_length {
            chunk.resize(frame_length, 0.0);
            let _ = tx.send(chunk);
        }
    });

    Ok(rx)
}

/// Extracts sample rate from WAV file header.
///
/// Lightweight metadata query without full sample loading.
///
/// # Parameters
/// - `path`: WAV file path (`AsRef<Path>`).
///
/// # Returns
/// - `Ok(u32)`: Sample rate in Hz.
/// - `Err(AudioError)`: I/O or format error.
pub fn get_samplerate<P: AsRef<Path>>(path: P) -> Result<u32, AudioError> {
    let wav_data = std::fs::read(&path)?;
    let reader = WavReader::new(Cursor::new(wav_data))?;
    Ok(reader.spec().sample_rate)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn create_test_wav() -> AudioData {
        AudioData::new(vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5], 44100, 1)
    }

    #[test]
    fn test_load() {
        let audio = create_test_wav();
        export("test.wav", &audio).unwrap();
        let loaded = load("test.wav", None, Some(true), None, None).unwrap();
        assert_eq!(loaded.samples, audio.samples);
        fs::remove_file("test.wav").unwrap();
    }

    #[test]
    fn test_load_segment() {
        let audio = create_test_wav();
        export("test.wav", &audio).unwrap();
        let loaded = load("test.wav", None, Some(true), Some(0.00004535147), Some(0.00004535147)).unwrap();
        assert_eq!(loaded.samples, vec![0.1, 0.2]);
        fs::remove_file("test.wav").unwrap();
    }

    #[test]
    fn test_export() {
        let audio = create_test_wav();
        export("test.wav", &audio).unwrap();
        let loaded = load("test.wav", None, Some(true), None, None).unwrap();
        assert_eq!(loaded.samples, audio.samples);
        fs::remove_file("test.wav").unwrap();
    }

    #[test]
    fn test_stream() {
        let audio = create_test_wav();
        export("test.wav", &audio).unwrap();
        let blocks: Vec<_> = stream("test.wav", 3, 2, Some(2)).unwrap().collect();
        assert_eq!(blocks, vec![vec![0.0, 0.1], vec![0.2, 0.3], vec![0.4, 0.5]]);
        fs::remove_file("test.wav").unwrap();
    }

    #[test]
    fn test_stream_lazy() {
        let audio = create_test_wav();
        export("test.wav", &audio).unwrap();
        let rx = stream_lazy("test.wav", 3, 2, Some(2)).unwrap();
        let blocks: Vec<_> = rx.into_iter().collect();
        assert_eq!(blocks, vec![vec![0.0, 0.1], vec![0.2, 0.3], vec![0.4, 0.5]]);
        fs::remove_file("test.wav").unwrap();
    }

    #[test]
    fn test_get_samplerate() {
        let audio = create_test_wav();
        export("test.wav", &audio).unwrap();
        assert_eq!(get_samplerate("test.wav").unwrap(), 44100);
        fs::remove_file("test.wav").unwrap();
    }
}