use hound::WavReader;
use std::path::Path;
use thiserror::Error;
use crate::signal_processing::to_mono;
use ndarray::ShapeError;

#[derive(Error, Debug)]
pub enum AudioError {
    #[error("Failed to open WAV file: {0}")]
    OpenError(#[from] hound::Error),
    #[error("Unsupported WAV format")]
    UnsupportedFormat,
    #[error("Invalid offset or duration")]
    InvalidRange,
    #[error("Audio IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("Hound error: {0}")]
    HoundError(hound::Error),
    #[error("Resample error: {0}")]
    ResampleError(#[from] crate::signal_processing::resampling::ResampleError),
    #[error("Streaming error")]
    StreamError,
    #[error("Array shape error: {0}")]
    ShapeError(#[from] ShapeError),
}


pub struct AudioData {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub channels: u16,
}

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

fn resample(samples: &[f32], orig_sr: u32, target_sr: u32) -> Result<Vec<f32>, AudioError> {
    crate::signal_processing::resample(samples, orig_sr, target_sr)
        .map_err(|e| AudioError::OpenError(hound::Error::IoError(std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))))
}

pub fn get_samplerate<P: AsRef<Path>>(path: P) -> Result<u32, AudioError> {
    let reader = WavReader::open(path)?;
    Ok(reader.spec().sample_rate)
}
