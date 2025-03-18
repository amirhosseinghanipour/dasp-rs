use hound::{WavReader, WavWriter};
use std::path::Path;
use crate::io::core::AudioError;
use crate::io::core::AudioData;

/// Gets metadata from a WAV file.
///
/// # Arguments
/// * `path` - Path to the WAV file
///
/// # Returns
/// Returns `Result<(u32, u16, f32), AudioError>` containing the sample rate, number of channels,
/// and duration in seconds, or an error if reading fails.
///
/// # Examples
/// ```
/// use std::path::Path;
/// let (sample_rate, channels, duration) = get_metadata(Path::new("audio.wav"))?;
/// ```
pub fn get_metadata<P: AsRef<Path>>(path: P) -> Result<(u32, u16, f32), AudioError> {
    let reader = WavReader::open(path)?;
    let spec = reader.spec();
    let duration = reader.duration() as f32 / spec.sample_rate as f32;

    Ok((spec.sample_rate, spec.channels, duration))
}

/// Splits audio data into chunks of a specified duration.
///
/// # Arguments
/// * `samples` - Input audio samples
/// * `sample_rate` - Sample rate in Hz
/// * `chunk_duration` - Duration of each chunk in seconds
///
/// # Returns
/// Returns a `Vec<Vec<f32>>` containing the audio chunks.
///
/// # Examples
/// ```
/// let chunks = split_into_chunks(&samples, 44100, 1.0);
/// ```
pub fn split_into_chunks(samples: &[f32], sample_rate: u32, chunk_duration: f32) -> Vec<Vec<f32>> {
    let chunk_size = (sample_rate as f32 * chunk_duration) as usize;
    samples.chunks(chunk_size).map(|chunk| chunk.to_vec()).collect()
}

/// Applies a gain to the audio data.
///
/// # Arguments
/// * `samples` - Input audio samples
/// * `gain` - Gain factor (e.g., 1.0 for no change, 0.5 for attenuation, 2.0 for amplification)
///
/// # Returns
/// Returns a `Vec<f32>` containing the audio samples with the applied gain.
///
/// # Examples
/// ```
/// let amplified_samples = apply_gain(&samples, 1.5);
/// ```
pub fn apply_gain(samples: &[f32], gain: f32) -> Vec<f32> {
    samples.iter().map(|&sample| sample * gain).collect()
}

/// Appends audio data to an existing WAV file.
///
/// # Arguments
/// * `path` - Path to the existing WAV file
/// * `audio_data` - Audio data to be appended
///
/// # Returns
/// Returns `Result<(), AudioError>` indicating success or failure.
///
/// # Examples
/// ```
/// let audio_data = AudioData {
///     samples: vec![0.4, 0.5, 0.6, 0.7],
///     sample_rate: 44100,
///     channels: 1,
/// };
/// append_audio("existing.wav", &audio_data)?;
/// ```
pub fn append_audio<P: AsRef<Path>>(path: P, audio_data: &AudioData) -> Result<(), AudioError> {
    let mut writer = WavWriter::append(path)?;

    for sample in &audio_data.samples {
        writer.write_sample(*sample)?;
    }

    writer.finalize()?;
    Ok(())
}