use crate::audio_io::AudioData;
use ndarray::Array2;

pub fn get_duration(audio: &AudioData) -> f32 {
    audio.samples.len() as f32 / audio.sample_rate as f32
}

pub fn get_duration_from_path<P: AsRef<std::path::Path>>(path: P) -> Result<f32, crate::audio_io::AudioError> {
    let audio = crate::audio_io::load(path, None, None, None, None)?;
    Ok(get_duration(&audio))
}

pub fn frames_to_samples(frames: &[usize], hop_length: Option<usize>, _n_fft: Option<usize>) -> Vec<usize> {
    let hop = hop_length.unwrap_or(512);
    frames.iter().map(|&f| f * hop).collect()
}

pub fn frames_to_time(frames: &[usize], sr: Option<u32>, hop_length: Option<usize>) -> Vec<f32> {
    let sample_rate = sr.unwrap_or(44100);
    let hop = hop_length.unwrap_or(512);
    frames.iter().map(|&f| f as f32 * hop as f32 / sample_rate as f32).collect()
}

pub fn samples_to_frames(samples: &[usize], hop_length: Option<usize>) -> Vec<usize> {
    let hop = hop_length.unwrap_or(512);
    samples.iter().map(|&s| s / hop).collect()
}

pub fn samples_to_time(samples: &[usize], sr: Option<u32>) -> Vec<f32> {
    let sample_rate = sr.unwrap_or(44100);
    samples.iter().map(|&s| s as f32 / sample_rate as f32).collect()
}

pub fn time_to_frames(times: &[f32], sr: Option<u32>, hop_length: Option<usize>, _n_fft: Option<usize>) -> Vec<usize> {
    let sample_rate = sr.unwrap_or(44100);
    let hop = hop_length.unwrap_or(512);
    times.iter().map(|&t| (t * sample_rate as f32 / hop as f32) as usize).collect()
}

pub fn time_to_samples(times: &[f32], sr: Option<u32>) -> Vec<usize> {
    let sample_rate = sr.unwrap_or(44100);
    times.iter().map(|&t| (t * sample_rate as f32) as usize).collect()
}

pub fn blocks_to_frames(blocks: &[usize], block_length: usize) -> Vec<usize> {
    blocks.iter().map(|&b| b * block_length).collect()
}

pub fn blocks_to_samples(blocks: &[usize], block_length: usize, hop_length: Option<usize>) -> Vec<usize> {
    let hop = hop_length.unwrap_or(512);
    blocks.iter().map(|&b| b * block_length * hop).collect()
}

pub fn blocks_to_time(blocks: &[usize], block_length: usize, hop_length: Option<usize>, sr: Option<u32>) -> Vec<f32> {
    let hop = hop_length.unwrap_or(512);
    let sample_rate = sr.unwrap_or(44100);
    blocks.iter().map(|&b| b as f32 * block_length as f32 * hop as f32 / sample_rate as f32).collect()
}

pub fn samples_like(X: &Array2<f32>, hop_length: Option<usize>, _n_fft: Option<usize>, _axis: Option<isize>) -> Vec<usize> {
    let hop = hop_length.unwrap_or(512);
    (0..X.shape()[1]).map(|i| i * hop).collect()
}

pub fn times_like(X: &Array2<f32>, sr: Option<u32>, hop_length: Option<usize>, _n_fft: Option<usize>, _axis: Option<isize>) -> Vec<f32> {
    let sample_rate = sr.unwrap_or(44100);
    let hop = hop_length.unwrap_or(512);
    (0..X.shape()[1]).map(|i| i as f32 * hop as f32 / sample_rate as f32).collect()
}