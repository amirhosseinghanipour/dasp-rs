/// Converts multi-channel audio samples to mono by averaging across channels.
///
/// # Arguments
/// * `samples` - Interleaved audio samples (e.g., [L1, R1, L2, R2, ...] for stereo)
/// * `channels` - Number of channels in the input samples
///
/// # Returns
/// Returns a `Vec<f32>` containing the mono audio signal, where each sample is the average
/// of the corresponding samples across all channels.
///
/// # Panics
/// Does not explicitly panic, but if `samples.len()` is not a multiple of `channels`,
/// the last incomplete chunk will be averaged over fewer samples, potentially leading
/// to unexpected results.
///
/// # Examples
/// ```
/// let stereo = vec![0.5, 0.7, 0.3, 0.9]; // [L1, R1, L2, R2]
/// let mono = to_mono(&stereo, 2);
/// assert_eq!(mono, vec![0.6, 0.6]); // [(0.5 + 0.7)/2, (0.3 + 0.9)/2]
/// ```
pub fn to_mono(samples: &[f32], channels: usize) -> Vec<f32> {
    let mut mono = Vec::with_capacity(samples.len() / channels);
    for chunk in samples.chunks(channels) {
        let sum: f32 = chunk.iter().sum();
        mono.push(sum / channels as f32);
    }
    mono
}