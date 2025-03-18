use ndarray::Array2;
use num_complex::Complex;

/// Reconstructs a time-domain signal from an STFT magnitude spectrogram using the Griffin-Lim algorithm.
///
/// Iteratively refines a signal estimate by enforcing consistency with the given magnitude spectrogram.
///
/// # Arguments
/// * `s` - Magnitude spectrogram (shape: `[n_freqs, n_frames]`).
/// * `n_iter` - Number of iterations (defaults to 32).
/// * `hop_length` - Hop length between frames (defaults to `n_fft / 4`, minimum 1).
///
/// # Returns
/// Reconstructed time-domain signal as a `Vec<f32>`.
///
/// # Examples
/// ```
/// use ndarray::Array2;
/// let mag_spectrogram = Array2::from_shape_vec((513, 10), vec![1.0; 513 * 10]).unwrap();
/// let signal = dasp_rs::signal_processing::spectral::griffinlim(&mag_spectrogram, None, None);
/// assert_eq!(signal.len(), 513 + 9 * 256); // n_fft + (n_frames - 1) * hop
/// ```
pub fn griffinlim(s: &Array2<f32>, n_iter: Option<usize>, hop_length: Option<usize>) -> Vec<f32> {
    let n_fft = (s.shape()[0] - 1) * 2;
    let hop = hop_length.unwrap_or(n_fft / 4).max(1);
    let n_iter = n_iter.unwrap_or(32);
    let signal_len = hop * (s.shape()[1] - 1) + n_fft;
    let mut y = crate::signal_generation::tone(440.0, None, Some(signal_len), None, None);
    for _ in 0..n_iter {
        let stft_y = crate::signal_processing::stft(&y, Some(n_fft), Some(hop), None).unwrap();
        let (mut mag, mut phase) = crate::signal_processing::magphase(&stft_y, None);
        for ((i, j), m) in mag.indexed_iter_mut() {
            *m = s[[i, j]].sqrt();
            let p = &mut phase[[i, j]];
            if m.abs() > 1e-10 {
                *p /= p.norm();
            }
        }
        let mag_complex = mag.mapv(|x| Complex::new(x, 0.0));
        let new_stft = mag_complex * phase;
        y = crate::signal_processing::istft(&new_stft, Some(hop), None, Some(signal_len));
    }
    y
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_griffinlim() {
        let n_freqs = 513;
        let n_frames = 10;
        let s = Array2::from_shape_vec((n_freqs, n_frames), vec![1.0; n_freqs * n_frames]).unwrap();
        let signal = griffinlim(&s, Some(5), Some(256));
        let expected_len = 256 * (n_frames - 1) + 1024;
        assert_eq!(signal.len(), expected_len);
        assert!(signal.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_griffinlim_zero_hop() {
        let s = Array2::from_shape_vec((513, 10), vec![1.0; 5130]).unwrap();
        let signal = griffinlim(&s, None, Some(0));
        assert!(signal.len() > 0);
    }
}