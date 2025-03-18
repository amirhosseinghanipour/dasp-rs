use ndarray::{Array2, ArrayView1};
use num_complex::Complex;
use rayon::prelude::*;
use thiserror::Error;

/// Error conditions for harmonic feature extraction and processing.
///
/// Enumerates specific failure modes in harmonic analysis and phase vocoding operations,
/// providing detailed diagnostics for DSP pipeline debugging.
#[derive(Error, Debug)]
pub enum HarmonicsError {
    /// Input arrays have mismatched lengths (e.g., amplitudes vs. frequency bins).
    #[error("Length mismatch: {0} vs {1}")]
    LengthMismatch(usize, usize),

    /// Invalid input parameter (e.g., empty array, negative rate).
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Numerical computation failure (e.g., division by zero, overflow).
    #[error("Computation failed: {0}")]
    ComputationFailed(String),
}

/// Interpolates harmonic amplitudes across frequency bins in parallel.
///
/// Performs linear interpolation of amplitude values at harmonic frequencies derived
/// from a fundamental frequency grid, leveraging parallel processing for efficiency.
///
/// # Parameters
/// - `x`: Amplitude spectrum as a slice of `f32` (frequency bin values).
/// - `freqs`: Frequency bins corresponding to `x` (monotonically increasing).
/// - `harmonics`: Harmonic multipliers (e.g., `[1.0, 2.0, 3.0]` for first three harmonics).
///
/// # Returns
/// - `Ok(Array2<f32>)`: Interpolated amplitudes, shape `(n_harmonics, n_bins)`.
/// - `Err(HarmonicsError)`: Failure due to length mismatch or invalid input.
///
/// # Constraints
/// - `x.len() == freqs.len()`; enforced via error return.
/// - `freqs` must be sorted in ascending order for correct interpolation.
/// - Harmonic frequencies exceeding `freqs.last()` are clamped to zero.
pub fn interp_harmonics(x: &[f32], freqs: &[f32], harmonics: &[f32]) -> Result<Array2<f32>, HarmonicsError> {
    if x.len() != freqs.len() {
        return Err(HarmonicsError::LengthMismatch(x.len(), freqs.len()));
    }
    if x.is_empty() {
        return Err(HarmonicsError::InvalidInput("Amplitude spectrum is empty".to_string()));
    }
    if !freqs.windows(2).all(|w| w[0] <= w[1]) {
        return Err(HarmonicsError::InvalidInput("Frequency bins must be sorted".to_string()));
    }

    let n_bins = freqs.len();
    let n_harmonics = harmonics.len();
    let mut result = Array2::zeros((n_harmonics, n_bins));

    result.axis_iter_mut(ndarray::Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(h_idx, mut row)| {
            let h = harmonics[h_idx];
            for (bin, &f) in freqs.iter().enumerate() {
                let target_freq = f * h;
                if target_freq <= *freqs.last().unwrap() {
                    let left_idx = freqs.binary_search_by(|&x| x.partial_cmp(&target_freq).unwrap())
                        .unwrap_or_else(|e| e.saturating_sub(1));
                    let left_idx = left_idx.min(n_bins - 2); // Ensure right_idx is valid
                    let right_idx = left_idx + 1;
                    let left_freq = freqs[left_idx];
                    let right_freq = freqs[right_idx];
                    let alpha = if left_freq == right_freq {
                        0.0
                    } else {
                        (target_freq - left_freq) / (right_freq - left_freq)
                    };
                    row[bin] = x[left_idx] * (1.0 - alpha) + x[right_idx] * alpha;
                }
            }
        });

    Ok(result)
}

/// Computes a salience map from a spectrogram via harmonic summation.
///
/// Sums weighted harmonic contributions across frequency bins and frames, producing
/// a salience map for pitch detection or harmonic analysis. Parallelized over frames.
///
/// # Parameters
/// - `s`: Spectrogram as `Array2<f32>`, shape `(n_bins, n_frames)`.
/// - `freqs`: Frequency bins corresponding to spectrogram rows (monotonically increasing).
/// - `harmonics`: Harmonic multipliers (e.g., `[1.0, 2.0]`).
/// - `weights`: Optional harmonic weights; defaults to uniform `1.0` if `None`.
///
/// # Returns
/// - `Ok(Array2<f32>)`: Salience map, shape `(n_bins, n_frames)`.
/// - `Err(HarmonicsError)`: Failure due to dimension mismatch or invalid input.
///
/// # Constraints
/// - `s.shape()[0] == freqs.len()`.
/// - `weights.len() == harmonics.len()` if provided.
pub fn salience(s: &Array2<f32>, freqs: &[f32], harmonics: &[f32], weights: Option<&[f32]>) -> Result<Array2<f32>, HarmonicsError> {
    let n_bins = s.shape()[0];
    let n_frames = s.shape()[1];
    if n_bins != freqs.len() {
        return Err(HarmonicsError::LengthMismatch(n_bins, freqs.len()));
    }
    if n_bins == 0 || n_frames == 0 {
        return Err(HarmonicsError::InvalidInput("Spectrogram is empty".to_string()));
    }
    if !freqs.windows(2).all(|w| w[0] <= w[1]) {
        return Err(HarmonicsError::InvalidInput("Frequency bins must be sorted".to_string()));
    }

    let n_harmonics = harmonics.len();
    let default_weights = vec![1.0; n_harmonics];
    let weights = weights.unwrap_or(&default_weights);
    if weights.len() != n_harmonics {
        return Err(HarmonicsError::LengthMismatch(weights.len(), n_harmonics));
    }

    let mut salience_map = Array2::zeros((n_bins, n_frames));
    salience_map.axis_iter_mut(ndarray::Axis(1))
        .into_par_iter()
        .enumerate()
        .for_each(|(frame, mut col)| {
            for (bin, &f) in freqs.iter().enumerate() {
                let mut total = 0.0;
                for (h_idx, &h) in harmonics.iter().enumerate() {
                    let harmonic_freq = f * h;
                    if harmonic_freq <= freqs[n_bins - 1] {
                        let left_idx = freqs.binary_search_by(|&x| x.partial_cmp(&harmonic_freq).unwrap())
                            .unwrap_or_else(|e| e.saturating_sub(1));
                        let left_idx = left_idx.min(n_bins - 2);
                        let right_idx = left_idx + 1;
                        let alpha = (harmonic_freq - freqs[left_idx]) / (freqs[right_idx] - freqs[left_idx]);
                        let interp = s[[left_idx, frame]] * (1.0 - alpha) + s[[right_idx, frame]] * alpha;
                        total += interp * weights[h_idx];
                    }
                }
                col[bin] = total;
            }
        });

    Ok(salience_map)
}

/// Extracts harmonic amplitudes from time-varying fundamental frequencies.
///
/// Interpolates amplitudes at harmonic frequencies based on frame-wise `f0` values,
/// parallelized across frames for performance.
///
/// # Parameters
/// - `x`: Amplitude spectrum as a slice of `f32`.
/// - `f0`: Fundamental frequencies per frame.
/// - `freqs`: Frequency bins corresponding to `x` (monotonically increasing).
/// - `harmonics`: Harmonic multipliers.
///
/// # Returns
/// - `Ok(Array2<f32>)`: Harmonic amplitudes, shape `(n_harmonics, n_frames)`.
/// - `Err(HarmonicsError)`: Failure due to length mismatch or invalid input.
///
/// # Constraints
/// - `x.len() == freqs.len()`.
/// - `f0` must be non-empty.
pub fn f0_harmonics(x: &[f32], f0: &[f32], freqs: &[f32], harmonics: &[f32]) -> Result<Array2<f32>, HarmonicsError> {
    if x.len() != freqs.len() {
        return Err(HarmonicsError::LengthMismatch(x.len(), freqs.len()));
    }
    if x.is_empty() || f0.is_empty() {
        return Err(HarmonicsError::InvalidInput("Input arrays cannot be empty".to_string()));
    }
    if !freqs.windows(2).all(|w| w[0] <= w[1]) {
        return Err(HarmonicsError::InvalidInput("Frequency bins must be sorted".to_string()));
    }

    let n_frames = f0.len();
    let n_harmonics = harmonics.len();
    let mut result = Array2::zeros((n_harmonics, n_frames));

    result.axis_iter_mut(ndarray::Axis(1))
        .into_par_iter()
        .enumerate()
        .for_each(|(frame, mut col)| {
            let fund = f0[frame];
            for (h_idx, &h) in harmonics.iter().enumerate() {
                let target_freq = fund * h;
                if target_freq <= freqs[freqs.len() - 1] {
                    let left_idx = freqs.binary_search_by(|&x| x.partial_cmp(&target_freq).unwrap())
                        .unwrap_or_else(|e| e.saturating_sub(1));
                    let left_idx = left_idx.min(freqs.len() - 2);
                    let right_idx = left_idx + 1;
                    let alpha = (target_freq - freqs[left_idx]) / (freqs[right_idx] - freqs[left_idx]);
                    col[h_idx] = x[left_idx] * (1.0 - alpha) + x[right_idx] * alpha;
                }
            }
        });

    Ok(result)
}

/// Performs phase vocoding for time-scale modification of a complex spectrogram.
///
/// Adjusts the temporal resolution of a spectrogram while preserving frequency content,
/// implementing phase unwrapping and accumulation for coherent resynthesis.
///
/// # Parameters
/// - `d`: Complex spectrogram as `Array2<Complex<f32>>`, shape `(n_bins, n_frames)`.
/// - `rate`: Time stretching factor (>1 stretches, <1 compresses).
/// - `hop_length`: Optional hop length between frames; defaults to `n_fft / 4`.
/// - `n_fft`: Optional FFT size; defaults to `(n_bins - 1) * 2`.
///
/// # Returns
/// - `Ok(Array2<Complex<f32>>)`: Time-scaled spectrogram with adjusted frame count.
/// - `Err(HarmonicsError)`: Failure due to invalid rate or dimensions.
///
/// # Constraints
/// - `rate > 0.0`.
/// - `hop_length > 0` if provided.
pub fn phase_vocoder(
    d: &Array2<Complex<f32>>,
    rate: f32,
    hop_length: Option<usize>,
    n_fft: Option<usize>,
) -> Result<Array2<Complex<f32>>, HarmonicsError> {
    if rate <= 0.0 {
        return Err(HarmonicsError::InvalidInput("Rate must be positive".to_string()));
    }
    let n_bins = d.shape()[0];
    let orig_frames = d.shape()[1];
    if n_bins == 0 || orig_frames == 0 {
        return Err(HarmonicsError::InvalidInput("Spectrogram is empty".to_string()));
    }

    let n_fft = n_fft.unwrap_or((n_bins - 1) * 2);
    let hop = hop_length.unwrap_or(n_fft / 4);
    if hop == 0 {
        return Err(HarmonicsError::InvalidInput("Hop length must be positive".to_string()));
    }

    let new_frames = ((orig_frames as f32 * hop as f32) / rate / hop as f32).ceil() as usize;
    let mut output = Array2::zeros((n_bins, new_frames));
    let mut phase_acc = Array2::zeros((n_bins, 1)); // Initial phase

    let omega = (0..n_bins).map(|k| 2.0 * std::f32::consts::PI * k as f32 / n_fft as f32).collect::<Vec<f32>>();

    for t in 0..new_frames {
        let orig_t = (t as f32 * rate * hop as f32 / hop as f32) as usize;
        let orig_t_next = ((t + 1) as f32 * rate * hop as f32 / hop as f32) as usize;

        if orig_t >= orig_frames || orig_t_next >= orig_frames {
            continue;
        }

        let mag = d.column(orig_t).mapv(|c| c.norm());
        let phase = d.column(orig_t).mapv(|c| c.arg());
        let phase_next = d.column(orig_t_next).mapv(|c| c.arg());
        let delta_phase = phase_next - phase - &ArrayView1::from(&omega) * hop as f32;

        let delta_phase_unwrapped: Vec<f32> = delta_phase.iter()
            .map(|&dp| dp - 2.0 * std::f32::consts::PI * (dp / (2.0 * std::f32::consts::PI)).round())
            .collect();

        let phase_advance = ArrayView1::from(&delta_phase_unwrapped).mapv(|x| x / hop as f32 * (hop as f32 * rate));
        phase_acc = phase_acc + phase_advance;

        output.column_mut(t).assign(&mag.mapv(|m| Complex::from_polar(m, phase_acc[[0, 0]])));
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_interp_harmonics() {
        let x = vec![0.1, 0.2, 0.3, 0.4];
        let freqs = vec![0.0, 100.0, 200.0, 300.0];
        let harmonics = vec![1.0, 2.0];
        let result = interp_harmonics(&x, &freqs, &harmonics).unwrap();
        assert_eq!(result.shape(), &[2, 4]);
        assert_eq!(result[[0, 0]], 0.1);
        assert_eq!(result[[1, 0]], 0.1);
        assert_eq!(result[[0, 1]], 0.2);
        assert_eq!(result[[1, 1]], 0.3);
    }

    #[test]
    fn test_interp_harmonics_mismatch() {
        let x = vec![0.1, 0.2];
        let freqs = vec![0.0, 100.0, 200.0];
        let harmonics = vec![1.0];
        let result = interp_harmonics(&x, &freqs, &harmonics);
        assert!(matches!(result, Err(HarmonicsError::LengthMismatch(2, 3))));
    }

    #[test]
    fn test_salience() {
        let s = array![[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9], [1.0, 1.1, 1.2]];
        let freqs = vec![0.0, 100.0, 200.0, 300.0];
        let harmonics = vec![1.0, 2.0];
        let weights: Option<&[f32]> = Some(&[1.0, 0.5]);
        let result = salience(&s, &freqs, &harmonics, weights).unwrap();
        assert_eq!(result.shape(), &[4, 3]);
        assert_eq!(result[[0, 0]], 0.1 + 0.1 * 0.5);
        assert_eq!(result[[1, 0]], 0.4 + 0.35);
    }

    #[test]
    fn test_salience_weight_mismatch() {
        let s = array![[0.1, 0.2], [0.3, 0.4]];
        let freqs = vec![0.0, 100.0];
        let harmonics = vec![1.0, 2.0];
        let weights: Option<&[f32]> = Some(&[1.0]);
        let result = salience(&s, &freqs, &harmonics, weights);
        assert!(matches!(result, Err(HarmonicsError::LengthMismatch(1, 2))));
    }

    #[test]
    fn test_f0_harmonics() {
        let x = vec![0.1, 0.2, 0.3, 0.4];
        let f0 = vec![100.0, 150.0];
        let freqs = vec![0.0, 100.0, 200.0, 300.0];
        let harmonics = vec![1.0, 2.0];
        let result = f0_harmonics(&x, &f0, &freqs, &harmonics).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result[[0, 0]], 0.2);
        assert_eq!(result[[1, 0]], 0.3);
        assert_eq!(result[[0, 1]], 0.25);
    }

    #[test]
    fn test_f0_harmonics_empty() {
        let x = vec![];
        let f0 = vec![100.0];
        let freqs = vec![];
        let harmonics = vec![1.0];
        let result = f0_harmonics(&x, &f0, &freqs, &harmonics);
        assert!(matches!(result, Err(HarmonicsError::InvalidInput(_))));
    }

    #[test]
    fn test_phase_vocoder() {
        let d = array![
            [Complex::new(1.0, 0.0), Complex::new(2.0, 0.0), Complex::new(3.0, 0.0), Complex::new(4.0, 0.0)],
            [Complex::new(5.0, 0.0), Complex::new(6.0, 0.0), Complex::new(7.0, 0.0), Complex::new(8.0, 0.0)],
        ];
        let result = phase_vocoder(&d, 0.5, Some(1), Some(4)).unwrap();
        assert_eq!(result.shape(), &[2, 8]);
        assert_eq!(result[[0, 0]].norm(), 1.0);
    }

    #[test]
    fn test_phase_vocoder_invalid_rate() {
        let d = array![[Complex::new(1.0, 0.0)]];
        let result = phase_vocoder(&d, 0.0, None, None);
        assert!(matches!(result, Err(HarmonicsError::InvalidInput(_))));
    }
}