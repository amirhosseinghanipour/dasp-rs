use ndarray::Array2;
use num_complex::Complex;

/// Interpolates harmonic amplitudes across frequency bins.
///
/// Performs linear interpolation of input amplitudes at harmonic frequencies
/// specified by the fundamental frequencies and harmonic multipliers.
///
/// # Arguments
/// * `x` - Input amplitude spectrum
/// * `freqs` - Frequency bins corresponding to `x`
/// * `harmonics` - Harmonic multipliers (e.g., [1.0, 2.0, 3.0] for first three harmonics)
///
/// # Returns
/// Returns a 2D array of shape `(n_harmonics, n_bins)` containing interpolated
/// amplitudes for each harmonic at each frequency bin.
///
/// # Panics
/// Panics if `x` and `freqs` have different lengths.
///
/// # Examples
/// ```
/// let x = vec![0.1, 0.2, 0.3, 0.4];
/// let freqs = vec![0.0, 100.0, 200.0, 300.0];
/// let harmonics = vec![1.0, 2.0];
/// let result = interp_harmonics(&x, &freqs, &harmonics);
/// ```
pub fn interp_harmonics(x: &[f32], freqs: &[f32], harmonics: &[f32]) -> Array2<f32> {
    assert_eq!(x.len(), freqs.len(), "x and freqs must have the same length");
    let n_bins = freqs.len();
    let n_harmonics = harmonics.len();
    let mut result = Array2::zeros((n_harmonics, n_bins));

    for (h_idx, &h) in harmonics.iter().enumerate() {
        for (bin, &f) in freqs.iter().enumerate() {
            let target_freq = f * h;
            if target_freq < freqs[freqs.len() - 1] {
                let left_idx = freqs.iter().position(|&x| x >= target_freq).unwrap_or(n_bins - 1);
                let left_idx = left_idx.saturating_sub(1);
                let right_idx = (left_idx + 1).min(n_bins - 1);
                let left_freq = freqs[left_idx];
                let right_freq = freqs[right_idx];
                if left_freq == right_freq {
                    result[[h_idx, bin]] = x[left_idx];
                } else {
                    let alpha = (target_freq - left_freq) / (right_freq - left_freq);
                    result[[h_idx, bin]] = x[left_idx] * (1.0 - alpha) + x[right_idx] * alpha;
                }
            }
        }
    }
    result
}

/// Computes a salience map from a spectrogram using harmonic summation.
///
/// Calculates salience by summing weighted harmonic contributions for each
/// frequency bin across time frames.
///
/// # Arguments
/// * `S` - Input spectrogram (n_bins × n_frames)
/// * `freqs` - Frequency bins corresponding to spectrogram rows
/// * `harmonics` - Harmonic multipliers
/// * `weights` - Optional weights for each harmonic (defaults to 1.0 for all)
///
/// # Returns
/// Returns a 2D array of shape `(n_bins, n_frames)` containing the salience map.
///
/// # Panics
/// Panics if `weights` length doesn't match `harmonics` length when provided.
///
/// # Examples
/// ```
/// use ndarray::Array2;
/// let S = Array2::from_shape_vec((4, 3), vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]).unwrap();
/// let freqs = vec![0.0, 100.0, 200.0, 300.0];
/// let harmonics = vec![1.0, 2.0];
/// let salience_map = salience(&S, &freqs, &harmonics, None);
/// ```
pub fn salience(S: &Array2<f32>, freqs: &[f32], harmonics: &[f32], weights: Option<&[f32]>) -> Array2<f32> {
    let n_bins = S.shape()[0];
    let n_frames = S.shape()[1];
    let n_harmonics = harmonics.len();
    let default_weights = vec![1.0; n_harmonics];
    let weights = weights.unwrap_or(&default_weights);
    assert_eq!(weights.len(), n_harmonics, "weights length must match harmonics");

    let mut salience_map = Array2::zeros((n_bins, n_frames));
    for frame in 0..n_frames {
        for (bin, &f) in freqs.iter().enumerate() {
            let mut total_salience = 0.0;
            for (h_idx, &h) in harmonics.iter().enumerate() {
                let harmonic_freq = f * h;
                if harmonic_freq < freqs[n_bins - 1] {
                    let nearest_bin = freqs.iter().position(|&x| x >= harmonic_freq).unwrap_or(n_bins - 1);
                    let nearest_bin = nearest_bin.saturating_sub(1);
                    let left_freq = freqs[nearest_bin];
                    let right_freq = freqs[(nearest_bin + 1).min(n_bins - 1)];
                    let alpha = (harmonic_freq - left_freq) / (right_freq - left_freq);
                    let interp_val = S[[nearest_bin, frame]] * (1.0 - alpha) + 
                                   S[[(nearest_bin + 1).min(n_bins - 1), frame]] * alpha;
                    total_salience += interp_val * weights[h_idx];
                }
            }
            salience_map[[bin, frame]] = total_salience;
        }
    }
    salience_map
}

/// Extracts harmonic amplitudes based on fundamental frequencies.
///
/// Interpolates amplitudes at harmonic frequencies derived from time-varying
/// fundamental frequencies.
///
/// # Arguments
/// * `x` - Input amplitude spectrum
/// * `f0` - Fundamental frequencies across frames
/// * `freqs` - Frequency bins corresponding to `x`
/// * `harmonics` - Harmonic multipliers
///
/// # Returns
/// Returns a 2D array of shape `(n_harmonics, n_frames)` containing interpolated
/// harmonic amplitudes.
///
/// # Panics
/// Panics if `x` and `freqs` have different lengths.
///
/// # Examples
/// ```
/// let x = vec![0.1, 0.2, 0.3, 0.4];
/// let f0 = vec![100.0, 110.0];
/// let freqs = vec![0.0, 100.0, 200.0, 300.0];
/// let harmonics = vec![1.0, 2.0];
/// let result = f0_harmonics(&x, &f0, &freqs, &harmonics);
/// ```
pub fn f0_harmonics(x: &[f32], f0: &[f32], freqs: &[f32], harmonics: &[f32]) -> Array2<f32> {
    assert_eq!(x.len(), freqs.len(), "x and freqs must have the same length");
    let n_frames = f0.len();
    let n_harmonics = harmonics.len();
    let mut result = Array2::zeros((n_harmonics, n_frames));

    for (frame, &fund) in f0.iter().enumerate() {
        for (h_idx, &h) in harmonics.iter().enumerate() {
            let target_freq = fund * h;
            if target_freq < freqs[freqs.len() - 1] {
                let left_idx = freqs.iter().position(|&x| x >= target_freq).unwrap_or(freqs.len() - 1);
                let left_idx = left_idx.saturating_sub(1);
                let right_idx = (left_idx + 1).min(freqs.len() - 1);
                let left_freq = freqs[left_idx];
                let right_freq = freqs[right_idx];
                if left_freq == right_freq {
                    result[[h_idx, frame]] = x[left_idx];
                } else {
                    let alpha = (target_freq - left_freq) / (right_freq - left_freq);
                    result[[h_idx, frame]] = x[left_idx] * (1.0 - alpha) + x[right_idx] * alpha;
                }
            }
        }
    }
    result
}

/// Performs phase vocoding for time stretching or compression.
///
/// Modifies the time scale of a complex spectrogram while preserving frequency content.
///
/// # Arguments
/// * `D` - Input complex spectrogram (n_bins × n_frames)
/// * `rate` - Time stretching factor (>1 for stretching, <1 for compression)
/// * `hop_length` - Optional hop length between frames (defaults to n_fft/4)
/// * `n_fft` - Optional FFT size (defaults to derived from spectrogram size)
///
/// # Returns
/// Returns a 2D array of complex values with adjusted frame count based on rate.
///
/// # Examples
/// ```
/// use ndarray::Array2;
/// use num_complex::Complex;
/// let D = Array2::from_shape_vec((3, 4), vec![
///     Complex::new(1.0, 0.0), Complex::new(2.0, 0.0),
///     Complex::new(3.0, 0.0), Complex::new(4.0, 0.0),
///     Complex::new(5.0, 0.0), Complex::new(6.0, 0.0),
///     Complex::new(7.0, 0.0), Complex::new(8.0, 0.0),
///     Complex::new(9.0, 0.0), Complex::new(10.0, 0.0),
///     Complex::new(11.0, 0.0), Complex::new(12.0, 0.0),
/// ]).unwrap();
/// let result = phase_vocoder(&D, 2.0, None, None);
/// ```
pub fn phase_vocoder(D: &Array2<Complex<f32>>, rate: f32, hop_length: Option<usize>, n_fft: Option<usize>) -> Array2<Complex<f32>> {
    let n = n_fft.unwrap_or((D.shape()[0] - 1) * 2);
    let hop = hop_length.unwrap_or(n / 4);
    let orig_frames = D.shape()[1];
    let new_frames = ((orig_frames as f32 * hop as f32) / rate / hop as f32).ceil() as usize;
    let mut output = Array2::zeros((D.shape()[0], new_frames));

    for new_idx in 0..new_frames {
        let orig_idx = ((new_idx as f32 * hop as f32 * rate) / hop as f32) as usize;
        if orig_idx < orig_frames {
            output.column_mut(new_idx).assign(&D.column(orig_idx));
        }
    }
    output
}