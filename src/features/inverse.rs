use ndarray::{Array2, Array1, Axis};
use crate::signal_processing::spectral::istft;
use crate::features::spectral::melspectrogram;
use crate::features::phase_recovery::griffinlim;

/// Computes Mel-frequency cepstral coefficients (MFCCs) from audio or spectrogram.
///
/// # Arguments
/// * `y` - Optional audio time series
/// * `sr` - Optional sample rate (required if `y` is provided)
/// * `S` - Optional pre-computed spectrogram
/// * `n_mfcc` - Optional number of MFCCs to return (defaults to 20)
/// * `dct_type` - Optional DCT type (1, 2, or 3; defaults to 2)
/// * `norm` - Optional normalization ("ortho" or None; defaults to None)
///
/// # Returns
/// Returns a 2D array of shape `(n_mfcc, n_frames)` containing MFCCs.
///
/// # Examples
/// ```
/// let y = vec![0.1, 0.2, 0.3, 0.4];
/// let mfcc = mfcc(Some(&y), Some(44100), None, None, None, None);
/// ```
pub fn mfcc(
    y: Option<&[f32]>,
    sr: Option<u32>,
    S: Option<&Array2<f32>>,
    n_mfcc: Option<usize>,
    dct_type: Option<i32>,
    norm: Option<&str>,
) -> Array2<f32> {
    let n_mfcc = n_mfcc.unwrap_or(20);
    let S = melspectrogram(y, sr, S, None, None, None, None, None);
    let log_S = S.mapv(|x| x.max(1e-10).ln());
    let mut mfcc = Array2::zeros((n_mfcc, S.shape()[1]));
    let dct_type = dct_type.unwrap_or(2);
    for t in 0..S.shape()[1] {
        for k in 0..n_mfcc {
            let mut sum = 0.0;
            for n in 0..S.shape()[0] {
                sum += log_S[[n, t]] * (std::f32::consts::PI * k as f32 * (n as f32 + 0.5) / S.shape()[0] as f32).cos();
            }
            mfcc[[k, t]] = sum * if dct_type == 2 && k == 0 { 1.0 / f32::sqrt(2.0) } else { 1.0 } * 2.0 / S.shape()[0] as f32;
        }
    }
    if norm == Some("ortho") {
        mfcc *= f32::sqrt(2.0 / S.shape()[0] as f32);
    }
    mfcc
}

/// Computes MFCCs and their first and second order deltas.
///
/// # Arguments
/// * `mfcc` - Input MFCC matrix
/// * `width` - Optional window width for delta computation (defaults to 9)
/// * `axis` - Optional axis along which to compute deltas (-1 for time, 0 for frequency; defaults to -1)
///
/// # Returns
/// Returns a tuple `(mfcc, delta, delta2)` where:
/// - `mfcc` is the original MFCC matrix
/// - `delta` is the first-order delta coefficients
/// - `delta2` is the second-order delta coefficients
///
/// # Panics
/// Panics if `width` is not a positive odd integer.
///
/// # Examples
/// ```
/// use ndarray::Array2;
/// let mfcc = Array2::from_shape_vec((4, 3), vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]).unwrap();
/// let (mfcc, delta, delta2) = deltas(&mfcc, None, None);
/// ```
pub fn deltas(
    mfcc: &Array2<f32>,
    width: Option<usize>,
    axis: Option<isize>,
) -> (Array2<f32>, Array2<f32>, Array2<f32>) {
    let width = width.unwrap_or(9);
    let axis = axis.unwrap_or(-1);
    let ax = if axis < 0 { 1 } else { 0 };

    assert!(width > 0 && width % 2 == 1, "Width must be a positive odd integer");

    let mut delta = Array2::zeros(mfcc.dim());
    let half_width = width / 2;
    let weights: Vec<f32> = (1..=half_width).map(|i| i as f32).collect();
    let norm = weights.iter().map(|x| x.powi(2)).sum::<f32>() * 2.0;

    for i in 0..mfcc.shape()[ax] {
        let slice = mfcc.index_axis(Axis(ax), i);
        for j in 0..slice.len() {
            let mut sum = 0.0;
            for (w_idx, &w) in weights.iter().enumerate() {
                let w = w_idx + 1;
                let left_idx = (j as isize - w as isize).max(0) as usize;
                let right_idx = (j + w).min(slice.len() - 1);
                sum += w as f32 * (slice[right_idx] - slice[left_idx]);
            }
            delta[[if ax == 1 { j } else { i }, if ax == 1 { i } else { j }]] = sum / norm;
        }
    }

    let mut delta2 = Array2::zeros(mfcc.dim());
    for i in 0..delta.shape()[ax] {
        let slice = delta.index_axis(Axis(ax), i);
        for j in 0..slice.len() {
            let mut sum = 0.0;
            for (w_idx, &w) in weights.iter().enumerate() {
                let w = w_idx + 1;
                let left_idx = (j as isize - w as isize).max(0) as usize;
                let right_idx = (j + w).min(slice.len() - 1);
                sum += w as f32 * (slice[right_idx] - slice[left_idx]);
            }
            delta2[[if ax == 1 { j } else { i }, if ax == 1 { i } else { j }]] = sum / norm;
        }
    }

    (mfcc.to_owned(), delta, delta2)
}

/// Computes first and second order delta coefficients from MFCCs.
///
/// # Arguments
/// * `mfcc` - Input MFCC matrix
/// * `width` - Optional window width for delta computation (defaults to 9)
/// * `axis` - Optional axis along which to compute deltas (-1 for time, 0 for frequency; defaults to -1)
///
/// # Returns
/// Returns a tuple `(delta, delta2)` containing first and second-order delta coefficients.
///
/// # Panics
/// Panics if `width` is not a positive odd integer.
///
/// # Examples
/// ```
/// use ndarray::Array2;
/// let mfcc = Array2::from_shape_vec((4, 3), vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]).unwrap();
/// let (delta, delta2) = mfcc_deltas(&mfcc, None, None);
/// ```
pub fn mfcc_deltas(
    mfcc: &Array2<f32>,
    width: Option<usize>,
    axis: Option<isize>,
) -> (Array2<f32>, Array2<f32>) {
    let width = width.unwrap_or(9);
    let axis = axis.unwrap_or(-1);
    let ax = if axis < 0 { 1 } else { 0 };

    assert!(width > 0 && width % 2 == 1, "Width must be a positive odd integer");

    let mut delta = Array2::zeros(mfcc.dim());
    let half_width = width / 2;
    let weights: Vec<f32> = (1..=half_width).map(|i| i as f32).collect();
    let norm = weights.iter().map(|x| x.powi(2)).sum::<f32>() * 2.0;

    for i in 0..mfcc.shape()[ax] {
        let slice = mfcc.index_axis(Axis(ax), i);
        for j in 0..slice.len() {
            let mut sum = 0.0;
            for (w_idx, &w) in weights.iter().enumerate() {
                let w = w_idx + 1;
                let left_idx = (j as isize - w as isize).max(0) as usize;
                let right_idx = (j + w).min(slice.len() - 1);
                sum += w as f32 * (slice[right_idx] - slice[left_idx]);
            }
            delta[[if ax == 1 { j } else { i }, if ax == 1 { i } else { j }]] = sum / norm;
        }
    }

    let mut delta2 = Array2::zeros(mfcc.dim());
    for i in 0..delta.shape()[ax] {
        let slice = delta.index_axis(Axis(ax), i);
        for j in 0..slice.len() {
            let mut sum = 0.0;
            for (w_idx, &w) in weights.iter().enumerate() {
                let w = w_idx + 1;
                let left_idx = (j as isize - w as isize).max(0) as usize;
                let right_idx = (j + w).min(slice.len() - 1);
                sum += w as f32 * (slice[right_idx] - slice[left_idx]);
            }
            delta2[[if ax == 1 { j } else { i }, if ax == 1 { i } else { j }]] = sum / norm;
        }
    }

    (delta, delta2)
}

/// Converts mel spectrogram to STFT magnitude spectrogram.
///
/// # Arguments
/// * `M` - Input mel spectrogram
/// * `sr` - Optional sample rate (defaults to 44100)
/// * `n_fft` - Optional FFT size (defaults to 2048)
/// * `power` - Optional power of input spectrogram (defaults to 2.0)
///
/// # Returns
/// Returns a 2D array of shape `(n_fft/2 + 1, n_frames)` containing the STFT magnitude spectrogram.
///
/// # Examples
/// ```
/// use ndarray::Array2;
/// let M = Array2::from_shape_vec((128, 10), vec![0.1; 128 * 10]).unwrap();
/// let S = mel_to_stft(&M, None, None, None);
/// ```
pub fn mel_to_stft(
    M: &Array2<f32>,
    sr: Option<u32>,
    n_fft: Option<usize>,
    power: Option<f32>,
) -> Array2<f32> {
    let sr = sr.unwrap_or(44100);
    let n_fft = n_fft.unwrap_or(2048);
    let power = power.unwrap_or(2.0);
    let mel_f = crate::frequencies::mel_frequencies(Some(M.shape()[0]), None, Some(sr as f32 / 2.0), None);
    let fft_f = crate::frequencies::fft_frequencies(Some(sr), Some(n_fft));
    let mut S = Array2::zeros((n_fft / 2 + 1, M.shape()[1]));
    for m in 0..M.shape()[0] {
        let f_low = if m == 0 { 0.0 } else { mel_f[m - 1] };
        let f_center = mel_f[m];
        let f_high = mel_f.get(m + 1).copied().unwrap_or(sr as f32 / 2.0);
        for (bin, &f) in fft_f.iter().enumerate() {
            let weight = if f >= f_low && f <= f_high {
                if f <= f_center { (f - f_low) / (f_center - f_low) } else { (f_high - f) / (f_high - f_center) }
            } else {
                0.0
            };
            for t in 0..M.shape()[1] {
                S[[bin, t]] += M[[m, t]] * weight.max(0.0);
            }
        }
    }
    S.mapv(|x| x.powf(1.0 / power))
}

/// Converts mel spectrogram to audio waveform.
///
/// # Arguments
/// * `M` - Input mel spectrogram
/// * `sr` - Optional sample rate (defaults to 44100)
/// * `n_fft` - Optional FFT size (defaults to 2048)
/// * `hop_length` - Optional hop length (defaults to n_fft/4)
///
/// # Returns
/// Returns a vector containing the reconstructed audio waveform.
///
/// # Examples
/// ```
/// use ndarray::Array2;
/// let M = Array2::from_shape_vec((128, 10), vec![0.1; 128 * 10]).unwrap();
/// let audio = mel_to_audio(&M, None, None, None);
/// ```
pub fn mel_to_audio(
    M: &Array2<f32>,
    sr: Option<u32>,
    n_fft: Option<usize>,
    hop_length: Option<usize>,
) -> Vec<f32> {
    let n_fft = n_fft.unwrap_or(2048);
    let hop = hop_length.unwrap_or(n_fft / 4);
    let S = mel_to_stft(M, sr, Some(n_fft), None);
    griffinlim(&S, None, Some(hop))
}

/// Converts MFCCs back to mel spectrogram.
///
/// # Arguments
/// * `mfcc` - Input MFCC matrix
/// * `n_mels` - Optional number of mel bins (defaults to 128)
/// * `dct_type` - Optional DCT type (defaults to 2)
///
/// # Returns
/// Returns a 2D array of shape `(n_mels, n_frames)` containing the mel spectrogram.
///
/// # Examples
/// ```
/// use ndarray::Array2;
/// let mfcc = Array2::from_shape_vec((20, 10), vec![0.1; 20 * 10]).unwrap();
/// let mel = mfcc_to_mel(&mfcc, None, None);
/// ```
pub fn mfcc_to_mel(
    mfcc: &Array2<f32>,
    n_mels: Option<usize>,
    dct_type: Option<i32>,
) -> Array2<f32> {
    let n_mels = n_mels.unwrap_or(128);
    let dct_type = dct_type.unwrap_or(2);
    let mut mel = Array2::zeros((n_mels, mfcc.shape()[1]));
    for t in 0..mfcc.shape()[1] {
        for n in 0..n_mels {
            let mut sum = 0.0;
            for k in 0..mfcc.shape()[0] {
                sum += mfcc[[k, t]] * (std::f32::consts::PI * k as f32 * (n as f32 + 0.5) / n_mels as f32).cos();
            }
            mel[[n, t]] = sum * if dct_type == 2 && n == 0 { f32::sqrt(2.0) } else { 1.0 } * n_mels as f32 / 2.0;
        }
    }
    mel.mapv(f32::exp)
}

/// Converts MFCCs to audio waveform.
///
/// # Arguments
/// * `mfcc` - Input MFCC matrix
/// * `n_mels` - Optional number of mel bins (defaults to 128)
/// * `sr` - Optional sample rate (defaults to 44100)
/// * `n_fft` - Optional FFT size (defaults to 2048)
/// * `hop_length` - Optional hop length (defaults to n_fft/4)
///
/// # Returns
/// Returns a vector containing the reconstructed audio waveform.
///
/// # Examples
/// ```
/// use ndarray::Array2;
/// let mfcc = Array2::from_shape_vec((20, 10), vec![0.1; 20 * 10]).unwrap();
/// let audio = mfcc_to_audio(&mfcc, None, None, None, None);
/// ```
pub fn mfcc_to_audio(
    mfcc: &Array2<f32>,
    n_mels: Option<usize>,
    sr: Option<u32>,
    n_fft: Option<usize>,
    hop_length: Option<usize>,
) -> Vec<f32> {
    let mel = mfcc_to_mel(mfcc, n_mels, None);
    mel_to_audio(&mel, sr, n_fft, hop_length)
}