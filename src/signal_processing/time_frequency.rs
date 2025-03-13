use rustfft::FftPlanner;
use num_complex::Complex;
use ndarray::{Array1, Array2, s};
use crate::{AudioError, utils::frequency::fft_frequencies};
use std::f32::consts::{PI, SQRT_2};

/// Computes the Short-Time Fourier Transform (STFT) of a signal.
///
/// # Arguments
/// * `y` - Input signal as a slice of `f32`
/// * `n_fft` - Optional FFT window size (defaults to 2048)
/// * `hop_length` - Optional hop length in samples (defaults to n_fft/4, minimum 1)
/// * `win_length` - Optional window length in samples (defaults to n_fft)
///
/// # Returns
/// Returns a `Result` containing an `Array2<Complex<f32>>` representing the STFT spectrogram,
/// with shape `(n_fft/2 + 1, n_frames)`, or an `AudioError` if array shaping fails.
///
/// # Examples
/// ```
/// let signal = vec![1.0, 2.0, 3.0, 4.0];
/// let spectrogram = stft(&signal, None, None, None).unwrap();
/// ```
pub fn stft(
    y: &[f32],
    n_fft: Option<usize>,
    hop_length: Option<usize>,
    win_length: Option<usize>,
) -> Result<Array2<Complex<f32>>, AudioError> {
    let n = n_fft.unwrap_or(2048);
    let hop = hop_length.unwrap_or(n / 4).max(1);
    let win = win_length.unwrap_or(n);
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);
    let mut buffer = vec![Complex::new(0.0, 0.0); n];
    let mut spectrogram = Vec::new();

    if y.len() < n {
        let mut padded = vec![0.0; n];
        padded[..y.len()].copy_from_slice(y);
        buffer[..n].copy_from_slice(&padded.iter().map(|&x| Complex::new(x * hamming(0, win), 0.0)).collect::<Vec<_>>());
        fft.process(&mut buffer);
        spectrogram.push(buffer.clone());
    } else {
        for i in (0..y.len()).step_by(hop) {
            let end = std::cmp::min(i + n, y.len());
            buffer.fill(Complex::new(0.0, 0.0));
            for (j, &sample) in y[i..end].iter().enumerate() {
                buffer[j] = Complex::new(sample * hamming(j, win), 0.0);
            }
            fft.process(&mut buffer);
            spectrogram.push(buffer.clone());
        }
    }

    let n_frames = spectrogram.len();
    Ok(Array2::from_shape_vec((n / 2 + 1, n_frames), spectrogram.into_iter().flat_map(|v| v.into_iter().take(n / 2 + 1)).collect())?)
}

/// Computes the inverse Short-Time Fourier Transform (iSTFT) to reconstruct a signal.
///
/// # Arguments
/// * `stft_matrix` - STFT spectrogram as an `Array2<Complex<f32>>`
/// * `hop_length` - Optional hop length in samples (defaults to n_fft/4, minimum 1)
/// * `win_length` - Optional window length in samples (defaults to n_fft)
/// * `length` - Optional output signal length in samples (defaults to maximum possible length)
///
/// # Returns
/// Returns a `Vec<f32>` containing the reconstructed time-domain signal.
///
/// # Examples
/// ```
/// use ndarray::arr2;
/// let stft_data = arr2(&[[Complex::new(1.0, 0.0)], [Complex::new(0.5, 0.0)]]);
/// let signal = istft(&stft_data, None, None, None);
/// ```
pub fn istft(
    stft_matrix: &Array2<Complex<f32>>,
    hop_length: Option<usize>,
    win_length: Option<usize>,
    length: Option<usize>,
) -> Vec<f32> {
    let n_fft = (stft_matrix.shape()[0] - 1) * 2;
    let hop = hop_length.unwrap_or(n_fft / 4).max(1);
    let win = win_length.unwrap_or(n_fft);
    let n_frames = stft_matrix.shape()[1];
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_inverse(n_fft);

    let max_len = hop * (n_frames - 1) + n_fft;
    let target_len = length.unwrap_or(max_len);
    let mut signal = vec![0.0; max_len];
    let mut window_sum = vec![0.0; max_len];
    let window = hamming_vec(win);

    for (frame_idx, frame) in stft_matrix.axis_iter(ndarray::Axis(1)).enumerate() {
        let mut buffer: Vec<Complex<f32>> = frame.to_vec();
        buffer.extend(vec![Complex::new(0.0, 0.0); n_fft - buffer.len()]);
        fft.process(&mut buffer);
        let start = frame_idx * hop;
        for (i, &val) in buffer.iter().enumerate().take(win) {
            if start + i < signal.len() {
                signal[start + i] += val.re * window[i];
                window_sum[start + i] += window[i];
            }
        }
    }

    for (i, &sum) in window_sum.iter().enumerate() {
        if sum > 1e-6 {
            signal[i] /= sum;
        }
    }

    signal.resize(target_len, 0.0);
    signal
}

/// Computes the Hamming window value at a given sample index.
///
/// # Arguments
/// * `n` - Sample index
/// * `win_length` - Total window length
///
/// # Returns
/// Returns a `f32` representing the Hamming window coefficient.
///
/// # Examples
/// ```
/// let value = hamming(0, 10);
/// assert!(value > 0.0 && value <= 1.0);
/// ```
fn hamming(n: usize, win_length: usize) -> f32 {
    0.54 - 0.46 * (2.0 * std::f32::consts::PI * n as f32 / (win_length - 1) as f32).cos()
}

/// Generates a Hamming window vector.
///
/// # Arguments
/// * `win_length` - Length of the window
///
/// # Returns
/// Returns a `Vec<f32>` containing the Hamming window coefficients.
///
/// # Examples
/// ```
/// let window = hamming_vec(5);
/// assert_eq!(window.len(), 5);
/// ```
fn hamming_vec(win_length: usize) -> Vec<f32> {
    (0..win_length).map(|n| hamming(n, win_length)).collect()
}

/// Separates magnitude and phase from a complex spectrogram.
///
/// # Arguments
/// * `D` - Input spectrogram as an `Array2<Complex<f32>>`
/// * `power` - Optional power to raise the magnitude (defaults to 1.0)
///
/// # Returns
/// Returns a tuple `(magnitude, phase)` where:
/// - `magnitude` is an `Array2<f32>` of magnitude values
/// - `phase` is an `Array2<Complex<f32>>` of unit-magnitude phase values
///
/// # Examples
/// ```
/// use ndarray::arr2;
/// let spectrogram = arr2(&[[Complex::new(3.0, 4.0)]]);
/// let (mag, phase) = magphase(&spectrogram, None);
/// assert_eq!(mag[[0, 0]], 5.0); // sqrt(3^2 + 4^2)
/// ```
pub fn magphase(D: &Array2<Complex<f32>>, power: Option<f32>) -> (Array2<f32>, Array2<Complex<f32>>) {
    let power_val = power.unwrap_or(1.0);
    let magnitude = D.mapv(|x| x.norm().powf(power_val));
    let phase = D.mapv(|x| x / x.norm());
    (magnitude, phase)
}

/// Computes a reassigned spectrogram for improved time-frequency resolution.
///
/// # Arguments
/// * `y` - Input signal as a slice of `f32`
/// * `sr` - Optional sample rate in Hz (defaults to 44100)
/// * `n_fft` - Optional FFT window size (defaults to 2048)
///
/// # Returns
/// Returns a `Result` containing an `Array2<f32>` representing the reassigned spectrogram,
/// or an `AudioError` if computation fails.
///
/// # Errors
/// * `AudioError::InsufficientData` - If signal length is less than `n_fft`.
/// * `AudioError::ComputationFailed` - If STFT computation fails.
///
/// # Examples
/// ```
/// let signal = vec![1.0; 4096];
/// let reassigned = reassigned_spectrogram(&signal, None, None).unwrap();
/// ```
pub fn reassigned_spectrogram(
    y: &[f32],
    sr: Option<u32>,
    n_fft: Option<usize>,
) -> Result<Array2<f32>, AudioError> {
    let sr = sr.unwrap_or(44100);
    let n_fft = n_fft.unwrap_or(2048);
    let hop_length = n_fft / 4;

    if y.len() < n_fft {
        return Err(AudioError::InsufficientData(format!("Signal too short: {} < {}", y.len(), n_fft)));
    }

    let S = stft(y, Some(n_fft), Some(hop_length), None)
        .map_err(|e| AudioError::ComputationFailed(format!("STFT failed: {}", e)))?;
    let S_time = stft_with_derivative(y, Some(n_fft), Some(hop_length), true)?;
    let S_freq = stft_with_derivative(y, Some(n_fft), Some(hop_length), false)?;

    let mut reassigned = Array2::zeros(S.dim());
    let freqs = fft_frequencies(Some(sr), Some(n_fft));
    let times = Array1::linspace(0.0, (y.len() as f32 - 1.0) / sr as f32, S.shape()[1]);

    for t in 0..S.shape()[1] {
        for f in 0..S.shape()[0] {
            let mag = S[[f, t]].norm();
            if mag > 1e-6 {
                let dphi_dt = S_time[[f, t]].im / mag;
                let t_reassigned = times[t] - dphi_dt * hop_length as f32 / sr as f32;
                let dphi_df = S_freq[[f, t]].im / mag;
                let f_reassigned = freqs[f] + dphi_df * sr as f32 / n_fft as f32;

                let t_idx = ((t_reassigned * sr as f32 / hop_length as f32).round() as usize).min(S.shape()[1] - 1);
                let f_idx = freqs.iter().position(|&x| x >= f_reassigned).unwrap_or(f).min(S.shape()[0] - 1);
                reassigned[[f_idx, t_idx]] += mag;
            }
        }
    }

    Ok(reassigned)
}

/// Computes the Constant-Q Transform (CQT) of a signal.
///
/// # Arguments
/// * `y` - Input signal as a slice of `f32`
/// * `sr` - Optional sample rate in Hz (defaults to 44100)
/// * `hop_length` - Optional hop length in samples (defaults to 512)
/// * `fmin` - Optional minimum frequency in Hz (defaults to 32.70, C1)
/// * `n_bins` - Optional number of frequency bins (defaults to 84)
///
/// # Returns
/// Returns a `Result` containing an `Array2<Complex<f32>>` representing the CQT spectrogram,
/// or an `AudioError` if computation fails.
///
/// # Errors
/// * `AudioError::InsufficientData` - If signal length is less than `hop_length`.
/// * `AudioError::InvalidInput` - If `fmin` is not positive.
/// * `AudioError::ComputationFailed` - If STFT computation fails.
///
/// # Examples
/// ```
/// let signal = vec![1.0; 1024];
/// let cqt_result = cqt(&signal, None, None, None, None).unwrap();
/// ```
pub fn cqt(
    y: &[f32],
    sr: Option<u32>,
    hop_length: Option<usize>,
    fmin: Option<f32>,
    n_bins: Option<usize>,
) -> Result<Array2<Complex<f32>>, AudioError> {
    let sr = sr.unwrap_or(44100);
    let hop_length = hop_length.unwrap_or(512);
    let fmin = fmin.unwrap_or(32.70);
    let n_bins = n_bins.unwrap_or(84);
    let bins_per_octave = 12;

    if y.len() < hop_length {
        return Err(AudioError::InsufficientData(format!("Signal too short: {} < {}", y.len(), hop_length)));
    }
    if fmin <= 0.0 {
        return Err(AudioError::InvalidInput("fmin must be positive".to_string()));
    }

    let n_fft = ((sr as f32 / fmin * 2.0) as u32).next_power_of_two() as usize;
    let S_stft = stft(y, Some(n_fft), Some(hop_length), None)
        .map_err(|e| AudioError::ComputationFailed(format!("STFT failed: {}", e)))?;
    let n_frames = S_stft.shape()[1];
    let mut S_cqt = Array2::zeros((n_bins, n_frames));

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n_fft);
    for k in 0..n_bins {
        let fk = fmin * 2.0f32.powf(k as f32 / bins_per_octave as f32);
        let n = (sr as f32 / fk).round() as usize;
        let mut kernel = Array1::zeros(n_fft);
        let window = hann_window(n);
        for i in 0..n {
            let phase = 2.0 * PI * fk * i as f32 / sr as f32;
            kernel[i] = Complex::new(window[i] * phase.cos(), window[i] * phase.sin()) / n as f32;
        }
        fft.process(&mut kernel.to_vec());

        for t in 0..n_frames {
            let stft_frame = S_stft.slice(s![.., t]);
            S_cqt[[k, t]] = stft_frame.iter().zip(kernel.iter()).map(|(&s, &k)| s * k.conj()).sum();
        }
    }

    Ok(S_cqt)
}

/// Computes the inverse Constant-Q Transform (iCQT) to reconstruct a signal.
///
/// # Arguments
/// * `C` - CQT spectrogram as an `Array2<Complex<f32>>`
/// * `sr` - Optional sample rate in Hz (defaults to 44100)
/// * `hop_length` - Optional hop length in samples (defaults to 512)
/// * `fmin` - Optional minimum frequency in Hz (defaults to 32.70, C1)
///
/// # Returns
/// Returns a `Result` containing a `Vec<f32>` of the reconstructed signal,
/// or an `AudioError` if computation fails.
///
/// # Errors
/// * `AudioError::InvalidInput` - If `fmin` is not positive.
///
/// # Examples
/// ```
/// use ndarray::arr2;
/// let cqt_data = arr2(&[[Complex::new(1.0, 0.0)]]);
/// let signal = icqt(&cqt_data, None, None, None).unwrap();
/// ```
pub fn icqt(
    C: &Array2<Complex<f32>>,
    sr: Option<u32>,
    hop_length: Option<usize>,
    fmin: Option<f32>,
) -> Result<Vec<f32>, AudioError> {
    let sr = sr.unwrap_or(44100);
    let hop_length = hop_length.unwrap_or(512);
    let fmin = fmin.unwrap_or(32.70);
    let n_bins = C.shape()[0];
    let n_frames = C.shape()[1];
    let bins_per_octave = 12;

    if fmin <= 0.0 {
        return Err(AudioError::InvalidInput("fmin must be positive".to_string()));
    }

    let n_fft = ((sr as f32 / fmin * 2.0) as u32).next_power_of_two() as usize;
    let n_samples = n_frames * hop_length;
    let mut y = vec![0.0; n_samples];
    let mut planner = FftPlanner::new();
    let ifft = planner.plan_fft_inverse(n_fft);

    for k in 0..n_bins {
        let fk = fmin * 2.0f32.powf(k as f32 / bins_per_octave as f32);
        let n = (sr as f32 / fk).round() as usize;
        let window = hann_window(n);
        let mut kernel = Array1::zeros(n_fft);
        for i in 0..n {
            let phase = 2.0 * PI * fk * i as f32 / sr as f32;
            kernel[i] = Complex::new(window[i] * phase.cos(), window[i] * phase.sin()) / n as f32;
        }
        ifft.process(&mut kernel.to_vec());

        for t in 0..n_frames {
            let mut frame = vec![Complex::new(C[[k, t]].re, C[[k, t]].im) * Complex::conj(&kernel[0]); n_fft];
            ifft.process(&mut frame);
            let start = t * hop_length;
            for i in 0..n.min(n_samples - start) {
                y[start + i] += frame[i].re * window[i];
            }
        }
    }

    let mut overlap = vec![0.0; n_samples];
    for t in 0..n_frames {
        let start = t * hop_length;
        for i in 0..n_fft.min(n_samples - start) {
            overlap[start + i] += hann_window(n_fft)[i].powi(2);
        }
    }
    for i in 0..n_samples {
        if overlap[i] > 1e-6 {
            y[i] /= overlap[i];
        }
    }

    Ok(y)
}

/// Computes a hybrid Constant-Q Transform (CQT) combining STFT and CQT properties.
///
/// # Arguments
/// * `y` - Input signal as a slice of `f32`
/// * `sr` - Optional sample rate in Hz (defaults to 44100)
/// * `hop_length` - Optional hop length in samples (defaults to 512)
/// * `fmin` - Optional minimum frequency in Hz (defaults to 32.70, C1)
///
/// # Returns
/// Returns a `Result` containing an `Array2<Complex<f32>>` representing the hybrid CQT,
/// or an `AudioError` if computation fails.
///
/// # Errors
/// * `AudioError::InsufficientData` - If signal length is less than `n_fft`.
/// * `AudioError::InvalidInput` - If `fmin` is not positive.
/// * `AudioError::ComputationFailed` - If STFT computation fails.
///
/// # Examples
/// ```
/// let signal = vec![1.0; 1024];
/// let hybrid = hybrid_cqt(&signal, None, None, None).unwrap();
/// ```
pub fn hybrid_cqt(
    y: &[f32],
    sr: Option<u32>,
    hop_length: Option<usize>,
    fmin: Option<f32>,
) -> Result<Array2<Complex<f32>>, AudioError> {
    let sr = sr.unwrap_or(44100);
    let hop_length = hop_length.unwrap_or(512);
    let fmin = fmin.unwrap_or(32.70);
    let n_fft = ((sr as f32 / fmin * 2.0) as u32).next_power_of_two() as usize;
    let n_bins = 84;

    if y.len() < n_fft {
        return Err(AudioError::InsufficientData(format!("Signal too short: {} < {}", y.len(), n_fft)));
    }
    if fmin <= 0.0 {
        return Err(AudioError::InvalidInput("fmin must be positive".to_string()));
    }

    let S_stft = stft(y, Some(n_fft), Some(hop_length), None)
        .map_err(|e| AudioError::ComputationFailed(format!("STFT failed: {}", e)))?;
    let mut S_hybrid = Array2::zeros((n_bins, S_stft.shape()[1]));
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n_fft);

    for k in 0..n_bins {
        let fk = fmin * 2.0f32.powf(k as f32 / 12.0);
        let n = (sr as f32 / fk).round() as usize;
        let mut kernel = Array1::zeros(n_fft);
        let window = hann_window(n);
        for i in 0..n {
            let phase = 2.0 * PI * fk * i as f32 / sr as f32;
            kernel[i] = Complex::new(window[i] * phase.cos(), window[i] * phase.sin()) / n as f32;
        }
        fft.process(&mut kernel.to_vec());

        for t in 0..S_stft.shape()[1] {
            S_hybrid[[k, t]] = S_stft.slice(s![.., t]).iter().zip(kernel.iter()).map(|(&s, &k)| s * k.conj()).sum();
        }
    }

    Ok(S_hybrid)
}

/// Computes a pseudo Constant-Q Transform (CQT) using STFT bin mapping.
///
/// # Arguments
/// * `y` - Input signal as a slice of `f32`
/// * `sr` - Optional sample rate in Hz (defaults to 44100)
/// * `hop_length` - Optional hop length in samples (defaults to 512)
/// * `fmin` - Optional minimum frequency in Hz (defaults to 32.70, C1)
///
/// # Returns
/// Returns a `Result` containing an `Array2<Complex<f32>>` representing the pseudo CQT,
/// or an `AudioError` if computation fails.
///
/// # Errors
/// * `AudioError::InsufficientData` - If signal length is less than `n_fft`.
/// * `AudioError::InvalidInput` - If `fmin` is not positive.
/// * `AudioError::ComputationFailed` - If STFT computation fails.
///
/// # Examples
/// ```
/// let signal = vec![1.0; 1024];
/// let pseudo = pseudo_cqt(&signal, None, None, None).unwrap();
/// ```
pub fn pseudo_cqt(
    y: &[f32],
    sr: Option<u32>,
    hop_length: Option<usize>,
    fmin: Option<f32>,
) -> Result<Array2<Complex<f32>>, AudioError> {
    let sr = sr.unwrap_or(44100);
    let hop_length = hop_length.unwrap_or(512);
    let fmin = fmin.unwrap_or(32.70);
    let n_fft = ((sr as f32 / fmin * 2.0) as u32).next_power_of_two() as usize;
    let n_bins = 84;

    if y.len() < n_fft {
        return Err(AudioError::InsufficientData(format!("Signal too short: {} < {}", y.len(), n_fft)));
    }
    if fmin <= 0.0 {
        return Err(AudioError::InvalidInput("fmin must be positive".to_string()));
    }

    let S_stft = stft(y, Some(n_fft), Some(hop_length), None)
        .map_err(|e| AudioError::ComputationFailed(format!("STFT failed: {}", e)))?;
    let mut S_pseudo = Array2::zeros((n_bins, S_stft.shape()[1]));
    let freqs = fft_frequencies(Some(sr), Some(n_fft));

    for t in 0..S_stft.shape()[1] {
        for k in 0..n_bins {
            let fk = fmin * 2.0f32.powf(k as f32 / 12.0);
            let idx = freqs.iter().position(|&f| f >= fk).unwrap_or(0);
            S_pseudo[[k, t]] = S_stft[[idx.min(S_stft.shape()[0] - 1), t]];
        }
    }

    Ok(S_pseudo)
}

/// Computes the Variable-Q Transform (VQT) of a signal.
///
/// # Arguments
/// * `y` - Input signal as a slice of `f32`
/// * `sr` - Optional sample rate in Hz (defaults to 44100)
/// * `hop_length` - Optional hop length in samples (defaults to 512)
/// * `fmin` - Optional minimum frequency in Hz (defaults to 32.70, C1)
/// * `n_bins` - Optional number of frequency bins (defaults to 84)
///
/// # Returns
/// Returns a `Result` containing an `Array2<Complex<f32>>` representing the VQT,
/// or an `AudioError` if computation fails.
///
/// # Errors
/// * `AudioError::InsufficientData` - If signal length is less than `hop_length`.
/// * `AudioError::InvalidInput` - If `fmin` is not positive.
/// * `AudioError::ComputationFailed` - If STFT computation fails.
///
/// # Examples
/// ```
/// let signal = vec![1.0; 1024];
/// let vqt_result = vqt(&signal, None, None, None, None).unwrap();
/// ```
pub fn vqt(
    y: &[f32],
    sr: Option<u32>,
    hop_length: Option<usize>,
    fmin: Option<f32>,
    n_bins: Option<usize>,
) -> Result<Array2<Complex<f32>>, AudioError> {
    let sr = sr.unwrap_or(44100);
    let hop_length = hop_length.unwrap_or(512);
    let fmin = fmin.unwrap_or(32.70);
    let n_bins = n_bins.unwrap_or(84);
    let gamma = 24.0;

    if y.len() < hop_length {
        return Err(AudioError::InsufficientData(format!("Signal too short: {} < {}", y.len(), hop_length)));
    }
    if fmin <= 0.0 {
        return Err(AudioError::InvalidInput("fmin must be positive".to_string()));
    }

    let n_fft = ((sr as f32 / fmin * 2.0) as u32).next_power_of_two() as usize;
    let S_stft = stft(y, Some(n_fft), Some(hop_length), None)
        .map_err(|e| AudioError::ComputationFailed(format!("STFT failed: {}", e)))?;
    let mut S_vqt = Array2::zeros((n_bins, S_stft.shape()[1]));
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n_fft);

    for k in 0..n_bins {
        let fk = fmin * 2.0f32.powf(k as f32 / 12.0);
        let q = gamma / (2.0f32.powf(1.0 / 12.0) - 1.0);
        let n = (sr as f32 * q / fk).round() as usize;
        let mut kernel = Array1::zeros(n_fft);
        let window = hann_window(n);
        for i in 0..n {
            let phase = 2.0 * PI * fk * i as f32 / sr as f32;
            kernel[i] = Complex::new(window[i] * phase.cos(), window[i] * phase.sin()) / n as f32;
        }
        fft.process(&mut kernel.to_vec());

        for t in 0..S_stft.shape()[1] {
            S_vqt[[k, t]] = S_stft.slice(s![.., t]).iter().zip(kernel.iter()).map(|(&s, &k)| s * k.conj()).sum();
        }
    }

    Ok(S_vqt)
}

/// Computes the Fourier Modulation Transform (FMT) of a signal.
///
/// # Arguments
/// * `y` - Input signal as a slice of `f32`
/// * `t_min` - Optional minimum time period in seconds (defaults to 0.005)
/// * `n_fmt` - Optional number of modulation frequencies (defaults to 5)
/// * `kind` - Optional transform kind ("cos" or others, defaults to "cos")
/// * `beta` - Optional power for magnitude scaling (defaults to 2.0)
///
/// # Returns
/// Returns a `Result` containing an `Array2<f32>` representing the FMT spectrogram,
/// or an `AudioError` if computation fails.
///
/// # Errors
/// * `AudioError::InsufficientData` - If signal length is less than `hop_length`.
/// * `AudioError::InvalidInput` - If `t_min` is not positive.
///
/// # Examples
/// ```
/// let signal = vec![1.0; 1024];
/// let fmt_result = fmt(&signal, None, None, None, None).unwrap();
/// ```
pub fn fmt(
    y: &[f32],
    t_min: Option<f32>,
    n_fmt: Option<usize>,
    kind: Option<&str>,
    beta: Option<f32>,
) -> Result<Array2<f32>, AudioError> {
    let sr = 44100;
    let t_min = t_min.unwrap_or(0.005);
    let n_fmt = n_fmt.unwrap_or(5);
    let kind = kind.unwrap_or("cos");
    let beta = beta.unwrap_or(2.0);
    let hop_length = (sr as f32 * t_min).round() as usize;

    if y.len() < hop_length {
        return Err(AudioError::InsufficientData(format!("Signal too short: {} < {}", y.len(), hop_length)));
    }
    if t_min <= 0.0 {
        return Err(AudioError::InvalidInput("t_min must be positive".to_string()));
    }

    let n_frames = (y.len() - hop_length) / hop_length + 1;
    let mut S = Array2::zeros((n_fmt, n_frames));
    let window = hann_window(hop_length);

    for t in 0..n_frames {
        let start = t * hop_length;
        let frame = &y[start..(start + hop_length).min(y.len())];
        for k in 0..n_fmt {
            let freq = (k + 1) as f32 / t_min;
            let mut sum_re = 0.0;
            let mut sum_im = 0.0;
            for (i, &sample) in frame.iter().enumerate() {
                let phase = 2.0 * PI * freq * i as f32 / sr as f32;
                let w = window[i];
                sum_re += sample * w * phase.cos();
                sum_im += sample * w * phase.sin();
            }
            let mag = Complex::new(sum_re, sum_im).norm() / hop_length as f32;
            S[[k, t]] = mag.powf(beta);
        }
    }

    Ok(S)
}

/// Generates a Hann window vector.
///
/// # Arguments
/// * `n` - Length of the window
///
/// # Returns
/// Returns a `Vec<f32>` containing the Hann window coefficients.
///
/// # Examples
/// ```
/// let window = hann_window(5);
/// assert_eq!(window.len(), 5);
/// ```
fn hann_window(n: usize) -> Vec<f32> {
    (0..n).map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / (n - 1) as f32).cos())).collect()
}

/// Computes STFT with time or frequency derivative for reassignment.
///
/// # Arguments
/// * `y` - Input signal as a slice of `f32`
/// * `n_fft` - Optional FFT window size (defaults to 2048)
/// * `hop_length` - Optional hop length in samples (defaults to n_fft/4)
/// * `time_derivative` - If true, computes time derivative; if false, frequency derivative
///
/// # Returns
/// Returns a `Result` containing an `Array2<Complex<f32>>` with derivative information,
/// or an `AudioError` if computation fails.
///
/// # Examples
/// ```
/// let signal = vec![1.0; 2048];
/// let deriv = stft_with_derivative(&signal, None, None, true).unwrap();
/// ```
fn stft_with_derivative(
    y: &[f32],
    n_fft: Option<usize>,
    hop_length: Option<usize>,
    time_derivative: bool,
) -> Result<Array2<Complex<f32>>, AudioError> {
    let n_fft = n_fft.unwrap_or(2048);
    let hop_length = hop_length.unwrap_or(n_fft / 4);
    let n_frames = (y.len() - n_fft) / hop_length + 1;
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n_fft);
    let mut S = Array2::zeros((n_fft / 2 + 1, n_frames));
    let window = hann_window(n_fft);
    let deriv_window = if time_derivative {
        (0..n_fft).map(|i| i as f32 * window[i]).collect::<Vec<_>>()
    } else {
        (0..n_fft).map(|i| window[i] * (2.0 * PI * i as f32 / n_fft as f32).sin()).collect::<Vec<_>>()
    };

    for t in 0..n_frames {
        let start = t * hop_length;
        let frame = &y[start..(start + n_fft).min(y.len())];
        let mut buffer = frame.iter().zip(deriv_window.iter()).map(|(&x, &w)| Complex::new(x * w, 0.0)).collect::<Vec<_>>();
        buffer.resize(n_fft, Complex::new(0.0, 0.0));
        fft.process(&mut buffer);
        for f in 0..n_fft / 2 + 1 {
            S[[f, t]] = buffer[f];
        }
    }
    Ok(S)
}

/// Designs a Butterworth bandpass filter.
///
/// # Arguments
/// * `lowcut` - Lower cutoff frequency in Hz
/// * `highcut` - Upper cutoff frequency in Hz
/// * `fs` - Sampling frequency in Hz
/// * `order` - Optional filter order (defaults to 2)
///
/// # Returns
/// Returns a `Result` containing a tuple `(b, a)` of numerator and denominator coefficients,
/// or an `AudioError` if frequencies are invalid.
///
/// # Errors
/// * `AudioError::InvalidInput` - If `lowcut` <= 0, `highcut` <= `lowcut`, or `highcut` >= `fs/2`.
///
/// # Examples
/// ```
/// let (b, a) = butterworth_bandpass(100.0, 1000.0, 44100.0, None).unwrap();
/// ```
fn butterworth_bandpass(lowcut: f32, highcut: f32, fs: f32, order: Option<usize>) -> Result<(Vec<f32>, Vec<f32>), AudioError> {
    if lowcut <= 0.0 || highcut <= lowcut || highcut >= fs / 2.0 {
        return Err(AudioError::InvalidInput(format!(
            "Invalid frequencies: lowcut={} must be > 0, highcut={} must be > lowcut and < fs/2={}",
            lowcut, highcut, fs / 2.0
        )));
    }

    let order = order.unwrap_or(2);
    let n = order as i32;

    let w_low = 2.0 * fs * (lowcut * PI / fs).tan();
    let w_high = 2.0 * fs * (highcut * PI / fs).tan();
    let w0 = (w_high * w_low).sqrt();
    let bw = w_high - w_low;

    let mut poles = Vec::new();
    for k in 0..n {
        let theta = PI * (2.0 * k as f32 + 1.0 + n as f32) / (2.0 * n as f32);
        let real = -bw / 2.0 * theta.sin();
        let imag = w0 * theta.cos();
        poles.push(Complex::new(real, imag));
        poles.push(Complex::new(real, -imag));
    }

    let mut z_poles = Vec::new();
    let fs2 = 2.0 * fs;
    for p in poles {
        let pz = (fs2 + p) / (fs2 - p);
        z_poles.push(pz);
    }

    let mut b = vec![1.0];
    let mut a = vec![1.0];
    for p in z_poles.iter() {
        b = convolve(&b, &[1.0, -p.re]);
        a = convolve(&a, &[1.0, -p.re]);
    }
    for _ in 0..n {
        b = convolve(&b, &[1.0, 0.0]);
    }

    let w_center = 2.0 * PI * (lowcut + highcut) / 2.0 / fs;
    let gain = evaluate_filter(&b, &a, w_center).norm();
    for b_k in b.iter_mut() {
        *b_k /= gain;
    }

    Ok((b, a))
}

/// Convolves two vectors.
///
/// # Arguments
/// * `a` - First input vector
/// * `b` - Second input vector
///
/// # Returns
/// Returns a `Vec<f32>` containing the convolution result.
///
/// # Examples
/// ```
/// let result = convolve(&[1.0, 2.0], &[3.0, 4.0]);
/// assert_eq!(result, vec![3.0, 10.0, 8.0]);
/// ```
fn convolve(a: &[f32], b: &[f32]) -> Vec<f32> {
    let mut result = vec![0.0; a.len() + b.len() - 1];
    for i in 0..a.len() {
        for j in 0..b.len() {
            result[i + j] += a[i] * b[j];
        }
    }
    result
}

/// Evaluates a digital filter's frequency response at a given frequency.
///
/// # Arguments
/// * `b` - Numerator coefficients
/// * `a` - Denominator coefficients
/// * `w` - Frequency in radians/sample
///
/// # Returns
/// Returns a `Complex<f32>` representing the filter's response.
///
/// # Examples
/// ```
/// let response = evaluate_filter(&[1.0], &[1.0, -0.5], 0.1);
/// ```
fn evaluate_filter(b: &[f32], a: &[f32], w: f32) -> Complex<f32> {
    let mut num = Complex::new(0.0, 0.0);
    let mut den = Complex::new(0.0, 0.0);
    for (k, &bk) in b.iter().enumerate() {
        let phase = -w * k as f32;
        num += Complex::new(bk * phase.cos(), bk * phase.sin());
    }
    for (k, &ak) in a.iter().enumerate() {
        let phase = -w * k as f32;
        den += Complex::new(ak * phase.cos(), ak * phase.sin());
    }
    num / den
}

/// Computes the Instantaneous Impulse Response Transform (IIRT) using bandpass filtering.
///
/// # Arguments
/// * `y` - Input signal as a slice of `f32`
/// * `sr` - Optional sample rate in Hz (defaults to 44100)
/// * `win_length` - Optional window length in samples (defaults to 2048)
/// * `hop_length` - Optional hop length in samples (defaults to win_length/4)
///
/// # Returns
/// Returns a `Result` containing an `Array2<f32>` representing the IIRT spectrogram,
/// or an `AudioError` if computation fails.
///
/// # Errors
/// * `AudioError::InsufficientData` - If signal length is less than `win_length`.
/// * `AudioError::InvalidInput` - If bandpass filter frequencies are invalid.
///
/// # Examples
/// ```
/// let signal = vec![1.0; 4096];
/// let iirt_result = iirt(&signal, None, None, None).unwrap();
/// ```
pub fn iirt(
    y: &[f32],
    sr: Option<u32>,
    win_length: Option<usize>,
    hop_length: Option<usize>,
) -> Result<Array2<f32>, AudioError> {
    let sr = sr.unwrap_or(44100);
    let win_length = win_length.unwrap_or(2048);
    let hop_length = hop_length.unwrap_or(win_length / 4);
    let n_bands = 12;

    if y.len() < win_length {
        return Err(AudioError::InsufficientData(format!("Signal too short: {} < {}", y.len(), win_length)));
    }

    let n_frames = (y.len() - win_length) / hop_length + 1;
    let mut S = Array2::zeros((n_bands, n_frames));
    let fmin = 32.70;

    for b in 0..n_bands {
        let fc = fmin * 2.0f32.powf(b as f32);
        let bw = fc / SQRT_2;
        let (b_coeffs, a_coeffs) = butterworth_bandpass(fc - bw / 2.0, fc + bw / 2.0, sr as f32, Some(4))?;
        
        for t in 0..n_frames {
            let start = t * hop_length;
            let frame = &y[start..(start + win_length).min(y.len())];
            let filtered = filter(frame, &b_coeffs, &a_coeffs);
            S[[b, t]] = filtered.iter().map(|&x| x.powi(2)).sum::<f32>().sqrt() / win_length as f32;
        }
    }

    Ok(S)
}

/// Applies an IIR filter to a signal.
///
/// # Arguments
/// * `x` - Input signal as a slice of `f32`
/// * `b` - Numerator coefficients
/// * `a` - Denominator coefficients
///
/// # Returns
/// Returns a `Vec<f32>` containing the filtered signal.
///
/// # Examples
/// ```
/// let signal = vec![1.0, 2.0, 3.0];
/// let filtered = filter(&signal, &[1.0, 0.0, 0.0], &[1.0, -0.5, 0.0]);
/// ```
fn filter(x: &[f32], b: &[f32], a: &[f32]) -> Vec<f32> {
    let mut y = vec![0.0; x.len()];
    for n in 0..x.len() {
        y[n] = b[0] * x[n] + b[1] * x.get(n - 1).unwrap_or(&0.0) + b[2] * x.get(n - 2).unwrap_or(&0.0)
            - a[1] * y.get(n - 1).unwrap_or(&0.0) - a[2] * y.get(n - 2).unwrap_or(&0.0);
    }
    y
}