use crate::audio_io::AudioError;
use ndarray::Array1;

/// Computes the autocorrelation of a signal.
///
/// # Arguments
/// * `y` - Input signal as a slice of `f32`
/// * `max_size` - Optional maximum lag size (defaults to signal length)
/// * `axis` - Optional axis parameter (currently unused, included for compatibility)
///
/// # Returns
/// Returns a `Vec<f32>` containing the autocorrelation values for lags from 0 to `max_size - 1`.
///
/// # Examples
/// ```
/// let signal = vec![1.0, 2.0, 3.0];
/// let autocorr = autocorrelate(&signal, Some(2), None);
/// assert_eq!(autocorr, vec![14.0, 8.0]); // [1*1 + 2*2 + 3*3, 1*2 + 2*3]
/// ```
pub fn autocorrelate(y: &[f32], max_size: Option<usize>, axis: Option<isize>) -> Vec<f32> {
    let max_lag = max_size.unwrap_or(y.len());
    let mut result = Vec::with_capacity(max_lag);
    for lag in 0..max_lag {
        let mut sum = 0.0;
        for i in 0..(y.len() - lag) {
            sum += y[i] * y[i + lag];
        }
        result.push(sum);
    }
    result
}

/// Computes Linear Predictive Coding (LPC) coefficients using the autocorrelation method.
///
/// # Arguments
/// * `y` - Input signal as a slice of `f32`
/// * `order` - LPC order (number of coefficients to compute, excluding the leading 1.0)
///
/// # Returns
/// Returns a `Result` containing a `Vec<f32>` of LPC coefficients,
/// or an `AudioError` if the signal length is too short.
///
/// # Errors
/// * `AudioError::InvalidRange` - If `y.len()` is less than or equal to `order`.
///
/// # Examples
/// ```
/// let signal = vec![1.0, 2.0, 3.0, 4.0];
/// let coeffs = lpc(&signal, 2).unwrap();
/// ```
pub fn lpc(y: &[f32], order: usize) -> Result<Vec<f32>, AudioError> {
    if y.len() <= order {
        return Err(AudioError::InvalidRange);
    }
    let r = autocorrelate(y, Some(order + 1), None);
    let mut a = vec![0.0; order + 1];
    a[0] = 1.0;
    let mut e = r[0];

    for i in 1..=order {
        let mut k = 0.0;
        for j in 0..i {
            k += a[j] * r[i - j];
        }
        k = -k / e;
        for j in 0..i {
            a[j] -= k * a[i - 1 - j];
        }
        a[i] = k;
        e *= 1.0 - k * k;
    }
    Ok(a)
}

/// Detects zero crossings in a signal.
///
/// # Arguments
/// * `y` - Input signal as a slice of `f32`
/// * `threshold` - Optional threshold value for zero crossing (defaults to 0.0)
/// * `pad` - Optional flag to pad with a zero crossing at index 0 if none are found (defaults to false)
///
/// # Returns
/// Returns a `Vec<usize>` containing the indices where zero crossings occur.
///
/// # Examples
/// ```
/// let signal = vec![1.0, -1.0, 2.0, -2.0];
/// let crossings = zero_crossings(&signal, None, None);
/// assert_eq!(crossings, vec![1, 3]);
/// ```
pub fn zero_crossings(y: &[f32], threshold: Option<f32>, pad: Option<bool>) -> Vec<usize> {
    let thresh = threshold.unwrap_or(0.0);
    let mut crossings = Vec::new();
    let mut prev_sign = y[0] >= thresh;
    for (i, &sample) in y.iter().enumerate().skip(1) {
        let sign = sample >= thresh;
        if sign != prev_sign {
            crossings.push(i);
        }
        prev_sign = sign;
    }
    if pad.unwrap_or(false) && crossings.is_empty() {
        crossings.push(0);
    }
    crossings
}

/// Applies μ-law compression to a signal.
///
/// # Arguments
/// * `x` - Input signal as a slice of `f32`
/// * `mu` - Optional μ-law parameter (defaults to 255.0)
/// * `quantize` - Optional flag to quantize the output to 8-bit levels (defaults to false)
///
/// # Returns
/// Returns a `Vec<f32>` containing the compressed signal.
///
/// # Examples
/// ```
/// let signal = vec![0.5, -0.5];
/// let compressed = mu_compress(&signal, None, None);
/// ```
pub fn mu_compress(x: &[f32], mu: Option<f32>, quantize: Option<bool>) -> Vec<f32> {
    let mu_val = mu.unwrap_or(255.0);
    x.iter().map(|&v| {
        let sign = if v >= 0.0 { 1.0 } else { -1.0 };
        let compressed = sign * (1.0 + mu_val.abs() * v.abs()).ln() / mu_val.ln();
        if quantize.unwrap_or(false) {
            (compressed * 255.0).round() / 255.0
        } else {
            compressed
        }
    }).collect()
}

/// Applies μ-law expansion to a compressed signal.
///
/// # Arguments
/// * `x` - Input compressed signal as a slice of `f32`
/// * `mu` - Optional μ-law parameter (defaults to 255.0)
/// * `quantize` - Optional flag (unused, included for symmetry with `mu_compress`)
///
/// # Returns
/// Returns a `Vec<f32>` containing the expanded signal.
///
/// # Examples
/// ```
/// let compressed = vec![0.5, -0.5];
/// let expanded = mu_expand(&compressed, None, None);
/// ```
pub fn mu_expand(x: &[f32], mu: Option<f32>, quantize: Option<bool>) -> Vec<f32> {
    let mu_val = mu.unwrap_or(255.0);
    x.iter().map(|&v| {
        let sign = if v >= 0.0 { 1.0 } else { -1.0 };
        sign * (mu_val.ln() * v.abs()).exp() / mu_val
    }).collect()
}

/// Computes the logarithmic energy of framed audio.
///
/// # Arguments
/// * `y` - Input signal as a slice of `f32`
/// * `frame_length` - Optional frame length in samples (defaults to 2048)
/// * `hop_length` - Optional hop length in samples (defaults to frame_length / 4)
///
/// # Returns
/// Returns an `Array1<f32>` containing the log energy for each frame.
///
/// # Examples
/// ```
/// let signal = vec![0.1, 0.2, 0.3, 0.4, 0.5];
/// let energy = log_energy(&signal, Some(2), Some(1));
/// ```
pub fn log_energy(
    y: &[f32],
    frame_length: Option<usize>,
    hop_length: Option<usize>,
) -> Array1<f32> {
    let frame_len = frame_length.unwrap_or(2048);
    let hop = hop_length.unwrap_or(frame_len / 4);
    let n_frames = (y.len() - frame_len) / hop + 1;
    let mut energy = Array1::zeros(n_frames);

    for i in 0..n_frames {
        let start = i * hop;
        let frame = &y[start..(start + frame_len).min(y.len())];
        let e = frame.iter().map(|&x| x.powi(2)).sum::<f32>();
        energy[i] = (e + 1e-10).ln();
    }

    energy
}