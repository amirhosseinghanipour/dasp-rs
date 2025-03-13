use ndarray::Array2;
use crate::AudioError;

/// Converts amplitude spectrogram to decibels (dB).
///
/// # Arguments
/// * `S` - Amplitude spectrogram as a 2D array
/// * `ref_val` - Optional reference amplitude (defaults to 1.0)
/// * `amin` - Optional minimum amplitude threshold (defaults to 1e-5)
/// * `top_db` - Optional maximum dB below reference (defaults to 80.0)
///
/// # Returns
/// Returns a `Result<Array2<f32>, AudioError>` containing the dB spectrogram.
///
/// # Errors
/// - `InsufficientData` if the spectrogram is empty
/// - `InvalidInput` if `ref_val`, `amin`, or `top_db` is non-positive, or if `S` contains negative values
///
/// # Examples
/// ```
/// use ndarray::arr2;
/// let S = arr2(&[[1.0, 2.0], [0.1, 0.01]]);
/// let S_db = amplitude_to_db(&S, None, None, None).unwrap();
/// assert!(S_db[[0, 0]] == 0.0); // 20 * log10(1.0 / 1.0)
/// assert!(S_db[[0, 1]] > 6.0 && S_db[[0, 1]] < 7.0); // ~6.021 dB
/// ```
pub fn amplitude_to_db(
    S: &Array2<f32>,
    ref_val: Option<f32>,
    amin: Option<f32>,
    top_db: Option<f32>,
) -> Result<Array2<f32>, AudioError> {
    let ref_val = ref_val.unwrap_or(1.0);
    let amin_val = amin.unwrap_or(1e-5);
    let top_db_val = top_db.unwrap_or(80.0);

    if S.is_empty() {
        return Err(AudioError::InsufficientData("Spectrogram is empty".to_string()));
    }
    if ref_val <= 0.0 {
        return Err(AudioError::InvalidInput("Reference value must be positive".to_string()));
    }
    if amin_val <= 0.0 {
        return Err(AudioError::InvalidInput("Minimum amplitude must be positive".to_string()));
    }
    if top_db_val <= 0.0 {
        return Err(AudioError::InvalidInput("Top dB must be positive".to_string()));
    }
    if S.iter().any(|&x| x < 0.0) {
        return Err(AudioError::InvalidInput("Spectrogram contains negative amplitudes".to_string()));
    }

    Ok(S.mapv(|x| {
        let x_clipped = x.max(amin_val);
        let db = 20.0 * (x_clipped / ref_val).log10();
        db.max(db.max(-top_db_val))
    }))
}

/// Converts decibel (dB) spectrogram to amplitude.
///
/// # Arguments
/// * `S_db` - Decibel spectrogram as a 2D array
/// * `ref_val` - Optional reference amplitude (defaults to 1.0)
///
/// # Returns
/// Returns a `Result<Array2<f32>, AudioError>` containing the amplitude spectrogram.
///
/// # Errors
/// - `InsufficientData` if the spectrogram is empty
/// - `InvalidInput` if `ref_val` is non-positive
///
/// # Examples
/// ```
/// use ndarray::arr2;
/// let S_db = arr2(&[[0.0, 6.021], [-20.0, -40.0]]);
/// let S = db_to_amplitude(&S_db, None).unwrap();
/// assert!(S[[0, 0]] == 1.0);
/// assert!(S[[0, 1]] > 1.995 && S[[0, 1]] < 2.005); // ~2.0
/// ```
pub fn db_to_amplitude(
    S_db: &Array2<f32>,
    ref_val: Option<f32>,
) -> Result<Array2<f32>, AudioError> {
    let ref_val = ref_val.unwrap_or(1.0);

    if S_db.is_empty() {
        return Err(AudioError::InsufficientData("Spectrogram is empty".to_string()));
    }
    if ref_val <= 0.0 {
        return Err(AudioError::InvalidInput("Reference value must be positive".to_string()));
    }

    Ok(S_db.mapv(|x| ref_val * 10.0f32.powf(x / 20.0)))
}

/// Converts power spectrogram to decibels (dB).
///
/// # Arguments
/// * `S` - Power spectrogram as a 2D array
/// * `ref_val` - Optional reference power (defaults to 1.0)
/// * `amin` - Optional minimum power threshold (defaults to 1e-10)
/// * `top_db` - Optional maximum dB below reference (defaults to 80.0)
///
/// # Returns
/// Returns a `Result<Array2<f32>, AudioError>` containing the dB spectrogram.
///
/// # Errors
/// - `InsufficientData` if the spectrogram is empty
/// - `InvalidInput` if `ref_val`, `amin`, or `top_db` is non-positive, or if `S` contains negative values
///
/// # Examples
/// ```
/// use ndarray::arr2;
/// let S = arr2(&[[1.0, 4.0], [0.1, 0.01]]);
/// let S_db = power_to_db(&S, None, None, None).unwrap();
/// assert!(S_db[[0, 0]] == 0.0); // 10 * log10(1.0 / 1.0)
/// assert!(S_db[[0, 1]] > 6.0 && S_db[[0, 1]] < 7.0); // ~6.021 dB
/// ```
pub fn power_to_db(
    S: &Array2<f32>,
    ref_val: Option<f32>,
    amin: Option<f32>,
    top_db: Option<f32>,
) -> Result<Array2<f32>, AudioError> {
    let ref_val = ref_val.unwrap_or(1.0);
    let amin_val = amin.unwrap_or(1e-10);
    let top_db_val = top_db.unwrap_or(80.0);

    if S.is_empty() {
        return Err(AudioError::InsufficientData("Spectrogram is empty".to_string()));
    }
    if ref_val <= 0.0 {
        return Err(AudioError::InvalidInput("Reference value must be positive".to_string()));
    }
    if amin_val <= 0.0 {
        return Err(AudioError::InvalidInput("Minimum power must be positive".to_string()));
    }
    if top_db_val <= 0.0 {
        return Err(AudioError::InvalidInput("Top dB must be positive".to_string()));
    }
    if S.iter().any(|&x| x < 0.0) {
        return Err(AudioError::InvalidInput("Spectrogram contains negative power values".to_string()));
    }

    Ok(S.mapv(|x| {
        let x_clipped = x.max(amin_val);
        let db = 10.0 * (x_clipped / ref_val).log10();
        db.max(db.max(-top_db_val))
    }))
}

/// Converts decibel (dB) spectrogram to power.
///
/// # Arguments
/// * `S_db` - Decibel spectrogram as a 2D array
/// * `ref_val` - Optional reference power (defaults to 1.0)
///
/// # Returns
/// Returns a `Result<Array2<f32>, AudioError>` containing the power spectrogram.
///
/// # Errors
/// - `InsufficientData` if the spectrogram is empty
/// - `InvalidInput` if `ref_val` is non-positive
///
/// # Examples
/// ```
/// use ndarray::arr2;
/// let S_db = arr2(&[[0.0, 6.021], [-10.0, -20.0]]);
/// let S = db_to_power(&S_db, None).unwrap();
/// assert!(S[[0, 0]] == 1.0);
/// assert!(S[[0, 1]] > 3.995 && S[[0, 1]] < 4.005); // ~4.0
/// ```
pub fn db_to_power(
    S_db: &Array2<f32>,
    ref_val: Option<f32>,
) -> Result<Array2<f32>, AudioError> {
    let ref_val = ref_val.unwrap_or(1.0);

    if S_db.is_empty() {
        return Err(AudioError::InsufficientData("Spectrogram is empty".to_string()));
    }
    if ref_val <= 0.0 {
        return Err(AudioError::InvalidInput("Reference value must be positive".to_string()));
    }

    Ok(S_db.mapv(|x| ref_val * 10.0f32.powf(x / 10.0)))
}

/// Applies perceptual frequency weighting to a spectrogram.
///
/// # Arguments
/// * `S` - Spectrogram as a 2D array (frequencies × time)
/// * `frequencies` - Array of frequencies corresponding to spectrogram rows
/// * `kind` - Optional weighting type ("A", "B", "C", "D"; defaults to "A")
///
/// # Returns
/// Returns a `Result<Array2<f32>, AudioError>` containing the weighted spectrogram.
///
/// # Errors
/// - `InsufficientData` if the spectrogram is empty
/// - `InvalidInput` if frequencies length mismatches spectrogram rows, spectrogram has negative values, or kind is unknown
///
/// # Examples
/// ```
/// use ndarray::arr2;
/// let S = arr2(&[[1.0, 1.0], [1.0, 1.0]]);
/// let freqs = vec![1000.0, 2000.0];
/// let S_weighted = perceptual_weighting(&S, &freqs, None).unwrap();
/// ```
pub fn perceptual_weighting(
    S: &Array2<f32>,
    frequencies: &[f32],
    kind: Option<&str>,
) -> Result<Array2<f32>, AudioError> {
    if S.is_empty() {
        return Err(AudioError::InsufficientData("Spectrogram is empty".to_string()));
    }
    if frequencies.len() != S.shape()[0] {
        return Err(AudioError::InvalidInput(format!(
            "Frequency length {} does not match spectrogram rows {}",
            frequencies.len(), S.shape()[0]
        )));
    }
    if S.iter().any(|&x| x < 0.0) {
        return Err(AudioError::InvalidInput("Spectrogram contains negative values".to_string()));
    }

    let weights = match kind.unwrap_or("A") {
        "A" => A_weighting(frequencies, None)?,
        "B" => B_weighting(frequencies, None)?,
        "C" => C_weighting(frequencies, None)?,
        "D" => D_weighting(frequencies, None)?,
        k => return Err(AudioError::InvalidInput(format!("Unknown weighting kind: {}", k))),
    };

    let mut S_weighted = Array2::zeros(S.dim());
    for f in 0..S.shape()[0] {
        let w = weights[f];
        for t in 0..S.shape()[1] {
            S_weighted[[f, t]] = S[[f, t]] * w;
        }
    }

    Ok(S_weighted)
}

/// Computes frequency weighting coefficients for a given type.
///
/// # Arguments
/// * `frequencies` - Array of frequencies
/// * `kind` - Optional weighting type ("A", "B", "C", "D"; defaults to "A")
///
/// # Returns
/// Returns a `Result<Vec<f32>, AudioError>` containing weighting coefficients.
///
/// # Errors
/// - `InvalidInput` if the weighting kind is unknown
///
/// # Examples
/// ```
/// let freqs = vec![1000.0, 2000.0];
/// let weights = frequency_weighting(&freqs, Some("A")).unwrap();
/// ```
pub fn frequency_weighting(
    frequencies: &[f32],
    kind: Option<&str>,
) -> Result<Vec<f32>, AudioError> {
    match kind.unwrap_or("A") {
        "A" => A_weighting(frequencies, None),
        "B" => B_weighting(frequencies, None),
        "C" => C_weighting(frequencies, None),
        "D" => D_weighting(frequencies, None),
        k => Err(AudioError::InvalidInput(format!("Unknown weighting kind: {}", k))),
    }
}

/// Computes multiple frequency weighting coefficients for various types.
///
/// # Arguments
/// * `frequencies` - Array of frequencies
/// * `kinds` - Array of weighting types (e.g., ["A", "C"])
///
/// # Returns
/// Returns a `Result<Vec<Vec<f32>>, AudioError>` containing weighting coefficients for each type.
///
/// # Errors
/// - `InsufficientData` if frequencies or kinds are empty
/// - `InvalidInput` if any weighting kind is unknown
///
/// # Examples
/// ```
/// let freqs = vec![1000.0, 2000.0];
/// let weights = multi_frequency_weighting(&freqs, &["A", "C"]).unwrap();
/// assert_eq!(weights.len(), 2);
/// ```
pub fn multi_frequency_weighting(
    frequencies: &[f32],
    kinds: &[&str],
) -> Result<Vec<Vec<f32>>, AudioError> {
    if frequencies.is_empty() {
        return Err(AudioError::InsufficientData("Frequency array is empty".to_string()));
    }
    if kinds.is_empty() {
        return Err(AudioError::InvalidInput("No weighting kinds provided".to_string()));
    }

    let mut results = Vec::with_capacity(kinds.len());
    for &kind in kinds {
        let weights = match kind {
            "A" => A_weighting(frequencies, None)?,
            "B" => B_weighting(frequencies, None)?,
            "C" => C_weighting(frequencies, None)?,
            "D" => D_weighting(frequencies, None)?,
            k => return Err(AudioError::InvalidInput(format!("Unknown weighting kind: {}", k))),
        };
        results.push(weights);
    }
    Ok(results)
}

/// Computes A-weighting coefficients for given frequencies.
///
/// # Arguments
/// * `frequencies` - Array of frequencies in Hz
/// * `min_db` - Optional minimum dB threshold (defaults to -80.0)
///
/// # Returns
/// Returns a `Result<Vec<f32>, AudioError>` containing A-weighting coefficients.
///
/// # Errors
/// - `InsufficientData` if frequencies array is empty
/// - `InvalidInput` if frequencies contain negative values
///
/// # Examples
/// ```
/// let freqs = vec![1000.0];
/// let weights = A_weighting(&freqs, None).unwrap();
/// assert!(weights[0] > 0.0);
/// ```
pub fn A_weighting(
    frequencies: &[f32],
    min_db: Option<f32>,
) -> Result<Vec<f32>, AudioError> {
    let min_db = min_db.unwrap_or(-80.0);

    if frequencies.is_empty() {
        return Err(AudioError::InsufficientData("Frequency array is empty".to_string()));
    }
    if frequencies.iter().any(|&f| f < 0.0) {
        return Err(AudioError::InvalidInput("Frequencies must be non-negative".to_string()));
    }

    let mut weights = Vec::with_capacity(frequencies.len());
    for &f in frequencies {
        if f < 1e-6 {
            weights.push(0.0);
            continue;
        }
        let f2 = f * f;
        let f4 = f2 * f2;
        let num = 12194.0_f32.powi(2) * f4;
        let den = (f2 + 20.6_f32.powi(2)) * (f2 + 12194.0_f32.powi(2)) * ((f2 + 107.7_f32.powi(2)) * (f2 + 737.9_f32.powi(2))).sqrt();
        let gain_db = 20.0 * (num / den).log10() + 2.0;
        let gain = if gain_db < min_db { 0.0 } else { 10.0_f32.powf(gain_db / 20.0) };
        weights.push(gain);
    }
    Ok(weights)
}

/// Computes B-weighting coefficients for given frequencies.
///
/// # Arguments
/// * `frequencies` - Array of frequencies in Hz
/// * `min_db` - Optional minimum dB threshold (defaults to -80.0)
///
/// # Returns
/// Returns a `Result<Vec<f32>, AudioError>` containing B-weighting coefficients.
///
/// # Errors
/// - `InsufficientData` if frequencies array is empty
/// - `InvalidInput` if frequencies contain negative values
///
/// # Examples
/// ```
/// let freqs = vec![1000.0];
/// let weights = B_weighting(&freqs, None).unwrap();
/// assert!(weights[0] > 0.0);
/// ```
pub fn B_weighting(
    frequencies: &[f32],
    min_db: Option<f32>,
) -> Result<Vec<f32>, AudioError> {
    let min_db = min_db.unwrap_or(-80.0);

    if frequencies.is_empty() {
        return Err(AudioError::InsufficientData("Frequency array is empty".to_string()));
    }
    if frequencies.iter().any(|&f| f < 0.0) {
        return Err(AudioError::InvalidInput("Frequencies must be non-negative".to_string()));
    }

    let mut weights = Vec::with_capacity(frequencies.len());
    for &f in frequencies {
        if f < 1e-6 {
            weights.push(0.0);
            continue;
        }
        let f2 = f * f;
        let num = 12194.0_f32.powi(2) * f2;
        let den = (f2 + 20.6_f32.powi(2)) * (f2 + 12194.0_f32.powi(2));
        let gain_db = 10.0 * (num / den + 1.0).log10();
        let gain = if gain_db < min_db { 0.0 } else { 10.0_f32.powf(gain_db / 20.0) };
        weights.push(gain);
    }
    Ok(weights)
}

/// Computes C-weighting coefficients for given frequencies.
///
/// # Arguments
/// * `frequencies` - Array of frequencies in Hz
/// * `min_db` - Optional minimum dB threshold (defaults to -80.0)
///
/// # Returns
/// Returns a `Result<Vec<f32>, AudioError>` containing C-weighting coefficients.
///
/// # Errors
/// - `InsufficientData` if frequencies array is empty
/// - `InvalidInput` if frequencies contain negative values
///
/// # Examples
/// ```
/// let freqs = vec![1000.0];
/// let weights = C_weighting(&freqs, None).unwrap();
/// assert!(weights[0] > 0.0);
/// ```
pub fn C_weighting(
    frequencies: &[f32],
    min_db: Option<f32>,
) -> Result<Vec<f32>, AudioError> {
    let min_db = min_db.unwrap_or(-80.0);

    if frequencies.is_empty() {
        return Err(AudioError::InsufficientData("Frequency array is empty".to_string()));
    }
    if frequencies.iter().any(|&f| f < 0.0) {
        return Err(AudioError::InvalidInput("Frequencies must be non-negative".to_string()));
    }

    let mut weights = Vec::with_capacity(frequencies.len());
    for &f in frequencies {
        if f < 1e-6 {
            weights.push(0.0);
            continue;
        }
        let f2 = f * f;
        let num = 12194.0_f32.powi(2) * f2;
        let den = (f2 + 20.6_f32.powi(2)) * (f2 + 12194.0_f32.powi(2));
        let gain_db = 10.0 * (num / den).log10() + 0.06;
        let gain = if gain_db < min_db { 0.0 } else { 10.0_f32.powf(gain_db / 20.0) };
        weights.push(gain);
    }
    Ok(weights)
}

/// Computes D-weighting coefficients for given frequencies.
///
/// # Arguments
/// * `frequencies` - Array of frequencies in Hz
/// * `min_db` - Optional minimum dB threshold (defaults to -80.0)
///
/// # Returns
/// Returns a `Result<Vec<f32>, AudioError>` containing D-weighting coefficients.
///
/// # Errors
/// - `InsufficientData` if frequencies array is empty
/// - `InvalidInput` if frequencies contain negative values
///
/// # Examples
/// ```
/// let freqs = vec![1000.0];
/// let weights = D_weighting(&freqs, None).unwrap();
/// assert!(weights[0] > 0.0);
/// ```
pub fn D_weighting(
    frequencies: &[f32],
    min_db: Option<f32>,
) -> Result<Vec<f32>, AudioError> {
    let min_db = min_db.unwrap_or(-80.0);

    if frequencies.is_empty() {
        return Err(AudioError::InsufficientData("Frequency array is empty".to_string()));
    }
    if frequencies.iter().any(|&f| f < 0.0) {
        return Err(AudioError::InvalidInput("Frequencies must be non-negative".to_string()));
    }

    let mut weights = Vec::with_capacity(frequencies.len());
    for &f in frequencies {
        if f < 1e-6 {
            weights.push(0.0);
            continue;
        }
        let f2 = f * f;
        let f4 = f2 * f2;
        let num = 6532.0_f32.powi(2) * f4;
        let den = (f2 + 148.0_f32.powi(2)) * (f2 + 6532.0_f32.powi(2)) * (f + 1087.0).powi(2);
        let gain_db = 10.0 * (num / den).log10();
        let gain = if gain_db < min_db { 0.0 } else { 10.0_f32.powf(gain_db / 20.0) };
        weights.push(gain);
    }
    Ok(weights)
}

/// Applies Per-Channel Energy Normalization (PCEN) to a spectrogram.
///
/// # Arguments
/// * `S` - Spectrogram as a 2D array (frequencies × time)
/// * `sr` - Optional sample rate in Hz (defaults to 44100)
/// * `hop_length` - Optional hop length in samples (defaults to 512)
/// * `gain` - Optional gain factor (defaults to 0.8)
/// * `bias` - Optional bias factor (defaults to 10.0)
///
/// # Returns
/// Returns a `Result<Array2<f32>, AudioError>` containing the normalized spectrogram.
///
/// # Errors
/// - `InsufficientData` if the spectrogram is empty
/// - `InvalidInput` if spectrogram has negative values or if `gain`/`bias` are negative
///
/// # Examples
/// ```
/// use ndarray::arr2;
/// let S = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
/// let P = pcen(&S, None, None, None, None).unwrap();
/// ```
pub fn pcen(
    S: &Array2<f32>,
    sr: Option<u32>,
    hop_length: Option<usize>,
    gain: Option<f32>,
    bias: Option<f32>,
) -> Result<Array2<f32>, AudioError> {
    let sr = sr.unwrap_or(44100);
    let hop_length = hop_length.unwrap_or(512);
    let gain = gain.unwrap_or(0.8);
    let bias = bias.unwrap_or(10.0);
    let eps = 1e-6;
    let s = 0.025;

    if S.is_empty() {
        return Err(AudioError::InsufficientData("Spectrogram is empty".to_string()));
    }
    if S.iter().any(|&x| x < 0.0) {
        return Err(AudioError::InvalidInput("Spectrogram contains negative values".to_string()));
    }
    if gain < 0.0 || bias < 0.0 {
        return Err(AudioError::InvalidInput("Gain and bias must be non-negative".to_string()));
    }

    let n_freqs = S.shape()[0];
    let n_frames = S.shape()[1];
    let alpha = (-s * sr as f32 / hop_length as f32).exp();
    let one_minus_alpha = 1.0 - alpha;

    let mut M = Array2::zeros((n_freqs, n_frames));
    let mut P = Array2::zeros((n_freqs, n_frames));

    for f in 0..n_freqs {
        M[[f, 0]] = S[[f, 0]];
        for t in 1..n_frames {
            M[[f, t]] = alpha * M[[f, t - 1]] + one_minus_alpha * S[[f, t]];
        }
    }

    for f in 0..n_freqs {
        for t in 0..n_frames {
            let m = M[[f, t]] + eps;
            P[[f, t]] = (S[[f, t]] / m).powf(gain) + bias - bias;
        }
    }

    Ok(P)
}