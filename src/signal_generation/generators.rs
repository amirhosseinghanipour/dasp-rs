use ndarray::Array1;

/// Generates a click signal at specified times or frame indices.
///
/// # Arguments
/// * `times` - Optional array of times in seconds where clicks should occur
/// * `frames` - Optional array of frame indices where clicks should occur
/// * `sr` - Optional sample rate in Hz (defaults to 44100 Hz)
/// * `hop_length` - Optional hop length in samples (defaults to 512)
///
/// # Returns
/// Returns a `Vec<f32>` representing the click signal, with 1.0 at click positions and 0.0 elsewhere.
///
/// # Notes
/// - If both `times` and `frames` are provided, `times` takes precedence.
/// - The signal length is determined by the maximum time or frame index, defaulting to 44100 samples if neither is provided.
/// - Clicks beyond the signal length are ignored.
///
/// # Examples
/// ```
/// let times = vec![0.1, 0.2, 0.3];
/// let signal = clicks(Some(&times), None, None, None);
/// ```
pub fn clicks(
    times: Option<&[f32]>,
    frames: Option<&[usize]>,
    sr: Option<u32>,
    hop_length: Option<usize>,
) -> Vec<f32> {
    let sample_rate = sr.unwrap_or(44100);
    let hop = hop_length.unwrap_or(512);
    let max_samples = times
        .map(|t| {
            (t.iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap()
                * sample_rate as f32) as usize
        })
        .or_else(|| frames.map(|f| f.iter().max().unwrap() * hop))
        .unwrap_or(44100);
    let mut signal = vec![0.0; max_samples];

    if let Some(ts) = times {
        for &t in ts {
            let idx = (t * sample_rate as f32) as usize;
            if idx < signal.len() {
                signal[idx] = 1.0;
            }
        }
    } else if let Some(fs) = frames {
        for &f in fs {
            let idx = f * hop;
            if idx < signal.len() {
                signal[idx] = 1.0;
            }
        }
    }
    signal
}

/// Generates a pure tone (sine wave) at a specified frequency.
///
/// # Arguments
/// * `frequency` - Frequency of the tone in Hz
/// * `sr` - Optional sample rate in Hz (defaults to 44100 Hz)
/// * `length` - Optional length in samples (overrides duration if provided)
/// * `duration` - Optional duration in seconds (defaults to 1.0 if length not provided)
/// * `phi` - Optional initial phase in radians (defaults to 0.0)
///
/// # Returns
/// Returns a `Vec<f32>` containing the generated sine wave.
///
/// # Examples
/// ```
/// let tone_signal = tone(440.0, None, None, Some(0.5), None); // 440 Hz tone for 0.5 seconds
/// ```
pub fn tone(
    frequency: f32,
    sr: Option<u32>,
    length: Option<usize>,
    duration: Option<f32>,
    phi: Option<f32>,
) -> Vec<f32> {
    let sample_rate = sr.unwrap_or(44100);
    let len = length.unwrap_or_else(|| (duration.unwrap_or(1.0) * sample_rate as f32) as usize);
    let phase = phi.unwrap_or(0.0);
    (0..len)
        .map(|n| {
            (2.0 * std::f32::consts::PI * frequency * n as f32 / sample_rate as f32 + phase).cos()
        })
        .collect()
}

/// Generates a linear chirp signal with frequency sweeping from fmin to fmax.
///
/// # Arguments
/// * `fmin` - Starting frequency in Hz
/// * `fmax` - Ending frequency in Hz
/// * `sr` - Optional sample rate in Hz (defaults to 44100 Hz)
/// * `length` - Optional length in samples (overrides duration if provided)
/// * `duration` - Optional duration in seconds (defaults to 1.0 if length not provided)
///
/// # Returns
/// Returns a `Vec<f32>` containing the chirp signal.
///
/// # Examples
/// ```
/// let chirp_signal = chirp(200.0, 800.0, None, None, Some(2.0)); // Chirp from 200 Hz to 800 Hz over 2 seconds
/// ```
pub fn chirp(
    fmin: f32,
    fmax: f32,
    sr: Option<u32>,
    length: Option<usize>,
    duration: Option<f32>,
) -> Vec<f32> {
    let sample_rate = sr.unwrap_or(44100);
    let len = length.unwrap_or_else(|| (duration.unwrap_or(1.0) * sample_rate as f32) as usize);
    let t = Array1::linspace(0.0, 1.0, len);
    let freq = Array1::linspace(fmin, fmax, len);
    t.iter()
        .zip(freq.iter())
        .map(|(&t, &f)| (2.0 * std::f32::consts::PI * f * t).cos())
        .collect()
}