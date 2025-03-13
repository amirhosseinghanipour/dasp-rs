use ndarray::{Array1, Array2, Axis};
use crate::signal_processing::time_frequency::stft;

/// Estimates the tempo (beats per minute) from audio or onset envelope.
///
/// # Arguments
/// * `y` - Optional audio time series
/// * `sr` - Optional sample rate (defaults to 44100)
/// * `onset_envelope` - Optional pre-computed onset strength envelope
/// * `hop_length` - Optional hop length in samples (defaults to 512)
///
/// # Returns
/// Returns a single `f32` value representing the estimated tempo in BPM.
///
/// # Panics
/// Panics if `y` is None and `onset_envelope` is None, or if STFT computation fails when `y` is provided.
///
/// # Examples
/// ```
/// let y = vec![0.1, 0.2, 0.3, 0.4];
/// let bpm = tempo(Some(&y), None, None, None);
/// ```
pub fn tempo(
    y: Option<&[f32]>,
    sr: Option<u32>,
    onset_envelope: Option<&Array1<f32>>,
    hop_length: Option<usize>,
) -> f32 {
    let sr = sr.unwrap_or(44100);
    let hop = hop_length.unwrap_or(512);
    let onset_owned = if onset_envelope.is_none() {
        let S = stft(
            y.expect("Audio signal required when onset_envelope is None"),
            None,
            Some(hop),
            None,
        )
        .expect("STFT computation failed")
        .mapv(|x| x.norm());
        S.map_axis(Axis(0), |row| row.iter().map(|&x| x.max(0.0)).sum::<f32>())
    } else {
        onset_envelope.unwrap().to_owned()
    };
    let onset = &onset_owned;
    let tempogram = tempogram(None, Some(sr), Some(onset), hop_length, None);
    tempogram.axis_iter(Axis(1)).map(|col| {
        let max_val = col.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(&0.0);
        let max_idx = col.iter().position(|&x| x == *max_val).unwrap_or(0);
        crate::tempo_frequencies(tempogram.shape()[0], Some(hop), Some(sr))[max_idx]
    }).sum::<f32>() / tempogram.shape()[1] as f32
}

/// Computes a tempogram (local autocorrelation of onset strength).
///
/// # Arguments
/// * `y` - Optional audio time series
/// * `sr` - Optional sample rate (defaults to 44100)
/// * `onset_envelope` - Optional pre-computed onset strength envelope
/// * `hop_length` - Optional hop length in samples (defaults to 512)
/// * `win_length` - Optional window length for autocorrelation (defaults to 384)
///
/// # Returns
/// Returns a 2D array of shape `(win_length/2 + 1, n_frames)` representing the tempogram.
///
/// # Panics
/// Panics if `y` is None and `onset_envelope` is None, or if STFT computation fails when `y` is provided.
///
/// # Examples
/// ```
/// let y = vec![0.1, 0.2, 0.3, 0.4];
/// let tgram = tempogram(Some(&y), None, None, None, None);
/// ```
pub fn tempogram(
    y: Option<&[f32]>,
    sr: Option<u32>,
    onset_envelope: Option<&Array1<f32>>,
    hop_length: Option<usize>,
    win_length: Option<usize>,
) -> Array2<f32> {
    let sr = sr.unwrap_or(44100);
    let hop = hop_length.unwrap_or(512);
    let win = win_length.unwrap_or(384);
    let onset_owned = if onset_envelope.is_none() {
        let S = stft(
            y.expect("Audio signal required when onset_envelope is None"),
            None, 
            Some(hop),
            None,
        )
        .expect("STFT computation failed")
        .mapv(|x| x.norm());
        S.map_axis(Axis(0), |row| row.iter().map(|&x| x.max(0.0)).sum::<f32>())
    } else {
        onset_envelope.unwrap().to_owned()
    };
    let onset = &onset_owned;
    let mut tempogram = Array2::zeros((win / 2 + 1, onset.len()));
    for t in 0..onset.len() {
        for lag in 0..(win / 2 + 1) {
            let past = (t as isize - lag as isize).max(0) as usize;
            tempogram[[lag, t]] = onset[t] * onset[past];
        }
    }
    tempogram
}

/// Computes a tempogram with harmonic ratio analysis.
///
/// # Arguments
/// * `y` - Optional audio time series
/// * `sr` - Optional sample rate (defaults to 44100)
/// * `onset_envelope` - Optional pre-computed onset strength envelope
/// * `hop_length` - Optional hop length in samples (defaults to 512)
/// * `ratios` - Optional array of tempo ratios to analyze (defaults to [2.0, 3.0, 4.0])
///
/// # Returns
/// Returns a 2D array of shape `(n_ratios, n_frames)` representing the ratio tempogram.
///
/// # Panics
/// Panics if `y` is None and `onset_envelope` is None, or if STFT computation fails when `y` is provided.
///
/// # Examples
/// ```
/// let y = vec![0.1, 0.2, 0.3, 0.4];
/// let ratio_tgram = tempogram_ratio(Some(&y), None, None, None, None);
/// ```
pub fn tempogram_ratio(
    y: Option<&[f32]>,
    sr: Option<u32>,
    onset_envelope: Option<&Array1<f32>>,
    hop_length: Option<usize>,
    ratios: Option<&[f32]>,
) -> Array2<f32> {
    let tempogram = tempogram(y, sr, onset_envelope, hop_length, None);
    let ratios = ratios.unwrap_or(&[2.0, 3.0, 4.0]);
    let mut ratio_map = Array2::zeros((ratios.len(), tempogram.shape()[1]));
    for (r_idx, &r) in ratios.iter().enumerate() {
        for t in 0..tempogram.shape()[1] {
            let mut sum = 0.0;
            for f in 0..tempogram.shape()[0] {
                let target_f = f as f32 * r;
                let bin = target_f.round() as usize;
                if bin < tempogram.shape()[0] {
                    sum += tempogram[[bin, t]];
                }
            }
            ratio_map[[r_idx, t]] = sum;
        }
    }
    ratio_map
}