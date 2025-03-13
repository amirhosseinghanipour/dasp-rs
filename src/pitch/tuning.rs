use ndarray::{Array1, Array2, Axis};
use crate::signal_processing::spectral::stft;
use crate::fft_frequencies;
use crate::AudioError;

pub fn pyin(
    y: &[f32],
    fmin: f32,
    fmax: f32,
    sr: Option<u32>,
    frame_length: Option<usize>,
) -> Result<Vec<f32>, AudioError> {
    let sr = sr.unwrap_or(44100);
    let frame_len = frame_length.unwrap_or(2048);
    let hop_length = frame_len / 4;
    let n_frames = if y.len() < frame_len { 1 } else { (y.len() - frame_len) / hop_length + 1 };

    if fmin >= fmax || fmin < 0.0 || fmax > sr as f32 / 2.0 {
        return Err(AudioError::InvalidInput("Invalid frequency range: fmin >= fmax or out of Nyquist bounds".to_string()));
    }
    if y.len() < frame_len {
        return Err(AudioError::InsufficientData("Signal too short for frame length".to_string()));
    }

    let mut pitches = Vec::with_capacity(n_frames);
    let lag_min = (sr as f32 / fmax).round() as usize;
    let lag_max = (sr as f32 / fmin).round() as usize;

    for i in 0..n_frames {
        let start = i * hop_length;
        let frame = &y[start..(start + frame_len).min(y.len())];
        
        let mut diff = vec![0.0; lag_max];
        for tau in 0..lag_max {
            let mut sum = 0.0;
            for j in 0..frame.len().saturating_sub(tau) {
                let d = frame[j] - frame[j + tau];
                sum += d * d;
            }
            diff[tau] = sum;
        }

        let mut cmnd = vec![1.0; lag_max];
        let mut running_sum = 0.0;
        for tau in 1..lag_max {
            running_sum += diff[tau];
            cmnd[tau] = if running_sum > 1e-6 { diff[tau] * tau as f32 / running_sum } else { 1.0 };
        }

        let mut min_idx = lag_min;
        let mut min_val = cmnd[lag_min];
        for tau in lag_min + 1..lag_max {
            if cmnd[tau] < min_val {
                min_val = cmnd[tau];
                min_idx = tau;
            }
        }

        let pitch = if min_val < 0.1 && min_idx > 1 && min_idx < lag_max - 1 {
            let a = cmnd[min_idx - 1];
            let b = cmnd[min_idx];
            let c = cmnd[min_idx + 1];
            let delta = (a - c) / (2.0 * (a - 2.0 * b + c) + 1e-6);
            sr as f32 / (min_idx as f32 + delta)
        } else {
            0.0
        };

        pitches.push(if pitch >= fmin && pitch <= fmax { pitch } else { 0.0 });
    }

    Ok(pitches)
}

pub fn yin(
    y: &[f32],
    fmin: f32,
    fmax: f32,
    sr: Option<u32>,
    frame_length: Option<usize>,
) -> Result<Vec<f32>, AudioError> {
    let sr = sr.unwrap_or(44100);
    let frame_len = frame_length.unwrap_or(2048);
    let hop_length = frame_len / 4;
    let n_frames = if y.len() < frame_len { 1 } else { (y.len() - frame_len) / hop_length + 1 };

    if fmin >= fmax || fmin < 0.0 || fmax > sr as f32 / 2.0 {
        return Err(AudioError::InvalidInput("Invalid frequency range".to_string()));
    }
    if y.len() < frame_len {
        return Err(AudioError::InsufficientData("Signal too short".to_string()));
    }

    let mut pitches = Vec::with_capacity(n_frames);
    let lag_min = (sr as f32 / fmax).round() as usize;
    let lag_max = (sr as f32 / fmin).round() as usize;

    for i in 0..n_frames {
        let start = i * hop_length;
        let frame = &y[start..(start + frame_len).min(y.len())];
        
        let mut diff = vec![0.0; lag_max];
        for tau in 0..lag_max {
            let mut sum = 0.0;
            for j in 0..frame.len().saturating_sub(tau) {
                let d = frame[j] - frame[j + tau];
                sum += d * d;
            }
            diff[tau] = sum;
        }

        let mut cmnd = vec![1.0; lag_max];
        let mut running_sum = 0.0;
        for tau in 1..lag_max {
            running_sum += diff[tau];
            cmnd[tau] = if running_sum > 1e-6 { diff[tau] * tau as f32 / running_sum } else { 1.0 };
        }

        let mut min_idx = lag_min;
        let mut min_val = cmnd[lag_min];
        for tau in lag_min + 1..lag_max {
            if cmnd[tau] < min_val {
                min_val = cmnd[tau];
                min_idx = tau;
            }
        }

        let pitch = if min_val < 0.5 && min_idx > 0 { sr as f32 / min_idx as f32 } else { 0.0 };
        pitches.push(if pitch >= fmin && pitch <= fmax { pitch } else { 0.0 });
    }

    Ok(pitches)
}

pub fn estimate_tuning(
    y: Option<&[f32]>,
    sr: Option<u32>,
    S: Option<&Array2<f32>>,
    n_fft: Option<usize>,
) -> Result<f32, AudioError> {
    let sr = sr.unwrap_or(44100);
    let n_fft = n_fft.unwrap_or(2048);
    let hop_length = n_fft / 4;

    let S = match (y, S) {
        (Some(y), None) => {
            if y.len() < n_fft {
                return Err(AudioError::InsufficientData("Signal too short for n_fft".to_string()));
            }
            stft(y, Some(n_fft), Some(hop_length), None)
                .map_err(|e| AudioError::ComputationFailed(format!("STFT failed: {}", e)))?
                .mapv(|x| x.norm())
        }
        (None, Some(S)) => S.to_owned(),
        _ => return Err(AudioError::InvalidInput("Must provide either y or S".to_string())),
    };

    let (pitches, mags) = piptrack(Some(&y.unwrap_or(&[])), Some(sr), Some(&S), Some(n_fft), Some(hop_length))?;
    let mut total_deviation = 0.0;
    let mut total_weight = 0.0;

    for t in 0..pitches.shape()[1] {
        for f in 0..pitches.shape()[0] {
            let freq = pitches[[f, t]];
            let mag = mags[[f, t]];
            if freq > 0.0 && mag > 1e-6 {
                let ref_freq = 440.0 * 2.0f32.powf((f32::log2(freq / 440.0)).floor());
                let deviation = 1200.0 * f32::log2(freq / ref_freq);
                total_deviation += deviation * mag;
                total_weight += mag;
            }
        }
    }

    Ok(if total_weight > 1e-6 { total_deviation / total_weight } else { 0.0 })
}

pub fn pitch_tuning(
    frequencies: &[f32],
    resolution: Option<f32>,
) -> Result<f32, AudioError> {
    let resolution = resolution.unwrap_or(1.0);
    if resolution <= 0.0 {
        return Err(AudioError::InvalidInput("Resolution must be positive".to_string()));
    }

    let valid_freqs: Vec<f32> = frequencies.iter().filter(|&&f| f > 0.0).copied().collect();
    if valid_freqs.is_empty() {
        return Ok(0.0);
    }

    let mut total_deviation = 0.0;
    for &freq in &valid_freqs {
        let ref_freq = 440.0 * 2.0f32.powf((f32::log2(freq / 440.0)).floor());
        let cents = 1200.0 * f32::log2(freq / ref_freq);
        total_deviation += (cents % resolution) - if cents % resolution > resolution / 2.0 { resolution } else { 0.0 };
    }

    Ok(total_deviation / valid_freqs.len() as f32)
}

pub fn piptrack(
    y: Option<&[f32]>,
    sr: Option<u32>,
    S: Option<&Array2<f32>>,
    n_fft: Option<usize>,
    hop_length: Option<usize>,
) -> Result<(Array2<f32>, Array2<f32>), AudioError> {
    let sr = sr.unwrap_or(44100);
    let n_fft = n_fft.unwrap_or(2048);
    let hop_length = hop_length.unwrap_or(n_fft / 4);

    let S = match (y, S) {
        (Some(y), None) => {
            if y.len() < n_fft {
                return Err(AudioError::InsufficientData("Signal too short for n_fft".to_string()));
            }
            stft(y, Some(n_fft), Some(hop_length), None)
                .map_err(|e| AudioError::ComputationFailed(format!("STFT failed: {}", e)))?
                .mapv(|x| x.norm())
        }
        (None, Some(S)) => S.to_owned(),
        _ => return Err(AudioError::InvalidInput("Must provide either y or S".to_string())),
    };

    let freqs = fft_frequencies(Some(sr), Some(n_fft));
    if freqs.len() != S.shape()[0] {
        return Err(AudioError::ComputationFailed("Frequency bins mismatch".to_string()));
    }

    let mut pitches = Array2::zeros(S.dim());
    let mut mags = Array2::zeros(S.dim());

    for t in 0..S.shape()[1] {
        let frame = S.column(t);
        let max_idx = frame.iter().position(|&x| x == *frame.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap()).unwrap_or(0);
        let peak_mag = frame[max_idx];
        if peak_mag > 1e-6 {
            let left = if max_idx > 0 { frame[max_idx - 1] } else { peak_mag };
            let right = if max_idx < frame.len() - 1 { frame[max_idx + 1] } else { peak_mag };
            let delta = (left - right) / (2.0 * (left - 2.0 * peak_mag + right) + 1e-6);
            pitches[[max_idx, t]] = freqs[max_idx] + delta * (freqs[1] - freqs[0]);
            mags[[max_idx, t]] = peak_mag;
        }
    }

    Ok((pitches, mags))
}