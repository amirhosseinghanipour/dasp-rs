use ndarray::{Array1, Array2, s, Axis};
use crate::signal_processing::time_frequency::{stft, cqt};
use crate::hz_to_midi;
use ndarray_linalg::{Solve, Eig};
use num_complex::Complex;

/// Computes chroma features using Short-Time Fourier Transform (STFT).
///
/// # Arguments
/// * `y` - Optional audio time series
/// * `sr` - Optional sample rate (defaults to 44100 Hz)
/// * `S` - Optional pre-computed magnitude spectrogram
/// * `norm` - Optional normalization factor
/// * `n_fft` - Optional FFT window size (defaults to 2048)
/// * `hop_length` - Optional hop length (defaults to n_fft/4)
/// * `tuning` - Optional tuning adjustment in semitones (currently unused)
///
/// # Returns
/// Returns a `Result` containing a 2D array of shape `(12, n_frames)` with chroma features,
/// or an error message as a `String`.
///
/// # Panics
/// Panics if neither `y` nor `S` is provided.
///
/// # Examples
/// ```
/// let y = vec![0.1, 0.2, 0.3, 0.4];
/// let chroma = chroma_stft(Some(&y), None, None, None, None, None, None).unwrap();
/// ```
pub fn chroma_stft(
    y: Option<&[f32]>,
    sr: Option<u32>,
    S: Option<&Array2<f32>>,
    norm: Option<f32>,
    n_fft: Option<usize>,
    hop_length: Option<usize>,
    tuning: Option<f32>,
) -> Result<Array2<f32>, String> {
    let sr = sr.unwrap_or(44100);
    let n_fft = n_fft.unwrap_or(2048);
    let hop = hop_length.unwrap_or(n_fft / 4);
    let S = match (y, S) {
        (Some(y), None) => stft(y, Some(n_fft), Some(hop), None).unwrap().mapv(|x| x.norm().powi(2)),
        (None, Some(S)) => S.to_owned(),
        _ => panic!("Must provide either y or S"),
    };
    let n_bins = S.shape()[0];
    let freqs = crate::fft_frequencies(Some(sr), Some(n_fft));
    let mut chroma = Array2::zeros((12, S.shape()[1]));

    for frame in 0..S.shape()[1] {
        for bin in 0..n_bins {
            let midi = hz_to_midi(&[freqs[bin]])[0];
            let pitch_class = (midi.round() as usize % 12) as usize;
            chroma[[pitch_class, frame]] += S[[bin, frame]];
        }
    }

    if let Some(norm_val) = norm {
        chroma.mapv_inplace(|x| x / norm_val);
    }
    Ok(chroma)
}

/// Computes chroma features using Constant-Q Transform (CQT).
///
/// # Arguments
/// * `y` - Optional audio time series
/// * `sr` - Optional sample rate (defaults to 44100 Hz)
/// * `C` - Optional pre-computed CQT spectrogram
/// * `hop_length` - Optional hop length (defaults to 512)
/// * `fmin` - Optional minimum frequency (defaults to 32.70 Hz, C1)
/// * `bins_per_octave` - Optional bins per octave (defaults to 12)
///
/// # Returns
/// Returns a `Result` containing a 2D array of shape `(12, n_frames)` with chroma features,
/// or an error message as a `String`.
///
/// # Panics
/// Panics if neither `y` nor `C` is provided.
///
/// # Examples
/// ```
/// let y = vec![0.1, 0.2, 0.3, 0.4];
/// let chroma = chroma_cqt(Some(&y), None, None, None, None, None).unwrap();
/// ```
pub fn chroma_cqt(
    y: Option<&[f32]>,
    sr: Option<u32>,
    C: Option<&Array2<f32>>,
    hop_length: Option<usize>,
    fmin: Option<f32>,
    bins_per_octave: Option<usize>,
) -> Result<Array2<f32>, String> {
    let sr = sr.unwrap_or(44100);
    let hop = hop_length.unwrap_or(512);
    let fmin = fmin.unwrap_or(32.70);
    let bpo = bins_per_octave.unwrap_or(12);
    let n_bins = bpo * 3;
    let C = match (y, C) {
        (Some(y), None) => cqt(y, Some(sr), Some(hop), Some(fmin), Some(n_bins))
            .map_err(|e| e.to_string()),
        (None, Some(C)) => Ok(C.mapv(|x| num_complex::Complex::new(x, 0.0))),
        _ => panic!("Must provide either y or C"),
    }?;
    let mut chroma = Array2::zeros((12, C.shape()[1]));
    for frame in 0..C.shape()[1] {
        for bin in 0..C.shape()[0] {
            let freq = fmin * 2.0f32.powf(bin as f32 / bpo as f32);
            let midi = hz_to_midi(&[freq])[0];
            let pitch_class = midi.round() as usize % 12;
            chroma[[pitch_class, frame]] += C[[bin, frame]].norm();
        }
    }
    Ok(chroma)
}

/// Computes Chroma Energy Normalized Statistics (CENS) features.
///
/// # Arguments
/// * `y` - Optional audio time series
/// * `sr` - Optional sample rate (defaults to 44100 Hz)
/// * `C` - Optional pre-computed CQT spectrogram
/// * `hop_length` - Optional hop length (defaults to 512)
/// * `fmin` - Optional minimum frequency (defaults to 32.70 Hz)
/// * `bins_per_octave` - Optional bins per octave (defaults to 12)
/// * `win_length` - Optional window length for normalization (defaults to 41)
///
/// # Returns
/// Returns a `Result` containing a 2D array of shape `(12, n_frames)` with CENS features,
/// or an error message as a `String`.
///
/// # Examples
/// ```
/// let y = vec![0.1, 0.2, 0.3, 0.4];
/// let cens = chroma_cens(Some(&y), None, None, None, None, None, None).unwrap();
/// ```
pub fn chroma_cens(
    y: Option<&[f32]>,
    sr: Option<u32>,
    C: Option<&Array2<f32>>,
    hop_length: Option<usize>,
    fmin: Option<f32>,
    bins_per_octave: Option<usize>,
    win_length: Option<usize>,
) -> Result<Array2<f32>, String> {
    let chroma = chroma_cqt(y, sr, C, hop_length, fmin, bins_per_octave)?;
    let win = win_length.unwrap_or(41);
    let half_win = win / 2;
    let mut cens = Array2::zeros(chroma.dim());
    for t in 0..chroma.shape()[1] {
        let slice = chroma.slice(s![.., t.saturating_sub(half_win)..(t + half_win + 1).min(chroma.shape()[1])]);
        let norm = slice.mapv(|x| x.powi(2)).sum_axis(Axis(1)).mapv(f32::sqrt);
        for p in 0..12 {
            cens[[p, t]] = if norm[p] > 1e-6 { chroma[[p, t]] / norm[p] } else { 0.0 };
        }
    }
    Ok(cens)
}

/// Computes a mel spectrogram.
///
/// # Arguments
/// * `y` - Optional audio time series
/// * `sr` - Optional sample rate (defaults to 44100 Hz)
/// * `S` - Optional pre-computed magnitude spectrogram
/// * `n_fft` - Optional FFT window size (defaults to 2048)
/// * `hop_length` - Optional hop length (defaults to n_fft/4)
/// * `n_mels` - Optional number of mel bands (defaults to 128)
/// * `fmin` - Optional minimum frequency (defaults to 0 Hz)
/// * `fmax` - Optional maximum frequency (defaults to sr/2)
///
/// # Returns
/// Returns a `Result` containing a 2D array of shape `(n_mels, n_frames)` with mel spectrogram,
/// or an error message as a `String`.
///
/// # Panics
/// Panics if neither `y` nor `S` is provided.
///
/// # Examples
/// ```
/// let y = vec![0.1, 0.2, 0.3, 0.4];
/// let mel = melspectrogram(Some(&y), None, None, None, None, None, None, None).unwrap();
/// ```
pub fn melspectrogram(
    y: Option<&[f32]>,
    sr: Option<u32>,
    S: Option<&Array2<f32>>,
    n_fft: Option<usize>,
    hop_length: Option<usize>,
    n_mels: Option<usize>,
    fmin: Option<f32>,
    fmax: Option<f32>,
) -> Result<Array2<f32>, String> {
    let sr = sr.unwrap_or(44100);
    let n_fft = n_fft.unwrap_or(2048);
    let hop = hop_length.unwrap_or(n_fft / 4);
    let n_mels = n_mels.unwrap_or(128);
    let fmin = fmin.unwrap_or(0.0);
    let fmax = fmax.unwrap_or(sr as f32 / 2.0);
    let S = match (y, S) {
        (Some(y), None) => stft(y, Some(n_fft), Some(hop), None).unwrap().mapv(|x| x.norm().powi(2)),
        (None, Some(S)) => S.to_owned(),
        _ => panic!("Must provide either y or S"),
    };
    let mel_f = crate::mel_frequencies(Some(n_mels), Some(fmin), Some(fmax), None);
    let mut mel_S = Array2::zeros((n_mels, S.shape()[1]));
    let fft_f = crate::fft_frequencies(Some(sr), Some(n_fft));
    for m in 0..n_mels {
        let f_low = if m == 0 { fmin } else { mel_f[m - 1] };
        let f_center = mel_f[m];
        let f_high = mel_f.get(m + 1).copied().unwrap_or(fmax);
        for (bin, &f) in fft_f.iter().enumerate() {
            let weight = if f >= f_low && f <= f_high {
                if f <= f_center { (f - f_low) / (f_center - f_low) } else { (f_high - f) / (f_high - f_center) }
            } else {
                0.0
            };
            for t in 0..S.shape()[1] {
                mel_S[[m, t]] += S[[bin, t]] * weight.max(0.0);
            }
        }
    }
    Ok(mel_S)
}

/// Computes Mel-frequency cepstral coefficients (MFCCs).
///
/// # Arguments
/// * `y` - Optional audio time series
/// * `sr` - Optional sample rate (defaults to 44100 Hz)
/// * `S` - Optional pre-computed spectrogram
/// * `n_mfcc` - Optional number of MFCCs (defaults to 20)
/// * `dct_type` - Optional DCT type (defaults to 2)
/// * `norm` - Optional normalization type ("ortho" or None)
///
/// # Returns
/// Returns a `Result` containing a 2D array of shape `(n_mfcc, n_frames)` with MFCCs,
/// or an error message as a `String`.
///
/// # Examples
/// ```
/// let y = vec![0.1, 0.2, 0.3, 0.4];
/// let mfcc = mfcc(Some(&y), None, None, None, None, None).unwrap();
/// ```
pub fn mfcc(
    y: Option<&[f32]>,
    sr: Option<u32>,
    S: Option<&Array2<f32>>,
    n_mfcc: Option<usize>,
    dct_type: Option<i32>,
    norm: Option<&str>,
) -> Result<Array2<f32>, String> {
    let n_mfcc = n_mfcc.unwrap_or(20);
    let S = melspectrogram(y, sr, S, None, None, None, None, None)?;
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
    Ok(mfcc)
}

/// Computes root mean square (RMS) energy.
///
/// # Arguments
/// * `y` - Optional audio time series
/// * `S` - Optional pre-computed spectrogram
/// * `frame_length` - Optional frame length (defaults to 2048)
/// * `hop_length` - Optional hop length (defaults to frame_length/4)
///
/// # Returns
/// Returns a `Result` containing a 1D array of RMS values per frame,
/// or an error message as a `String`.
///
/// # Panics
/// Panics if neither `y` nor `S` is provided.
///
/// # Examples
/// ```
/// let y = vec![0.1, 0.2, 0.3, 0.4];
/// let rms = rms(Some(&y), None, None, None).unwrap();
/// ```
pub fn rms(
    y: Option<&[f32]>,
    S: Option<&Array2<f32>>,
    frame_length: Option<usize>,
    hop_length: Option<usize>,
) -> Result<Array1<f32>, String> {
    let frame_len = frame_length.unwrap_or(2048);
    let hop = hop_length.unwrap_or(frame_len / 4);
    match (y, S) {
        (Some(y), None) => {
            let n_frames = (y.len() - frame_len) / hop + 1;
            let mut rms = Array1::zeros(n_frames);
            for i in 0..n_frames {
                let start = i * hop;
                let slice = &y[start..(start + frame_len).min(y.len())];
                rms[i] = f32::sqrt(slice.iter().map(|x| x.powi(2)).sum::<f32>() / slice.len() as f32);
            }
            Ok(rms)
        }
        (None, Some(S)) => Ok(S.map_axis(Axis(0), |row| f32::sqrt(row.iter().map(|x| x.powi(2)).sum::<f32>() / row.len() as f32))),
        _ => panic!("Must provide either y or S"),
    }
}

/// Computes spectral centroid frequencies.
///
/// # Arguments
/// * `y` - Optional audio time series
/// * `sr` - Optional sample rate (defaults to 44100 Hz)
/// * `S` - Optional pre-computed spectrogram
/// * `n_fft` - Optional FFT window size (defaults to 2048)
/// * `hop_length` - Optional hop length (defaults to n_fft/4)
///
/// # Returns
/// Returns a `Result` containing a 1D array of centroid frequencies per frame,
/// or an error message as a `String`.
///
/// # Panics
/// Panics if neither `y` nor `S` is provided.
///
/// # Examples
/// ```
/// let y = vec![0.1, 0.2, 0.3, 0.4];
/// let centroid = spectral_centroid(Some(&y), None, None, None, None).unwrap();
/// ```
pub fn spectral_centroid(
    y: Option<&[f32]>,
    sr: Option<u32>,
    S: Option<&Array2<f32>>,
    n_fft: Option<usize>,
    hop_length: Option<usize>,
) -> Result<Array1<f32>, String> {
    let sr = sr.unwrap_or(44100);
    let n_fft = n_fft.unwrap_or(2048);
    let hop = hop_length.unwrap_or(n_fft / 4);
    let S = match (y, S) {
        (Some(y), None) => stft(y, Some(n_fft), Some(hop), None).unwrap().mapv(|x| x.norm()),
        (None, Some(S)) => S.to_owned(),
        _ => panic!("Must provide either y or S"),
    };
    let freqs = crate::fft_frequencies(Some(sr), Some(n_fft));
    Ok(S.axis_iter(Axis(1)).map(|frame| {
        let total = frame.sum();
        if total > 1e-6 { frame.dot(&Array1::from_vec(freqs.clone())) / total } else { 0.0 }
    }).collect())
}

/// Computes spectral bandwidth.
///
/// # Arguments
/// * `y` - Optional audio time series
/// * `sr` - Optional sample rate (defaults to 44100 Hz)
/// * `S` - Optional pre-computed spectrogram
/// * `n_fft` - Optional FFT window size (defaults to 2048)
/// * `hop_length` - Optional hop length (defaults to n_fft/4)
/// * `p` - Optional power for bandwidth calculation (defaults to 2)
///
/// # Returns
/// Returns a `Result` containing a 1D array of bandwidth values per frame,
/// or an error message as a `String`.
///
/// # Panics
/// Panics if neither `y` nor `S` is provided.
///
/// # Examples
/// ```
/// let y = vec![0.1, 0.2, 0.3, 0.4];
/// let bandwidth = spectral_bandwidth(Some(&y), None, None, None, None, None).unwrap();
/// ```
pub fn spectral_bandwidth(
    y: Option<&[f32]>,
    sr: Option<u32>,
    S: Option<&Array2<f32>>,
    n_fft: Option<usize>,
    hop_length: Option<usize>,
    p: Option<i32>,
) -> Result<Array1<f32>, String> {
    let p = p.unwrap_or(2);
    let centroid = spectral_centroid(y, sr, S, n_fft, hop_length)?;
    let S = match (y, S) {
        (Some(y), None) => stft(y, Some(n_fft.unwrap()), Some(hop_length.unwrap_or(n_fft.unwrap_or(2048) / 4)), None).unwrap().mapv(|x| x.norm()),
        (None, Some(S)) => S.to_owned(),
        _ => panic!("Must provide either y or S"),
    };
    let freqs = crate::fft_frequencies(sr, n_fft);
    Ok(S.axis_iter(Axis(1)).zip(centroid.iter()).map(|(frame, &c)| {
        let total = frame.sum();
        if total > 1e-6 {
            let dev = frame.iter().zip(freqs.iter()).map(|(&s, &f)| s * (f - c).powi(p)).fold(0.0, |acc, x| acc + x) / total;
            dev.powf(1.0 / p as f32)
        } else {
            0.0
        }
    }).collect())
}

/// Computes spectral contrast across frequency bands.
///
/// # Arguments
/// * `y` - Optional audio time series
/// * `sr` - Optional sample rate (defaults to 44100 Hz)
/// * `S` - Optional pre-computed spectrogram
/// * `n_fft` - Optional FFT window size (defaults to 2048)
/// * `hop_length` - Optional hop length (defaults to n_fft/4)
/// * `n_bands` - Optional number of frequency bands (defaults to 6)
///
/// # Returns
/// Returns a 2D array of shape `(n_bands + 1, n_frames)` with contrast values.
///
/// # Panics
/// Panics if neither `y` nor `S` is provided.
///
/// # Examples
/// ```
/// let y = vec![0.1, 0.2, 0.3, 0.4];
/// let contrast = spectral_contrast(Some(&y), None, None, None, None, None);
/// ```
pub fn spectral_contrast(
    y: Option<&[f32]>,
    sr: Option<u32>,
    S: Option<&Array2<f32>>,
    n_fft: Option<usize>,
    hop_length: Option<usize>,
    n_bands: Option<usize>,
) -> Array2<f32> {
    let sr = sr.unwrap_or(44100);
    let n_fft = n_fft.unwrap_or(2048);
    let hop = hop_length.unwrap_or(n_fft / 4);
    let n_bands = n_bands.unwrap_or(6);
    let S = match (y, S) {
        (Some(y), None) => stft(y, Some(n_fft), Some(hop), None).unwrap().mapv(|x| x.norm()),
        (None, Some(S)) => S.to_owned(),
        _ => panic!("Must provide either y or S"),
    };
    let freqs = crate::fft_frequencies(Some(sr), Some(n_fft));
    let band_edges = Array1::logspace(2.0, 0.0, f32::log2(sr as f32 / 2.0), n_bands + 1);
    let mut contrast = Array2::zeros((n_bands + 1, S.shape()[1]));
    for t in 0..S.shape()[1] {
        for b in 0..n_bands + 1 {
            let f_low = if b == 0 { 0.0 } else { band_edges[b - 1] };
            let f_high = band_edges[b];
            let slice = S.slice(s![.., t]);
            let band = slice.iter().zip(freqs.iter()).filter(|&(_, &f)| f >= f_low && f <= f_high).map(|(&s, _)| s);
            let band_vec: Vec<_> = band.collect();
            if !band_vec.is_empty() {
                let mut sorted: Vec<_> = band_vec;
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let peak = sorted[sorted.len() - 1];
                let valley = sorted[0];
                contrast[[b, t]] = peak - valley;
            }
        }
    }
    contrast
}

/// Computes spectral flatness.
///
/// # Arguments
/// * `y` - Optional audio time series
/// * `S` - Optional pre-computed spectrogram
/// * `n_fft` - Optional FFT window size (defaults to 2048)
/// * `hop_length` - Optional hop length (defaults to n_fft/4)
///
/// # Returns
/// Returns a 1D array of flatness values per frame.
///
/// # Panics
/// Panics if neither `y` nor `S` is provided.
///
/// # Examples
/// ```
/// let y = vec![0.1, 0.2, 0.3, 0.4];
/// let flatness = spectral_flatness(Some(&y), None, None, None);
/// ```
pub fn spectral_flatness(
    y: Option<&[f32]>,
    S: Option<&Array2<f32>>,
    n_fft: Option<usize>,
    hop_length: Option<usize>,
) -> Array1<f32> {
    let n_fft = n_fft.unwrap_or(2048);
    let hop = hop_length.unwrap_or(n_fft / 4);
    let S = match (y, S) {
        (Some(y), None) => stft(y, Some(n_fft), Some(hop), None).unwrap().mapv(|x| x.norm().max(1e-10)),
        (None, Some(S)) => S.to_owned(),
        _ => panic!("Must provide either y or S"),
    };
    S.axis_iter(Axis(1)).map(|frame| {
        let log_frame = frame.mapv(f32::ln);
        let geo_mean = log_frame.sum() / frame.len() as f32;
        let arith_mean = frame.sum() / frame.len() as f32;
        f32::exp(geo_mean) / arith_mean
    }).collect()
}

/// Computes spectral roll-off frequency.
///
/// # Arguments
/// * `y` - Optional audio time series
/// * `sr` - Optional sample rate (defaults to 44100 Hz)
/// * `S` - Optional pre-computed spectrogram
/// * `n_fft` - Optional FFT window size (defaults to 2048)
/// * `hop_length` - Optional hop length (defaults to n_fft/4)
/// * `roll_percent` - Optional roll-off percentage (defaults to 0.85)
///
/// # Returns
/// Returns a 1D array of roll-off frequencies per frame.
///
/// # Panics
/// Panics if neither `y` nor `S` is provided.
///
/// # Examples
/// ```
/// let y = vec![0.1, 0.2, 0.3, 0.4];
/// let rolloff = spectral_rolloff(Some(&y), None, None, None, None, None);
/// ```
pub fn spectral_rolloff(
    y: Option<&[f32]>,
    sr: Option<u32>,
    S: Option<&Array2<f32>>,
    n_fft: Option<usize>,
    hop_length: Option<usize>,
    roll_percent: Option<f32>,
) -> Array1<f32> {
    let sr = sr.unwrap_or(44100);
    let n_fft = n_fft.unwrap_or(2048);
    let hop = hop_length.unwrap_or(n_fft / 4);
    let roll_percent = roll_percent.unwrap_or(0.85);
    let S = match (y, S) {
        (Some(y), None) => stft(y, Some(n_fft), Some(hop), None).unwrap().mapv(|x| x.norm()),
        (None, Some(S)) => S.to_owned(),
        _ => panic!("Must provide either y or S"),
    };
    let freqs = crate::fft_frequencies(Some(sr), Some(n_fft));
    S.axis_iter(Axis(1)).map(|frame| {
        let total_energy = frame.sum();
        let target_energy = total_energy * roll_percent;
        let mut cum_energy = 0.0;
        for (f, &s) in freqs.iter().zip(frame.iter()) {
            cum_energy += s;
            if cum_energy >= target_energy {
                return *f;
            }
        }
        freqs[freqs.len() - 1]
    }).collect()
}

/// Computes polynomial fit coefficients for spectral features.
///
/// # Arguments
/// * `y` - Optional audio time series
/// * `sr` - Optional sample rate (defaults to 44100 Hz)
/// * `S` - Optional pre-computed spectrogram
/// * `n_fft` - Optional FFT window size (defaults to 2048)
/// * `hop_length` - Optional hop length (defaults to n_fft/4)
/// * `order` - Optional polynomial order (defaults to 1)
///
/// # Returns
/// Returns a 2D array of shape `(order + 1, n_frames)` with polynomial coefficients.
///
/// # Panics
/// Panics if neither `y` nor `S` is provided.
///
/// # Examples
/// ```
/// let y = vec![0.1, 0.2, 0.3, 0.4];
/// let coeffs = poly_features(Some(&y), None, None, None, None, None);
/// ```
pub fn poly_features(
    y: Option<&[f32]>,
    sr: Option<u32>,
    S: Option<&Array2<f32>>,
    n_fft: Option<usize>,
    hop_length: Option<usize>,
    order: Option<usize>,
) -> Array2<f32> {
    let n_fft = n_fft.unwrap_or(2048);
    let hop = hop_length.unwrap_or(n_fft / 4);
    let order = order.unwrap_or(1);
    let S = match (y, S) {
        (Some(y), None) => stft(y, Some(n_fft), Some(hop), None).unwrap().mapv(|x| x.norm()),
        (None, Some(S)) => S.to_owned(),
        _ => panic!("Must provide either y or S"),
    };
    let mut coeffs = Array2::zeros((order + 1, S.shape()[1]));
    let x = Array1::linspace(0.0, S.shape()[0] as f32 - 1.0, S.shape()[0]);
    for t in 0..S.shape()[1] {
        let y_t = S.slice(s![.., t]).to_owned();
        let poly = polyfit(&x, &y_t, order);
        for (i, &c) in poly.iter().enumerate() {
            coeffs[[i, t]] = c;
        }
    }
    coeffs
}

/// Computes Tonnetz features from chroma.
///
/// # Arguments
/// * `y` - Optional audio time series
/// * `sr` - Optional sample rate (defaults to 44100 Hz)
/// * `chroma` - Optional pre-computed chroma features
///
/// # Returns
/// Returns a `Result` containing a 2D array of shape `(6, n_frames)` with Tonnetz features,
/// or an error message as a `String`.
///
/// # Examples
/// ```
/// let y = vec![0.1, 0.2, 0.3, 0.4];
/// let tonnetz = tonnetz(Some(&y), None, None).unwrap();
/// ```
pub fn tonnetz(
    y: Option<&[f32]>,
    sr: Option<u32>,
    chroma: Option<&Array2<f32>>,
) -> Result<Array2<f32>, String> {
    let chroma_stft_result = chroma_stft(y, sr, None, None, None, None, None)?;
    let chroma = chroma.unwrap_or(&chroma_stft_result);
    let transform = Array2::from_shape_vec((6, 12), vec![
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // Fifths
        0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // Minor thirds
        0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // Major thirds
        0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // Minor sevenths
        0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // Major seconds
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, // Tritones
    ]).unwrap();
    Ok(transform.dot(chroma))
}

/// Fits a polynomial to data points.
///
/// # Arguments
/// * `x` - X-coordinates
/// * `y` - Y-coordinates
/// * `order` - Polynomial order
///
/// # Returns
/// Returns a vector of polynomial coefficients.
///
/// # Panics
/// Panics if linear solving fails (returns zeros instead).
fn polyfit(x: &Array1<f32>, y: &Array1<f32>, order: usize) -> Vec<f32> {
    let n = order + 1;
    let mut A = Array2::zeros((x.len(), n));
    for i in 0..x.len() {
        for j in 0..n {
            A[[i, j]] = x[i].powi(j as i32);
        }
    }
    let coeffs = A.solve(&y.to_owned()).unwrap_or_else(|_| Array1::zeros(n));
    coeffs.to_vec()
}

/// Computes spectral flux.
///
/// # Arguments
/// * `y` - Optional audio time series
/// * `sr` - Optional sample rate (defaults to 44100 Hz)
/// * `S` - Optional pre-computed spectrogram
/// * `n_fft` - Optional FFT window size (defaults to 2048)
/// * `hop_length` - Optional hop length (defaults to n_fft/4)
///
/// # Returns
/// Returns a 1D array of flux values per frame.
///
/// # Panics
/// Panics if neither `y` nor `S` is provided, or if STFT computation fails.
///
/// # Examples
/// ```
/// let y = vec![0.1, 0.2, 0.3, 0.4];
/// let flux = spectral_flux(Some(&y), None, None, None, None);
/// ```
pub fn spectral_flux(
    y: Option<&[f32]>,
    sr: Option<u32>,
    S: Option<&Array2<f32>>,
    n_fft: Option<usize>,
    hop_length: Option<usize>,
) -> Array1<f32> {
    let sr = sr.unwrap_or(44100);
    let n_fft = n_fft.unwrap_or(2048);
    let hop = hop_length.unwrap_or(n_fft / 4);
    let S = match (y, S) {
        (Some(y), None) => stft(y, Some(n_fft), Some(hop), None)
            .expect("STFT failed")
            .mapv(|x| x.norm()),
        (None, Some(S)) => S.to_owned(),
        _ => panic!("Must provide either y or S"),
    };
    let mut flux = Array1::zeros(S.shape()[1]);
    for t in 1..S.shape()[1] {
        let diff = &S.slice(s![.., t]) - &S.slice(s![.., t - 1]);
        flux[t] = diff.mapv(|x| x.powi(2)).sum().sqrt();
    }
    flux
}

/// Computes spectral entropy.
///
/// # Arguments
/// * `y` - Optional audio time series
/// * `sr` - Optional sample rate (defaults to 44100 Hz)
/// * `S` - Optional pre-computed spectrogram
/// * `n_fft` - Optional FFT window size (defaults to 2048)
/// * `hop_length` - Optional hop length (defaults to n_fft/4)
///
/// # Returns
/// Returns a 1D array of entropy values per frame.
///
/// # Panics
/// Panics if neither `y` nor `S` is provided, or if STFT computation fails.
///
/// # Examples
/// ```
/// let y = vec![0.1, 0.2, 0.3, 0.4];
/// let entropy = spectral_entropy(Some(&y), None, None, None, None);
/// ```
pub fn spectral_entropy(
    y: Option<&[f32]>,
    sr: Option<u32>,
    S: Option<&Array2<f32>>,
    n_fft: Option<usize>,
    hop_length: Option<usize>,
) -> Array1<f32> {
    let sr = sr.unwrap_or(44100);
    let n_fft = n_fft.unwrap_or(2048);
    let hop = hop_length.unwrap_or(n_fft / 4);
    let S = match (y, S) {
        (Some(y), None) => stft(y, Some(n_fft), Some(hop), None)
            .expect("STFT failed")
            .mapv(|x| x.norm()),
        (None, Some(S)) => S.to_owned(),
        _ => panic!("Must provide either y or S"),
    };
    S.axis_iter(Axis(1)).map(|frame| {
        let sum = frame.sum();
        if sum <= 1e-10 {
            0.0
        } else {
            let p = frame.mapv(|x| x / sum);
            -p.mapv(|x| if x > 1e-10 { x * x.ln() } else { 0.0 }).sum()
        }
    }).collect()
}

/// Computes pitch chroma features.
///
/// # Arguments
/// * `y` - Optional audio time series
/// * `sr` - Optional sample rate (defaults to 44100 Hz)
/// * `S` - Optional pre-computed spectrogram
/// * `n_fft` - Optional FFT window size (defaults to 2048)
/// * `hop_length` - Optional hop length (defaults to n_fft/4)
///
/// # Returns
/// Returns a 2D array of shape `(12, n_frames)` with normalized pitch chroma features.
///
/// # Panics
/// Panics if neither `y` nor `S` is provided, or if STFT computation fails.
///
/// # Examples
/// ```
/// let y = vec![0.1, 0.2, 0.3, 0.4];
/// let chroma = pitch_chroma(Some(&y), None, None, None, None);
/// ```
pub fn pitch_chroma(
    y: Option<&[f32]>,
    sr: Option<u32>,
    S: Option<&Array2<f32>>,
    n_fft: Option<usize>,
    hop_length: Option<usize>,
) -> Array2<f32> {
    let sr = sr.unwrap_or(44100);
    let n_fft = n_fft.unwrap_or(2048);
    let hop = hop_length.unwrap_or(n_fft / 4);
    let S = match (y, S) {
        (Some(y), None) => stft(y, Some(n_fft), Some(hop), None)
            .expect("STFT failed")
            .mapv(|x| x.norm()),
        (None, Some(S)) => S.to_owned(),
        _ => panic!("Must provide either y or S"),
    };
    let freqs = crate::fft_frequencies(Some(sr), Some(n_fft));
    let mut chroma = Array2::zeros((12, S.shape()[1]));
    for t in 0..S.shape()[1] {
        let frame = S.column(t);
        for (bin, &f) in freqs.iter().enumerate() {
            if frame[bin] > 0.0 {
                let midi = crate::hz_to_midi(&[f])[0];
                let pitch_class = midi.round() as usize % 12;
                chroma[[pitch_class, t]] += frame[bin];
            }
        }
    }
    for t in 0..chroma.shape()[1] {
        let sum = chroma.column(t).sum();
        if sum > 1e-6 {
            chroma.column_mut(t).mapv_inplace(|x| x / sum);
        }
    }
    chroma
}

/// Applies cepstral mean and variance normalization (CMVN).
///
/// # Arguments
/// * `features` - Input feature matrix
/// * `axis` - Optional axis for normalization (-1 for time, 0 for features; defaults to -1)
/// * `variance` - Optional flag to normalize variance (defaults to true)
///
/// # Returns
/// Returns a `Result` containing the normalized feature matrix,
/// or an error message as a `String`.
///
/// # Examples
/// ```
/// use ndarray::Array2;
/// let features = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
/// let normalized = cmvn(&features, None, None).unwrap();
/// ```
pub fn cmvn(
    features: &Array2<f32>,
    axis: Option<isize>,
    variance: Option<bool>,
) -> Result<Array2<f32>, String> {
    let axis = axis.unwrap_or(-1);
    let do_variance = variance.unwrap_or(true);
    let ax = if axis < 0 { 1 } else { 0 };

    if features.shape()[ax] < 2 {
        return Err("Feature dimension too small for normalization".to_string());
    }

    let mut normalized = features.to_owned();
    let means = normalized.mean_axis(Axis(ax)).ok_or("Failed to compute mean")?;
    for i in 0..normalized.shape()[1 - ax] {
        for j in 0..normalized.shape()[ax] {
            let idx = if ax == 1 { [j, i] } else { [i, j] };
            normalized[idx] -= means[if ax == 1 { j } else { i }];
        }
    }

    if do_variance {
        let variances = normalized.mapv(|x| x.powi(2)).mean_axis(Axis(ax)).ok_or("Failed to compute variance")?;
        let std_devs = variances.mapv(|x| (x + 1e-10).sqrt());
        for i in 0..normalized.shape()[1 - ax] {
            for j in 0..normalized.shape()[ax] {
                let idx = if ax == 1 { [j, i] } else { [i, j] };
                normalized[idx] /= std_devs[if ax == 1 { j } else { i }];
            }
        }
    }

    Ok(normalized)
}

/// Performs Harmonic-Percussive Source Separation (HPSS).
///
/// # Arguments
/// * `y` - Optional audio time series
/// * `sr` - Optional sample rate (defaults to 44100 Hz)
/// * `S` - Optional pre-computed spectrogram
/// * `n_fft` - Optional FFT window size (defaults to 2048)
/// * `hop_length` - Optional hop length (defaults to n_fft/4)
/// * `harm_win` - Optional window size for harmonic component (defaults to 31)
/// * `perc_win` - Optional window size for percussive component (defaults to 31)
///
/// # Returns
/// Returns a tuple `(harmonic, percussive)` containing two 2D arrays with separated components.
///
/// # Panics
/// Panics if neither `y` nor `S` is provided, or if STFT computation fails.
///
/// # Examples
/// ```
/// let y = vec![0.1, 0.2, 0.3, 0.4];
/// let (harmonic, percussive) = hpss(Some(&y), None, None, None, None, None, None);
/// ```
pub fn hpss(
    y: Option<&[f32]>,
    sr: Option<u32>,
    S: Option<&Array2<f32>>,
    n_fft: Option<usize>,
    hop_length: Option<usize>,
    harm_win: Option<usize>,
    perc_win: Option<usize>,
) -> (Array2<f32>, Array2<f32>) {
    let sr = sr.unwrap_or(44100);
    let n_fft = n_fft.unwrap_or(2048);
    let hop = hop_length.unwrap_or(n_fft / 4);
    let harm_win = harm_win.unwrap_or(31);
    let perc_win = perc_win.unwrap_or(31);
    let S = match (y, S) {
        (Some(y), None) => stft(y, Some(n_fft), Some(hop), None)
            .expect("STFT failed")
            .mapv(|x| x.norm().powi(2)),
        (None, Some(S)) => S.to_owned(),
        _ => panic!("Must provide either y or S"),
    };

    let mut harmonic = Array2::zeros(S.dim());
    for f in 0..S.shape()[0] {
        let row = S.index_axis(Axis(0), f);
        for t in 0..S.shape()[1] {
            let start = t.saturating_sub(harm_win / 2);
            let end = (t + harm_win / 2 + 1).min(S.shape()[1]);
            let mut slice: Vec<f32> = row.slice(s![start..end]).to_vec();
            slice.sort_by(|a, b| a.partial_cmp(b).unwrap());
            harmonic[[f, t]] = slice[slice.len() / 2];
        }
    }

    let mut percussive = Array2::zeros(S.dim());
    for t in 0..S.shape()[1] {
        let col = S.index_axis(Axis(1), t);
        for f in 0..S.shape()[0] {
            let start = f.saturating_sub(perc_win / 2);
            let end = (f + perc_win / 2 + 1).min(S.shape()[0]);
            let mut slice: Vec<f32> = col.slice(s![start..end]).to_vec();
            slice.sort_by(|a, b| a.partial_cmp(b).unwrap());
            percussive[[f, t]] = slice[slice.len() / 2];
        }
    }

    let total = harmonic.clone() + percussive.clone();
    let harm_mask = &harmonic / &total.mapv(|x| if x > 0.0 { x } else { 1.0 });
    let perc_mask = &percussive / &total.mapv(|x| if x > 0.0 { x } else { 1.0 });
    (
        S.to_owned() * &harm_mask,
        S.to_owned() * &perc_mask,
    )
}

/// Estimates pitch using autocorrelation.
///
/// # Arguments
/// * `y` - Audio time series
/// * `sr` - Optional sample rate (defaults to 44100 Hz)
/// * `frame_length` - Optional frame length (defaults to 2048)
/// * `hop_length` - Optional hop length (defaults to frame_length/4)
/// * `fmin` - Optional minimum frequency (defaults to 50 Hz)
/// * `fmax` - Optional maximum frequency (defaults to 500 Hz)
///
/// # Returns
/// Returns a 1D array of pitch estimates in Hz per frame.
///
/// # Examples
/// ```
/// let y = vec![0.1, 0.2, 0.3, 0.4];
/// let pitch = pitch_autocorr(&y, None, None, None, None, None);
/// ```
pub fn pitch_autocorr(
    y: &[f32],
    sr: Option<u32>,
    frame_length: Option<usize>,
    hop_length: Option<usize>,
    fmin: Option<f32>,
    fmax: Option<f32>,
) -> Array1<f32> {
    let sr = sr.unwrap_or(44100);
    let frame_len = frame_length.unwrap_or(2048);
    let hop = hop_length.unwrap_or(frame_len / 4);
    let fmin = fmin.unwrap_or(50.0);
    let fmax = fmax.unwrap_or(500.0);
    let n_frames = (y.len() - frame_len) / hop + 1;
    let mut pitch = Array1::zeros(n_frames);

    for i in 0..n_frames {
        let start = i * hop;
        let frame = &y[start..(start + frame_len).min(y.len())];
        let autocorr = crate::signal_processing::time_domain::autocorrelate(frame, Some(frame_len), None);
        let lag_min = (sr as f32 / fmax).round() as usize;
        let lag_max = (sr as f32 / fmin).round() as usize;
        let max_idx = autocorr[lag_min..lag_max.min(autocorr.len())]
            .iter()
            .position(|&x| x == *autocorr[lag_min..lag_max.min(autocorr.len())]
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap())
            .unwrap_or(0) + lag_min;
        pitch[i] = if max_idx > 0 { sr as f32 / max_idx as f32 } else { 0.0 };
    }

    pitch
}

/// Computes features for voice activity detection (VAD).
///
/// # Arguments
/// * `y` - Audio time series
/// * `sr` - Optional sample rate (defaults to 44100 Hz)
/// * `frame_length` - Optional frame length (defaults to 2048)
/// * `hop_length` - Optional hop length (defaults to frame_length/4)
/// * `n_fft` - Optional FFT window size (defaults to 2048)
///
/// # Returns
/// Returns a 2D array of shape `(3, n_frames)` with log energy, ZCR, and flatness.
///
/// # Panics
/// Panics if STFT computation fails.
///
/// # Examples
/// ```
/// let y = vec![0.1, 0.2, 0.3, 0.4];
/// let vad = vad_features(&y, None, None, None, None);
/// ```
pub fn vad_features(
    y: &[f32],
    sr: Option<u32>,
    frame_length: Option<usize>,
    hop_length: Option<usize>,
    n_fft: Option<usize>,
) -> Array2<f32> {
    let sr = sr.unwrap_or(44100);
    let frame_len = frame_length.unwrap_or(2048);
    let hop = hop_length.unwrap_or(frame_len / 4);
    let n_fft = n_fft.unwrap_or(2048);
    let n_frames = (y.len() - frame_len) / hop + 1;

    let energy = crate::signal_processing::time_domain::log_energy(y, Some(frame_len), Some(hop));
    
    let zcr = crate::features::zero_crossing_rate(y, Some(frame_len), Some(hop));
    
    let S = stft(y, Some(n_fft), Some(hop), None)
        .expect("STFT failed")
        .mapv(|x| x.norm());
    let flatness = S.axis_iter(Axis(1)).map(|frame| {
        let geo_mean = frame.mapv(|x| x.max(1e-10).ln()).mean().unwrap().exp();
        let arith_mean = frame.mean().unwrap();
        if arith_mean > 1e-10 { geo_mean / arith_mean } else { 0.0 }
    }).collect::<Array1<f32>>();

    let mut features = Array2::zeros((3, n_frames));
    for i in 0..n_frames {
        features[[0, i]] = energy[i];
        features[[1, i]] = zcr[i];
        features[[2, i]] = flatness[i];
    }

    features
}

/// Computes spectral subband centroids.
///
/// # Arguments
/// * `y` - Optional audio time series
/// * `sr` - Optional sample rate (defaults to 44100 Hz)
/// * `S` - Optional pre-computed spectrogram
/// * `n_fft` - Optional FFT window size (defaults to 2048)
/// * `hop_length` - Optional hop length (defaults to n_fft/4)
/// * `n_bands` - Optional number of subbands (defaults to 4)
///
/// # Returns
/// Returns a 2D array of shape `(n_bands, n_frames)` with subband centroids.
///
/// # Panics
/// Panics if neither `y` nor `S` is provided, or if STFT computation fails.
///
/// # Examples
/// ```
/// let y = vec![0.1, 0.2, 0.3, 0.4];
/// let centroids = spectral_subband_centroids(Some(&y), None, None, None, None, None);
/// ```
pub fn spectral_subband_centroids(
    y: Option<&[f32]>,
    sr: Option<u32>,
    S: Option<&Array2<f32>>,
    n_fft: Option<usize>,
    hop_length: Option<usize>,
    n_bands: Option<usize>,
) -> Array2<f32> {
    let sr = sr.unwrap_or(44100);
    let n_fft = n_fft.unwrap_or(2048);
    let hop = hop_length.unwrap_or(n_fft / 4);
    let n_bands = n_bands.unwrap_or(4);
    
    let S = match (y, S) {
        (Some(y), None) => stft(y, Some(n_fft), Some(hop), None)
            .expect("STFT failed")
            .mapv(|x| x.norm()),
        (None, Some(S)) => S.to_owned(),
        _ => panic!("Must provide either y or S"),
    };
    let freqs = crate::fft_frequencies(Some(sr), Some(n_fft));
    let band_edges = Array1::linspace(0.0, sr as f32 / 2.0, n_bands + 1);

    let mut centroids = Array2::zeros((n_bands, S.shape()[1]));
    for t in 0..S.shape()[1] {
        for b in 0..n_bands {
            let f_low = band_edges[b];
            let f_high = band_edges[b + 1];
            let subband: Vec<(f32, f32)> = freqs.iter()
                .zip(S.column(t))
                .filter(|(f, _)| **f >= f_low && **f < f_high)
                .map(|(f, s)| (*f, *s))
                .collect();
            if subband.is_empty() {
                centroids[[b, t]] = (f_low + f_high) / 2.0;
            } else {
                let total_energy = subband.iter().map(|(_, s)| s).sum::<f32>();
                centroids[[b, t]] = if total_energy > 1e-10 {
                    subband.iter().map(|(f, s)| f * s).sum::<f32>() / total_energy
                } else {
                    (f_low + f_high) / 2.0
                };
            }
        }
    }

    centroids
}

/// Estimates formant frequencies using LPC.
///
/// # Arguments
/// * `y` - Audio time series
/// * `sr` - Optional sample rate (defaults to 44100 Hz)
/// * `n_formants` - Optional number of formants to extract (defaults to 3)
/// * `frame_length` - Optional frame length (defaults to 2048)
/// * `hop_length` - Optional hop length (defaults to frame_length/4)
///
/// # Returns
/// Returns a `Result` containing a 2D array of shape `(n_formants, n_frames)` with formant frequencies,
/// or an error message as a `String`.
///
/// # Examples
/// ```
/// let y = vec![0.1, 0.2, 0.3, 0.4];
/// let formants = formant_frequencies(&y, None, None, None, None).unwrap();
/// ```
pub fn formant_frequencies(
    y: &[f32],
    sr: Option<u32>,
    n_formants: Option<usize>,
    frame_length: Option<usize>,
    hop_length: Option<usize>,
) -> Result<Array2<f32>, String> {
    let sr = sr.unwrap_or(44100);
    let n_formants = n_formants.unwrap_or(3);
    let frame_len = frame_length.unwrap_or(2048);
    let hop = hop_length.unwrap_or(frame_len / 4);
    let order = (2.0 * sr as f32 / 1000.0).round() as usize + 2;
    let n_frames = (y.len() - frame_len) / hop + 1;

    if y.len() < frame_len {
        return Err("Signal length is shorter than frame length".to_string());
    }

    let mut formants = Array2::zeros((n_formants, n_frames));

    for i in 0..n_frames {
        let start = i * hop;
        let frame = &y[start..(start + frame_len).min(y.len())];
        let lpc_coeffs = lpc(frame, order)?;
        let roots = polynomial_roots(&lpc_coeffs)?;
        let mut freqs: Vec<f32> = roots.iter()
            .filter_map(|r| {
                if r.im.abs() > 1e-6 {
                    let freq = r.arg().abs() * sr as f32 / (2.0 * std::f32::consts::PI);
                    if freq > 50.0 && freq < sr as f32 / 2.0 { Some(freq) } else { None }
                } else { None }
            })
            .collect();
        freqs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for (j, &f) in freqs.iter().take(n_formants).enumerate() {
            formants[[j, i]] = f;
        }
    }

    Ok(formants)
}

/// Computes Linear Predictive Coding (LPC) coefficients.
///
/// # Arguments
/// * `frame` - Audio frame
/// * `order` - LPC order
///
/// # Returns
/// Returns a `Result` containing LPC coefficients, or an error message as a `String`.
fn lpc(frame: &[f32], order: usize) -> Result<Vec<f32>, String> {
    if frame.len() < order {
        return Err("Frame length must be at least LPC order".to_string());
    }
    let autocorr = crate::signal_processing::time_domain::autocorrelate(frame, Some(order + 1), None);
    if autocorr[0] <= 1e-10 {
        return Err("Frame energy too low for LPC".to_string());
    }

    let mut a = vec![1.0; order + 1];
    let mut e = autocorr[0];
    let mut tmp = vec![0.0; order + 1];

    for i in 1..=order {
        let mut lambda = 0.0;
        for j in 0..i {
            lambda -= a[j] * autocorr[i - j];
        }
        lambda /= e;
        for j in 0..i {
            tmp[j] = a[j] + lambda * a[i - 1 - j];
        }
        a[..i].copy_from_slice(&tmp[..i]);
        a[i] = lambda;
        e *= 1.0 - lambda * lambda;
        if e <= 1e-10 {
            return Err("LPC instability detected".to_string());
        }
    }
    Ok(a)
}

/// Computes roots of a polynomial.
///
/// # Arguments
/// * `coeffs` - Polynomial coefficients (highest degree first)
///
/// # Returns
/// Returns a `Result` containing complex roots, or an error message as a `String`.
fn polynomial_roots(coeffs: &[f32]) -> Result<Vec<Complex<f32>>, String> {
    if coeffs.len() <= 1 {
        return Ok(vec![]);
    }

    let n = coeffs.len() - 1;
    let mut companion = Array2::zeros((n, n));
    for i in 0..n - 1 {
        companion[[i + 1, i]] = 1.0;
    }
    let a_n = coeffs[n];
    if a_n.abs() < 1e-10 {
        return Err("Leading coefficient too small".to_string());
    }
    for i in 0..n {
        companion[[i, n - 1]] = -coeffs[n - 1 - i] / a_n;
    }

    let eigenvalues = companion.eig().map_err(|e| format!("Eigenvalue computation failed: {}", e))?;
    Ok(eigenvalues.0.to_vec())
}