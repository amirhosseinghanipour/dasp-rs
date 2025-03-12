use ndarray::{Array2, Array1};
use crate::signal_processing::spectral::istft;
use crate::features::spectral::melspectrogram;
use crate::features::phase_recovery::griffinlim;

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