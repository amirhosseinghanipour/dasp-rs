use ndarray::{Array2, Array1};
use crate::signal_processing::spectral::istft;
use crate::features::spectral::melspectrogram;
use crate::features::phase_recovery::griffinlim;

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