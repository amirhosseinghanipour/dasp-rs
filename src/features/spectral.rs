use ndarray::{Array1, Array2, s, Axis};
use crate::signal_processing::spectral::{stft, cqt};
use crate::frequencies::{hz_to_midi, midi_to_note};

pub fn chroma_stft(
    y: Option<&[f32]>,
    sr: Option<u32>,
    S: Option<&Array2<f32>>,
    norm: Option<f32>,
    n_fft: Option<usize>,
    hop_length: Option<usize>,
    tuning: Option<f32>,
) -> Array2<f32> {
    let sr = sr.unwrap_or(44100);
    let n_fft = n_fft.unwrap_or(2048);
    let hop = hop_length.unwrap_or(n_fft / 4);
    let S = match (y, S) {
        (Some(y), None) => stft(y, Some(n_fft), Some(hop), None).unwrap().mapv(|x| x.norm().powi(2)),
        (None, Some(S)) => S.to_owned(),
        _ => panic!("Must provide either y or S"),
    };
    let n_bins = S.shape()[0];
    let freqs = crate::frequencies::fft_frequencies(Some(sr), Some(n_fft));
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
    chroma
}

pub fn chroma_cqt(
    y: Option<&[f32]>,
    sr: Option<u32>,
    C: Option<&Array2<f32>>,
    hop_length: Option<usize>,
    fmin: Option<f32>,
    bins_per_octave: Option<usize>,
) -> Array2<f32> {
    let sr = sr.unwrap_or(44100);
    let hop = hop_length.unwrap_or(512);
    let fmin = fmin.unwrap_or(32.70);
    let bpo = bins_per_octave.unwrap_or(12);
    let n_bins = bpo * 3;
    let C = match (y, C) {
        (Some(y), None) => cqt(y, sr, Some(hop), Some(fmin), Some(n_bins), Some(bpo)).unwrap(),
        (None, Some(C)) => C.to_owned(),
        _ => panic!("Must provide either y or C"),
    };
    let mut chroma = Array2::zeros((12, C.shape()[1]));
    for frame in 0..C.shape()[1] {
        for bin in 0..C.shape()[0] {
            let freq = fmin * 2.0f32.powf(bin as f32 / bpo as f32);
            let midi = hz_to_midi(&[freq])[0];
            let pitch_class = (midi.round() as usize % 12) as usize;
            chroma[[pitch_class, frame]] += C[[bin, frame]].norm();
        }
    }
    chroma
}

pub fn chroma_cens(
    y: Option<&[f32]>,
    sr: Option<u32>,
    C: Option<&Array2<f32>>,
    hop_length: Option<usize>,
    fmin: Option<f32>,
    bins_per_octave: Option<usize>,
    win_length: Option<usize>,
) -> Array2<f32> {
    let mut chroma = chroma_cqt(y, sr, C, hop_length, fmin, bins_per_octave);
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
    cens
}

pub fn melspectrogram(
    y: Option<&[f32]>,
    sr: Option<u32>,
    S: Option<&Array2<f32>>,
    n_fft: Option<usize>,
    hop_length: Option<usize>,
    n_mels: Option<usize>,
    fmin: Option<f32>,
    fmax: Option<f32>,
) -> Array2<f32> {
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
    let mel_f = crate::frequencies::mel_frequencies(Some(n_mels), Some(fmin), Some(fmax), None);
    let mut mel_S = Array2::zeros((n_mels, S.shape()[1]));
    let fft_f = crate::frequencies::fft_frequencies(Some(sr), Some(n_fft));
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
    mel_S
}

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

pub fn rms(
    y: Option<&[f32]>,
    S: Option<&Array2<f32>>,
    frame_length: Option<usize>,
    hop_length: Option<usize>,
) -> Array1<f32> {
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
            rms
        }
        (None, Some(S)) => S.map_axis(Axis(0), |row| f32::sqrt(row.iter().map(|x| x.powi(2)).sum::<f32>() / row.len() as f32)),
        _ => panic!("Must provide either y or S"),
    }
}

pub fn spectral_centroid(
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
        (Some(y), None) => stft(y, Some(n_fft), Some(hop), None).unwrap().mapv(|x| x.norm()),
        (None, Some(S)) => S.to_owned(),
        _ => panic!("Must provide either y or S"),
    };
    let freqs = crate::frequencies::fft_frequencies(Some(sr), Some(n_fft));
    S.axis_iter(Axis(1)).map(|frame| {
        let total = frame.sum();
        if total > 1e-6 { frame.dot(&Array1::from_vec(freqs.clone())) / total } else { 0.0 }
    }).collect()
}

pub fn spectral_bandwidth(
    y: Option<&[f32]>,
    sr: Option<u32>,
    S: Option<&Array2<f32>>,
    n_fft: Option<usize>,
    hop_length: Option<usize>,
    p: Option<i32>,
) -> Array1<f32> {
    let p = p.unwrap_or(2);
    let centroid = spectral_centroid(y, sr, S, n_fft, hop_length);
    let S = match (y, S) {
        (Some(y), None) => stft(y, Some(n_fft), Some(hop_length.unwrap_or(n_fft.unwrap_or(2048) / 4)), None).unwrap().mapv(|x| x.norm()),
        (None, Some(S)) => S.to_owned(),
        _ => panic!("Must provide either y or S"),
    };
    let freqs = crate::frequencies::fft_frequencies(sr, n_fft);
    S.axis_iter(Axis(1)).zip(centroid.iter()).map(|(frame, &c)| {
        let total = frame.sum();
        if total > 1e-6 {
            let dev = frame.iter().zip(freqs.iter()).map(|(&s, &f)| s * (f - c).powi(p)).sum::<f32>() / total;
            dev.powf(1.0 / p as f32)
        } else {
            0.0
        }
    }).collect()
}

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
    let freqs = crate::frequencies::fft_frequencies(Some(sr), Some(n_fft));
    let band_edges = Array1::logspace(2.0, 0.0, f32::log2(sr as f32 / 2.0), n_bands + 1);
    let mut contrast = Array2::zeros((n_bands + 1, S.shape()[1]));
    for t in 0..S.shape()[1] {
        for b in 0..n_bands + 1 {
            let f_low = if b == 0 { 0.0 } else { band_edges[b - 1] };
            let f_high = band_edges[b];
            let band = S.slice(s![..;..;t]).iter().zip(freqs.iter()).filter(|&(_, &f)| f >= f_low && f <= f_high).map(|(&s, _)| s);
            let band_vec: Vec<_> = band.collect();
            if !band_vec.is_empty() {
                let sorted: Vec<_> = band_vec.into_iter().sorted_by(|a, b| a.partial_cmp(b).unwrap()).collect();
                let peak = sorted[sorted.len() - 1];
                let valley = sorted[0];
                contrast[[b, t]] = peak - valley;
            }
        }
    }
    contrast
}