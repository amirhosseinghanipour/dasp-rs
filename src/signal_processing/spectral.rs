use rustfft::FftPlanner;
use num_complex::Complex;
use ndarray::Array2;
use crate::audio_io::AudioError;

pub fn stft(y: &[f32], n_fft: Option<usize>, hop_length: Option<usize>, win_length: Option<usize>) -> Result<Array2<Complex<f32>>, AudioError> {
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

pub fn istft(stft_matrix: &Array2<Complex<f32>>, hop_length: Option<usize>, win_length: Option<usize>, length: Option<usize>) -> Vec<f32> {
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

fn hamming(n: usize, win_length: usize) -> f32 {
    0.54 - 0.46 * (2.0 * std::f32::consts::PI * n as f32 / (win_length - 1) as f32).cos()
}

fn hamming_vec(win_length: usize) -> Vec<f32> {
    (0..win_length).map(|n| hamming(n, win_length)).collect()
}

pub fn reassigned_spectrogram(_y: &[f32], _sr: Option<u32>, _n_fft: Option<usize>) -> Array2<f32> { unimplemented!() }
pub fn cqt(_y: &[f32], _sr: Option<u32>, _hop_length: Option<usize>, _fmin: Option<f32>, _n_bins: Option<usize>) -> Array2<Complex<f32>> { unimplemented!() }
pub fn icqt(_C: &Array2<Complex<f32>>, _sr: Option<u32>, _hop_length: Option<usize>, _fmin: Option<f32>) -> Vec<f32> { unimplemented!() }
pub fn hybrid_cqt(_y: &[f32], _sr: Option<u32>, _hop_length: Option<usize>, _fmin: Option<f32>) -> Array2<Complex<f32>> { unimplemented!() }
pub fn pseudo_cqt(_y: &[f32], _sr: Option<u32>, _hop_length: Option<usize>, _fmin: Option<f32>) -> Array2<Complex<f32>> { unimplemented!() }
pub fn vqt(_y: &[f32], _sr: Option<u32>, _hop_length: Option<usize>, _fmin: Option<f32>, _n_bins: Option<usize>) -> Array2<Complex<f32>> { unimplemented!() }
pub fn iirt(_y: &[f32], _sr: Option<u32>, _win_length: Option<usize>, _hop_length: Option<usize>) -> Array2<f32> { unimplemented!() }
pub fn fmt(_y: &[f32], _t_min: Option<f32>, _n_fmt: Option<usize>, _kind: Option<&str>, _beta: Option<f32>) -> Array2<f32> { unimplemented!() }

pub fn magphase(D: &Array2<Complex<f32>>, power: Option<f32>) -> (Array2<f32>, Array2<Complex<f32>>) {
    let power_val = power.unwrap_or(1.0);
    let magnitude = D.mapv(|x| x.norm().powf(power_val));
    let phase = D.mapv(|x| x / x.norm());
    (magnitude, phase)
}