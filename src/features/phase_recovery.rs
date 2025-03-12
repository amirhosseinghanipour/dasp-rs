use ndarray::{Array2, Array1};
use num_complex::Complex;

pub fn griffinlim(S: &Array2<f32>, n_iter: Option<usize>, hop_length: Option<usize>) -> Vec<f32> {
    let n_fft = (S.shape()[0] - 1) * 2;
    let hop = hop_length.unwrap_or(n_fft / 4).max(1);
    let n_iter = n_iter.unwrap_or(32);
    let signal_len = hop * (S.shape()[1] - 1) + n_fft;
    let mut y = Array1::from_vec(crate::signal_generation::tone(440.0, None, Some(signal_len), None, None));
    for _ in 0..n_iter {
        let stft_y = crate::signal_processing::stft(y.as_slice().unwrap(), Some(n_fft), Some(hop), None).unwrap();
        let (mut mag, mut phase) = crate::signal_processing::magphase(&stft_y, None);
        for ((i, j), m) in mag.indexed_iter_mut() {
            *m = S[[i, j]].sqrt();
            let p = &mut phase[[i, j]];
            if m.abs() > 1e-10 { 
                *p = *p / p.norm();
            }
        }
        let mag_complex = mag.mapv(|x| Complex::new(x, 0.0));
        let new_stft = mag_complex * phase;
        y = Array1::from_vec(crate::signal_processing::istft(&new_stft, Some(hop), None, Some(signal_len)));
    }
    y.to_vec()
}

pub fn griffinlim_cqt(_C: &Array2<f32>, _n_iter: Option<usize>, _sr: Option<u32>) -> Vec<f32> { unimplemented!() }