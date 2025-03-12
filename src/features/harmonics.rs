use ndarray::Array2;
use num_complex::Complex;

pub fn interp_harmonics(_x: &[f32], _freqs: &[f32], _harmonics: &[f32]) -> Array2<f32> { unimplemented!() }
pub fn salience(_S: &Array2<f32>, _freqs: &[f32], _harmonics: &[f32], _weights: Option<&[f32]>) -> Array2<f32> { unimplemented!() }
pub fn f0_harmonics(_x: &[f32], _f0: &[f32], _freqs: &[f32], _harmonics: &[f32]) -> Array2<f32> { unimplemented!() }

pub fn phase_vocoder(D: &Array2<Complex<f32>>, rate: f32, hop_length: Option<usize>, n_fft: Option<usize>) -> Array2<Complex<f32>> {
    let n = n_fft.unwrap_or((D.shape()[0] - 1) * 2);
    let hop = hop_length.unwrap_or(n / 4);
    let orig_frames = D.shape()[1];
    let new_frames = ((orig_frames as f32 * hop as f32) / rate / hop as f32).ceil() as usize;
    let mut output = Array2::zeros((D.shape()[0], new_frames));

    for new_idx in 0..new_frames {
        let orig_idx = ((new_idx as f32 * hop as f32 * rate) / hop as f32) as usize;
        if orig_idx < orig_frames {
            output.column_mut(new_idx).assign(&D.column(orig_idx));
        }
    }
    output
}