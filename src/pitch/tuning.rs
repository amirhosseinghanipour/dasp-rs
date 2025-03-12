pub fn pyin(_y: &[f32], _fmin: f32, _fmax: f32, _sr: Option<u32>, _frame_length: Option<usize>) -> Vec<f32> { unimplemented!() }
pub fn yin(_y: &[f32], _fmin: f32, _fmax: f32, _sr: Option<u32>, _frame_length: Option<usize>) -> Vec<f32> { unimplemented!() }
pub fn estimate_tuning(_y: Option<&[f32]>, _sr: Option<u32>, _S: Option<&ndarray::Array2<f32>>, _n_fft: Option<usize>) -> f32 { unimplemented!() }
pub fn pitch_tuning(_frequencies: &[f32], _resolution: Option<f32>) -> f32 { unimplemented!() }
pub fn piptrack(_y: Option<&[f32]>, _sr: Option<u32>, _S: Option<&ndarray::Array2<f32>>, _n_fft: Option<usize>, _hop_length: Option<usize>) -> (ndarray::Array2<f32>, ndarray::Array2<f32>) { unimplemented!() }