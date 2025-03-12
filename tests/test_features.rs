use aurust::{griffinlim, phase_vocoder};
use ndarray::arr2;
use num_complex::Complex;

#[test]
fn test_griffinlim() {
    let S = arr2(&[[1.0, 1.0], [0.5, 0.5]]);
    let y = griffinlim(&S, None, None);
    let n_fft = (S.shape()[0] - 1) * 2;
    let hop = (n_fft / 4).max(1);
    let expected_len = hop * (S.shape()[1] - 1) + n_fft;
    assert_eq!(y.len(), expected_len, "Griffin-Lim output length should match expected");
}

#[test]
fn test_phase_vocoder() {
    let D = arr2(&[[Complex::new(1.0, 0.0)], [Complex::new(0.5, 0.0)]]);
    let vocoded = phase_vocoder(&D, 2.0, None, None);
    assert!(vocoded.shape()[1] < D.shape()[1]);
}