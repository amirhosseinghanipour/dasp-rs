use aurust::{to_mono, resample, autocorrelate, lpc, zero_crossings, mu_compress, mu_expand, stft, istft, magphase};
use ndarray::arr2;
use approx::assert_abs_diff_eq;
use num_complex::Complex;

#[test]
fn test_to_mono() {
    let stereo = vec![0.5, 0.5, 0.3, 0.7];
    let mono = to_mono(&stereo, 2);
    assert_eq!(mono, vec![0.5, 0.5]);
}

#[test]
fn test_resample() {
    let samples = vec![0.0, 1.0, 0.0, -1.0];
    let resampled = resample(&samples, 44100, 22050).unwrap();
    assert!(resampled.len() < samples.len());
}

#[test]
fn test_autocorrelate() {
    let y = vec![1.0, 0.0, 1.0];
    let ac = autocorrelate(&y, Some(2), None);
    assert_eq!(ac.len(), 2);
}

#[test]
fn test_lpc() {
    let y = vec![1.0, 0.5, 0.25];
    let coeffs = lpc(&y, 2).unwrap();
    assert_eq!(coeffs.len(), 3);
}

#[test]
fn test_zero_crossings() {
    let y = vec![1.0, -1.0, 1.0];
    let crossings = zero_crossings(&y, None, None);
    assert_eq!(crossings, vec![1, 2]);
}

#[test]
fn test_mu_compress_expand() {
    let x = vec![0.5, -0.5];
    let compressed = mu_compress(&x, None, None);
    let expanded = mu_expand(&compressed, None, None);
    for (orig, exp) in x.iter().zip(expanded.iter()) {
        assert_abs_diff_eq!(orig, exp, epsilon = 0.01);
    }
}

#[test]
fn test_stft_istft() {
    let y = vec![1.0, 0.0, 1.0, 0.0];
    let stft_res = stft(&y, Some(4), None, None).unwrap();
    let y_recon = istft(&stft_res, None, None, Some(y.len())); // Pass original length
    assert_eq!(y_recon.len(), y.len(), "Reconstructed signal length should match original");
}

#[test]
fn test_magphase() {
    let D = arr2(&[[Complex::new(1.0, 1.0)], [Complex::new(0.0, 1.0)]]);
    let (mag, phase) = magphase(&D, None);
    assert_abs_diff_eq!(mag[[0, 0]], 2.0f32.sqrt(), epsilon = 0.01);
}