use aurust::{amplitude_to_db, db_to_amplitude, power_to_db, db_to_power, A_weighting};
use ndarray::arr2;
use approx::assert_abs_diff_eq;

#[test]
fn test_amplitude_to_db() {
    let S = arr2(&[[1.0], [0.5]]);
    let db = amplitude_to_db(&S, None, None, None);
    assert!(db[[0, 0]] > db[[1, 0]]);
}

#[test]
fn test_db_to_amplitude() {
    let S_db = arr2(&[[0.0], [-6.0]]);
    let amp = db_to_amplitude(&S_db, None);
    assert_abs_diff_eq!(amp[[0, 0]], 1.0, epsilon = 0.01);
}

#[test]
fn test_power_to_db() {
    let S = arr2(&[[1.0], [0.25]]);
    let db = power_to_db(&S, None, None, None);
    assert!(db[[0, 0]] > db[[1, 0]]);
}

#[test]
fn test_db_to_power() {
    let S_db = arr2(&[[0.0], [-6.0]]);
    let power = db_to_power(&S_db, None);
    assert_abs_diff_eq!(power[[0, 0]], 1.0, epsilon = 0.01);
}

#[test]
fn test_A_weighting() {
    let freqs = vec![1000.0];
    let weights = A_weighting(&freqs, None);
    assert_abs_diff_eq!(weights[0], 0.0, epsilon = 1.0);
}