use aurust::{clicks, tone, chirp};

#[test]
fn test_clicks() {
    let signal = clicks(Some(&[0.1, 0.2]), None, None, None);
    assert!(!signal.is_empty());
    assert_eq!(signal[(0.1 * 44100.0) as usize], 1.0);
}

#[test]
fn test_tone() {
    let signal = tone(440.0, None, Some(10), None, None);
    assert_eq!(signal.len(), 10);
}

#[test]
fn test_chirp() {
    let signal = chirp(440.0, 880.0, None, Some(10), None);
    assert_eq!(signal.len(), 10);
}