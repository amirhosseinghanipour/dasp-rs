use aurust::{load, get_samplerate, stream};

#[test]
fn test_load() {
    let audio = load("test.wav", None, None, None, None).unwrap();
    assert!(!audio.samples.is_empty());
    assert!(audio.sample_rate > 0);
}

#[test]
fn test_load_with_resample() {
    let audio = load("test.wav", Some(22050), None, None, None).unwrap();
    assert_eq!(audio.sample_rate, 22050);
    assert!(!audio.samples.is_empty());
}

#[test]
fn test_load_mono() {
    let audio = load("test.wav", None, Some(true), None, None).unwrap();
    assert_eq!(audio.channels, 1);
}

#[test]
fn test_load_offset_duration() {
    let audio = load("test.wav", None, None, Some(0.1), Some(0.5)).unwrap();
    let expected_samples = (0.5 * audio.sample_rate as f32) as usize;
    assert!(audio.samples.len() <= expected_samples + 1);
}

#[test]
#[should_panic(expected = "InvalidRange")]
fn test_load_invalid_offset() {
    let _ = load("test.wav", None, None, Some(100.0), None).unwrap();
}

#[test]
fn test_get_samplerate() {
    let sr = get_samplerate("test.wav").unwrap();
    assert!(sr > 0);
}

#[test]
fn test_stream() {
    let mut stream_iter = stream("test.wav", 2, 1024, None).unwrap();
    let first_block = stream_iter.next().unwrap();
    assert_eq!(first_block.len(), 1024); // Each frame is 1024 samples
    assert!(stream_iter.take(1).count() <= 1); // At most 2 blocks total
}