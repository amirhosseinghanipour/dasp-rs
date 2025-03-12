use aurust::{get_duration, frames_to_samples, hz_to_midi, midi_to_hz, note_to_midi, hz_to_mel};
use approx::assert_abs_diff_eq;

#[test]
fn test_get_duration() {
    let audio = aurust::load("test.wav", None, None, None, None).unwrap();
    let duration = get_duration(&audio);
    assert!(duration > 0.0);
}

#[test]
fn test_frames_to_samples() {
    let frames = vec![0, 1, 2];
    let samples = frames_to_samples(&frames, None, None);
    assert_eq!(samples, vec![0, 512, 1024]);
}

#[test]
fn test_hz_to_midi() {
    let freqs = vec![440.0];
    let midi = hz_to_midi(&freqs);
    assert_abs_diff_eq!(midi[0], 69.0, epsilon = 0.01);
}

#[test]
fn test_midi_to_hz() {
    let midi = vec![69.0];
    let freqs = midi_to_hz(&midi);
    assert_abs_diff_eq!(freqs[0], 440.0, epsilon = 0.01);
}

#[test]
fn test_note_to_midi() {
    let notes = vec!["A4"];
    let midi = note_to_midi(&notes, None);
    assert_abs_diff_eq!(midi[0], 69.0, epsilon = 0.01);
}

#[test]
fn test_hz_to_mel() {
    let freqs = vec![1000.0];
    let mels = hz_to_mel(&freqs, None);
    assert!(mels[0] > 0.0);
}