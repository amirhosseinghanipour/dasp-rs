pub fn key_to_notes(key: &str, _unicode: Option<bool>, _natural: Option<bool>) -> Vec<String> {
    let notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
    notes.iter().map(|&n| n.to_string()).collect()
}

pub fn key_to_degrees(_key: &str) -> Vec<usize> { unimplemented!() }
pub fn mela_to_svara(_mela: usize, _abbr: Option<bool>, _unicode: Option<bool>) -> Vec<String> { unimplemented!() }
pub fn mela_to_degrees(_mela: usize) -> Vec<usize> { unimplemented!() }
pub fn thaat_to_degrees(_thaat: &str) -> Vec<usize> { unimplemented!() }
pub fn list_mela() -> Vec<(usize, String)> { unimplemented!() }
pub fn list_thaat() -> Vec<String> { unimplemented!() }
pub fn fifths_to_note(_unison: &str, _fifths: i32, _unicode: Option<bool>) -> String { unimplemented!() }
pub fn interval_to_fjs(_interval: f32, _unison: Option<f32>) -> String { unimplemented!() }
pub fn interval_frequencies(_n_bins: usize, _fmin: f32, _intervals: &[f32]) -> Vec<f32> { unimplemented!() }
pub fn pythagorean_intervals(_bins_per_octave: Option<usize>) -> Vec<f32> { unimplemented!() }
pub fn plimit_intervals(_primes: &[usize]) -> Vec<f32> { unimplemented!() }