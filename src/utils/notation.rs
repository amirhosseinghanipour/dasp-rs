pub fn key_to_notes(key: &str, _unicode: Option<bool>, _natural: Option<bool>) -> Vec<String> {
    let notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
    notes.iter().map(|&n| n.to_string()).collect()
}

pub fn key_to_degrees(key: &str) -> Vec<usize> {
    let key = key.to_lowercase();
    let (tonic, mode) = key.split_once(':').unwrap_or((&key, "maj"));
    let tonic_shift = match tonic {
        "c" => 0, "c#" | "db" => 1, "d" => 2, "d#" | "eb" => 3, "e" => 4,
        "f" => 5, "f#" | "gb" => 6, "g" => 7, "g#" | "ab" => 8, "a" => 9,
        "a#" | "bb" => 10, "b" => 11, _ => 0,
    };
    let major = vec![0, 2, 4, 5, 7, 9, 11];
    let minor = vec![0, 2, 3, 5, 7, 8, 10];
    let degrees = match mode {
        "maj" | "major" => major,
        "min" | "minor" => minor,
        _ => major,
    };
    degrees.into_iter().map(|d| (d + tonic_shift) % 12).collect()
}
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