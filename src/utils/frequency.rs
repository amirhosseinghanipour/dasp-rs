use ndarray::Array1;

pub fn hz_to_note(frequencies: &[f32]) -> Vec<String> {
    frequencies.iter().map(|&f| {
        let midi = hz_to_midi(&[f])[0];
        midi_to_note(&[midi], None, None, None)[0].clone()
    }).collect()
}

pub fn hz_to_midi(frequencies: &[f32]) -> Vec<f32> {
    frequencies.iter().map(|&f| 12.0 * (f / 440.0).log2() + 69.0).collect()
}

pub fn hz_to_svara_h(frequencies: &[f32], Sa: f32, abbr: Option<bool>) -> Vec<String> {
    let abbr = abbr.unwrap_or(false);
    let midi_Sa = hz_to_midi(&[Sa])[0];
    let midi_notes = hz_to_midi(frequencies);
    let svara_names = if abbr {
        vec!["S", "R1", "R2", "G1", "G2", "M1", "M2", "P", "D1", "D2", "N1", "N2"]
    } else {
        vec!["Shadjam", "Shuddha Rishabham", "Chatushruti Rishabham",
             "Shuddha Gandharam", "Sadharana Gandharam", "Shuddha Madhyamam",
             "Prati Madhyamam", "Panchamam", "Shuddha Dhaivatam", "Chatushruti Dhaivatam",
             "Shuddha Nishadam", "Kaisiki Nishadam"]
    };
    midi_notes.iter().map(|&m| {
        let degree = ((m - midi_Sa + 0.5).round() as i32 % 12 + 12) % 12;
        svara_names[degree as usize].to_string()
    }).collect()
}

pub fn hz_to_svara_c(_frequencies: &[f32], _Sa: f32, _mela: Option<usize>) -> Vec<String> { unimplemented!() }
pub fn hz_to_fjs(_frequencies: &[f32], _fmin: Option<f32>, _unison: Option<f32>) -> Vec<String> { unimplemented!() }

pub fn midi_to_hz(notes: &[f32]) -> Vec<f32> {
    notes.iter().map(|&n| 440.0 * 2.0f32.powf((n - 69.0) / 12.0)).collect()
}

pub fn midi_to_note(midi: &[f32], octave: Option<bool>, _cents: Option<bool>, _key: Option<&str>) -> Vec<String> {
    let note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
    midi.iter().map(|&m| {
        let note_idx = (m.round() as usize) % 12;
        let oct = if octave.unwrap_or(true) { format!("{}", (m.round() as i32 - 12) / 12) } else { "".to_string() };
        format!("{}{}", note_names[note_idx], oct)
    }).collect()
}

pub fn midi_to_svara_h(_midi: &[f32], _Sa: f32, _abbr: Option<bool>, _octave: Option<bool>) -> Vec<String> { unimplemented!() }
pub fn midi_to_svara_c(_midi: &[f32], _Sa: f32, _mela: Option<usize>, _abbr: Option<bool>) -> Vec<String> { unimplemented!() }

pub fn note_to_hz(note: &[&str]) -> Vec<f32> {
    note.iter().map(|&n| {
        let midi = note_to_midi(&[n], None)[0];
        midi_to_hz(&[midi])[0]
    }).collect()
}

pub fn note_to_midi(note: &[&str], round_midi: Option<bool>) -> Vec<f32> {
    let note_map = [("C", 0), ("C#", 1), ("Db", 1), ("D", 2), ("D#", 3), ("Eb", 3), ("E", 4), ("F", 5), ("F#", 6), ("Gb", 6), ("G", 7), ("G#", 8), ("Ab", 8), ("A", 9), ("A#", 10), ("Bb", 10), ("B", 11)];
    note.iter().map(|&n| {
        let (note_part, octave_part) = n.split_at(n.find(|c: char| c.is_digit(10)).unwrap_or(n.len()));
        let note_val = note_map.iter().find(|&&(name, _)| name == note_part).map(|&(_, val)| val).unwrap_or(0) as f32;
        let octave = octave_part.parse::<i32>().unwrap_or(4);
        let midi = note_val + (octave + 1) as f32 * 12.0;
        if round_midi.unwrap_or(true) { midi.round() } else { midi }
    }).collect()
}

pub fn note_to_svara_h(_notes: &[&str], _Sa: f32, _abbr: Option<bool>) -> Vec<String> { unimplemented!() }
pub fn note_to_svara_c(_notes: &[&str], _Sa: f32, _mela: Option<usize>, _abbr: Option<bool>) -> Vec<String> { unimplemented!() }

pub fn hz_to_mel(frequencies: &[f32], htk: Option<bool>) -> Vec<f32> {
    if htk.unwrap_or(false) {
        frequencies.iter().map(|&f| 2595.0 * (1.0 + f / 700.0).log10()).collect()
    } else {
        frequencies.iter().map(|&f| 1125.0 * (1.0 + f / 700.0).ln()).collect()
    }
}

pub fn hz_to_octs(frequencies: &[f32], tuning: Option<f32>) -> Vec<f32> {
    let tune = tuning.unwrap_or(0.0);
    frequencies.iter().map(|&f| (f / (440.0 * 2.0f32.powf(tune / 12.0))).log2() + 4.0).collect()
}

pub fn mel_to_hz(mels: &[f32], htk: Option<bool>) -> Vec<f32> {
    if htk.unwrap_or(false) {
        mels.iter().map(|&m| 700.0 * (10.0f32.powf(m / 2595.0) - 1.0)).collect()
    } else {
        mels.iter().map(|&m| 700.0 * (m / 1125.0).exp() - 700.0).collect()
    }
}

pub fn octs_to_hz(octs: &[f32], tuning: Option<f32>, _bins_per_octave: Option<usize>) -> Vec<f32> {
    let tune = tuning.unwrap_or(0.0);
    octs.iter().map(|&o| 440.0 * 2.0f32.powf(o - 4.0 + tune / 12.0)).collect()
}

pub fn A4_to_tuning(A4: f32, _bins_per_octave: Option<usize>) -> f32 {
    12.0 * (A4 / 440.0).log2()
}

pub fn tuning_to_A4(tuning: f32, _bins_per_octave: Option<usize>) -> f32 {
    440.0 * 2.0f32.powf(tuning / 12.0)
}

pub fn fft_frequencies(sr: Option<u32>, n_fft: Option<usize>) -> Vec<f32> {
    let sample_rate = sr.unwrap_or(44100);
    let n = n_fft.unwrap_or(2048);
    Array1::linspace(0.0, sample_rate as f32 / 2.0, n / 2 + 1).to_vec()
}

pub fn cqt_frequencies(_n_bins: usize, _fmin: Option<f32>) -> Vec<f32> { unimplemented!() }
pub fn mel_frequencies(n_mels: Option<usize>, fmin: Option<f32>, fmax: Option<f32>, _htk: Option<bool>) -> Vec<f32> {
    let n = n_mels.unwrap_or(128);
    let min_freq = fmin.unwrap_or(0.0);
    let max_freq = fmax.unwrap_or(11025.0);
    let min_mel = hz_to_mel(&[min_freq], None)[0];
    let max_mel = hz_to_mel(&[max_freq], None)[0];
    let mel_steps = Array1::linspace(min_mel, max_mel, n);
    mel_to_hz(&mel_steps.to_vec(), None)
}

pub fn tempo_frequencies(_n_bins: usize, _hop_length: Option<usize>, _sr: Option<u32>) -> Vec<f32> { unimplemented!() }
pub fn fourier_tempo_frequencies(_sr: Option<u32>) -> Vec<f32> { unimplemented!() }