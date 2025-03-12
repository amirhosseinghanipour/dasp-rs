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

pub fn mela_to_svara(mela: usize, abbr: Option<bool>, unicode: Option<bool>) -> Vec<String> {
    let abbr = abbr.unwrap_or(false);
    let unicode = unicode.unwrap_or(false);
    let degrees = mela_to_degrees(mela);
    let svara_full = if unicode {
        vec!["ṣaḍjam", "ṛṣabham", "gāndhāram", "madhyamam", "pañcamam", "dhaivatam", "niṣādam"]
    } else {
        vec!["shadjam", "rishabham", "gandharam", "madhyamam", "panchamam", "dhaivatam", "nishadam"]
    };
    let svara_abbr = ["S", "R", "G", "M", "P", "D", "N"];
    let mut result = Vec::new();
    for (i, &deg) in degrees.iter().enumerate() {
        let base = match i {
            0 => "S", 1..=3 => "R", 4..=6 => "G", 7 => "M", 8 => "P", 9..=11 => "D", 12..=14 => "N",
            _ => "S",
        };
        let variant = match deg % 12 {
            1 => "1", 2 => "2", 3 => "3", 5 => "1", 6 => "2", 7 => "3", 8 => "1", 9 => "2", 10 => "3", _ => "",
        };
        let name = if abbr {
            format!("{}{}", base, variant)
        } else {
            let idx = match base {
                "S" => 0, "R" => 1, "G" => 2, "M" => 3, "P" => 4, "D" => 5, "N" => 6, _ => 0,
            };
            format!("{}{}", svara_full[idx], if variant.is_empty() { "" } else { variant })
        };
        result.push(name);
    }
    result
}

pub fn mela_to_degrees(mela: usize) -> Vec<usize> {
    if !(1..=72).contains(&mela) { return vec![0, 2, 4, 5, 7, 9, 11]; }
    let mela = mela - 1;
    let r = (mela / 36) % 2;
    let g = (mela / 18) % 2;
    let m = (mela / 9) % 2;
    let d = (mela / 3) % 3;
    let n = mela % 3;
    vec![
        0,
        if r == 0 { 1 } else { 2 + g },
        if r == 0 { 2 + g } else { 4 },
        5 + m,
        7,
        8 + d,
        10 + n,
    ]
}

pub fn thaat_to_degrees(thaat: &str) -> Vec<usize> {
    match thaat.to_lowercase().as_str() {
        "bilaval" => vec![0, 2, 4, 5, 7, 9, 11],
        "kalyani" => vec![0, 2, 4, 6, 7, 9, 11],
        "khamaj" => vec![0, 2, 4, 5, 7, 9, 10],
        "bhairav" => vec![0, 1, 4, 5, 6, 9, 11],
        "purvi" => vec![0, 1, 4, 6, 7, 9, 11],
        "marwa" => vec![0, 1, 3, 6, 7, 9, 11],
        "kafi" => vec![0, 2, 3, 5, 7, 9, 10],
        "asavari" => vec![0, 2, 3, 5, 7, 8, 10],
        "todi" => vec![0, 1, 3, 6, 7, 8, 11],
        "bhoopali" => vec![0, 2, 4, 7, 9],
        _ => vec![0, 2, 4, 5, 7, 9, 11],
    }
}

pub fn list_mela() -> Vec<(usize, String)> {
    let names = vec![
        "Kanakangi", "Ratnangi", "Ganamurti", "Vanaspati", "Manavati", "Tanarupi",
        "Senavati", "Hanumatodi", "Dhenuka", "Natakapriya", "Kokilapriya", "Rupavati",
        "Gayakapriya", "Vakulabharanam", "Mayamalavagowla", "Chakravakam", "Suryakantam",
        "Hatakambari", "Jhankaradhwani", "Natabhairavi", "Keeravani", "Kharaharapriya",
        "Gourimanohari", "Varunapriya", "Mararanjani", "Charukesi", "Sarasangi",
        "Harikambhoji", "Dheerasankarabharanam", "Naganandini", "Yagapriya", "Ragavardhini",
        "Gangeyabhushani", "Vagadheeswari", "Shulini", "Chalanata", "Salagam", "Jalarnavam",
        "Jhalavarali", "Navaneetam", "Pavani", "Raghupriya", "Gavambodhi", "Bhavapriya",
        "Shubhapantuvarali", "Shadvidamargini", "Suvarnangi", "Divyamani", "Dhavalambari",
        "Namanarayani", "Kamavardhini", "Ramapriya", "Gamanashrama", "Vishwambari",
        "Shamalangi", "Shanmukhapriya", "Simhendramadhyamam", "Hemavati", "Dharmavati",
        "Neetimati", "Kantamani", "Rishabhapriya", "Latangi", "Vachaspati", "Mechakalyani",
        "Chitrambari", "Sucharitra", "Jyotiswarupini", "Dhatuvardhani", "Nasikabhushani",
        "Kosalam", "Rasikapriya",
    ];
    names.into_iter().enumerate().map(|(i, name)| (i + 1, name.to_string())).collect()
}

pub fn list_thaat() -> Vec<String> {
    vec![
        "Bilaval".to_string(),
        "Kalyani".to_string(),
        "Khamaj".to_string(),
        "Bhairav".to_string(),
        "Purvi".to_string(),
        "Marwa".to_string(),
        "Kafi".to_string(),
        "Asavari".to_string(),
        "Todi".to_string(),
        "Bhoopali".to_string(),
    ]
}

pub fn fifths_to_note(unison: &str, fifths: i32, unicode: Option<bool>) -> String {
    let unicode = unicode.unwrap_or(false);
    let semitones = (fifths * 7) % 12;
    let octave_shift = (fifths * 7) / 12;
    let base = match unison.to_lowercase().as_str() {
        "c" => 0, "c#" | "db" => 1, "d" => 2, "d#" | "eb" => 3, "e" => 4,
        "f" => 5, "f#" | "gb" => 6, "g" => 7, "g#" | "ab" => 8, "a" => 9,
        "a#" | "bb" => 10, "b" => 11, _ => 0,
    };
    let note_idx = (base + semitones + 12) % 12;
    let note = match note_idx {
        0 => "C", 1 => if unicode { "C♯" } else { "C#" }, 2 => "D",
        3 => if unicode { "D♯" } else { "D#" }, 4 => "E", 5 => "F",
        6 => if unicode { "F♯" } else { "F#" }, 7 => "G",
        8 => if unicode { "G♯" } else { "G#" }, 9 => "A",
        10 => if unicode { "A♯" } else { "A#" }, 11 => "B",
        _ => "C",
    };
    format!("{}{}", note, if octave_shift != 0 { octave_shift.to_string() } else { "".to_string() })
}

pub fn interval_to_fjs(interval: f32, unison: Option<f32>) -> String {
    let unison = unison.unwrap_or(1.0);
    let ratio = interval / unison;
    match ratio {
        r if (r - 1.0).abs() < 1e-6 => "1/1".to_string(),
        r if (r - 3.0/2.0).abs() < 1e-6 => "3/2".to_string(),
        r if (r - 4.0/3.0).abs() < 1e-6 => "4/3".to_string(),
        r if (r - 5.0/4.0).abs() < 1e-6 => "5/4".to_string(),
        r if (r - 6.0/5.0).abs() < 1e-6 => "6/5".to_string(),
        _ => format!("{:.2}/1", ratio),
    }
}

pub fn interval_frequencies(n_bins: usize, fmin: f32, intervals: &[f32]) -> Vec<f32> {
    let mut freqs = Vec::with_capacity(n_bins);
    let mut f = fmin;
    let mut interval_idx = 0;
    for _ in 0..n_bins {
        freqs.push(f);
        f *= intervals[interval_idx % intervals.len()];
        interval_idx += 1;
    }
    freqs
}

pub fn pythagorean_intervals(bins_per_octave: Option<usize>) -> Vec<f32> {
    let bins = bins_per_octave.unwrap_or(12);
    let mut intervals = Vec::with_capacity(bins);
    let fifth = 3.0 / 2.0;
    let mut ratio = 1.0;
    for i in 0..bins {
        intervals.push(ratio);
        ratio *= if i % 2 == 0 { fifth } else { 1.0 / fifth };
        while ratio > 2.0 { ratio /= 2.0; }
        while ratio < 1.0 { ratio *= 2.0; }
    }
    intervals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    intervals
}

pub fn plimit_intervals(primes: &[usize]) -> Vec<f32> {
    let mut intervals = vec![1.0];
    for &p in primes {
        let mut new_intervals = Vec::new();
        for &i in &intervals {
            let mut n = i;
            while n < 2.0 {
                new_intervals.push(n);
                n *= p as f32;
            }
            let mut d = i;
            while d > 0.5 {
                new_intervals.push(d);
                d /= p as f32;
            }
        }
        intervals.extend(new_intervals);
    }
    intervals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    intervals.dedup_by(|a, b| (*a - *b).abs() < 1e-6);
    intervals.retain(|&x| (1.0..=2.0).contains(&x));
    intervals
}