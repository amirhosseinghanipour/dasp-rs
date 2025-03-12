use ndarray::Array1;

pub fn clicks(times: Option<&[f32]>, frames: Option<&[usize]>, sr: Option<u32>, hop_length: Option<usize>) -> Vec<f32> {
    let sample_rate = sr.unwrap_or(44100);
    let hop = hop_length.unwrap_or(512);
    let max_samples = times.map(|t| (t.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap() * sample_rate as f32) as usize)
        .or_else(|| frames.map(|f| f.iter().max().unwrap() * hop)).unwrap_or(44100);
    let mut signal = vec![0.0; max_samples];

    if let Some(ts) = times {
        for &t in ts {
            let idx = (t * sample_rate as f32) as usize;
            if idx < signal.len() { signal[idx] = 1.0; }
        }
    } else if let Some(fs) = frames {
        for &f in fs {
            let idx = f * hop;
            if idx < signal.len() { signal[idx] = 1.0; }
        }
    }
    signal
}

pub fn tone(frequency: f32, sr: Option<u32>, length: Option<usize>, duration: Option<f32>, phi: Option<f32>) -> Vec<f32> {
    let sample_rate = sr.unwrap_or(44100);
    let len = length.unwrap_or_else(|| (duration.unwrap_or(1.0) * sample_rate as f32) as usize);
    let phase = phi.unwrap_or(0.0);
    (0..len).map(|n| (2.0 * std::f32::consts::PI * frequency * n as f32 / sample_rate as f32 + phase).cos()).collect()
}

pub fn chirp(fmin: f32, fmax: f32, sr: Option<u32>, length: Option<usize>, duration: Option<f32>) -> Vec<f32> {
    let sample_rate = sr.unwrap_or(44100);
    let len = length.unwrap_or_else(|| (duration.unwrap_or(1.0) * sample_rate as f32) as usize);
    let t = Array1::linspace(0.0, 1.0, len);
    let freq = Array1::linspace(fmin, fmax, len);
    t.iter().zip(freq.iter()).map(|(&t, &f)| (2.0 * std::f32::consts::PI * f * t).cos()).collect()
}