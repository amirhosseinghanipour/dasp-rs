use crate::audio_io::AudioError;

pub fn autocorrelate(y: &[f32], max_size: Option<usize>, axis: Option<isize>) -> Vec<f32> {
    let max_lag = max_size.unwrap_or(y.len());
    let mut result = Vec::with_capacity(max_lag);
    for lag in 0..max_lag {
        let mut sum = 0.0;
        for i in 0..(y.len() - lag) {
            sum += y[i] * y[i + lag];
        }
        result.push(sum);
    }
    result
}

pub fn lpc(y: &[f32], order: usize) -> Result<Vec<f32>, AudioError> {
    if y.len() <= order {
        return Err(AudioError::InvalidRange);
    }
    let mut r = autocorrelate(y, Some(order + 1), None);
    let mut a = vec![0.0; order + 1];
    a[0] = 1.0;
    let mut e = r[0];

    for i in 1..=order {
        let mut k = 0.0;
        for j in 0..i {
            k += a[j] * r[i - j];
        }
        k = -k / e;
        for j in 0..i {
            a[j] -= k * a[i - 1 - j];
        }
        a[i] = k;
        e *= 1.0 - k * k;
    }
    Ok(a)
}

pub fn zero_crossings(y: &[f32], threshold: Option<f32>, pad: Option<bool>) -> Vec<usize> {
    let thresh = threshold.unwrap_or(0.0);
    let mut crossings = Vec::new();
    let mut prev_sign = y[0] >= thresh;
    for (i, &sample) in y.iter().enumerate().skip(1) {
        let sign = sample >= thresh;
        if sign != prev_sign {
            crossings.push(i);
        }
        prev_sign = sign;
    }
    if pad.unwrap_or(false) && crossings.is_empty() {
        crossings.push(0);
    }
    crossings
}

pub fn mu_compress(x: &[f32], mu: Option<f32>, quantize: Option<bool>) -> Vec<f32> {
    let mu_val = mu.unwrap_or(255.0);
    x.iter().map(|&v| {
        let sign = if v >= 0.0 { 1.0 } else { -1.0 };
        let compressed = sign * (1.0 + mu_val.abs() * v.abs()).ln() / mu_val.ln();
        if quantize.unwrap_or(false) {
            (compressed * 255.0).round() / 255.0
        } else {
            compressed
        }
    }).collect()
}

pub fn mu_expand(x: &[f32], mu: Option<f32>, quantize: Option<bool>) -> Vec<f32> {
    let mu_val = mu.unwrap_or(255.0);
    x.iter().map(|&v| {
        let sign = if v >= 0.0 { 1.0 } else { -1.0 };
        sign * (mu_val.ln() * v.abs()).exp() / mu_val
    }).collect()
}