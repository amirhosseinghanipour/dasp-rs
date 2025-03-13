use ndarray::{Array1, Array2, Axis};

/// Computes delta coefficients of arbitrary order from a 2D array.
///
/// # Arguments
/// * `data` - Input 2D array (typically features × time)
/// * `width` - Optional window width for delta computation (defaults to 9)
/// * `order` - Optional order of delta (defaults to 1)
/// * `axis` - Optional axis along which to compute deltas (-1 for time, 0 for features; defaults to -1)
///
/// # Returns
/// Returns a 2D array of the same shape as `data` containing the delta coefficients.
///
/// # Examples
/// ```
/// use ndarray::Array2;
/// let data = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
/// let delta = delta(&data, None, None, None);
/// ```
pub fn delta(
    data: &Array2<f32>,
    width: Option<usize>,
    order: Option<usize>,
    axis: Option<isize>,
) -> Array2<f32> {
    let width = width.unwrap_or(9);
    let order = order.unwrap_or(1);
    let axis = axis.unwrap_or(-1);
    let mut result = data.to_owned();
    for _ in 0..order {
        let mut delta = Array2::zeros(result.dim());
        let half_width = width / 2;
        let weights: Vec<f32> = (1..=half_width).map(|i| i as f32).collect();
        let norm = weights.iter().map(|x| x.powi(2)).sum::<f32>();
        for i in 0..result.shape()[axis.unsigned_abs()] {
            let slice = result.index_axis(Axis(axis.unsigned_abs()), i);
            for j in 0..slice.len() {
                let mut sum = 0.0;
                for w in 0..weights.len() {
                    let left = (j as isize - w as isize - 1).max(0) as usize;
                    let right = (j + w + 1).min(slice.len() - 1);
                    sum += weights[w] * (slice[right] - slice[left]);
                }
                delta[[if axis < 0 { j } else { i }, if axis < 0 { i } else { j }]] = sum / norm;
            }
        }
        result = delta;
    }
    result
}

/// Stacks delayed copies of a 2D array for temporal context.
///
/// # Arguments
/// * `data` - Input 2D array (features × time)
/// * `n_steps` - Optional number of delayed copies (defaults to 2)
/// * `delay` - Optional delay between steps in frames (defaults to 1)
///
/// # Returns
/// Returns a 2D array of shape `(n_features * n_steps, n_frames)` containing stacked features.
///
/// # Examples
/// ```
/// use ndarray::Array2;
/// let data = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
/// let stacked = stack_memory(&data, None, None);
/// ```
pub fn stack_memory(
    data: &Array2<f32>,
    n_steps: Option<usize>,
    delay: Option<usize>,
) -> Array2<f32> {
    let n_steps = n_steps.unwrap_or(2);
    let delay = delay.unwrap_or(1);
    let n_frames = data.shape()[1];
    let n_features = data.shape()[0];
    let mut stacked = Array2::zeros((n_features * n_steps, n_frames));
    for step in 0..n_steps {
        let offset = step * delay;
        for t in 0..n_frames {
            let src_t = (t as isize - offset as isize).max(0) as usize;
            for f in 0..n_features {
                stacked[[f + step * n_features, t]] = data[[f, src_t]];
            }
        }
    }
    stacked
}

/// Computes temporal kurtosis from audio or spectrogram.
///
/// Kurtosis measures the "tailedness" of the distribution in each frame.
///
/// # Arguments
/// * `y` - Optional audio time series
/// * `S` - Optional spectrogram (features × time)
/// * `frame_length` - Optional frame length for audio (defaults to 2048)
/// * `hop_length` - Optional hop length for audio (defaults to frame_length/4)
///
/// # Returns
/// Returns a 1D array containing kurtosis values for each frame.
///
/// # Panics
/// Panics if neither `y` nor `S` is provided, or if both are provided.
///
/// # Examples
/// ```
/// let y = vec![0.1, 0.2, 0.3, 0.4, 0.5];
/// let kurtosis = temporal_kurtosis(Some(&y), None, None, None);
/// ```
pub fn temporal_kurtosis(
    y: Option<&[f32]>,
    S: Option<&Array2<f32>>,
    frame_length: Option<usize>,
    hop_length: Option<usize>,
) -> Array1<f32> {
    let frame_len = frame_length.unwrap_or(2048);
    let hop = hop_length.unwrap_or(frame_len / 4);
    match (y, S) {
        (Some(y), None) => {
            let n_frames = (y.len() - frame_len) / hop + 1;
            let mut kurtosis = Array1::zeros(n_frames);
            for i in 0..n_frames {
                let start = i * hop;
                let frame = &y[start..(start + frame_len).min(y.len())];
                let mean = frame.iter().sum::<f32>() / frame.len() as f32;
                let m2 = frame.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / frame.len() as f32;
                let m4 = frame.iter().map(|&x| (x - mean).powi(4)).sum::<f32>() / frame.len() as f32;
                kurtosis[i] = if m2 > 1e-10 { m4 / m2.powi(2) - 3.0 } else { 0.0 };
            }
            kurtosis
        }
        (None, Some(S)) => S.axis_iter(Axis(1)).map(|frame| {
            let mean = frame.mean().unwrap_or(0.0);
            let m2 = frame.mapv(|x| (x - mean).powi(2)).mean().unwrap_or(0.0);
            let m4 = frame.mapv(|x| (x - mean).powi(4)).mean().unwrap_or(0.0);
            if m2 > 1e-10 { m4 / m2.powi(2) - 3.0 } else { 0.0 }
        }).collect(),
        _ => panic!("Must provide either y or S"),
    }
}

/// Computes zero-crossing rate from an audio signal.
///
/// Measures the rate at which the signal changes sign in each frame.
///
/// # Arguments
/// * `y` - Input audio time series
/// * `frame_length` - Optional frame length (defaults to 2048)
/// * `hop_length` - Optional hop length (defaults to frame_length/4)
///
/// # Returns
/// Returns a 1D array containing zero-crossing rates for each frame.
///
/// # Examples
/// ```
/// let y = vec![1.0, -1.0, 2.0, -2.0, 1.0];
/// let zcr = zero_crossing_rate(&y, None, None);
/// ```
pub fn zero_crossing_rate(
    y: &[f32],
    frame_length: Option<usize>,
    hop_length: Option<usize>,
) -> Array1<f32> {
    let frame_len = frame_length.unwrap_or(2048);
    let hop = hop_length.unwrap_or(frame_len / 4);
    let n_frames = (y.len() - frame_len) / hop + 1;
    let mut zcr = Array1::zeros(n_frames);
    for i in 0..n_frames {
        let start = i * hop;
        let slice = &y[start..(start + frame_len).min(y.len())];
        let count = slice.windows(2).filter(|w| w[0] * w[1] < 0.0).count();
        zcr[i] = count as f32 / frame_len as f32;
    }
    zcr
}