use ndarray::{Array2, Axis};

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