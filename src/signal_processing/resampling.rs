use rubato::{Resampler, SincFixedIn, SincInterpolationType, SincInterpolationParameters, WindowFunction};
use thiserror::Error;
use crate::audio_io::AudioError;

#[derive(Error, Debug)]
pub enum ResampleError {
    #[error("Resampling failed: {0}")]
    RubatoError(String),
}

pub fn resample(samples: &[f32], orig_sr: u32, target_sr: u32) -> Result<Vec<f32>, AudioError> {
    if orig_sr == target_sr {
        return Ok(samples.to_vec());
    }

    if samples.is_empty() {
        return Ok(vec![]);
    }

    let ratio = target_sr as f64 / orig_sr as f64;

    let sinc_len = 256;
    let f_cutoff = 0.95;
    let oversampling_factor = 256;
    let interpolation_params = SincInterpolationParameters {
        sinc_len,
        f_cutoff,
        oversampling_factor,
        interpolation: SincInterpolationType::Linear,
        window: WindowFunction::BlackmanHarris,
    };

    let mut resampler = SincFixedIn::<f32>::new(
        ratio,
        1.0,
        interpolation_params,
        samples.len(),
        1,
    ).map_err(|e: rubato::ResamplerConstructionError| ResampleError::RubatoError(format!("Resampler initialization failed: {}", e)))?;

    let input = vec![samples.to_vec()];

    let output = resampler.process(&input, None)
        .map_err(|e| ResampleError::RubatoError(format!("Resampling failed: {}", e)))?;
    Ok(output[0].clone())
}