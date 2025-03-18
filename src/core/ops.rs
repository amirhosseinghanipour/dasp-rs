use crate::core::AudioData;
use thiserror::Error;
use rayon::prelude::*;

/// Enumerates error conditions for signal operation failures in DSP workflows.
///
/// Provides detailed diagnostics for binary and scalar operations on audio signals,
/// tailored for debugging and error recovery in production-grade audio processing pipelines.
#[derive(Error, Debug)]
pub enum SignalOpError {
    /// Signals have incompatible sample lengths for binary operations.
    #[error("Sample length mismatch: {0} vs {0}")]
    LengthMismatch(usize, usize),

    /// Division by zero encountered at a specific sample index.
    #[error("Division by zero at sample index {0}")]
    DivisionByZero(usize),

    /// Input validation failure (e.g., empty signal array, mismatched metadata).
    #[error("Invalid input parameter: {0}")]
    InvalidInput(String),

    /// Numerical computation failure (e.g., overflow, NaN result).
    #[error("Computation failed: {0}")]
    ComputationFailed(String),
}

/// Mixes multiple audio signals by averaging their samples in parallel.
///
/// Computes the sample-wise mean of an array of `AudioData` signals, producing a new
/// `AudioData` instance. All signals must share identical sample lengths, sample rates,
/// and channel counts. Parallelized using `rayon` for multi-core efficiency.
///
/// # Parameters
/// - `signals`: Slice of `AudioData` references to mix.
///
/// # Returns
/// - `Ok(AudioData)`: Mixed signal with averaged samples.
/// - `Err(SignalOpError)`: Failure due to empty input, length mismatch, or metadata inconsistency.
pub fn mix_signals(signals: &[&AudioData]) -> Result<AudioData, SignalOpError> {
    if signals.is_empty() {
        return Err(SignalOpError::InvalidInput("Signal array is empty".to_string()));
    }

    let length = signals[0].samples.len();
    let sample_rate = signals[0].sample_rate;
    let channels = signals[0].channels;

    for &signal in signals.iter().skip(1) {
        if signal.samples.len() != length {
            return Err(SignalOpError::LengthMismatch(length, signal.samples.len()));
        }
        if signal.sample_rate != sample_rate || signal.channels != channels {
            return Err(SignalOpError::InvalidInput(
                format!(
                    "Metadata mismatch: expected {} Hz, {} channels; got {} Hz, {} channels",
                    sample_rate, channels, signal.sample_rate, signal.channels
                )
            ));
        }
    }

    let mixed_samples: Vec<f32> = (0..length)
        .into_par_iter()
        .map(|i| {
            let sum: f32 = signals.iter().map(|s| s.samples[i]).sum();
            sum / signals.len() as f32
        })
        .collect();

    Ok(AudioData::new(mixed_samples, sample_rate, channels))
}

/// Subtracts one audio signal from another with parallel sample processing.
///
/// Performs sample-wise subtraction (`signal1 - signal2`), producing a new `AudioData`.
/// Signals must have identical sample lengths, sample rates, and channel counts.
///
/// # Parameters
/// - `signal1`: Base signal (minuend).
/// - `signal2`: Signal to subtract (subtrahend).
///
/// # Returns
/// - `Ok(AudioData)`: Resulting difference signal.
/// - `Err(SignalOpError)`: Failure due to length or metadata mismatch.
pub fn subtract_signals(signal1: &AudioData, signal2: &AudioData) -> Result<AudioData, SignalOpError> {
    if signal1.samples.len() != signal2.samples.len() {
        return Err(SignalOpError::LengthMismatch(signal1.samples.len(), signal2.samples.len()));
    }
    if signal1.sample_rate != signal2.sample_rate || signal1.channels != signal2.channels {
        return Err(SignalOpError::InvalidInput(
            format!(
                "Metadata mismatch: expected {} Hz, {} channels; got {} Hz, {} channels",
                signal1.sample_rate, signal1.channels, signal2.sample_rate, signal2.channels
            )
        ));
    }

    let samples: Vec<f32> = signal1.samples
        .par_iter()
        .zip(&signal2.samples)
        .map(|(&s1, &s2)| s1 - s2)
        .collect();

    Ok(AudioData::new(samples, signal1.sample_rate, signal1.channels))
}

/// Multiplies two audio signals sample-wise in parallel (e.g., amplitude modulation).
///
/// Computes the product of corresponding samples from `signal1` and `signal2`, producing
/// a new `AudioData`. Suitable for modulation effects. Signals must match in length,
/// sample rate, and channels.
///
/// # Parameters
/// - `signal1`: First signal (carrier or base).
/// - `signal2`: Second signal (modulator).
///
/// # Returns
/// - `Ok(AudioData)`: Product signal.
/// - `Err(SignalOpError)`: Failure due to length or metadata mismatch.
pub fn multiply_signals(signal1: &AudioData, signal2: &AudioData) -> Result<AudioData, SignalOpError> {
    if signal1.samples.len() != signal2.samples.len() {
        return Err(SignalOpError::LengthMismatch(signal1.samples.len(), signal2.samples.len()));
    }
    if signal1.sample_rate != signal2.sample_rate || signal1.channels != signal2.channels {
        return Err(SignalOpError::InvalidInput(
            format!(
                "Metadata mismatch: expected {} Hz, {} channels; got {} Hz, {} channels",
                signal1.sample_rate, signal1.channels, signal2.sample_rate, signal2.channels
            )
        ));
    }

    let samples: Vec<f32> = signal1.samples
        .par_iter()
        .zip(&signal2.samples)
        .map(|(&s1, &s2)| s1 * s2)
        .collect();

    if samples.iter().any(|&s| !s.is_finite()) {
        return Err(SignalOpError::ComputationFailed("Non-finite result detected".to_string()));
    }

    Ok(AudioData::new(samples, signal1.sample_rate, signal1.channels))
}

/// Divides one audio signal by another with parallel processing and zero handling.
///
/// Performs sample-wise division (`signal1 / signal2`), producing a new `AudioData`.
/// Handles division by zero by clamping to 0.0 and logs a warning. Signals must match
/// in length, sample rate, and channels.
///
/// # Parameters
/// - `signal1`: Numerator signal.
/// - `signal2`: Denominator signal.
///
/// # Returns
/// - `Ok(AudioData)`: Quotient signal.
/// - `Err(SignalOpError)`: Failure due to length or metadata mismatch.
pub fn divide_signals(signal1: &AudioData, signal2: &AudioData) -> Result<AudioData, SignalOpError> {
    if signal1.samples.len() != signal2.samples.len() {
        return Err(SignalOpError::LengthMismatch(signal1.samples.len(), signal2.samples.len()));
    }
    if signal1.sample_rate != signal2.sample_rate || signal1.channels != signal2.channels {
        return Err(SignalOpError::InvalidInput(
            format!(
                "Metadata mismatch: expected {} Hz, {} channels; got {} Hz, {} channels",
                signal1.sample_rate, signal1.channels, signal2.sample_rate, signal2.channels
            )
        ));
    }

    let samples: Vec<f32> = signal1.samples
        .par_iter()
        .zip(&signal2.samples)
        .enumerate()
        .map(|(i, (&s1, &s2))| {
            if s2 == 0.0 {
                eprintln!("Warning: Division by zero at index {}, clamping to 0.0", i);
                0.0
            } else {
                s1 / s2
            }
        })
        .collect();

    if samples.iter().any(|&s| !s.is_finite()) {
        return Err(SignalOpError::ComputationFailed("Non-finite result detected".to_string()));
    }

    Ok(AudioData::new(samples, signal1.sample_rate, signal1.channels))
}

/// Applies a scalar operation to an audio signal in parallel.
///
/// Performs element-wise addition, subtraction, multiplication, or division between
/// a signal’s samples and a scalar value, producing a new `AudioData`. Division by zero
/// is explicitly rejected.
///
/// # Parameters
/// - `signal`: Input signal.
/// - `scalar`: Scalar value for operation.
/// - `op`: Operation type: `"add"`, `"subtract"`, `"multiply"`, or `"divide"`.
///
/// # Returns
/// - `Ok(AudioData)`: Resulting signal.
/// - `Err(SignalOpError)`: Failure due to invalid operation or division by zero.
pub fn scalar_operation(signal: &AudioData, scalar: f32, op: &str) -> Result<AudioData, SignalOpError> {
    let samples: Vec<f32> = match op.to_lowercase().as_str() {
        "add" => signal.samples.par_iter().map(|&s| s + scalar).collect(),
        "subtract" => signal.samples.par_iter().map(|&s| s - scalar).collect(),
        "multiply" => signal.samples.par_iter().map(|&s| s * scalar).collect(),
        "divide" => {
            if scalar == 0.0 {
                return Err(SignalOpError::DivisionByZero(0));
            }
            signal.samples.par_iter().map(|&s| s / scalar).collect()
        }
        _ => return Err(SignalOpError::InvalidInput(format!("Unsupported operation: {}", op))),
    };

    if samples.iter().any(|&s| !s.is_finite()) {
        return Err(SignalOpError::ComputationFailed("Non-finite result detected".to_string()));
    }

    Ok(AudioData::new(samples, signal.sample_rate, signal.channels))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_signal(samples: Vec<f32>) -> AudioData {
        AudioData::new(samples, 44100, 1)
    }

    #[test]
    fn test_mix_signals_basic() {
        let s1 = test_signal(vec![1.0, 2.0, 3.0]);
        let s2 = test_signal(vec![2.0, 4.0, 6.0]);
        let mixed = mix_signals(&[&s1, &s2]).unwrap();
        assert_eq!(mixed.samples, vec![1.5, 3.0, 4.5]);
        assert_eq!(mixed.sample_rate, 44100);
        assert_eq!(mixed.channels, 1);
    }

    #[test]
    fn test_mix_signals_empty() {
        let result = mix_signals(&[]);
        assert!(matches!(result, Err(SignalOpError::InvalidInput(_))));
    }

    #[test]
    fn test_mix_signals_length_mismatch() {
        let s1 = test_signal(vec![1.0, 2.0]);
        let s2 = test_signal(vec![2.0, 4.0, 6.0]);
        let result = mix_signals(&[&s1, &s2]);
        assert!(matches!(result, Err(SignalOpError::LengthMismatch(2, 3))));
    }

    #[test]
    fn test_subtract_signals() {
        let s1 = test_signal(vec![2.0, 4.0, 6.0]);
        let s2 = test_signal(vec![1.0, 2.0, 3.0]);
        let result = subtract_signals(&s1, &s2).unwrap();
        assert_eq!(result.samples, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_subtract_signals_mismatch() {
        let s1 = test_signal(vec![2.0, 4.0]);
        let s2 = test_signal(vec![1.0, 2.0, 3.0]);
        let result = subtract_signals(&s1, &s2);
        assert!(matches!(result, Err(SignalOpError::LengthMismatch(2, 3))));
    }

    #[test]
    fn test_multiply_signals() {
        let s1 = test_signal(vec![1.0, 2.0, 3.0]);
        let s2 = test_signal(vec![2.0, 2.0, 2.0]);
        let result = multiply_signals(&s1, &s2).unwrap();
        assert_eq!(result.samples, vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_multiply_signals_overflow() {
        let s1 = test_signal(vec![f32::MAX, 2.0]);
        let s2 = test_signal(vec![2.0, 2.0]);
        let result = multiply_signals(&s1, &s2);
        assert!(matches!(result, Err(SignalOpError::ComputationFailed(_))));
    }

    #[test]
    fn test_divide_signals() {
        let s1 = test_signal(vec![4.0, 6.0, 8.0]);
        let s2 = test_signal(vec![2.0, 0.0, 4.0]);
        let result = divide_signals(&s1, &s2).unwrap();
        assert_eq!(result.samples, vec![2.0, 0.0, 2.0]);
    }

    #[test]
    fn test_divide_signals_infinity() {
        let s1 = test_signal(vec![f32::MAX, 1.0]);
        let s2 = test_signal(vec![0.001, 1.0]);
        let result = divide_signals(&s1, &s2);
        assert!(matches!(result, Err(SignalOpError::ComputationFailed(_))));
    }

    #[test]
    fn test_scalar_operation_add() {
        let s = test_signal(vec![1.0, 2.0, 3.0]);
        let result = scalar_operation(&s, 1.0, "add").unwrap();
        assert_eq!(result.samples, vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_scalar_operation_multiply() {
        let s = test_signal(vec![1.0, 2.0, 3.0]);
        let result = scalar_operation(&s, 2.0, "multiply").unwrap();
        assert_eq!(result.samples, vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_scalar_operation_divide_by_zero() {
        let s = test_signal(vec![1.0, 2.0]);
        let result = scalar_operation(&s, 0.0, "divide");
        assert!(matches!(result, Err(SignalOpError::DivisionByZero(0))));
    }

    #[test]
    fn test_scalar_operation_invalid_op() {
        let s = test_signal(vec![1.0, 2.0]);
        let result = scalar_operation(&s, 1.0, "invalid");
        assert!(matches!(result, Err(SignalOpError::InvalidInput(_))));
    }
}