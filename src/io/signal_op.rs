use crate::io::core::AudioData;
use thiserror::Error;

/// Custom error types for signal operation failures.
///
/// This enum defines errors that can occur during signal operations such as
/// mismatched lengths, division by zero, or invalid input.
#[derive(Error, Debug)]
pub enum SignalOpError {
    /// Error when signal lengths do not match for binary operations.
    #[error("Signal lengths mismatch: {0} vs {1}")]
    LengthMismatch(usize, usize),

    /// Error when attempting to divide by zero in signal division.
    #[error("Division by zero detected at index {0}")]
    DivisionByZero(usize),

    /// Error when input signals are invalid (e.g., empty).
    #[error("Invalid input: {0}")]
    InvalidInput(String),
}

/// Mixes two or more audio signals together by averaging their samples.
///
/// This function takes a slice of `AudioData` references and produces a new `AudioData`
/// where each sample is the average of the corresponding samples from all input signals.
/// All signals must have the same length, sample rate, and number of channels.
///
/// # Arguments
/// * `signals` - A slice of `AudioData` references to mix.
///
/// # Returns
/// Returns `Result<AudioData, SignalOpError>` containing the mixed signal or an error.
///
/// # Examples
/// ```
/// let signal1 = AudioData { samples: vec![1.0, 2.0, 3.0], sample_rate: 44100, channels: 1 };
/// let signal2 = AudioData { samples: vec![2.0, 4.0, 6.0], sample_rate: 44100, channels: 1 };
/// let mixed = mix_signals(&[signal1, signal2])?;
/// assert_eq!(mixed.samples, vec![1.5, 3.0, 4.5]);
/// ```
pub fn mix_signals(signals: &[&AudioData]) -> Result<AudioData, SignalOpError> {
    if signals.is_empty() {
        return Err(SignalOpError::InvalidInput("No signals provided".to_string()));
    }

    let length = signals[0].samples.len();
    let sample_rate = signals[0].sample_rate;
    let channels = signals[0].channels;

    for signal in signals.iter().skip(1) {
        if signal.samples.len() != length {
            return Err(SignalOpError::LengthMismatch(length, signal.samples.len()));
        }
        if signal.sample_rate != sample_rate || signal.channels != channels {
            return Err(SignalOpError::InvalidInput(
                "Sample rate or channels mismatch".to_string(),
            ));
        }
    }

    let mixed_samples: Vec<f32> = (0..length)
        .map(|i| {
            let sum: f32 = signals.iter().map(|s| s.samples[i]).sum();
            sum / signals.len() as f32
        })
        .collect();

    Ok(AudioData {
        samples: mixed_samples,
        sample_rate,
        channels,
    })
}

/// Subtracts one audio signal from another.
///
/// This function subtracts the samples of `signal2` from `signal1`, producing a new
/// `AudioData`. Both signals must have the same length, sample rate, and channels.
///
/// # Arguments
/// * `signal1` - The base signal.
/// * `signal2` - The signal to subtract.
///
/// # Returns
/// Returns `Result<AudioData, SignalOpError>` containing the subtracted signal or an error.
///
/// # Examples
/// ```
/// let signal1 = AudioData { samples: vec![2.0, 4.0, 6.0], sample_rate: 44100, channels: 1 };
/// let signal2 = AudioData { samples: vec![1.0, 2.0, 3.0], sample_rate: 44100, channels: 1 };
/// let result = subtract_signals(&signal1, &signal2)?;
/// assert_eq!(result.samples, vec![1.0, 2.0, 3.0]);
/// ```
pub fn subtract_signals(signal1: &AudioData, signal2: &AudioData) -> Result<AudioData, SignalOpError> {
    if signal1.samples.len() != signal2.samples.len() {
        return Err(SignalOpError::LengthMismatch(signal1.samples.len(), signal2.samples.len()));
    }
    if signal1.sample_rate != signal2.sample_rate || signal1.channels != signal2.channels {
        return Err(SignalOpError::InvalidInput(
            "Sample rate or channels mismatch".to_string(),
        ));
    }

    let samples: Vec<f32> = signal1
        .samples
        .iter()
        .zip(&signal2.samples)
        .map(|(&s1, &s2)| s1 - s2)
        .collect();

    Ok(AudioData {
        samples,
        sample_rate: signal1.sample_rate,
        channels: signal1.channels,
    })
}

/// Multiplies two audio signals together (e.g., for amplitude modulation).
///
/// This function multiplies corresponding samples of two signals, producing a new
/// `AudioData`. Useful for effects like amplitude modulation. Signals must match in length,
/// sample rate, and channels.
///
/// # Arguments
/// * `signal1` - The first signal.
/// * `signal2` - The second signal (e.g., modulator).
///
/// # Returns
/// Returns `Result<AudioData, SignalOpError>` containing the multiplied signal or an error.
///
/// # Examples
/// ```
/// let signal1 = AudioData { samples: vec![1.0, 2.0, 3.0], sample_rate: 44100, channels: 1 };
/// let signal2 = AudioData { samples: vec![2.0, 2.0, 2.0], sample_rate: 44100, channels: 1 };
/// let result = multiply_signals(&signal1, &signal2)?;
/// assert_eq!(result.samples, vec![2.0, 4.0, 6.0]);
/// ```
pub fn multiply_signals(signal1: &AudioData, signal2: &AudioData) -> Result<AudioData, SignalOpError> {
    if signal1.samples.len() != signal2.samples.len() {
        return Err(SignalOpError::LengthMismatch(signal1.samples.len(), signal2.samples.len()));
    }
    if signal1.sample_rate != signal2.sample_rate || signal1.channels != signal2.channels {
        return Err(SignalOpError::InvalidInput(
            "Sample rate or channels mismatch".to_string(),
        ));
    }

    let samples: Vec<f32> = signal1
        .samples
        .iter()
        .zip(&signal2.samples)
        .map(|(&s1, &s2)| s1 * s2)
        .collect();

    Ok(AudioData {
        samples,
        sample_rate: signal1.sample_rate,
        channels: signal1.channels,
    })
}

/// Divides one audio signal by another with safeguards against division by zero.
///
/// This function divides the samples of `signal1` by `signal2`, replacing any division-by-zero
/// with a default value (0.0). Signals must match in length, sample rate, and channels.
///
/// # Arguments
/// * `signal1` - The numerator signal.
/// * `signal2` - The denominator signal.
///
/// # Returns
/// Returns `Result<AudioData, SignalOpError>` containing the divided signal or an error.
///
/// # Examples
/// ```
/// let signal1 = AudioData { samples: vec![4.0, 6.0, 8.0], sample_rate: 44100, channels: 1 };
/// let signal2 = AudioData { samples: vec![2.0, 0.0, 4.0], sample_rate: 44100, channels: 1 };
/// let result = divide_signals(&signal1, &signal2)?;
/// assert_eq!(result.samples, vec![2.0, 0.0, 2.0]); // 0.0 where division by zero occurs
/// ```
pub fn divide_signals(signal1: &AudioData, signal2: &AudioData) -> Result<AudioData, SignalOpError> {
    if signal1.samples.len() != signal2.samples.len() {
        return Err(SignalOpError::LengthMismatch(signal1.samples.len(), signal2.samples.len()));
    }
    if signal1.sample_rate != signal2.sample_rate || signal1.channels != signal2.channels {
        return Err(SignalOpError::InvalidInput(
            "Sample rate or channels mismatch".to_string(),
        ));
    }

    let samples: Vec<f32> = signal1
        .samples
        .iter()
        .zip(&signal2.samples)
        .enumerate()
        .map(|(i, (&s1, &s2))| {
            if s2 == 0.0 {
                eprintln!("Warning: Division by zero at index {}", i);
                0.0
            } else {
                s1 / s2
            }
        })
        .collect();

    Ok(AudioData {
        samples,
        sample_rate: signal1.sample_rate,
        channels: signal1.channels,
    })
}

/// Applies a scalar operation to an audio signal.
///
/// This function performs addition, subtraction, multiplication, or division on a signal
/// with a constant scalar value. Division by zero is handled by returning an error.
///
/// # Arguments
/// * `signal` - The input signal.
/// * `scalar` - The constant value to apply.
/// * `op` - The operation to perform: "add", "subtract", "multiply", or "divide".
///
/// # Returns
/// Returns `Result<AudioData, SignalOpError>` containing the modified signal or an error.
///
/// # Examples
/// ```
/// let signal = AudioData { samples: vec![1.0, 2.0, 3.0], sample_rate: 44100, channels: 1 };
/// let result = scalar_operation(&signal, 2.0, "multiply")?;
/// assert_eq!(result.samples, vec![2.0, 4.0, 6.0]);
/// ```
pub fn scalar_operation(signal: &AudioData, scalar: f32, op: &str) -> Result<AudioData, SignalOpError> {
    let samples = match op.to_lowercase().as_str() {
        "add" => signal.samples.iter().map(|&s| s + scalar).collect(),
        "subtract" => signal.samples.iter().map(|&s| s - scalar).collect(),
        "multiply" => signal.samples.iter().map(|&s| s * scalar).collect(),
        "divide" => {
            if scalar == 0.0 {
                return Err(SignalOpError::DivisionByZero(0));
            }
            signal.samples.iter().map(|&s| s / scalar).collect()
        }
        _ => return Err(SignalOpError::InvalidInput(format!("Unknown operation: {}", op))),
    };

    Ok(AudioData {
        samples,
        sample_rate: signal.sample_rate,
        channels: signal.channels,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mix_signals() {
        let signal1 = AudioData {
            samples: vec![1.0, 2.0, 3.0],
            sample_rate: 44100,
            channels: 1,
        };
        let signal2 = AudioData {
            samples: vec![2.0, 4.0, 6.0],
            sample_rate: 44100,
            channels: 1,
        };
        let mixed = mix_signals(&[&signal1, &signal2]).unwrap();
        assert_eq!(mixed.samples, vec![1.5, 3.0, 4.5]);
    }

    #[test]
    fn test_subtract_signals() {
        let signal1 = AudioData {
            samples: vec![2.0, 4.0, 6.0],
            sample_rate: 44100,
            channels: 1,
        };
        let signal2 = AudioData {
            samples: vec![1.0, 2.0, 3.0],
            sample_rate: 44100,
            channels: 1,
        };
        let result = subtract_signals(&signal1, &signal2).unwrap();
        assert_eq!(result.samples, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_multiply_signals() {
        let signal1 = AudioData {
            samples: vec![1.0, 2.0, 3.0],
            sample_rate: 44100,
            channels: 1,
        };
        let signal2 = AudioData {
            samples: vec![2.0, 2.0, 2.0],
            sample_rate: 44100,
            channels: 1,
        };
        let result = multiply_signals(&signal1, &signal2).unwrap();
        assert_eq!(result.samples, vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_divide_signals() {
        let signal1 = AudioData {
            samples: vec![4.0, 6.0, 8.0],
            sample_rate: 44100,
            channels: 1,
        };
        let signal2 = AudioData {
            samples: vec![2.0, 0.0, 4.0],
            sample_rate: 44100,
            channels: 1,
        };
        let result = divide_signals(&signal1, &signal2).unwrap();
        assert_eq!(result.samples, vec![2.0, 0.0, 2.0]);
    }

    #[test]
    fn test_scalar_operation() {
        let signal = AudioData {
            samples: vec![1.0, 2.0, 3.0],
            sample_rate: 44100,
            channels: 1,
        };
        let result = scalar_operation(&signal, 2.0, "multiply").unwrap();
        assert_eq!(result.samples, vec![2.0, 4.0, 6.0]);

        let result = scalar_operation(&signal, 1.0, "add").unwrap();
        assert_eq!(result.samples, vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_scalar_division_by_zero() {
        let signal = AudioData {
            samples: vec![1.0, 2.0, 3.0],
            sample_rate: 44100,
            channels: 1,
        };
        let result = scalar_operation(&signal, 0.0, "divide");
        assert!(matches!(result, Err(SignalOpError::DivisionByZero(0))));
    }
}