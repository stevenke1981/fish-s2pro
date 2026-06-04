use std::fs;
use std::path::Path;

#[derive(Debug, thiserror::Error)]
pub enum ParityError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("{0}")]
    Message(String),
}

pub type Result<T> = std::result::Result<T, ParityError>;

#[derive(Debug, Clone)]
pub struct WavMetrics {
    pub sample_rate: u32,
    pub channels: u16,
    pub bits_per_sample: u16,
    pub duration_seconds: f64,
    pub rms: f64,
    pub peak: f64,
    pub envelope_rms: Vec<f64>,
}

#[derive(Debug, Clone, Copy)]
pub struct ParityTolerance {
    pub max_duration_delta_seconds: f64,
    pub max_rms_delta: f64,
    pub max_envelope_mae: f64,
}

impl Default for ParityTolerance {
    fn default() -> Self {
        Self {
            max_duration_delta_seconds: 0.10,
            max_rms_delta: 0.03,
            max_envelope_mae: 0.04,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ParityReport {
    pub passed: bool,
    pub duration_delta_seconds: f64,
    pub rms_delta: f64,
    pub envelope_mae: f64,
    pub failures: Vec<String>,
}

pub fn metrics_from_wav_file(path: impl AsRef<Path>) -> Result<WavMetrics> {
    let bytes = fs::read(path)?;
    metrics_from_wav_bytes(&bytes, 50)
}

pub fn compare_wav_files(
    expected: impl AsRef<Path>,
    actual: impl AsRef<Path>,
    tolerance: ParityTolerance,
) -> Result<ParityReport> {
    let expected = metrics_from_wav_file(expected)?;
    let actual = metrics_from_wav_file(actual)?;
    Ok(compare_metrics(&expected, &actual, tolerance))
}

pub fn compare_metrics(
    expected: &WavMetrics,
    actual: &WavMetrics,
    tolerance: ParityTolerance,
) -> ParityReport {
    let duration_delta_seconds = (expected.duration_seconds - actual.duration_seconds).abs();
    let rms_delta = (expected.rms - actual.rms).abs();
    let envelope_mae = envelope_mae(&expected.envelope_rms, &actual.envelope_rms);

    let mut failures = Vec::new();
    if expected.sample_rate != actual.sample_rate {
        failures.push(format!(
            "sample rate mismatch: expected {}, actual {}",
            expected.sample_rate, actual.sample_rate
        ));
    }
    if expected.channels != actual.channels {
        failures.push(format!(
            "channel count mismatch: expected {}, actual {}",
            expected.channels, actual.channels
        ));
    }
    if duration_delta_seconds > tolerance.max_duration_delta_seconds {
        failures.push(format!(
            "duration delta {duration_delta_seconds:.4}s exceeds {:.4}s",
            tolerance.max_duration_delta_seconds
        ));
    }
    if rms_delta > tolerance.max_rms_delta {
        failures.push(format!(
            "RMS delta {rms_delta:.6} exceeds {:.6}",
            tolerance.max_rms_delta
        ));
    }
    if envelope_mae > tolerance.max_envelope_mae {
        failures.push(format!(
            "envelope MAE {envelope_mae:.6} exceeds {:.6}",
            tolerance.max_envelope_mae
        ));
    }

    ParityReport {
        passed: failures.is_empty(),
        duration_delta_seconds,
        rms_delta,
        envelope_mae,
        failures,
    }
}

pub fn metrics_from_wav_bytes(bytes: &[u8], frame_ms: u32) -> Result<WavMetrics> {
    let wav = ParsedWav::parse(bytes)?;
    let samples = decode_samples(&wav)?;
    if samples.is_empty() {
        return Err(ParityError::Message("WAV contains no samples".into()));
    }

    let sum_square: f64 = samples.iter().map(|s| f64::from(*s) * f64::from(*s)).sum();
    let rms = (sum_square / samples.len() as f64).sqrt();
    let peak = samples.iter().map(|s| s.abs()).fold(0.0_f32, f32::max) as f64;
    let frame_count = wav.data.len() as f64 / wav.block_align as f64;
    let duration_seconds = frame_count / wav.sample_rate as f64;
    let envelope_rms = rms_envelope(&samples, wav.sample_rate, wav.channels, frame_ms);

    Ok(WavMetrics {
        sample_rate: wav.sample_rate,
        channels: wav.channels,
        bits_per_sample: wav.bits_per_sample,
        duration_seconds,
        rms,
        peak,
        envelope_rms,
    })
}

fn rms_envelope(samples: &[f32], sample_rate: u32, channels: u16, frame_ms: u32) -> Vec<f64> {
    let samples_per_frame =
        ((sample_rate as usize * channels as usize * frame_ms as usize) / 1000).max(1);
    samples
        .chunks(samples_per_frame)
        .map(|chunk| {
            let sum_square: f64 = chunk.iter().map(|s| f64::from(*s) * f64::from(*s)).sum();
            (sum_square / chunk.len() as f64).sqrt()
        })
        .collect()
}

fn envelope_mae(expected: &[f64], actual: &[f64]) -> f64 {
    let len = expected.len().max(actual.len());
    if len == 0 {
        return 0.0;
    }
    let sum: f64 = (0..len)
        .map(|i| {
            let pos = if len == 1 {
                0.0
            } else {
                i as f64 / (len - 1) as f64
            };
            (sample_envelope(expected, pos) - sample_envelope(actual, pos)).abs()
        })
        .sum();
    sum / len as f64
}

fn sample_envelope(values: &[f64], pos: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    if values.len() == 1 {
        return values[0];
    }
    let scaled = pos.clamp(0.0, 1.0) * (values.len() - 1) as f64;
    let lo = scaled.floor() as usize;
    let hi = scaled.ceil() as usize;
    if lo == hi {
        values[lo]
    } else {
        let t = scaled - lo as f64;
        values[lo] * (1.0 - t) + values[hi] * t
    }
}

#[derive(Debug)]
struct ParsedWav<'a> {
    audio_format: u16,
    channels: u16,
    sample_rate: u32,
    block_align: u16,
    bits_per_sample: u16,
    data: &'a [u8],
}

impl<'a> ParsedWav<'a> {
    fn parse(bytes: &'a [u8]) -> Result<Self> {
        if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
            return Err(ParityError::Message("not a RIFF/WAVE file".into()));
        }

        let mut pos = 12;
        let mut fmt = None;
        let mut data = None;
        while pos + 8 <= bytes.len() {
            let id = &bytes[pos..pos + 4];
            let size = read_u32(&bytes[pos + 4..pos + 8]) as usize;
            pos += 8;
            if pos + size > bytes.len() {
                return Err(ParityError::Message("truncated WAV chunk".into()));
            }
            let chunk = &bytes[pos..pos + size];
            match id {
                b"fmt " => {
                    if chunk.len() < 16 {
                        return Err(ParityError::Message("fmt chunk too short".into()));
                    }
                    fmt = Some((
                        read_u16(&chunk[0..2]),
                        read_u16(&chunk[2..4]),
                        read_u32(&chunk[4..8]),
                        read_u16(&chunk[12..14]),
                        read_u16(&chunk[14..16]),
                    ));
                }
                b"data" => data = Some(chunk),
                _ => {}
            }
            pos += size + (size % 2);
        }

        let (audio_format, channels, sample_rate, block_align, bits_per_sample) =
            fmt.ok_or_else(|| ParityError::Message("missing fmt chunk".into()))?;
        let data = data.ok_or_else(|| ParityError::Message("missing data chunk".into()))?;
        if channels == 0 || sample_rate == 0 || block_align == 0 {
            return Err(ParityError::Message("invalid WAV format values".into()));
        }

        Ok(Self {
            audio_format,
            channels,
            sample_rate,
            block_align,
            bits_per_sample,
            data,
        })
    }
}

fn decode_samples(wav: &ParsedWav<'_>) -> Result<Vec<f32>> {
    match (wav.audio_format, wav.bits_per_sample) {
        (1, 8) => Ok(wav
            .data
            .iter()
            .map(|b| (*b as f32 - 128.0) / 128.0)
            .collect()),
        (1, 16) => chunks_exact(wav.data, 2)?
            .map(|c| Ok(i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0))
            .collect(),
        (1, 24) => chunks_exact(wav.data, 3)?
            .map(|c| {
                let value =
                    i32::from_le_bytes([c[0], c[1], c[2], if c[2] & 0x80 == 0 { 0 } else { 0xff }]);
                Ok(value as f32 / 8_388_608.0)
            })
            .collect(),
        (1, 32) => chunks_exact(wav.data, 4)?
            .map(|c| Ok(i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32 / 2_147_483_648.0))
            .collect(),
        (3, 32) => chunks_exact(wav.data, 4)?
            .map(|c| Ok(f32::from_le_bytes([c[0], c[1], c[2], c[3]]).clamp(-1.0, 1.0)))
            .collect(),
        _ => Err(ParityError::Message(format!(
            "unsupported WAV format: audio_format={}, bits_per_sample={}",
            wav.audio_format, wav.bits_per_sample
        ))),
    }
}

fn chunks_exact<'a>(bytes: &'a [u8], width: usize) -> Result<std::slice::ChunksExact<'a, u8>> {
    if bytes.len() % width != 0 {
        return Err(ParityError::Message(
            "WAV data is not sample-aligned".into(),
        ));
    }
    Ok(bytes.chunks_exact(width))
}

fn read_u16(bytes: &[u8]) -> u16 {
    u16::from_le_bytes([bytes[0], bytes[1]])
}

fn read_u32(bytes: &[u8]) -> u32 {
    u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_pcm16_wav_metrics() {
        let wav = test_wav(&[0, 16_384, -16_384, 0], 24_000, 1);
        let metrics = metrics_from_wav_bytes(&wav, 50).unwrap();
        assert_eq!(metrics.sample_rate, 24_000);
        assert_eq!(metrics.channels, 1);
        assert_eq!(metrics.bits_per_sample, 16);
        assert!((metrics.duration_seconds - (4.0 / 24_000.0)).abs() < 0.000001);
        assert!(metrics.rms > 0.35 && metrics.rms < 0.36);
    }

    #[test]
    fn reports_duration_delta_failure() {
        let a = metrics_from_wav_bytes(&test_wav(&[0; 24_000], 24_000, 1), 50).unwrap();
        let b = metrics_from_wav_bytes(&test_wav(&[0; 12_000], 24_000, 1), 50).unwrap();
        let report = compare_metrics(&a, &b, ParityTolerance::default());
        assert!(!report.passed);
        assert!(report.failures.iter().any(|f| f.contains("duration delta")));
    }

    #[test]
    #[ignore = "requires FISH_S2_PARITY=1 plus golden/candidate WAV paths"]
    fn compares_env_candidate_to_golden() {
        if std::env::var("FISH_S2_PARITY").ok().as_deref() != Some("1") {
            eprintln!("set FISH_S2_PARITY=1 to enable the local parity gate");
            return;
        }
        let golden =
            std::env::var("FISH_S2_GOLDEN_WAV").unwrap_or_else(|_| "output/golden.wav".to_string());
        let candidate = std::env::var("FISH_S2_CANDIDATE_WAV")
            .unwrap_or_else(|_| "output/candidate.wav".to_string());
        let report = compare_wav_files(golden, candidate, ParityTolerance::default()).unwrap();
        assert!(report.passed, "{report:#?}");
    }

    fn test_wav(samples: &[i16], sample_rate: u32, channels: u16) -> Vec<u8> {
        let data_len = samples.len() * 2;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&(36 + data_len as u32).to_le_bytes());
        bytes.extend_from_slice(b"WAVE");
        bytes.extend_from_slice(b"fmt ");
        bytes.extend_from_slice(&16_u32.to_le_bytes());
        bytes.extend_from_slice(&1_u16.to_le_bytes());
        bytes.extend_from_slice(&channels.to_le_bytes());
        bytes.extend_from_slice(&sample_rate.to_le_bytes());
        bytes.extend_from_slice(&(sample_rate * channels as u32 * 2).to_le_bytes());
        bytes.extend_from_slice(&(channels * 2).to_le_bytes());
        bytes.extend_from_slice(&16_u16.to_le_bytes());
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&(data_len as u32).to_le_bytes());
        for sample in samples {
            bytes.extend_from_slice(&sample.to_le_bytes());
        }
        bytes
    }
}
