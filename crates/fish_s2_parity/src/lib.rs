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

#[derive(Debug, Clone, serde::Deserialize)]
pub struct SlowArDump {
    pub transformer: String,
    pub layer: usize,
    pub position: usize,
    pub hidden_size: usize,
    pub head_count: usize,
    pub head_count_kv: usize,
    pub head_dim: usize,
    pub normalized: TensorStats,
    pub query: TensorStats,
    pub key: TensorStats,
    pub value: TensorStats,
    pub attention: TensorStats,
    pub projected: TensorStats,
    pub hidden: TensorStats,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct TensorStats {
    pub len: usize,
    pub l2: f64,
    pub mean_abs: f64,
    pub max_abs: f64,
    pub first8: Vec<f64>,
}

#[derive(Debug, Clone, Copy)]
pub struct SlowArTensorTolerance {
    pub max_l2_delta: f64,
    pub max_mean_abs_delta: f64,
    pub max_max_abs_delta: f64,
    pub max_first8_mae: f64,
}

impl Default for SlowArTensorTolerance {
    fn default() -> Self {
        Self {
            max_l2_delta: 1e-3,
            max_mean_abs_delta: 1e-5,
            max_max_abs_delta: 1e-3,
            max_first8_mae: 1e-5,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SlowArTensorDelta {
    pub name: &'static str,
    pub l2_delta: f64,
    pub mean_abs_delta: f64,
    pub max_abs_delta: f64,
    pub first8_mae: f64,
}

#[derive(Debug, Clone)]
pub struct SlowArParityReport {
    pub passed: bool,
    pub tensor_deltas: Vec<SlowArTensorDelta>,
    pub failures: Vec<String>,
}

pub fn slow_ar_dump_from_file(path: impl AsRef<Path>) -> Result<SlowArDump> {
    let bytes = fs::read(path)?;
    serde_json::from_slice(&bytes).map_err(|err| ParityError::Message(err.to_string()))
}

pub fn compare_slow_ar_dump_files(
    expected: impl AsRef<Path>,
    actual: impl AsRef<Path>,
    tolerance: SlowArTensorTolerance,
) -> Result<SlowArParityReport> {
    let expected = slow_ar_dump_from_file(expected)?;
    let actual = slow_ar_dump_from_file(actual)?;
    Ok(compare_slow_ar_dumps(&expected, &actual, tolerance))
}

pub fn compare_slow_ar_dumps(
    expected: &SlowArDump,
    actual: &SlowArDump,
    tolerance: SlowArTensorTolerance,
) -> SlowArParityReport {
    let mut failures = Vec::new();
    compare_metadata(expected, actual, &mut failures);
    let pairs = [
        ("normalized", &expected.normalized, &actual.normalized),
        ("query", &expected.query, &actual.query),
        ("key", &expected.key, &actual.key),
        ("value", &expected.value, &actual.value),
        ("attention", &expected.attention, &actual.attention),
        ("projected", &expected.projected, &actual.projected),
        ("hidden", &expected.hidden, &actual.hidden),
    ];

    let mut tensor_deltas = Vec::with_capacity(pairs.len());
    for (name, expected, actual) in pairs {
        let delta = compare_tensor_stats(name, expected, actual, tolerance, &mut failures);
        tensor_deltas.push(delta);
    }

    SlowArParityReport {
        passed: failures.is_empty(),
        tensor_deltas,
        failures,
    }
}

fn compare_metadata(expected: &SlowArDump, actual: &SlowArDump, failures: &mut Vec<String>) {
    let checks = [
        ("layer", expected.layer, actual.layer),
        ("position", expected.position, actual.position),
        ("hidden_size", expected.hidden_size, actual.hidden_size),
        ("head_count", expected.head_count, actual.head_count),
        (
            "head_count_kv",
            expected.head_count_kv,
            actual.head_count_kv,
        ),
        ("head_dim", expected.head_dim, actual.head_dim),
    ];
    for (name, expected, actual) in checks {
        if expected != actual {
            failures.push(format!(
                "{name} mismatch: expected {expected}, actual {actual}"
            ));
        }
    }
}

fn compare_tensor_stats(
    name: &'static str,
    expected: &TensorStats,
    actual: &TensorStats,
    tolerance: SlowArTensorTolerance,
    failures: &mut Vec<String>,
) -> SlowArTensorDelta {
    if expected.len != actual.len {
        failures.push(format!(
            "{name}.len mismatch: expected {}, actual {}",
            expected.len, actual.len
        ));
    }
    let l2_delta = (expected.l2 - actual.l2).abs();
    let mean_abs_delta = (expected.mean_abs - actual.mean_abs).abs();
    let max_abs_delta = (expected.max_abs - actual.max_abs).abs();
    let first8_mae = first8_mae(&expected.first8, &actual.first8);
    if l2_delta > tolerance.max_l2_delta {
        failures.push(format!(
            "{name}.l2 delta {l2_delta:.8} exceeds {:.8}",
            tolerance.max_l2_delta
        ));
    }
    if mean_abs_delta > tolerance.max_mean_abs_delta {
        failures.push(format!(
            "{name}.mean_abs delta {mean_abs_delta:.8} exceeds {:.8}",
            tolerance.max_mean_abs_delta
        ));
    }
    if max_abs_delta > tolerance.max_max_abs_delta {
        failures.push(format!(
            "{name}.max_abs delta {max_abs_delta:.8} exceeds {:.8}",
            tolerance.max_max_abs_delta
        ));
    }
    if first8_mae > tolerance.max_first8_mae {
        failures.push(format!(
            "{name}.first8 MAE {first8_mae:.8} exceeds {:.8}",
            tolerance.max_first8_mae
        ));
    }
    SlowArTensorDelta {
        name,
        l2_delta,
        mean_abs_delta,
        max_abs_delta,
        first8_mae,
    }
}

fn first8_mae(expected: &[f64], actual: &[f64]) -> f64 {
    let len = expected.len().max(actual.len());
    if len == 0 {
        return 0.0;
    }
    let sum: f64 = (0..len)
        .map(|index| {
            let expected = expected.get(index).copied().unwrap_or(0.0);
            let actual = actual.get(index).copied().unwrap_or(0.0);
            (expected - actual).abs()
        })
        .sum();
    sum / len as f64
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
    fn compares_identical_slow_ar_dumps() {
        let dump: SlowArDump = serde_json::from_str(test_slow_ar_dump_json()).unwrap();
        let report = compare_slow_ar_dumps(&dump, &dump, SlowArTensorTolerance::default());
        assert!(report.passed);
        assert!(report.failures.is_empty());
        assert_eq!(report.tensor_deltas.len(), 7);
    }

    #[test]
    fn reports_slow_ar_tensor_delta_failure() {
        let expected: SlowArDump = serde_json::from_str(test_slow_ar_dump_json()).unwrap();
        let mut actual = expected.clone();
        actual.query.l2 += 0.1;
        let report = compare_slow_ar_dumps(&expected, &actual, SlowArTensorTolerance::default());
        assert!(!report.passed);
        assert!(report.failures.iter().any(|f| f.contains("query.l2")));
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

    fn test_slow_ar_dump_json() -> &'static str {
        r#"{
          "transformer": "model.gguf",
          "layer": 0,
          "position": 0,
          "hidden_size": 4,
          "head_count": 2,
          "head_count_kv": 1,
          "head_dim": 2,
          "normalized": {"len": 4, "l2": 1.0, "mean_abs": 0.25, "max_abs": 1.0, "first8": [1.0, 0.0, 0.0, 0.0]},
          "query": {"len": 4, "l2": 2.0, "mean_abs": 0.5, "max_abs": 1.0, "first8": [1.0, 0.0, 0.0, 1.0]},
          "key": {"len": 2, "l2": 1.0, "mean_abs": 0.5, "max_abs": 1.0, "first8": [1.0, 0.0]},
          "value": {"len": 2, "l2": 3.0, "mean_abs": 1.5, "max_abs": 3.0, "first8": [3.0, 0.0]},
          "attention": {"len": 4, "l2": 4.0, "mean_abs": 1.5, "max_abs": 3.0, "first8": [3.0, 0.0, 3.0, 0.0]},
          "projected": {"len": 4, "l2": 5.0, "mean_abs": 2.25, "max_abs": 6.0, "first8": [3.0, 6.0, 0.0, 0.0]},
          "hidden": {"len": 4, "l2": 6.0, "mean_abs": 2.5, "max_abs": 6.0, "first8": [4.0, 6.0, 0.0, 0.0]}
        }"#
    }
}
