use std::io::Write;
use std::path::{Path, PathBuf};

use fish_s2_core::gguf::GgufFile;
use fish_s2_infer::{
    forward_codec_encoder_frontend, models_dir, project_root, CodecEncoderF16Weights, InferError,
    CODEC_FRAME_LENGTH, CODEC_SAMPLE_RATE,
};

#[derive(Debug, serde::Serialize)]
struct EncoderStageDump {
    backend: &'static str,
    input_samples: u32,
    padded_samples: u32,
    output_frames: u32,
    hidden_dim: usize,
    hidden_len: usize,
    hidden_l2: f64,
    hidden_mean_abs: f64,
    hidden_max_abs: f64,
    hidden_first8: Vec<f64>,
}

fn main() -> fish_s2_infer::Result<()> {
    let args = Args::parse()?;
    let gguf = GgufFile::open(&args.codec).map_err(|err| InferError::Message(err.to_string()))?;
    let weights = CodecEncoderF16Weights::from_gguf(&gguf)?;
    let audio = match args.wav_input.as_ref() {
        Some(path) => read_wav_mono_f32(path)?,
        None => synthetic_pcm(args.samples),
    };
    let result = forward_codec_encoder_frontend(&audio, &weights)?;
    let dump = EncoderStageDump {
        backend: "rust",
        input_samples: result.input_samples,
        padded_samples: result.padded_samples,
        output_frames: result.output_frames,
        hidden_dim: result.hidden_dim,
        hidden_len: result.hidden.len(),
        hidden_l2: l2(&result.hidden),
        hidden_mean_abs: mean_abs(&result.hidden),
        hidden_max_abs: max_abs(&result.hidden),
        hidden_first8: result
            .hidden
            .iter()
            .take(8)
            .map(|value| f64::from(*value))
            .collect(),
    };
    write_json(&args.output, &dump)?;
    println!(
        "wrote {} ({} -> {} samples, {} frames x {} hidden)",
        args.output.display(),
        dump.input_samples,
        dump.padded_samples,
        dump.output_frames,
        dump.hidden_dim
    );
    Ok(())
}

struct Args {
    codec: PathBuf,
    output: PathBuf,
    wav_input: Option<PathBuf>,
    samples: usize,
}

impl Args {
    fn parse() -> fish_s2_infer::Result<Self> {
        let mut codec = models_dir().join("s2-pro-f16-codec-only.gguf");
        let mut output = project_root()
            .join("output")
            .join("encoder_stage_synthetic_rust.json");
        let mut wav_input = None;
        let mut samples = CODEC_FRAME_LENGTH;
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--codec" => codec = PathBuf::from(args.next().ok_or_missing("--codec")?),
                "--output" => output = PathBuf::from(args.next().ok_or_missing("--output")?),
                "--wav-input" => {
                    wav_input = Some(PathBuf::from(args.next().ok_or_missing("--wav-input")?))
                }
                "--samples" => {
                    let raw = args.next().ok_or_missing("--samples")?;
                    samples = raw.parse::<usize>().map_err(|err| {
                        InferError::Message(format!("invalid --samples {raw}: {err}"))
                    })?;
                }
                "--help" | "-h" => {
                    print_usage();
                    std::process::exit(0);
                }
                other => return Err(InferError::Message(format!("unknown argument: {other}"))),
            }
        }
        Ok(Self {
            codec,
            output,
            wav_input,
            samples,
        })
    }
}

trait MissingArg {
    fn ok_or_missing(self, flag: &str) -> fish_s2_infer::Result<String>;
}

impl MissingArg for Option<String> {
    fn ok_or_missing(self, flag: &str) -> fish_s2_infer::Result<String> {
        self.ok_or_else(|| InferError::Message(format!("missing {flag}")))
    }
}

fn synthetic_pcm(samples: usize) -> Vec<f32> {
    (0..samples)
        .map(|index| (((index % 97) as f32) - 48.0) / 4096.0)
        .collect()
}

fn read_wav_mono_f32(path: &Path) -> fish_s2_infer::Result<Vec<f32>> {
    let bytes = std::fs::read(path)?;
    let wav = ParsedWav::parse(&bytes)?;
    if wav.sample_rate != CODEC_SAMPLE_RATE {
        return Err(InferError::Message(format!(
            "expected {CODEC_SAMPLE_RATE} Hz WAV, got {} Hz",
            wav.sample_rate
        )));
    }
    let samples = decode_wav_samples(&wav)?;
    if samples.is_empty() {
        return Err(InferError::Message("WAV contains no samples".into()));
    }
    if wav.channels == 1 {
        return Ok(samples);
    }
    let channels = usize::from(wav.channels);
    if samples.len() % channels != 0 {
        return Err(InferError::Message(
            "WAV sample count is not divisible by channel count".into(),
        ));
    }
    Ok(samples
        .chunks_exact(channels)
        .map(|frame| frame.iter().sum::<f32>() / wav.channels as f32)
        .collect())
}

struct ParsedWav<'a> {
    audio_format: u16,
    channels: u16,
    sample_rate: u32,
    bits_per_sample: u16,
    data: &'a [u8],
}

impl<'a> ParsedWav<'a> {
    fn parse(bytes: &'a [u8]) -> fish_s2_infer::Result<Self> {
        if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
            return Err(InferError::Message("not a RIFF/WAVE file".into()));
        }
        let mut offset = 12usize;
        let mut fmt = None;
        let mut data = None;
        while offset.checked_add(8).is_some_and(|end| end <= bytes.len()) {
            let id = &bytes[offset..offset + 4];
            let len = read_u32(bytes, offset + 4)? as usize;
            offset += 8;
            let end = offset
                .checked_add(len)
                .ok_or_else(|| InferError::Message("WAV chunk length overflow".into()))?;
            if end > bytes.len() {
                return Err(InferError::Message(
                    "WAV chunk extends past file end".into(),
                ));
            }
            match id {
                b"fmt " => {
                    if len < 16 {
                        return Err(InferError::Message("WAV fmt chunk too short".into()));
                    }
                    fmt = Some((
                        read_u16(bytes, offset)?,
                        read_u16(bytes, offset + 2)?,
                        read_u32(bytes, offset + 4)?,
                        read_u16(bytes, offset + 14)?,
                    ));
                }
                b"data" => data = Some(&bytes[offset..end]),
                _ => {}
            }
            offset = end + (len % 2);
        }
        let (audio_format, channels, sample_rate, bits_per_sample) =
            fmt.ok_or_else(|| InferError::Message("missing WAV fmt chunk".into()))?;
        let data = data.ok_or_else(|| InferError::Message("missing WAV data chunk".into()))?;
        if channels == 0 {
            return Err(InferError::Message("WAV has zero channels".into()));
        }
        Ok(Self {
            audio_format,
            channels,
            sample_rate,
            bits_per_sample,
            data,
        })
    }
}

fn decode_wav_samples(wav: &ParsedWav<'_>) -> fish_s2_infer::Result<Vec<f32>> {
    match (wav.audio_format, wav.bits_per_sample) {
        (1, 8) => Ok(wav
            .data
            .iter()
            .map(|byte| (*byte as f32 - 128.0) / 128.0)
            .collect()),
        (1, 16) => chunks_exact(wav.data, 2)?
            .map(|chunk| Ok(i16::from_le_bytes([chunk[0], chunk[1]]) as f32 / 32768.0))
            .collect(),
        (1, 24) => chunks_exact(wav.data, 3)?
            .map(|chunk| {
                let value = i32::from_le_bytes([
                    chunk[0],
                    chunk[1],
                    chunk[2],
                    if chunk[2] & 0x80 == 0 { 0 } else { 0xff },
                ]);
                Ok(value as f32 / 8_388_608.0)
            })
            .collect(),
        (1, 32) => chunks_exact(wav.data, 4)?
            .map(|chunk| {
                Ok(
                    i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as f32
                        / 2_147_483_648.0,
                )
            })
            .collect(),
        (3, 32) => chunks_exact(wav.data, 4)?
            .map(|chunk| Ok(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])))
            .collect(),
        _ => Err(InferError::Message(format!(
            "unsupported WAV format: audio_format={}, bits_per_sample={}",
            wav.audio_format, wav.bits_per_sample
        ))),
    }
}

fn chunks_exact(
    bytes: &[u8],
    chunk_size: usize,
) -> fish_s2_infer::Result<std::slice::ChunksExact<'_, u8>> {
    let chunks = bytes.chunks_exact(chunk_size);
    if !chunks.remainder().is_empty() {
        return Err(InferError::Message(format!(
            "WAV data length is not divisible by {chunk_size}"
        )));
    }
    Ok(chunks)
}

fn read_u16(bytes: &[u8], offset: usize) -> fish_s2_infer::Result<u16> {
    let slice = bytes
        .get(offset..offset + 2)
        .ok_or_else(|| InferError::Message("unexpected EOF reading WAV u16".into()))?;
    Ok(u16::from_le_bytes([slice[0], slice[1]]))
}

fn read_u32(bytes: &[u8], offset: usize) -> fish_s2_infer::Result<u32> {
    let slice = bytes
        .get(offset..offset + 4)
        .ok_or_else(|| InferError::Message("unexpected EOF reading WAV u32".into()))?;
    Ok(u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]))
}

fn write_json(path: &Path, dump: &EncoderStageDump) -> fish_s2_infer::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut file = std::fs::File::create(path)?;
    let json = serde_json::to_string_pretty(dump)?;
    file.write_all(json.as_bytes())?;
    file.write_all(b"\n")?;
    Ok(())
}

fn l2(values: &[f32]) -> f64 {
    values
        .iter()
        .map(|value| {
            let v = f64::from(*value);
            v * v
        })
        .sum::<f64>()
        .sqrt()
}

fn mean_abs(values: &[f32]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values
        .iter()
        .map(|value| f64::from(value.abs()))
        .sum::<f64>()
        / values.len() as f64
}

fn max_abs(values: &[f32]) -> f64 {
    values
        .iter()
        .map(|value| f64::from(value.abs()))
        .fold(0.0, f64::max)
}

fn print_usage() {
    eprintln!(
        "Usage: fish_s2_encoder_stage_dump [--codec codec.gguf] [--output encoder_stage.json] \
         [--samples 2048] [--wav-input reference.wav]"
    );
}
