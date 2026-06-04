use std::io::Write;
use std::path::{Path, PathBuf};

use fish_s2_core::gguf::GgufFile;
use fish_s2_infer::{
    forward_codec_encoder_frontend, models_dir, project_root, CodecEncoderF16Weights, InferError,
    CODEC_FRAME_LENGTH,
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
    let audio = synthetic_pcm(args.samples);
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
    samples: usize,
}

impl Args {
    fn parse() -> fish_s2_infer::Result<Self> {
        let mut codec = models_dir().join("s2-pro-f16-codec-only.gguf");
        let mut output = project_root()
            .join("output")
            .join("encoder_stage_synthetic_rust.json");
        let mut samples = CODEC_FRAME_LENGTH;
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--codec" => codec = PathBuf::from(args.next().ok_or_missing("--codec")?),
                "--output" => output = PathBuf::from(args.next().ok_or_missing("--output")?),
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
         [--samples 2048]"
    );
}
