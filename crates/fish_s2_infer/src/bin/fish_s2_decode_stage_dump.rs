use std::io::Write;
use std::path::{Path, PathBuf};

use fish_s2_core::gguf::GgufFile;
use fish_s2_infer::{
    forward_codec_post_module, forward_codec_upsample, models_dir, project_root, rvq_lookup_codes,
    CodecF16Weights, CodecPostModuleF16Weights, CodecUpsampleF16Weights, InferError,
};

#[derive(Debug, serde::Deserialize)]
struct GeneratedCodesInput {
    text: Option<String>,
    num_codebooks: u32,
    n_frames: u32,
    codes: Vec<i32>,
}

#[derive(Debug, serde::Serialize)]
struct DecodeStageDump {
    backend: &'static str,
    text: Option<String>,
    num_codebooks: u32,
    input_frames: u32,
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
    let input = read_generated_codes(&args.codes)?;
    let gguf = GgufFile::open(&args.codec).map_err(|err| InferError::Message(err.to_string()))?;
    let rvq_weights = CodecF16Weights::from_gguf(&gguf)?;
    let post_weights = CodecPostModuleF16Weights::from_gguf(&gguf)?;
    let upsample_weights = CodecUpsampleF16Weights::from_gguf(&gguf)?;
    let rvq = rvq_lookup_codes(
        &input.codes,
        input.num_codebooks,
        input.n_frames,
        &rvq_weights,
    )?;
    let post = forward_codec_post_module(&rvq.latents, rvq.n_frames, &post_weights)?;
    let upsample = forward_codec_upsample(&post.hidden, post.n_frames, &upsample_weights)?;
    let dump = DecodeStageDump {
        backend: "rust",
        text: input.text,
        num_codebooks: input.num_codebooks,
        input_frames: upsample.input_frames,
        output_frames: upsample.output_frames,
        hidden_dim: upsample.hidden_dim,
        hidden_len: upsample.hidden.len(),
        hidden_l2: l2(&upsample.hidden),
        hidden_mean_abs: mean_abs(&upsample.hidden),
        hidden_max_abs: max_abs(&upsample.hidden),
        hidden_first8: upsample
            .hidden
            .iter()
            .take(8)
            .map(|value| f64::from(*value))
            .collect(),
    };
    write_json(&args.output, &dump)?;
    println!(
        "wrote {} ({} -> {} frames x {} hidden)",
        args.output.display(),
        dump.input_frames,
        dump.output_frames,
        dump.hidden_dim
    );
    Ok(())
}

struct Args {
    codec: PathBuf,
    codes: PathBuf,
    output: PathBuf,
}

impl Args {
    fn parse() -> fish_s2_infer::Result<Self> {
        let mut codec = models_dir().join("s2-pro-f16-codec-only.gguf");
        let mut codes = project_root()
            .join("output")
            .join("generated_codes_hi_rust.json");
        let mut output = project_root()
            .join("output")
            .join("decode_stage_hi_rust.json");
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--codec" => codec = PathBuf::from(args.next().ok_or_missing("--codec")?),
                "--codes" => codes = PathBuf::from(args.next().ok_or_missing("--codes")?),
                "--output" => output = PathBuf::from(args.next().ok_or_missing("--output")?),
                "--help" | "-h" => {
                    print_usage();
                    std::process::exit(0);
                }
                other => return Err(InferError::Message(format!("unknown argument: {other}"))),
            }
        }
        Ok(Self {
            codec,
            codes,
            output,
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

fn read_generated_codes(path: &Path) -> fish_s2_infer::Result<GeneratedCodesInput> {
    let bytes = std::fs::read(path)?;
    serde_json::from_slice(&bytes).map_err(InferError::from)
}

fn write_json(path: &Path, dump: &DecodeStageDump) -> fish_s2_infer::Result<()> {
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
        "Usage: fish_s2_decode_stage_dump [--codec codec.gguf] [--codes generated_codes.json] \
         [--output decode_stage.json]"
    );
}
