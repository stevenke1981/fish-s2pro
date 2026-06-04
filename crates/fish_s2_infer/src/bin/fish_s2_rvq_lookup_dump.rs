use std::io::Write;
use std::path::{Path, PathBuf};

use fish_s2_core::gguf::GgufFile;
use fish_s2_infer::{models_dir, project_root, rvq_lookup_codes, CodecF16Weights, InferError};

#[derive(Debug, serde::Deserialize)]
struct GeneratedCodesInput {
    text: Option<String>,
    num_codebooks: u32,
    n_frames: u32,
    codes: Vec<i32>,
}

#[derive(Debug, serde::Serialize)]
struct RvqLookupDump {
    backend: &'static str,
    text: Option<String>,
    num_codebooks: u32,
    n_frames: u32,
    latent_dim: usize,
    latent_len: usize,
    latent_l2: f64,
    latent_mean_abs: f64,
    latent_max_abs: f64,
    latent_first8: Vec<f64>,
}

fn main() -> fish_s2_infer::Result<()> {
    let args = Args::parse()?;
    let input = read_generated_codes(&args.codes)?;
    let gguf = GgufFile::open(&args.codec).map_err(|err| InferError::Message(err.to_string()))?;
    let weights = CodecF16Weights::from_gguf(&gguf)?;
    let result = rvq_lookup_codes(&input.codes, input.num_codebooks, input.n_frames, &weights)?;
    let dump = RvqLookupDump {
        backend: "rust",
        text: input.text,
        num_codebooks: result.num_codebooks,
        n_frames: result.n_frames,
        latent_dim: result.latent_dim,
        latent_len: result.latents.len(),
        latent_l2: l2(&result.latents),
        latent_mean_abs: mean_abs(&result.latents),
        latent_max_abs: max_abs(&result.latents),
        latent_first8: result
            .latents
            .iter()
            .take(8)
            .map(|value| f64::from(*value))
            .collect(),
    };
    write_json(&args.output, &dump)?;
    println!(
        "wrote {} ({} frames x {} latent)",
        args.output.display(),
        dump.n_frames,
        dump.latent_dim
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
            .join("rvq_lookup_hi_rust.json");
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

fn write_json(path: &Path, dump: &RvqLookupDump) -> fish_s2_infer::Result<()> {
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
        "Usage: fish_s2_rvq_lookup_dump [--codec codec.gguf] [--codes generated_codes.json] \
         [--output rvq_lookup.json]"
    );
}
