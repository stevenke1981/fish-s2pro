use std::io::Write;
use std::path::{Path, PathBuf};

use fish_s2_core::gguf::GgufFile;
use fish_s2_infer::{
    decode_waveform, decode_waveform_to_wav, models_dir, project_root, CodecDecoderF16Weights,
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
struct WaveformDump {
    backend: &'static str,
    text: Option<String>,
    num_codebooks: u32,
    input_frames: u32,
    latent_frames: u32,
    sample_rate: u32,
    num_samples: usize,
    samples_l2: f64,
    samples_mean_abs: f64,
    samples_max_abs: f64,
    samples_first8: Vec<f64>,
}

fn main() -> fish_s2_infer::Result<()> {
    let args = Args::parse()?;
    let input = read_generated_codes(&args.codes)?;
    let gguf = GgufFile::open(&args.codec).map_err(|err| InferError::Message(err.to_string()))?;
    let waveform = decode_waveform(
        &input.codes,
        input.num_codebooks,
        input.n_frames,
        &CodecF16Weights::from_gguf(&gguf)?,
        &CodecPostModuleF16Weights::from_gguf(&gguf)?,
        &CodecUpsampleF16Weights::from_gguf(&gguf)?,
        &CodecDecoderF16Weights::from_gguf(&gguf)?,
    )?;
    if let Some(wav_path) = args.wav.as_ref() {
        let wav = decode_waveform_to_wav(
            &input.codes,
            input.num_codebooks,
            input.n_frames,
            &CodecF16Weights::from_gguf(&gguf)?,
            &CodecPostModuleF16Weights::from_gguf(&gguf)?,
            &CodecUpsampleF16Weights::from_gguf(&gguf)?,
            &CodecDecoderF16Weights::from_gguf(&gguf)?,
        )?;
        if let Some(parent) = wav_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(wav_path, wav)?;
        println!("wrote {}", wav_path.display());
    }
    let dump = WaveformDump {
        backend: "rust",
        text: input.text,
        num_codebooks: input.num_codebooks,
        input_frames: waveform.input_frames,
        latent_frames: waveform.latent_frames,
        sample_rate: waveform.sample_rate,
        num_samples: waveform.num_samples,
        samples_l2: l2(&waveform.samples),
        samples_mean_abs: mean_abs(&waveform.samples),
        samples_max_abs: max_abs(&waveform.samples),
        samples_first8: waveform
            .samples
            .iter()
            .take(8)
            .map(|value| f64::from(*value))
            .collect(),
    };
    write_json(&args.output, &dump)?;
    println!(
        "wrote {} ({} samples @ {} Hz)",
        args.output.display(),
        dump.num_samples,
        dump.sample_rate
    );
    Ok(())
}

struct Args {
    codec: PathBuf,
    codes: PathBuf,
    output: PathBuf,
    wav: Option<PathBuf>,
}

impl Args {
    fn parse() -> fish_s2_infer::Result<Self> {
        let mut codec = models_dir().join("s2-pro-f16-codec-only.gguf");
        let mut codes = project_root()
            .join("output")
            .join("generated_codes_hi_rust.json");
        let mut output = project_root().join("output").join("waveform_hi_rust.json");
        let mut wav = None;
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--codec" => codec = PathBuf::from(args.next().ok_or_missing("--codec")?),
                "--codes" => codes = PathBuf::from(args.next().ok_or_missing("--codes")?),
                "--output" => output = PathBuf::from(args.next().ok_or_missing("--output")?),
                "--wav" => wav = Some(PathBuf::from(args.next().ok_or_missing("--wav")?)),
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
            wav,
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

fn write_json(path: &Path, dump: &WaveformDump) -> fish_s2_infer::Result<()> {
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
        "Usage: fish_s2_waveform_dump [--codec codec.gguf] [--codes generated_codes.json] \
         [--output waveform.json] [--wav out.wav]"
    );
}
