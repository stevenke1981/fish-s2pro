use std::io::Write;
use std::path::{Path, PathBuf};

use fish_s2_core::gguf::GgufFile;
use fish_s2_infer::{
    encode_reference_audio, encode_reference_wav_file, models_dir, project_root,
    CodecReferenceAudioResult, CodecReferenceEncoderF16Weights, InferError, CODEC_FRAME_LENGTH,
};

#[derive(Debug, serde::Serialize)]
struct ReferenceCodesDump {
    backend: &'static str,
    code_layout: &'static str,
    input_samples: u32,
    padded_samples: u32,
    encoder_frames: u32,
    n_frames: u32,
    num_codebooks: u32,
    codes_len: usize,
    codes: Vec<i32>,
    final_residual_l2: Vec<f32>,
}

fn main() -> fish_s2_infer::Result<()> {
    let args = Args::parse()?;
    let gguf = GgufFile::open(&args.codec).map_err(|err| InferError::Message(err.to_string()))?;
    let weights = CodecReferenceEncoderF16Weights::from_gguf(&gguf)?;
    let result = match args.wav_input.as_ref() {
        Some(path) => encode_reference_wav_file(path, &weights)?,
        None => encode_reference_audio(&synthetic_pcm(args.samples), &weights)?,
    };
    let dump = ReferenceCodesDump::from_result(result);
    write_json(&args.output, &dump)?;
    println!(
        "wrote {} ({} codebooks x {} frames)",
        args.output.display(),
        dump.num_codebooks,
        dump.n_frames
    );
    Ok(())
}

impl ReferenceCodesDump {
    fn from_result(result: CodecReferenceAudioResult) -> Self {
        Self {
            backend: "rust",
            code_layout: "codebook_major",
            input_samples: result.input_samples,
            padded_samples: result.padded_samples,
            encoder_frames: result.encoder_frames,
            n_frames: result.quantizer_frames,
            num_codebooks: result.num_codebooks,
            codes_len: result.codes.len(),
            codes: result.codes,
            final_residual_l2: result.final_residual_l2,
        }
    }
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
            .join("reference_codes_synthetic_rust.json");
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

fn write_json(path: &Path, dump: &ReferenceCodesDump) -> fish_s2_infer::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut file = std::fs::File::create(path)?;
    let json = serde_json::to_string_pretty(dump)?;
    file.write_all(json.as_bytes())?;
    file.write_all(b"\n")?;
    Ok(())
}

fn print_usage() {
    eprintln!(
        "Usage: fish_s2_reference_codes_dump [--codec codec.gguf] [--wav-input reference.wav] \
         [--samples N] [--output reference_codes.json]"
    );
}
