use std::io::Write;
use std::path::PathBuf;

use fish_s2_infer::{
    default_tokenizer_path, load_prompt_codes_file, models_dir, project_root, GenerateParams,
    InferError, RustPipeline, RustPipelineConfig, RustSynthesisOptions,
};

#[derive(Debug, serde::Serialize)]
struct E2eWavDump {
    backend: &'static str,
    text: String,
    prompt_text: Option<String>,
    prompt_code_cols: Option<usize>,
    temperature: f32,
    top_p: f32,
    top_k: i32,
    max_new_tokens: u32,
    min_tokens_before_end: u32,
    seed: u64,
    num_codebooks: u32,
    n_frames: u32,
    input_frames: u32,
    latent_frames: u32,
    sample_rate: u32,
    num_samples: usize,
    wav_bytes: usize,
}

#[derive(Debug, serde::Serialize)]
struct GeneratedCodesDump {
    backend: &'static str,
    text: String,
    prompt_text: Option<String>,
    prompt_code_cols: Option<usize>,
    temperature: f32,
    top_p: f32,
    top_k: i32,
    max_new_tokens: u32,
    min_tokens_before_end: u32,
    num_codebooks: u32,
    n_frames: u32,
    codes: Vec<i32>,
}

fn main() -> fish_s2_infer::Result<()> {
    let args = Args::parse()?;
    let prompt_file = args
        .prompt_codes
        .as_ref()
        .map(load_prompt_codes_file)
        .transpose()?;
    let prompt_text = match (&args.prompt_text, &prompt_file) {
        (Some(text), _) => Some(text.clone()),
        (None, Some(file)) => Some(file.prompt_text.clone()),
        (None, None) => None,
    };
    let prompt_codes = prompt_file
        .as_ref()
        .map(|file| file.to_prompt_codes())
        .transpose()?;
    if args.prompt_text.is_some() && prompt_codes.is_none() {
        return Err(InferError::Message(
            "--prompt-text requires --prompt-codes".into(),
        ));
    }

    let params = GenerateParams {
        max_new_tokens: args.max_new_tokens,
        temperature: args.temperature,
        top_p: args.top_p,
        top_k: args.top_k,
        min_tokens_before_end: args.min_tokens_before_end,
    };
    let options = RustSynthesisOptions {
        text: args.text.clone(),
        prompt_text: prompt_text.clone(),
        prompt_codes,
        params,
        seed: args.seed,
    };
    let mut pipeline = RustPipeline::load(
        RustPipelineConfig::new(&args.transformer, &args.codec).with_tokenizer(&args.tokenizer),
    )?;
    let result = pipeline.synthesize(&options)?;

    write_bytes(&args.wav, &result.wav_bytes)?;
    write_json(
        &args.codes,
        &GeneratedCodesDump {
            backend: "rust",
            text: args.text.clone(),
            prompt_text: prompt_text.clone(),
            prompt_code_cols: options.prompt_codes.as_ref().map(|codes| codes.cols),
            temperature: args.temperature,
            top_p: args.top_p,
            top_k: args.top_k,
            max_new_tokens: args.max_new_tokens,
            min_tokens_before_end: args.min_tokens_before_end,
            num_codebooks: result.codes.num_codebooks,
            n_frames: result.codes.n_frames,
            codes: result.codes.codes.clone(),
        },
    )?;
    write_json(
        &args.output,
        &E2eWavDump {
            backend: "rust",
            text: args.text,
            prompt_text,
            prompt_code_cols: options.prompt_codes.as_ref().map(|codes| codes.cols),
            temperature: args.temperature,
            top_p: args.top_p,
            top_k: args.top_k,
            max_new_tokens: args.max_new_tokens,
            min_tokens_before_end: args.min_tokens_before_end,
            seed: args.seed,
            num_codebooks: result.codes.num_codebooks,
            n_frames: result.codes.n_frames,
            input_frames: result.waveform.input_frames,
            latent_frames: result.waveform.latent_frames,
            sample_rate: result.waveform.sample_rate,
            num_samples: result.waveform.num_samples,
            wav_bytes: result.wav_bytes.len(),
        },
    )?;
    println!(
        "wrote {} ({} codebooks x {} frames, {} samples @ {} Hz)",
        args.output.display(),
        result.codes.num_codebooks,
        result.codes.n_frames,
        result.waveform.num_samples,
        result.waveform.sample_rate
    );
    println!("wrote {}", args.codes.display());
    println!("wrote {}", args.wav.display());
    Ok(())
}

struct Args {
    transformer: PathBuf,
    codec: PathBuf,
    tokenizer: PathBuf,
    text: String,
    prompt_text: Option<String>,
    prompt_codes: Option<PathBuf>,
    output: PathBuf,
    codes: PathBuf,
    wav: PathBuf,
    max_new_tokens: u32,
    temperature: f32,
    top_p: f32,
    top_k: i32,
    min_tokens_before_end: u32,
    seed: u64,
}

impl Args {
    fn parse() -> fish_s2_infer::Result<Self> {
        let mut transformer = models_dir().join("s2-pro-f16-transformer-only.gguf");
        let mut codec = models_dir().join("s2-pro-f16-codec-only.gguf");
        let mut tokenizer = default_tokenizer_path();
        let mut text = "hi".to_string();
        let mut prompt_text = None;
        let mut prompt_codes = None;
        let mut output = project_root().join("output").join("e2e_wav_hi_rust.json");
        let mut codes = project_root().join("output").join("e2e_codes_hi_rust.json");
        let mut wav = project_root().join("output").join("e2e_hi_rust.wav");
        let mut max_new_tokens = 1u32;
        let mut temperature = 0.0f32;
        let mut top_p = 1.0f32;
        let mut top_k = 0i32;
        let mut min_tokens_before_end = 0u32;
        let mut seed = 0u64;
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--transformer" => {
                    transformer = PathBuf::from(args.next().ok_or_missing("--transformer")?)
                }
                "--codec" => codec = PathBuf::from(args.next().ok_or_missing("--codec")?),
                "--tokenizer" => {
                    tokenizer = PathBuf::from(args.next().ok_or_missing("--tokenizer")?)
                }
                "--text" => text = args.next().ok_or_missing("--text")?,
                "--prompt-text" => prompt_text = Some(args.next().ok_or_missing("--prompt-text")?),
                "--prompt-codes" => {
                    prompt_codes = Some(PathBuf::from(args.next().ok_or_missing("--prompt-codes")?))
                }
                "--output" => output = PathBuf::from(args.next().ok_or_missing("--output")?),
                "--codes" => codes = PathBuf::from(args.next().ok_or_missing("--codes")?),
                "--wav" => wav = PathBuf::from(args.next().ok_or_missing("--wav")?),
                "--max-new-tokens" => {
                    max_new_tokens = args
                        .next()
                        .ok_or_missing("--max-new-tokens")?
                        .parse()
                        .map_err(parse_int_err)?;
                }
                "--temperature" => {
                    temperature = args
                        .next()
                        .ok_or_missing("--temperature")?
                        .parse()
                        .map_err(parse_float_err)?;
                }
                "--top-p" => {
                    top_p = args
                        .next()
                        .ok_or_missing("--top-p")?
                        .parse()
                        .map_err(parse_float_err)?;
                }
                "--top-k" => {
                    top_k = args
                        .next()
                        .ok_or_missing("--top-k")?
                        .parse()
                        .map_err(parse_int_err)?;
                }
                "--min-tokens-before-end" => {
                    min_tokens_before_end = args
                        .next()
                        .ok_or_missing("--min-tokens-before-end")?
                        .parse()
                        .map_err(parse_int_err)?;
                }
                "--seed" => {
                    seed = args
                        .next()
                        .ok_or_missing("--seed")?
                        .parse()
                        .map_err(parse_int_err)?;
                }
                "--help" | "-h" => {
                    print_usage();
                    std::process::exit(0);
                }
                other => return Err(InferError::Message(format!("unknown argument: {other}"))),
            }
        }
        Ok(Self {
            transformer,
            codec,
            tokenizer,
            text,
            prompt_text,
            prompt_codes,
            output,
            codes,
            wav,
            max_new_tokens,
            temperature,
            top_p,
            top_k,
            min_tokens_before_end,
            seed,
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

fn write_bytes(path: &PathBuf, bytes: &[u8]) -> fish_s2_infer::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, bytes)?;
    Ok(())
}

fn write_json<T: serde::Serialize>(path: &PathBuf, value: &T) -> fish_s2_infer::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut file = std::fs::File::create(path)?;
    let json = serde_json::to_string_pretty(value)?;
    file.write_all(json.as_bytes())?;
    file.write_all(b"\n")?;
    Ok(())
}

fn parse_int_err(err: std::num::ParseIntError) -> InferError {
    InferError::Message(err.to_string())
}

fn parse_float_err(err: std::num::ParseFloatError) -> InferError {
    InferError::Message(err.to_string())
}

fn print_usage() {
    eprintln!(
        "Usage: fish_s2_e2e_wav_dump [--transformer model.gguf] [--codec codec.gguf] \
         [--tokenizer tokenizer.json] [--text hi] [--prompt-text <ref>] \
         [--prompt-codes reference_prompt_codes.json] [--output dump.json] \
         [--codes generated_codes.json] [--wav out.wav] [--max-new-tokens 1] \
         [--temperature 0] [--top-p 1] [--top-k 0] [--min-tokens-before-end 0] [--seed 0]"
    );
}
