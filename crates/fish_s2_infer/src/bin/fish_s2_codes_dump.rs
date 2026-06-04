use std::path::PathBuf;

use fish_s2_core::gguf::GgufFile;
use fish_s2_infer::{
    build_prompt, default_tokenizer_path, generate_codes, load_prompt_codes_file, FastArWeights,
    GenerateParams, PromptBuildOptions, S2Tokenizer, SeededRng, SlowArState,
    TransformerTensorRegistry,
};

#[derive(Debug, serde::Serialize)]
struct GeneratedCodesDump {
    backend: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_code_cols: Option<i32>,
    text: String,
    temperature: f32,
    top_p: f32,
    top_k: i32,
    max_new_tokens: u32,
    min_tokens_before_end: u32,
    prompt_cols: i32,
    num_codebooks: u32,
    n_frames: u32,
    codes: Vec<i32>,
}

fn main() -> fish_s2_infer::Result<()> {
    let args = Args::parse()?;
    let tokenizer = S2Tokenizer::from_file(&args.tokenizer)?;
    let mut state = SlowArState::open_default_max_seq_len(&args.transformer)?;
    let graph = state.graph_spec().clone();
    let gguf = GgufFile::open(&args.transformer)
        .map_err(|err| fish_s2_infer::InferError::Message(err.to_string()))?;
    let registry = TransformerTensorRegistry::from_gguf(&gguf)?;
    let fast_weights = FastArWeights::from_gguf(&gguf, &registry)?;
    let (prompt_text, prompt_codes) = if let Some(path) = &args.prompt_codes {
        let file = load_prompt_codes_file(path)?;
        let ref_text = args
            .prompt_text
            .as_deref()
            .unwrap_or(file.prompt_text.as_str());
        (Some(ref_text.to_string()), Some(file.to_prompt_codes()?))
    } else {
        if args.prompt_text.is_some() {
            return Err(fish_s2_infer::InferError::Message(
                "--prompt-text requires --prompt-codes".into(),
            ));
        }
        (None, None)
    };
    let prompt = build_prompt(
        &tokenizer,
        PromptBuildOptions {
            text: &args.text,
            prompt_text: prompt_text.as_deref(),
            prompt_codes: prompt_codes.as_ref(),
            graph: &graph,
        },
    )?;
    let params = GenerateParams {
        max_new_tokens: args.max_new_tokens,
        temperature: args.temperature,
        top_p: args.top_p,
        top_k: args.top_k,
        min_tokens_before_end: args.min_tokens_before_end,
    };
    let mut rng = SeededRng::new(args.seed);
    let result = generate_codes(
        &mut state,
        &tokenizer.config(),
        &graph,
        &prompt,
        &params,
        &fast_weights,
        &mut rng,
    )?;
    let dump = GeneratedCodesDump {
        backend: "rust",
        prompt_text,
        prompt_code_cols: prompt_codes
            .as_ref()
            .map(|codes| {
                i32::try_from(codes.cols).map_err(|_| {
                    fish_s2_infer::InferError::Message("prompt cols overflows i32".into())
                })
            })
            .transpose()?,
        text: args.text,
        temperature: args.temperature,
        top_p: args.top_p,
        top_k: args.top_k,
        max_new_tokens: args.max_new_tokens,
        min_tokens_before_end: args.min_tokens_before_end,
        prompt_cols: i32::try_from(prompt.cols)
            .map_err(|_| fish_s2_infer::InferError::Message("prompt.cols overflows i32".into()))?,
        num_codebooks: result.num_codebooks,
        n_frames: result.n_frames,
        codes: result.codes,
    };
    let json = serde_json::to_string_pretty(&dump)
        .map_err(|err| fish_s2_infer::InferError::Message(err.to_string()))?;
    std::fs::write(&args.output, json)
        .map_err(|err| fish_s2_infer::InferError::Message(err.to_string()))?;
    println!(
        "wrote {} ({} codebooks x {} frames)",
        args.output.display(),
        dump.num_codebooks,
        dump.n_frames
    );
    Ok(())
}

struct Args {
    transformer: PathBuf,
    tokenizer: PathBuf,
    text: String,
    prompt_text: Option<String>,
    prompt_codes: Option<PathBuf>,
    output: PathBuf,
    max_new_tokens: u32,
    temperature: f32,
    top_p: f32,
    top_k: i32,
    min_tokens_before_end: u32,
    seed: u64,
}

impl Args {
    fn parse() -> fish_s2_infer::Result<Self> {
        let mut transformer = None;
        let mut tokenizer = None;
        let mut text = "hi".to_string();
        let mut prompt_text = None;
        let mut prompt_codes = None;
        let mut output = None;
        let mut max_new_tokens = 2u32;
        let mut temperature = 0.0f32;
        let mut top_p = 1.0f32;
        let mut top_k = 0i32;
        let mut min_tokens_before_end = 0u32;
        let mut seed = 0u64;
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--transformer" => {
                    transformer = Some(PathBuf::from(args.next().ok_or_missing("--transformer")?));
                }
                "--tokenizer" => {
                    tokenizer = Some(PathBuf::from(args.next().ok_or_missing("--tokenizer")?));
                }
                "--text" => text = args.next().ok_or_missing("--text")?,
                "--prompt-text" => {
                    prompt_text = Some(args.next().ok_or_missing("--prompt-text")?);
                }
                "--prompt-codes" => {
                    prompt_codes =
                        Some(PathBuf::from(args.next().ok_or_missing("--prompt-codes")?));
                }
                "--output" => output = Some(PathBuf::from(args.next().ok_or_missing("--output")?)),
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
                other => {
                    return Err(fish_s2_infer::InferError::Message(format!(
                        "unknown argument: {other}"
                    )));
                }
            }
        }
        Ok(Self {
            transformer: transformer.ok_or_else(|| {
                fish_s2_infer::InferError::Message("missing --transformer".into())
            })?,
            tokenizer: tokenizer.unwrap_or_else(default_tokenizer_path),
            text,
            prompt_text,
            prompt_codes,
            output: output
                .ok_or_else(|| fish_s2_infer::InferError::Message("missing --output".into()))?,
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
        self.ok_or_else(|| fish_s2_infer::InferError::Message(format!("missing {flag}")))
    }
}

fn parse_int_err(err: std::num::ParseIntError) -> fish_s2_infer::InferError {
    fish_s2_infer::InferError::Message(err.to_string())
}

fn parse_float_err(err: std::num::ParseFloatError) -> fish_s2_infer::InferError {
    fish_s2_infer::InferError::Message(err.to_string())
}

fn print_usage() {
    eprintln!(
        "Usage: fish_s2_codes_dump --transformer <gguf> --output <codes.json> \
         [--tokenizer tokenizer.json] [--text hi] [--prompt-text <ref>] [--prompt-codes <codes.json>] \
         [--max-new-tokens 2] [--temperature 0] [--top-p 1] [--top-k 0] \
         [--min-tokens-before-end 0] [--seed 0]"
    );
}
