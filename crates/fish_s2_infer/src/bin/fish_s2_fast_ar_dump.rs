use std::path::PathBuf;

use fish_s2_core::gguf::GgufFile;
use fish_s2_infer::{
    build_prompt, default_tokenizer_path, generate_fast_ar_first_frame, FastArWeights,
    GenerateParams, PromptBuildOptions, S2Tokenizer, SeededRng, SlowArState,
    TransformerTensorRegistry,
};

#[derive(Debug, serde::Serialize)]
struct FastArFrameDump {
    backend: &'static str,
    text: String,
    temperature: f32,
    top_p: f32,
    top_k: i32,
    min_tokens_before_end: u32,
    prompt_cols: i32,
    main_token_id: u32,
    slow_hidden: Vec<f32>,
    codebook_ids: Vec<u32>,
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
    let prompt = build_prompt(
        &tokenizer,
        PromptBuildOptions {
            text: &args.text,
            prompt_text: None,
            prompt_codes: None,
            graph: &graph,
        },
    )?;
    let gen = GenerateParams {
        max_new_tokens: 0,
        temperature: args.temperature,
        top_p: args.top_p,
        top_k: args.top_k,
        min_tokens_before_end: args.min_tokens_before_end,
    };
    let mut rng = SeededRng::new(args.seed);
    let result = generate_fast_ar_first_frame(
        &mut state,
        &tokenizer.config(),
        &graph,
        &prompt,
        &gen,
        &fast_weights,
        &mut rng,
    )?;
    let dump = FastArFrameDump {
        backend: "rust",
        text: args.text,
        temperature: args.temperature,
        top_p: args.top_p,
        top_k: args.top_k,
        min_tokens_before_end: args.min_tokens_before_end,
        prompt_cols: i32::try_from(result.prompt_cols)
            .map_err(|_| fish_s2_infer::InferError::Message("prompt.cols overflows i32".into()))?,
        main_token_id: result.main_token_id,
        slow_hidden: result.slow_hidden,
        codebook_ids: result.codebook_ids,
    };
    let json = serde_json::to_string_pretty(&dump)
        .map_err(|err| fish_s2_infer::InferError::Message(err.to_string()))?;
    std::fs::write(&args.output, json)
        .map_err(|err| fish_s2_infer::InferError::Message(err.to_string()))?;
    println!(
        "wrote {} ({} codebooks)",
        args.output.display(),
        dump.codebook_ids.len()
    );
    Ok(())
}

struct Args {
    transformer: PathBuf,
    tokenizer: PathBuf,
    text: String,
    output: PathBuf,
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
        let mut output = None;
        let mut temperature = 0.0f32;
        let mut top_p = 1.0f32;
        let mut top_k = 0i32;
        let mut min_tokens_before_end = 0u32;
        let mut seed = 0u64;
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--transformer" => {
                    transformer = Some(PathBuf::from(args.next().ok_or_else(|| {
                        fish_s2_infer::InferError::Message("missing --transformer value".into())
                    })?));
                }
                "--tokenizer" => {
                    tokenizer = Some(PathBuf::from(args.next().ok_or_else(|| {
                        fish_s2_infer::InferError::Message("missing --tokenizer value".into())
                    })?));
                }
                "--output" => {
                    output = Some(PathBuf::from(args.next().ok_or_else(|| {
                        fish_s2_infer::InferError::Message("missing --output value".into())
                    })?));
                }
                "--text" => {
                    text = args.next().ok_or_else(|| {
                        fish_s2_infer::InferError::Message("missing --text value".into())
                    })?;
                }
                "--temperature" => {
                    temperature = args
                        .next()
                        .ok_or_else(|| {
                            fish_s2_infer::InferError::Message("missing --temperature".into())
                        })?
                        .parse()
                        .map_err(|err| {
                            fish_s2_infer::InferError::Message(format!("--temperature: {err}"))
                        })?;
                }
                "--top-p" => {
                    top_p = args
                        .next()
                        .ok_or_else(|| {
                            fish_s2_infer::InferError::Message("missing --top-p".into())
                        })?
                        .parse()
                        .map_err(|err| {
                            fish_s2_infer::InferError::Message(format!("--top-p: {err}"))
                        })?;
                }
                "--top-k" => {
                    top_k = args
                        .next()
                        .ok_or_else(|| {
                            fish_s2_infer::InferError::Message("missing --top-k".into())
                        })?
                        .parse()
                        .map_err(|err| {
                            fish_s2_infer::InferError::Message(format!("--top-k: {err}"))
                        })?;
                }
                "--min-tokens-before-end" => {
                    min_tokens_before_end = args
                        .next()
                        .ok_or_else(|| {
                            fish_s2_infer::InferError::Message(
                                "missing --min-tokens-before-end".into(),
                            )
                        })?
                        .parse()
                        .map_err(|err| {
                            fish_s2_infer::InferError::Message(format!(
                                "--min-tokens-before-end: {err}"
                            ))
                        })?;
                }
                "--seed" => {
                    seed = args
                        .next()
                        .ok_or_else(|| fish_s2_infer::InferError::Message("missing --seed".into()))?
                        .parse()
                        .map_err(|err| {
                            fish_s2_infer::InferError::Message(format!("--seed: {err}"))
                        })?;
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
            output: output
                .ok_or_else(|| fish_s2_infer::InferError::Message("missing --output".into()))?,
            temperature,
            top_p,
            top_k,
            min_tokens_before_end,
            seed,
        })
    }
}

fn print_usage() {
    eprintln!(
        "Usage: fish_s2_fast_ar_dump --transformer <gguf> --output <frame.json> \
         [--tokenizer tokenizer.json] [--text hi] [--temperature 0] [--top-p 1] [--top-k 0] \
         [--min-tokens-before-end 0] [--seed 0]"
    );
}
