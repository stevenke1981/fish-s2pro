use std::path::PathBuf;

use fish_s2_infer::{
    build_prompt, default_tokenizer_path, generate_semantic_tokens, GenerateParams,
    PromptBuildOptions, S2Tokenizer, SeededRng, SlowArState,
};

#[derive(Debug, serde::Serialize)]
struct SemanticTokenDump {
    backend: &'static str,
    text: String,
    temperature: f32,
    top_p: f32,
    top_k: i32,
    max_new_tokens: u32,
    min_tokens_before_end: u32,
    prompt_cols: i32,
    main_token_ids: Vec<u32>,
}

fn main() -> fish_s2_infer::Result<()> {
    let args = Args::parse()?;
    let tokenizer = S2Tokenizer::from_file(&args.tokenizer)?;
    let mut state = SlowArState::open_default_max_seq_len(&args.transformer)?;
    let graph = state.graph_spec().clone();
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
        max_new_tokens: args.max_new_tokens,
        temperature: args.temperature,
        top_p: args.top_p,
        top_k: args.top_k,
        min_tokens_before_end: args.min_tokens_before_end,
    };
    let mut rng = SeededRng::new(args.seed);
    let result = generate_semantic_tokens(
        &mut state,
        &tokenizer.config(),
        &graph,
        &prompt,
        &gen,
        &mut rng,
    )?;
    let dump = SemanticTokenDump {
        backend: "rust",
        text: args.text,
        temperature: args.temperature,
        top_p: args.top_p,
        top_k: args.top_k,
        max_new_tokens: args.max_new_tokens,
        min_tokens_before_end: args.min_tokens_before_end,
        prompt_cols: i32::try_from(prompt.cols)
            .map_err(|_| fish_s2_infer::InferError::Message("prompt.cols overflows i32".into()))?,
        main_token_ids: result.token_ids,
    };
    let json = serde_json::to_string_pretty(&dump)
        .map_err(|err| fish_s2_infer::InferError::Message(err.to_string()))?;
    std::fs::write(&args.output, json)
        .map_err(|err| fish_s2_infer::InferError::Message(err.to_string()))?;
    println!(
        "wrote {} ({} tokens)",
        args.output.display(),
        dump.main_token_ids.len()
    );
    Ok(())
}

struct Args {
    transformer: PathBuf,
    tokenizer: PathBuf,
    text: String,
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
        let mut output = None;
        let mut max_new_tokens = 4u32;
        let mut temperature = 0.0f32;
        let mut top_p = 1.0f32;
        let mut top_k = 0i32;
        let mut min_tokens_before_end = 0u32;
        let mut seed = 0u64;
        let mut argv = std::env::args().skip(1);
        while let Some(arg) = argv.next() {
            match arg.as_str() {
                "--transformer" => {
                    transformer = Some(PathBuf::from(argv.next().ok_or_missing("--transformer")?));
                }
                "--tokenizer" => {
                    tokenizer = Some(PathBuf::from(argv.next().ok_or_missing("--tokenizer")?));
                }
                "--text" => text = argv.next().ok_or_missing("--text")?,
                "--output" => output = Some(PathBuf::from(argv.next().ok_or_missing("--output")?)),
                "--max-new-tokens" => {
                    max_new_tokens = argv
                        .next()
                        .ok_or_missing("--max-new-tokens")?
                        .parse()
                        .map_err(parse_err)?;
                }
                "--temperature" => {
                    temperature = argv
                        .next()
                        .ok_or_missing("--temperature")?
                        .parse()
                        .map_err(parse_f32_err)?;
                }
                "--top-p" => {
                    top_p = argv
                        .next()
                        .ok_or_missing("--top-p")?
                        .parse()
                        .map_err(parse_f32_err)?;
                }
                "--top-k" => {
                    top_k = argv
                        .next()
                        .ok_or_missing("--top-k")?
                        .parse()
                        .map_err(parse_err)?;
                }
                "--min-tokens-before-end" => {
                    min_tokens_before_end = argv
                        .next()
                        .ok_or_missing("--min-tokens-before-end")?
                        .parse()
                        .map_err(parse_err)?;
                }
                "--seed" => {
                    seed = argv
                        .next()
                        .ok_or_missing("--seed")?
                        .parse()
                        .map_err(parse_err)?;
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

fn parse_err(err: std::num::ParseIntError) -> fish_s2_infer::InferError {
    fish_s2_infer::InferError::Message(err.to_string())
}

fn parse_f32_err(err: std::num::ParseFloatError) -> fish_s2_infer::InferError {
    fish_s2_infer::InferError::Message(err.to_string())
}

fn print_usage() {
    eprintln!(
        "Usage: fish_s2_semantic_dump --transformer <gguf> --output <tokens.json> \
         [--tokenizer tokenizer.json] [--text hi] [--max-new-tokens 4] [--temperature 0] \
         [--top-p 1] [--top-k 0] [--min-tokens-before-end 0] [--seed 0]"
    );
}
