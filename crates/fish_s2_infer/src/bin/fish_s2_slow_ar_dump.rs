use std::path::PathBuf;

use fish_s2_core::gguf::GgufFile;
use fish_s2_infer::{
    forward_slow_ar_block_prefill_layers, SlowArLayerBlockOutput, SlowArLayerShape,
    SlowArOutputHeadF16Weights, TransformerTensorRegistry,
};

#[derive(Debug, serde::Serialize)]
struct Dump {
    transformer: String,
    layer: usize,
    layer_count: usize,
    position: usize,
    token_count: usize,
    hidden_size: usize,
    head_count: usize,
    head_count_kv: usize,
    head_dim: usize,
    normalized: TensorStats,
    query: TensorStats,
    key: TensorStats,
    value: TensorStats,
    attention: TensorStats,
    projected: TensorStats,
    hidden: TensorStats,
    ffn_normalized: TensorStats,
    ffn_gate: TensorStats,
    ffn_up: TensorStats,
    ffn_activated: TensorStats,
    ffn_projected: TensorStats,
    block_hidden: TensorStats,
    #[serde(skip_serializing_if = "Option::is_none")]
    final_normalized: Option<TensorStats>,
    #[serde(skip_serializing_if = "Option::is_none")]
    logits: Option<TensorStats>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    top_logits: Vec<TopLogit>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    sequence: Vec<TokenDump>,
}

#[derive(Debug, Clone, serde::Serialize)]
struct TokenDump {
    position: usize,
    normalized: TensorStats,
    query: TensorStats,
    key: TensorStats,
    value: TensorStats,
    attention: TensorStats,
    projected: TensorStats,
    hidden: TensorStats,
    ffn_normalized: TensorStats,
    ffn_gate: TensorStats,
    ffn_up: TensorStats,
    ffn_activated: TensorStats,
    ffn_projected: TensorStats,
    block_hidden: TensorStats,
}

#[derive(Debug, Clone, serde::Serialize)]
struct TopLogit {
    token_id: usize,
    value: f32,
}

#[derive(Debug, Clone, serde::Serialize)]
struct TensorStats {
    len: usize,
    l2: f32,
    mean_abs: f32,
    max_abs: f32,
    first8: Vec<f32>,
}

fn main() -> fish_s2_infer::Result<()> {
    let args = Args::parse()?;
    let gguf = GgufFile::open(&args.transformer)
        .map_err(|err| fish_s2_infer::InferError::Message(err.to_string()))?;
    let registry = TransformerTensorRegistry::from_gguf(&gguf)?;
    let graph = registry.graph_spec();
    let shape = SlowArLayerShape::from_ar_graph_spec(&graph.slow)?;

    let hidden_tokens = (0..args.tokens)
        .map(|index| hidden_fixture(shape.hidden_size, index))
        .collect::<Vec<_>>();
    let outputs = forward_slow_ar_block_prefill_layers(
        &gguf,
        &registry,
        args.layer,
        args.layers,
        &hidden_tokens,
        args.position,
    )?;

    let logits = if args.logits {
        let last_hidden = outputs
            .last()
            .ok_or_else(|| fish_s2_infer::InferError::Message("no Slow-AR outputs".into()))?
            .hidden
            .clone();
        Some(
            SlowArOutputHeadF16Weights::from_gguf(&gguf)?
                .forward_logits(&last_hidden, shape.rms_norm_eps)?,
        )
    } else {
        None
    };

    let dump = build_dump(&args, shape, outputs, logits.as_ref());
    let json = serde_json::to_string_pretty(&dump)?;
    if let Some(path) = args.output {
        std::fs::write(path, json)?;
    } else {
        println!("{json}");
    }
    Ok(())
}

fn build_dump(
    args: &Args,
    shape: SlowArLayerShape,
    outputs: Vec<SlowArLayerBlockOutput>,
    logits: Option<&fish_s2_infer::SlowArLogitsOutput>,
) -> Dump {
    let mut sequence = outputs
        .iter()
        .enumerate()
        .map(|(offset, output)| build_token_dump(args.position + offset, output))
        .collect::<Vec<_>>();
    let first = sequence
        .first()
        .expect("at least one Slow-AR dump token")
        .clone();
    if args.tokens == 1 {
        sequence.clear();
    }
    let final_normalized = logits.map(|logits| stats(&logits.normalized));
    let logits_stats = logits.map(|logits| stats(&logits.logits));
    let top_logits = logits.map_or_else(Vec::new, |logits| top_logits(&logits.logits, args.top_k));
    Dump {
        transformer: args.transformer.display().to_string(),
        layer: args.layer,
        layer_count: args.layers,
        position: args.position,
        token_count: args.tokens,
        hidden_size: shape.hidden_size,
        head_count: shape.head_count,
        head_count_kv: shape.head_count_kv,
        head_dim: shape.head_dim,
        normalized: first.normalized,
        query: first.query,
        key: first.key,
        value: first.value,
        attention: first.attention,
        projected: first.projected,
        hidden: first.hidden,
        ffn_normalized: first.ffn_normalized,
        ffn_gate: first.ffn_gate,
        ffn_up: first.ffn_up,
        ffn_activated: first.ffn_activated,
        ffn_projected: first.ffn_projected,
        block_hidden: first.block_hidden,
        final_normalized,
        logits: logits_stats,
        top_logits,
        sequence,
    }
}

fn build_token_dump(position: usize, output: &SlowArLayerBlockOutput) -> TokenDump {
    TokenDump {
        position,
        normalized: stats(&output.attention.normalized),
        query: stats(&output.attention.query),
        key: stats(&output.attention.key),
        value: stats(&output.attention.value),
        attention: stats(&output.attention.attention),
        projected: stats(&output.attention.projected),
        hidden: stats(&output.attention.hidden),
        ffn_normalized: stats(&output.feed_forward.normalized),
        ffn_gate: stats(&output.feed_forward.gate),
        ffn_up: stats(&output.feed_forward.up),
        ffn_activated: stats(&output.feed_forward.activated),
        ffn_projected: stats(&output.feed_forward.projected),
        block_hidden: stats(&output.hidden),
    }
}

fn hidden_fixture(hidden_size: usize, token_index: usize) -> Vec<f32> {
    let mut hidden = vec![0.0f32; hidden_size];
    hidden[0] = 1.0;
    hidden[1] = -0.5 + token_index as f32;
    hidden[hidden_size - 1] = 0.25 + token_index as f32 * 0.125;
    hidden
}

fn stats(values: &[f32]) -> TensorStats {
    let len = values.len();
    let l2 = values.iter().map(|value| value * value).sum::<f32>().sqrt();
    let mean_abs = if values.is_empty() {
        0.0
    } else {
        values.iter().map(|value| value.abs()).sum::<f32>() / len as f32
    };
    let max_abs = values
        .iter()
        .map(|value| value.abs())
        .fold(0.0f32, f32::max);
    TensorStats {
        len,
        l2,
        mean_abs,
        max_abs,
        first8: values.iter().take(8).copied().collect(),
    }
}

fn top_logits(values: &[f32], count: usize) -> Vec<TopLogit> {
    let mut top = values
        .iter()
        .enumerate()
        .map(|(token_id, value)| TopLogit {
            token_id,
            value: *value,
        })
        .collect::<Vec<_>>();
    top.sort_by(|left, right| {
        right
            .value
            .total_cmp(&left.value)
            .then_with(|| left.token_id.cmp(&right.token_id))
    });
    top.truncate(count.min(top.len()));
    top
}

#[derive(Debug)]
struct Args {
    transformer: PathBuf,
    output: Option<PathBuf>,
    layer: usize,
    layers: usize,
    position: usize,
    tokens: usize,
    logits: bool,
    top_k: usize,
}

impl Args {
    fn parse() -> fish_s2_infer::Result<Self> {
        let mut transformer = None;
        let mut output = None;
        let mut layer = 0usize;
        let mut layers = 1usize;
        let mut position = 0usize;
        let mut tokens = 1usize;
        let mut logits = false;
        let mut top_k = 8usize;
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--transformer" => transformer = args.next().map(PathBuf::from),
                "--output" => output = args.next().map(PathBuf::from),
                "--layer" => layer = parse_usize("--layer", args.next())?,
                "--layers" => layers = parse_nonzero_usize("--layers", args.next())?,
                "--position" => position = parse_usize("--position", args.next())?,
                "--tokens" => tokens = parse_nonzero_usize("--tokens", args.next())?,
                "--logits" => logits = true,
                "--top-k" => top_k = parse_nonzero_usize("--top-k", args.next())?,
                "--help" | "-h" => {
                    print_help();
                    std::process::exit(0);
                }
                other => {
                    return Err(fish_s2_infer::InferError::Message(format!(
                        "unknown argument: {other}"
                    )));
                }
            }
        }
        let transformer = transformer.ok_or_else(|| {
            fish_s2_infer::InferError::Message("missing --transformer <path>".into())
        })?;
        Ok(Self {
            transformer,
            output,
            layer,
            layers,
            position,
            tokens,
            logits,
            top_k,
        })
    }
}

fn parse_usize(name: &str, value: Option<String>) -> fish_s2_infer::Result<usize> {
    let value = value
        .ok_or_else(|| fish_s2_infer::InferError::Message(format!("missing value for {name}")))?;
    value.parse().map_err(|err| {
        fish_s2_infer::InferError::Message(format!("invalid {name} value {value}: {err}"))
    })
}

fn parse_nonzero_usize(name: &str, value: Option<String>) -> fish_s2_infer::Result<usize> {
    let value = parse_usize(name, value)?;
    if value == 0 {
        return Err(fish_s2_infer::InferError::Message(format!(
            "{name} must be greater than zero"
        )));
    }
    Ok(value)
}

fn print_help() {
    eprintln!(
        "Usage: fish_s2_slow_ar_dump --transformer <s2-pro-*-transformer-only.gguf> [--output output.json] [--layer 0] [--layers 1] [--position 0] [--tokens 1] [--logits] [--top-k 8]"
    );
}
