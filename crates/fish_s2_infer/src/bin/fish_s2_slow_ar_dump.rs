use std::path::PathBuf;

use fish_s2_core::gguf::GgufFile;
use fish_s2_infer::{
    SlowArKvCache, SlowArLayerF16Weights, SlowArLayerForwardOutput, SlowArLayerShape,
    TransformerTensorRegistry,
};

#[derive(Debug, serde::Serialize)]
struct Dump {
    transformer: String,
    layer: usize,
    position: usize,
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
}

#[derive(Debug, serde::Serialize)]
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
    let weights = SlowArLayerF16Weights::from_gguf_layer(&gguf, &registry, args.layer)?;

    let mut hidden = vec![0.0f32; shape.hidden_size];
    hidden[0] = 1.0;
    hidden[1] = -0.5;
    hidden[shape.hidden_size - 1] = 0.25;

    let mut cache = SlowArKvCache::new(graph.kv_cache, args.position + 1)?;
    let output = weights.skeleton(shape).forward_decode_token(
        &hidden,
        &mut cache,
        args.layer,
        args.position,
    )?;

    let dump = build_dump(&args, shape, output);
    let json = serde_json::to_string_pretty(&dump)?;
    if let Some(path) = args.output {
        std::fs::write(path, json)?;
    } else {
        println!("{json}");
    }
    Ok(())
}

fn build_dump(args: &Args, shape: SlowArLayerShape, output: SlowArLayerForwardOutput) -> Dump {
    Dump {
        transformer: args.transformer.display().to_string(),
        layer: args.layer,
        position: args.position,
        hidden_size: shape.hidden_size,
        head_count: shape.head_count,
        head_count_kv: shape.head_count_kv,
        head_dim: shape.head_dim,
        normalized: stats(&output.normalized),
        query: stats(&output.query),
        key: stats(&output.key),
        value: stats(&output.value),
        attention: stats(&output.attention),
        projected: stats(&output.projected),
        hidden: stats(&output.hidden),
    }
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

#[derive(Debug)]
struct Args {
    transformer: PathBuf,
    output: Option<PathBuf>,
    layer: usize,
    position: usize,
}

impl Args {
    fn parse() -> fish_s2_infer::Result<Self> {
        let mut transformer = None;
        let mut output = None;
        let mut layer = 0usize;
        let mut position = 0usize;
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--transformer" => transformer = args.next().map(PathBuf::from),
                "--output" => output = args.next().map(PathBuf::from),
                "--layer" => layer = parse_usize("--layer", args.next())?,
                "--position" => position = parse_usize("--position", args.next())?,
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
            position,
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

fn print_help() {
    eprintln!(
        "Usage: fish_s2_slow_ar_dump --transformer <s2-pro-*-transformer-only.gguf> [--output output.json] [--layer 0] [--position 0]"
    );
}
