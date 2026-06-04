use std::io::Write;
use std::path::{Path, PathBuf};

use fish_s2_core::gguf::GgufFile;
use fish_s2_infer::{
    format_codec_dimensions, models_dir, project_root, CodecTensorDumpRow, CodecTensorRegistry,
    InferError,
};

fn main() -> fish_s2_infer::Result<()> {
    let args = Args::parse()?;
    let gguf = GgufFile::open(&args.codec).map_err(|err| InferError::Message(err.to_string()))?;
    let registry = CodecTensorRegistry::from_gguf(&gguf)?;
    let rows = registry.dump_rows(gguf.tensor_data_start)?;
    write_tensor_dump(&args.output, &rows, args.contains.as_deref(), args.limit)?;
    if let Some(path) = &args.metadata_output {
        write_metadata_dump(path, &registry.metadata)?;
    }
    eprintln!("codec={}", gguf.path);
    eprintln!("architecture={}", registry.architecture);
    eprintln!("tensor_count={}", registry.tensor_count);
    for (prefix, count) in registry.prefix_counts() {
        eprintln!("prefix.{prefix}={count}");
    }
    eprintln!("wrote {}", args.output.display());
    Ok(())
}

struct Args {
    codec: PathBuf,
    output: PathBuf,
    metadata_output: Option<PathBuf>,
    contains: Option<String>,
    limit: Option<usize>,
}

impl Args {
    fn parse() -> fish_s2_infer::Result<Self> {
        let mut codec = default_codec_path();
        let mut output = default_output_path();
        let mut metadata_output = None;
        let mut contains = None;
        let mut limit = None;
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--codec" => codec = PathBuf::from(args.next().ok_or_missing("--codec")?),
                "--output" => output = PathBuf::from(args.next().ok_or_missing("--output")?),
                "--metadata-output" => {
                    metadata_output = Some(PathBuf::from(
                        args.next().ok_or_missing("--metadata-output")?,
                    ));
                }
                "--contains" => contains = Some(args.next().ok_or_missing("--contains")?),
                "--limit" => {
                    limit = Some(
                        args.next()
                            .ok_or_missing("--limit")?
                            .parse()
                            .map_err(|err| {
                                InferError::Message(format!("invalid --limit: {err}"))
                            })?,
                    );
                }
                "--help" | "-h" => {
                    print_usage();
                    std::process::exit(0);
                }
                other => {
                    return Err(InferError::Message(format!("unknown argument: {other}")));
                }
            }
        }
        Ok(Self {
            codec,
            output,
            metadata_output,
            contains,
            limit,
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

fn default_codec_path() -> PathBuf {
    models_dir().join("s2-pro-f16-codec-only.gguf")
}

fn default_output_path() -> PathBuf {
    project_root()
        .join("output")
        .join("s2-pro-f16-codec-registry.tsv")
}

fn write_tensor_dump(
    path: &Path,
    rows: &[CodecTensorDumpRow],
    contains: Option<&str>,
    limit: Option<usize>,
) -> fish_s2_infer::Result<()> {
    create_parent(path)?;
    let mut file = std::fs::File::create(path)?;
    writeln!(
        file,
        "index\tcomponent\trole\tmodule\tlayer\tquantizer_index\tname\ttype\tdimensions\telements\tbytes\trelative_offset\tabsolute_offset"
    )?;
    let mut written = 0usize;
    for row in rows {
        if contains.is_some_and(|needle| !row.name.contains(needle)) {
            continue;
        }
        if limit.is_some_and(|max| written >= max) {
            break;
        }
        writeln!(
            file,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:?}\t{}\t{}\t{}\t{}\t{}",
            row.index,
            row.component,
            row.role,
            row.module.as_deref().unwrap_or(""),
            optional_usize(row.layer),
            optional_usize(row.quantizer_index),
            row.name,
            row.ggml_type,
            format_codec_dimensions(&row.dimensions),
            row.elements,
            row.bytes,
            row.relative_offset,
            row.absolute_offset,
        )?;
        written += 1;
    }
    Ok(())
}

fn write_metadata_dump(path: &Path, metadata: &[(String, String)]) -> fish_s2_infer::Result<()> {
    create_parent(path)?;
    let mut file = std::fs::File::create(path)?;
    writeln!(file, "key\tvalue")?;
    for (key, value) in metadata {
        writeln!(file, "{key}\t{value}")?;
    }
    Ok(())
}

fn create_parent(path: &Path) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    Ok(())
}

fn optional_usize(value: Option<usize>) -> String {
    value.map(|v| v.to_string()).unwrap_or_default()
}

fn print_usage() {
    eprintln!(
        "Usage: fish_s2_codec_dump [--codec codec.gguf] [--output codec-registry.tsv] \
         [--metadata-output codec-metadata.tsv] [--contains TEXT] [--limit N]"
    );
}
