use std::env;
use std::fs::File;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use fish_s2_core::gguf::GgufFile;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let mut args = env::args().skip(1);
    let path = args.next().ok_or_else(usage)?;
    let mut limit = None;
    let mut contains = None;
    let mut output = None;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--limit" => {
                let value = args.next().ok_or_else(usage)?;
                limit = Some(
                    value
                        .parse::<usize>()
                        .map_err(|err| format!("invalid --limit value: {err}"))?,
                );
            }
            "--contains" => {
                contains = Some(args.next().ok_or_else(usage)?);
            }
            "--output" => {
                output = Some(PathBuf::from(args.next().ok_or_else(usage)?));
            }
            _ => return Err(usage()),
        }
    }

    let gguf = GgufFile::open(&path).map_err(|err| err.to_string())?;
    eprintln!("path={}", gguf.path);
    eprintln!("version={}", gguf.version);
    eprintln!("tensor_count={}", gguf.tensors.len());
    eprintln!("metadata_count={}", gguf.metadata.len());
    eprintln!("tensor_data_start={}", gguf.tensor_data_start);
    if let Some(arch) = gguf
        .metadata
        .iter()
        .find(|(key, _)| key == "general.architecture")
        .map(|(_, value)| value)
    {
        eprintln!("general.architecture={arch}");
    }

    let mut writer: Box<dyn Write> = match output {
        Some(path) => Box::new(create_parented(&path)?),
        None => Box::new(io::stdout()),
    };
    writeln!(
        writer,
        "index\tname\ttype\tdimensions\telements\tbytes\trelative_offset\tabsolute_offset"
    )
    .map_err(|err| err.to_string())?;

    let mut written = 0usize;
    for (index, tensor) in gguf.tensors.iter().enumerate() {
        if contains
            .as_deref()
            .is_some_and(|needle| !tensor.name.contains(needle))
        {
            continue;
        }
        if limit.is_some_and(|max| written >= max) {
            break;
        }
        writeln!(
            writer,
            "{index}\t{}\t{:?}\t{}\t{}\t{}\t{}\t{}",
            tensor.name,
            tensor.ggml_type,
            format_dimensions(&tensor.dimensions),
            tensor.element_count().map_err(|err| err.to_string())?,
            tensor.byte_len().map_err(|err| err.to_string())?,
            tensor.relative_offset,
            tensor.absolute_offset(gguf.tensor_data_start),
        )
        .map_err(|err| err.to_string())?;
        written += 1;
    }

    Ok(())
}

fn create_parented(path: &Path) -> Result<File, String> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|err| err.to_string())?;
    }
    File::create(path).map_err(|err| err.to_string())
}

fn format_dimensions(dimensions: &[u64]) -> String {
    dimensions
        .iter()
        .map(u64::to_string)
        .collect::<Vec<_>>()
        .join("x")
}

fn usage() -> String {
    "usage: fish_s2_gguf_dump <model.gguf> [--limit N] [--contains TEXT] [--output tensors.tsv]"
        .to_string()
}
