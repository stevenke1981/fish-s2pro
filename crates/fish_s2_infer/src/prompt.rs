//! Prompt tensor construction aligned with `s2.cpp` `build_prompt()`.

use std::path::Path;

use crate::error::{InferError, Result};
use crate::registry::DualArGraphSpec;
use crate::tokenizer::S2Tokenizer;

/// Codebook-major prompt codes: `data[cb * cols + t]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromptCodes {
    pub num_codebooks: u32,
    pub cols: usize,
    pub data: Vec<u32>,
}

/// JSON fixture for reference-prompt parity (`cols` = `T_prompt` timesteps).
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize, PartialEq, Eq)]
pub struct PromptCodesFile {
    pub prompt_text: String,
    pub num_codebooks: u32,
    pub cols: usize,
    pub codes: Vec<u32>,
}

impl PromptCodesFile {
    pub fn to_prompt_codes(&self) -> Result<PromptCodes> {
        let expected = self
            .cols
            .checked_mul(self.num_codebooks as usize)
            .ok_or_else(|| InferError::Message("prompt codes length overflow".into()))?;
        if self.codes.len() != expected {
            return Err(InferError::Message(format!(
                "prompt codes length mismatch: expected {expected} (cols={} * num_codebooks={}), got {}",
                self.cols,
                self.num_codebooks,
                self.codes.len()
            )));
        }
        Ok(PromptCodes {
            num_codebooks: self.num_codebooks,
            cols: self.cols,
            data: self.codes.clone(),
        })
    }
}

pub fn load_prompt_codes_file(path: impl AsRef<Path>) -> Result<PromptCodesFile> {
    let bytes = std::fs::read(path.as_ref())
        .map_err(|err| InferError::Message(format!("read prompt codes: {err}")))?;
    serde_json::from_slice(&bytes)
        .map_err(|err| InferError::Message(format!("parse prompt codes json: {err}")))
}

pub fn load_prompt_codes(path: impl AsRef<Path>) -> Result<PromptCodes> {
    load_prompt_codes_file(path)?.to_prompt_codes()
}

/// Row-major prompt layout from `build_prompt()`:
/// - row 0: vocabulary-space token ids
/// - rows 1..num_codebooks: codebook values (0 for plain text positions)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromptTensor {
    pub rows: u32,
    pub cols: usize,
    pub data: Vec<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PromptBuildOptions<'a> {
    pub text: &'a str,
    pub prompt_text: Option<&'a str>,
    pub prompt_codes: Option<&'a PromptCodes>,
    pub graph: &'a DualArGraphSpec,
}

/// Transpose `(rows, cols)` codebook-major storage to s2.cpp time-major flat layout:
/// `flat[t * rows + r] = data[r * cols + t]`.
pub fn transpose_to_time_major(tensor: &PromptTensor) -> Result<Vec<i32>> {
    let rows = usize::try_from(tensor.rows)
        .map_err(|_| InferError::Message("prompt rows overflows usize".into()))?;
    if rows == 0 {
        return Err(InferError::Message("prompt rows must be non-zero".into()));
    }
    let cols = tensor.cols;
    let expected = rows
        .checked_mul(cols)
        .ok_or_else(|| InferError::Message("prompt data length overflow".into()))?;
    if tensor.data.len() != expected {
        return Err(InferError::Message(format!(
            "prompt data length mismatch: expected {expected}, got {}",
            tensor.data.len()
        )));
    }
    let mut flat = vec![0i32; expected];
    for t in 0..cols {
        for r in 0..rows {
            flat[t * rows + r] = i32::try_from(tensor.data[r * cols + t])
                .map_err(|_| InferError::Message("prompt token id does not fit i32".into()))?;
        }
    }
    Ok(flat)
}

pub fn build_prompt(
    tokenizer: &S2Tokenizer,
    options: PromptBuildOptions<'_>,
) -> Result<PromptTensor> {
    let num_codebooks = options.graph.num_codebooks;
    let rows = num_codebooks + 1;
    let sem_begin = options.graph.semantic_begin_id;
    let im_end_id = tokenizer.config().im_end_id;
    let voice_id = tokenizer.config().voice_id;

    let (prompt_text, prompt_codes) = match (options.prompt_text, options.prompt_codes) {
        (Some(text), Some(codes)) => (text, Some(codes)),
        (None, None) => ("", None),
        _ => {
            return Err(InferError::Message(
                "prompt_text and prompt_codes must both be set or both omitted".into(),
            ));
        }
    };

    let has_reference = prompt_codes.is_some_and(|codes| {
        !prompt_text.is_empty() && codes.cols > 0 && codes.num_codebooks == num_codebooks
    });

    let newline = tokenizer.encode_newline()?;
    let mut sys_pre = Vec::new();
    let mut sys_post = Vec::new();

    if has_reference {
        let codes = prompt_codes.expect("has_reference implies prompt_codes");
        append_ids(
            &mut sys_pre,
            &tokenizer.encode_s2cpp("<|im_start|>system")?.ids,
        );
        append_ids(&mut sys_pre, &newline);
        append_ids(
            &mut sys_pre,
            &tokenizer
                .encode_s2cpp(
                    "convert the provided text to speech reference to the following:\n\nText:\n",
                )?
                .ids,
        );
        append_ids(&mut sys_pre, &tokenizer.encode_s2cpp("<|speaker:0|>")?.ids);
        append_ids(&mut sys_pre, &tokenizer.encode_s2cpp(prompt_text)?.ids);
        append_ids(&mut sys_pre, &tokenizer.encode_s2cpp("\n\nSpeech:\n")?.ids);

        let t_prompt = codes.cols;
        append_u32(&mut sys_post, im_end_id);
        append_ids(&mut sys_post, &newline);
        append_ids(
            &mut sys_post,
            &tokenizer.encode_s2cpp("<|im_start|>user")?.ids,
        );
        append_ids(&mut sys_post, &newline);
        append_ids(&mut sys_post, &tokenizer.encode_s2cpp(options.text)?.ids);
        append_u32(&mut sys_post, im_end_id);
        append_ids(&mut sys_post, &newline);
        append_ids(
            &mut sys_post,
            &tokenizer.encode_s2cpp("<|im_start|>assistant")?.ids,
        );
        append_ids(&mut sys_post, &newline);
        append_u32(&mut sys_post, voice_id);

        let total_len = sys_pre.len() + t_prompt + sys_post.len();
        let mut data = vec![0u32; rows as usize * total_len];
        let mut pos = 0usize;

        write_row0(&mut data, &mut pos, &sys_pre);
        for t in 0..t_prompt {
            data[pos + t] = codes.data[t]
                .checked_add(sem_begin)
                .ok_or_else(|| InferError::Message("prompt semantic id overflow".into()))?;
        }
        for cb in 0..num_codebooks as usize {
            for t in 0..t_prompt {
                data[(cb + 1) * total_len + pos + t] = codes.data[cb * t_prompt + t];
            }
        }
        pos += t_prompt;
        write_row0(&mut data, &mut pos, &sys_post);

        return Ok(PromptTensor {
            rows,
            cols: total_len,
            data,
        });
    }

    append_ids(
        &mut sys_post,
        &tokenizer.encode_s2cpp("<|im_start|>user")?.ids,
    );
    append_ids(&mut sys_post, &newline);
    append_ids(&mut sys_post, &tokenizer.encode_s2cpp(options.text)?.ids);
    append_u32(&mut sys_post, im_end_id);
    append_ids(&mut sys_post, &newline);
    append_ids(
        &mut sys_post,
        &tokenizer.encode_s2cpp("<|im_start|>assistant")?.ids,
    );
    append_ids(&mut sys_post, &newline);
    append_u32(&mut sys_post, voice_id);

    let total_len = sys_post.len();
    let mut data = vec![0u32; rows as usize * total_len];
    let mut pos = 0usize;
    write_row0(&mut data, &mut pos, &sys_post);

    Ok(PromptTensor {
        rows,
        cols: total_len,
        data,
    })
}

fn write_row0(data: &mut [u32], pos: &mut usize, ids: &[u32]) {
    for (offset, id) in ids.iter().enumerate() {
        data[*pos + offset] = *id;
    }
    *pos += ids.len();
}

fn append_ids(dst: &mut Vec<u32>, src: &[u32]) {
    dst.extend_from_slice(src);
}

fn append_u32(dst: &mut Vec<u32>, value: u32) {
    dst.push(value);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_codes_file_validates_length() {
        let file = PromptCodesFile {
            prompt_text: "ref".into(),
            num_codebooks: 2,
            cols: 3,
            codes: vec![1, 2, 3, 4, 5, 6],
        };
        let codes = file.to_prompt_codes().unwrap();
        assert_eq!(codes.cols, 3);
        assert_eq!(codes.data.len(), 6);
    }

    #[test]
    #[ignore = "requires local models/tokenizer.json and transformer GGUF"]
    fn reference_prompt_cols_matches_cpp_golden() {
        use crate::registry::DualArGraphSpec;
        use crate::tokenizer::S2Tokenizer;
        use fish_s2_core::gguf::GgufFile;
        use std::path::PathBuf;

        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        let tokenizer = S2Tokenizer::from_file(root.join("models/tokenizer.json")).unwrap();
        let gguf = GgufFile::open(root.join("models/s2-pro-f16-transformer-only.gguf")).unwrap();
        let graph = DualArGraphSpec::from_gguf(&gguf).unwrap();
        let file = load_prompt_codes_file(
            root.join("crates/fish_s2_parity/tests/fixtures/reference_prompt_codes.json"),
        )
        .unwrap();
        let codes = file.to_prompt_codes().unwrap();
        let prompt = build_prompt(
            &tokenizer,
            PromptBuildOptions {
                text: "hi",
                prompt_text: Some(file.prompt_text.as_str()),
                prompt_codes: Some(&codes),
                graph: &graph,
            },
        )
        .unwrap();
        // s2.cpp reference_generate_codes_dump with the same fixture.
        assert_eq!(
            prompt.cols, 61,
            "prompt_cols must match s2.cpp build_prompt"
        );
    }

    #[test]
    fn transpose_matches_time_major_layout() {
        let tensor = PromptTensor {
            rows: 3,
            cols: 2,
            data: vec![
                10, 20, // row0
                1, 2, // row1
                3, 4, // row2
            ],
        };
        assert_eq!(
            transpose_to_time_major(&tensor).unwrap(),
            vec![10, 1, 3, 20, 2, 4]
        );
    }
}
