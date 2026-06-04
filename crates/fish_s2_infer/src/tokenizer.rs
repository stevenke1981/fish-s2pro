use std::path::Path;

use tokenizers::Tokenizer;

use crate::error::{InferError, Result};
use crate::paths::default_tokenizer_path;
use crate::tokenizer_s2cpp::S2CppBpeTokenizer;

/// Qwen newline token id used by `s2.cpp` (`NEWLINE = { 198 }`).
pub const S2CPP_NEWLINE_TOKEN_ID: u32 = 198;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenizedText {
    pub ids: Vec<u32>,
    pub tokens: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct S2TokenizerConfig {
    pub im_start_id: u32,
    pub im_end_id: u32,
    pub voice_id: u32,
}

pub struct S2Tokenizer {
    tokenizer: Tokenizer,
    cpp_bpe: S2CppBpeTokenizer,
    config: S2TokenizerConfig,
}

impl S2Tokenizer {
    pub fn from_default_path() -> Result<Self> {
        Self::from_file(default_tokenizer_path())
    }

    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes = std::fs::read(path)
            .map_err(|err| InferError::Message(format!("read tokenizer: {err}")))?;
        let tokenizer = Tokenizer::from_bytes(&bytes).map_err(|err| {
            InferError::Message(format!(
                "failed to load tokenizer {}: {err}",
                path.display()
            ))
        })?;
        let cpp_bpe = S2CppBpeTokenizer::from_json_bytes(&bytes)?;
        let mut instance = Self {
            tokenizer,
            cpp_bpe,
            config: S2TokenizerConfig {
                im_start_id: 0,
                im_end_id: 0,
                voice_id: 0,
            },
        };
        instance.config = S2TokenizerConfig {
            im_start_id: instance.cpp_special_token_id("<|im_start|>")?,
            im_end_id: instance.cpp_special_token_id("<|im_end|>")?,
            voice_id: instance.cpp_special_token_id("<|voice|>")?,
        };
        Ok(instance)
    }

    pub fn config(&self) -> S2TokenizerConfig {
        self.config
    }

    pub fn encode_newline(&self) -> Result<Vec<u32>> {
        Ok(vec![S2CPP_NEWLINE_TOKEN_ID])
    }

    /// Encode with the same BPE path as `s2.cpp` (`build_prompt`, generation dumps).
    pub fn encode_s2cpp(&self, text: &str) -> Result<TokenizedText> {
        Ok(TokenizedText {
            ids: self.cpp_bpe.encode(text),
            tokens: Vec::new(),
        })
    }

    pub fn encode(&self, text: &str) -> Result<TokenizedText> {
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|err| InferError::Message(format!("tokenizer encode failed: {err}")))?;
        Ok(TokenizedText {
            ids: encoding.get_ids().to_vec(),
            tokens: encoding.get_tokens().to_vec(),
        })
    }

    fn cpp_special_token_id(&self, token: &str) -> Result<u32> {
        self.cpp_bpe
            .token_to_id(token)
            .ok_or_else(|| InferError::Message(format!("missing special token id for {token:?}")))
    }
}

pub fn gpt2_byte_to_unicode() -> [char; 256] {
    let mut table = ['\0'; 256];
    let mut bytes = Vec::new();
    bytes.extend(b'!'..=b'~');
    bytes.extend(0xA1..=0xAC);
    bytes.extend(0xAE..=0xFF);

    let mut code_points: Vec<u32> = bytes.iter().map(|byte| u32::from(*byte)).collect();
    let mut next = 0u32;
    for byte in 0u8..=255 {
        if !bytes.contains(&byte) {
            bytes.push(byte);
            code_points.push(256 + next);
            next += 1;
        }
    }

    for (byte, code_point) in bytes.into_iter().zip(code_points) {
        table[byte as usize] = char::from_u32(code_point).expect("valid GPT-2 bytelevel char");
    }
    table
}

pub fn bytelevel_encode_utf8(text: &str) -> String {
    let table = gpt2_byte_to_unicode();
    text.as_bytes()
        .iter()
        .map(|byte| table[*byte as usize])
        .collect()
}

pub fn parse_token_ids_csv(value: &str) -> Result<Vec<u32>> {
    value
        .split(',')
        .filter(|part| !part.trim().is_empty())
        .map(|part| {
            part.trim().parse::<u32>().map_err(|err| {
                InferError::Message(format!("invalid token id '{part}' in fixture: {err}"))
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bytelevel_table_matches_gpt2_known_values() {
        let table = gpt2_byte_to_unicode();
        assert_eq!(table[b'!' as usize], '!');
        assert_eq!(table[b' ' as usize], 'Ġ');
        assert_eq!(table[b'\n' as usize], 'Ċ');
        assert_eq!(table[0], 'Ā');
        assert_eq!(table[0xFF], 'ÿ');
    }

    #[test]
    fn bytelevel_encodes_utf8_bytes() {
        assert_eq!(bytelevel_encode_utf8(" hello\n"), "ĠhelloĊ");
        assert_eq!(bytelevel_encode_utf8("é"), "Ã©");
    }

    #[test]
    fn parses_token_id_fixture_csv() {
        assert_eq!(parse_token_ids_csv("1, 2,3,").unwrap(), vec![1, 2, 3]);
        assert!(parse_token_ids_csv("1,nope").is_err());
    }

    #[test]
    #[ignore = "requires local models/tokenizer.json"]
    fn encodes_local_tokenizer_golden_cases() {
        let tokenizer = S2Tokenizer::from_default_path().unwrap();
        for raw_line in include_str!("../tests/fixtures/tokenizer_golden.tsv").lines() {
            let line = raw_line.strip_suffix('\r').unwrap_or(raw_line);
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let (text, expected) = line.split_once('\t').expect("fixture text<TAB>ids");
            let text = unescape_fixture_text(text);
            let expected = parse_token_ids_csv(expected).unwrap();
            let actual = tokenizer.encode(&text).unwrap();
            assert_eq!(actual.ids, expected, "text={}", text.escape_debug());
        }
    }

    fn unescape_fixture_text(value: &str) -> String {
        let mut out = String::new();
        let mut chars = value.chars();
        while let Some(ch) = chars.next() {
            if ch == '\\' {
                match chars.next() {
                    Some('n') => out.push('\n'),
                    Some('t') => out.push('\t'),
                    Some('\\') => out.push('\\'),
                    Some(other) => {
                        out.push('\\');
                        out.push(other);
                    }
                    None => out.push('\\'),
                }
            } else {
                out.push(ch);
            }
        }
        out
    }
}
