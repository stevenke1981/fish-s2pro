//! BPE encode path aligned with `output/s2.cpp-src/src/s2_tokenizer.cpp` for prompt parity.

use std::collections::HashMap;

use serde::Deserialize;

use crate::error::{InferError, Result};
use crate::tokenizer::bytelevel_encode_utf8;

#[derive(Debug, Clone)]
pub struct S2CppBpeTokenizer {
    vocab: HashMap<String, u32>,
    merge_rank: HashMap<String, i32>,
    special_tokens: Vec<(String, u32)>,
}

#[derive(Debug, Deserialize)]
struct TokenizerJson {
    added_tokens: Option<Vec<AddedToken>>,
    model: Option<TokenizerModel>,
}

#[derive(Debug, Deserialize)]
struct AddedToken {
    content: String,
    id: u32,
    #[serde(default)]
    special: bool,
}

#[derive(Debug, Deserialize)]
struct TokenizerModel {
    vocab: HashMap<String, u32>,
    merges: Vec<MergeEntry>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum MergeEntry {
    Pair(String, String),
    Line(String),
}

impl S2CppBpeTokenizer {
    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self> {
        let doc: TokenizerJson = serde_json::from_slice(bytes)
            .map_err(|err| InferError::Message(format!("parse tokenizer json: {err}")))?;

        let mut vocab = HashMap::new();
        let mut special_tokens = Vec::new();

        if let Some(added) = doc.added_tokens {
            for tok in added {
                if tok.content.is_empty() {
                    continue;
                }
                vocab.insert(tok.content.clone(), tok.id);
                if tok.special {
                    special_tokens.push((tok.content, tok.id));
                }
            }
        }

        let mut merges = Vec::new();
        if let Some(model) = doc.model {
            for (key, id) in model.vocab {
                vocab.entry(key).or_insert(id);
            }
            for merge_item in model.merges {
                match merge_item {
                    MergeEntry::Pair(a, b) => merges.push((a, b)),
                    MergeEntry::Line(line) => {
                        let (a, b) = line
                            .split_once(' ')
                            .ok_or_else(|| InferError::Message("invalid merge line".into()))?;
                        merges.push((a.to_string(), b.to_string()));
                    }
                }
            }
        }

        let mut merge_rank = HashMap::new();
        for (rank, (a, b)) in merges.into_iter().enumerate() {
            merge_rank.insert(format!("{a}{b}"), i32::try_from(rank).unwrap());
        }

        special_tokens.sort_by(|left, right| right.0.len().cmp(&left.0.len()));

        Ok(Self {
            vocab,
            merge_rank,
            special_tokens,
        })
    }

    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.vocab.get(token).copied()
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return Vec::new();
        }
        let (segments, special_ids) = split_on_specials(text, &self.special_tokens);
        let mut ids = Vec::new();
        for (segment, special_id) in segments.into_iter().zip(special_ids) {
            if let Some(id) = special_id {
                ids.push(id);
            } else if !segment.is_empty() {
                for word in pre_tokenize(&segment) {
                    ids.extend(self.bpe_encode_word(&word));
                }
            }
        }
        ids
    }

    fn bpe_encode_word(&self, word: &str) -> Vec<u32> {
        if word.is_empty() {
            return Vec::new();
        }
        let bl = bytelevel_encode_utf8(word);
        if let Some(&id) = self.vocab.get(&bl) {
            return vec![id];
        }

        let mut symbols = utf8_chars(&bl);
        while symbols.len() > 1 {
            let mut best_rank = i32::MAX;
            let mut best_pos = None;
            for i in 0..symbols.len().saturating_sub(1) {
                let pair = format!("{}{}", symbols[i], symbols[i + 1]);
                if let Some(&rank) = self.merge_rank.get(&pair) {
                    if rank < best_rank {
                        best_rank = rank;
                        best_pos = Some(i);
                    }
                }
            }
            let Some(pos) = best_pos else { break };
            let right = symbols.remove(pos + 1);
            symbols[pos].push_str(&right);
        }

        symbols
            .iter()
            .filter_map(|sym| self.vocab.get(sym).copied())
            .collect()
    }
}

fn split_on_specials(text: &str, specials: &[(String, u32)]) -> (Vec<String>, Vec<Option<u32>>) {
    #[derive(Clone, Copy)]
    struct Match {
        pos: usize,
        len: usize,
        id: u32,
    }

    let mut matches = Vec::new();
    for (token, id) in specials {
        let mut pos = 0usize;
        while let Some(found) = text[pos..].find(token.as_str()) {
            let abs = pos + found;
            matches.push(Match {
                pos: abs,
                len: token.len(),
                id: *id,
            });
            pos = abs + token.len();
        }
    }

    matches.sort_by(|a, b| a.pos.cmp(&b.pos).then_with(|| b.len.cmp(&a.len)));

    let mut filtered = Vec::new();
    let mut last_end = 0usize;
    for m in matches {
        if m.pos >= last_end {
            filtered.push(m);
            last_end = m.pos + m.len;
        }
    }

    let mut segments = Vec::new();
    let mut special_ids = Vec::new();
    let mut pos = 0usize;
    for m in filtered {
        if m.pos > pos {
            segments.push(text[pos..m.pos].to_string());
            special_ids.push(None);
        }
        segments.push(text[m.pos..m.pos + m.len].to_string());
        special_ids.push(Some(m.id));
        pos = m.pos + m.len;
    }
    if pos < text.len() {
        segments.push(text[pos..].to_string());
        special_ids.push(None);
    }

    (segments, special_ids)
}

fn pre_tokenize(text: &str) -> Vec<String> {
    let chars = utf8_chars(text);
    let mut words = Vec::new();
    let mut current = String::new();
    for ch in chars {
        if ch.len() == 1 {
            let b = ch.as_bytes()[0];
            if matches!(b, b' ' | b'\t' | b'\n' | b'\r') {
                if !current.is_empty() {
                    words.push(std::mem::take(&mut current));
                }
                current = ch;
                continue;
            }
        }
        current.push_str(&ch);
    }
    if !current.is_empty() {
        words.push(current);
    }
    words
}

fn utf8_chars(text: &str) -> Vec<String> {
    let mut chars = Vec::new();
    let mut i = 0usize;
    let bytes = text.as_bytes();
    while i < bytes.len() {
        let mut len = 1usize;
        let c = bytes[i];
        if (c & 0xF8) == 0xF0 {
            len = 4;
        } else if (c & 0xF0) == 0xE0 {
            len = 3;
        } else if (c & 0xE0) == 0xC0 {
            len = 2;
        }
        if i + len > bytes.len() {
            len = 1;
        }
        chars.push(String::from_utf8_lossy(&bytes[i..i + len]).into_owned());
        i += len;
    }
    chars
}
