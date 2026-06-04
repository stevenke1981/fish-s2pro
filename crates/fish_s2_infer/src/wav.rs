use std::path::Path;

use byteorder::{LittleEndian, WriteBytesExt};

use crate::error::{InferError, Result};

pub fn pcm_to_wav(samples: &[f32], sample_rate: u32) -> Vec<u8> {
    let num_samples = samples.len() as u32;
    let bits_per_sample: u16 = 16;
    let num_channels: u16 = 1;
    let bytes_per_sample = bits_per_sample / 8;
    let block_align = num_channels * bytes_per_sample;
    let byte_rate = sample_rate * block_align as u32;
    let data_size = num_samples * bytes_per_sample as u32;
    let header_size = 44u32;
    let file_size = header_size + data_size;

    let mut out = Vec::with_capacity((header_size + data_size) as usize);
    out.extend_from_slice(b"RIFF");
    out.write_u32::<LittleEndian>(file_size).unwrap();
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.write_u32::<LittleEndian>(16).unwrap();
    out.write_u16::<LittleEndian>(1).unwrap();
    out.write_u16::<LittleEndian>(num_channels).unwrap();
    out.write_u32::<LittleEndian>(sample_rate).unwrap();
    out.write_u32::<LittleEndian>(byte_rate).unwrap();
    out.write_u16::<LittleEndian>(block_align).unwrap();
    out.write_u16::<LittleEndian>(bits_per_sample).unwrap();
    out.extend_from_slice(b"data");
    out.write_u32::<LittleEndian>(data_size).unwrap();

    for &s in samples {
        let clamped = s.clamp(-1.0, 1.0);
        let pcm = (clamped * 32767.0) as i16;
        out.write_i16::<LittleEndian>(pcm).unwrap();
    }
    out
}

pub fn read_wav_mono_f32(path: &Path, expected_sample_rate: u32) -> Result<Vec<f32>> {
    let bytes = std::fs::read(path)?;
    wav_mono_f32_from_bytes(&bytes, expected_sample_rate)
}

pub fn wav_mono_f32_from_bytes(bytes: &[u8], expected_sample_rate: u32) -> Result<Vec<f32>> {
    let wav = ParsedWav::parse(bytes)?;
    if wav.sample_rate != expected_sample_rate {
        return Err(InferError::Message(format!(
            "expected {expected_sample_rate} Hz WAV, got {} Hz",
            wav.sample_rate
        )));
    }
    let samples = decode_wav_samples(&wav)?;
    if samples.is_empty() {
        return Err(InferError::Message("WAV contains no samples".into()));
    }
    if wav.channels == 1 {
        return Ok(samples);
    }
    let channels = usize::from(wav.channels);
    if samples.len() % channels != 0 {
        return Err(InferError::Message(
            "WAV sample count is not divisible by channel count".into(),
        ));
    }
    Ok(samples
        .chunks_exact(channels)
        .map(|frame| frame.iter().sum::<f32>() / wav.channels as f32)
        .collect())
}

struct ParsedWav<'a> {
    audio_format: u16,
    channels: u16,
    sample_rate: u32,
    bits_per_sample: u16,
    data: &'a [u8],
}

impl<'a> ParsedWav<'a> {
    fn parse(bytes: &'a [u8]) -> Result<Self> {
        if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
            return Err(InferError::Message("not a RIFF/WAVE file".into()));
        }
        let mut offset = 12usize;
        let mut fmt = None;
        let mut data = None;
        while offset.checked_add(8).is_some_and(|end| end <= bytes.len()) {
            let id = &bytes[offset..offset + 4];
            let len = read_u32(bytes, offset + 4)? as usize;
            offset += 8;
            let end = offset
                .checked_add(len)
                .ok_or_else(|| InferError::Message("WAV chunk length overflow".into()))?;
            if end > bytes.len() {
                return Err(InferError::Message(
                    "WAV chunk extends past file end".into(),
                ));
            }
            match id {
                b"fmt " => {
                    if len < 16 {
                        return Err(InferError::Message("WAV fmt chunk too short".into()));
                    }
                    fmt = Some((
                        read_u16(bytes, offset)?,
                        read_u16(bytes, offset + 2)?,
                        read_u32(bytes, offset + 4)?,
                        read_u16(bytes, offset + 14)?,
                    ));
                }
                b"data" => data = Some(&bytes[offset..end]),
                _ => {}
            }
            offset = end + (len % 2);
        }
        let (audio_format, channels, sample_rate, bits_per_sample) =
            fmt.ok_or_else(|| InferError::Message("missing WAV fmt chunk".into()))?;
        let data = data.ok_or_else(|| InferError::Message("missing WAV data chunk".into()))?;
        if channels == 0 {
            return Err(InferError::Message("WAV has zero channels".into()));
        }
        Ok(Self {
            audio_format,
            channels,
            sample_rate,
            bits_per_sample,
            data,
        })
    }
}

fn decode_wav_samples(wav: &ParsedWav<'_>) -> Result<Vec<f32>> {
    match (wav.audio_format, wav.bits_per_sample) {
        (1, 8) => Ok(wav
            .data
            .iter()
            .map(|byte| (*byte as f32 - 128.0) / 128.0)
            .collect()),
        (1, 16) => chunks_exact(wav.data, 2)?
            .map(|chunk| Ok(i16::from_le_bytes([chunk[0], chunk[1]]) as f32 / 32768.0))
            .collect(),
        (1, 24) => chunks_exact(wav.data, 3)?
            .map(|chunk| {
                let value = i32::from_le_bytes([
                    chunk[0],
                    chunk[1],
                    chunk[2],
                    if chunk[2] & 0x80 == 0 { 0 } else { 0xff },
                ]);
                Ok(value as f32 / 8_388_608.0)
            })
            .collect(),
        (1, 32) => chunks_exact(wav.data, 4)?
            .map(|chunk| {
                Ok(
                    i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as f32
                        / 2_147_483_648.0,
                )
            })
            .collect(),
        (3, 32) => chunks_exact(wav.data, 4)?
            .map(|chunk| Ok(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])))
            .collect(),
        _ => Err(InferError::Message(format!(
            "unsupported WAV format: audio_format={}, bits_per_sample={}",
            wav.audio_format, wav.bits_per_sample
        ))),
    }
}

fn chunks_exact(bytes: &[u8], chunk_size: usize) -> Result<std::slice::ChunksExact<'_, u8>> {
    let chunks = bytes.chunks_exact(chunk_size);
    if !chunks.remainder().is_empty() {
        return Err(InferError::Message(format!(
            "WAV data length is not divisible by {chunk_size}"
        )));
    }
    Ok(chunks)
}

fn read_u16(bytes: &[u8], offset: usize) -> Result<u16> {
    let slice = bytes
        .get(offset..offset + 2)
        .ok_or_else(|| InferError::Message("unexpected EOF reading WAV u16".into()))?;
    Ok(u16::from_le_bytes([slice[0], slice[1]]))
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32> {
    let slice = bytes
        .get(offset..offset + 4)
        .ok_or_else(|| InferError::Message("unexpected EOF reading WAV u32".into()))?;
    Ok(u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_pcm_to_wav_roundtrip() {
        let samples = [-1.0, -0.25, 0.0, 0.25, 0.999];
        let bytes = pcm_to_wav(&samples, 44_100);
        let decoded = wav_mono_f32_from_bytes(&bytes, 44_100).expect("decode wav");
        assert_eq!(decoded.len(), samples.len());
        for (actual, expected) in decoded.iter().zip(samples) {
            assert!((actual - expected).abs() <= 2.0 / 32768.0);
        }
    }
}
