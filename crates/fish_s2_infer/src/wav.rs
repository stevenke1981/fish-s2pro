use byteorder::{LittleEndian, WriteBytesExt};

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
