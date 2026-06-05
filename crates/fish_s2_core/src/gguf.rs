use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use byteorder::{LittleEndian, ReadBytesExt};

use crate::error::{CoreError, Result};

const GGUF_MAGIC: &[u8; 4] = b"GGUF";
const GGUF_ALIGNMENT: u64 = 32;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GgufSummary {
    pub path: String,
    pub version: u32,
    pub tensor_count: u64,
    pub metadata: Vec<(String, String)>,
    pub architecture: Option<String>,
    pub kind: Option<String>,
    pub parameter_count: Option<u64>,
    pub file_size_bytes: u64,
}

#[derive(Debug, Clone)]
pub struct GgufFile {
    pub path: String,
    pub version: u32,
    pub metadata: Vec<(String, String)>,
    pub tensors: Vec<GgufTensorInfo>,
    pub tensor_data_start: u64,
    pub file_size_bytes: u64,
}

impl GgufFile {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let file_size_bytes = std::fs::metadata(path).map_err(CoreError::Io)?.len();
        let mut file = File::open(path).map_err(CoreError::Io)?;
        let mut magic = [0u8; 4];
        file.read_exact(&mut magic).map_err(CoreError::Io)?;
        if &magic != GGUF_MAGIC {
            return Err(CoreError::Message(format!(
                "not a GGUF file: {}",
                path.display()
            )));
        }
        let version = file.read_u32::<LittleEndian>().map_err(CoreError::Io)?;
        let tensor_count = file.read_u64::<LittleEndian>().map_err(CoreError::Io)?;
        let metadata_kv_count = file.read_u64::<LittleEndian>().map_err(CoreError::Io)?;

        let mut metadata = Vec::with_capacity(metadata_kv_count as usize);
        for _ in 0..metadata_kv_count {
            let key = read_gguf_string(&mut file)?;
            let value = read_gguf_value_as_string(&mut file)?;
            metadata.push((key, value));
        }

        let mut tensors = Vec::with_capacity(tensor_count as usize);
        for _ in 0..tensor_count {
            tensors.push(read_tensor_info(&mut file)?);
        }
        let tensor_data_start = align_to(file.stream_position().map_err(CoreError::Io)?);
        validate_tensor_bounds(&tensors, tensor_data_start, file_size_bytes)?;

        Ok(Self {
            path: path.display().to_string(),
            version,
            metadata,
            tensors,
            tensor_data_start,
            file_size_bytes,
        })
    }

    pub fn tensor(&self, name: &str) -> Option<&GgufTensorInfo> {
        self.tensors.iter().find(|tensor| tensor.name == name)
    }

    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.tensors.iter().map(|tensor| tensor.name.as_str())
    }

    pub fn tensor_bytes(&self, name: &str) -> Result<Vec<u8>> {
        let tensor = self
            .tensor(name)
            .ok_or_else(|| CoreError::Message(format!("tensor not found: {name}")))?;
        self.read_tensor_bytes(tensor, tensor.byte_len()?)
    }

    pub fn tensor_bytes_prefix(&self, name: &str, max_len: usize) -> Result<Vec<u8>> {
        let tensor = self
            .tensor(name)
            .ok_or_else(|| CoreError::Message(format!("tensor not found: {name}")))?;
        self.read_tensor_bytes(tensor, tensor.byte_len()?.min(max_len))
    }

    pub fn mapped_tensor_view(&self, name: &str) -> Result<MappedTensorView> {
        let tensor = self
            .tensor(name)
            .ok_or_else(|| CoreError::Message(format!("tensor not found: {name}")))?;
        Ok(MappedTensorView {
            file_path: self.path.clone(),
            name: tensor.name.clone(),
            dimensions: tensor.dimensions.clone(),
            ggml_type: tensor.ggml_type,
            absolute_offset: tensor.absolute_offset(self.tensor_data_start),
            byte_len: tensor.byte_len()?,
        })
    }

    fn read_tensor_bytes(&self, tensor: &GgufTensorInfo, len: usize) -> Result<Vec<u8>> {
        let mut file = File::open(&self.path).map_err(CoreError::Io)?;
        file.seek(SeekFrom::Start(
            tensor.absolute_offset(self.tensor_data_start),
        ))
        .map_err(CoreError::Io)?;
        let mut bytes = vec![0u8; len];
        file.read_exact(&mut bytes).map_err(CoreError::Io)?;
        Ok(bytes)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct MappedTensorView {
    pub file_path: String,
    pub name: String,
    pub dimensions: Vec<u64>,
    pub ggml_type: GgmlType,
    pub absolute_offset: u64,
    pub byte_len: usize,
}

impl MappedTensorView {
    pub fn read_bytes(&self) -> Result<Vec<u8>> {
        self.read_bytes_prefix(self.byte_len)
    }

    pub fn read_bytes_prefix(&self, max_len: usize) -> Result<Vec<u8>> {
        let len = self.byte_len.min(max_len);
        let mut file = File::open(&self.file_path).map_err(CoreError::Io)?;
        file.seek(SeekFrom::Start(self.absolute_offset))
            .map_err(CoreError::Io)?;
        let mut bytes = vec![0u8; len];
        file.read_exact(&mut bytes).map_err(CoreError::Io)?;
        Ok(bytes)
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GgufTensorInfo {
    pub name: String,
    pub dimensions: Vec<u64>,
    pub ggml_type: GgmlType,
    pub relative_offset: u64,
}

impl GgufTensorInfo {
    pub fn element_count(&self) -> Result<u64> {
        self.dimensions.iter().try_fold(1u64, |acc, dim| {
            acc.checked_mul(*dim).ok_or_else(|| {
                CoreError::Message(format!("tensor element count overflow: {}", self.name))
            })
        })
    }

    pub fn byte_len(&self) -> Result<usize> {
        self.ggml_type.byte_len(self.element_count()?)
    }

    pub fn absolute_offset(&self, tensor_data_start: u64) -> u64 {
        tensor_data_start + self.relative_offset
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[repr(u32)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Q8K = 15,
    I8 = 16,
    I16 = 17,
    I32 = 18,
    I64 = 19,
    F64 = 20,
    IQ1S = 21,
    IQ1M = 22,
    BF16 = 30,
}

impl GgmlType {
    pub fn from_u32(value: u32) -> Result<Self> {
        match value {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            2 => Ok(Self::Q4_0),
            3 => Ok(Self::Q4_1),
            6 => Ok(Self::Q5_0),
            7 => Ok(Self::Q5_1),
            8 => Ok(Self::Q8_0),
            9 => Ok(Self::Q8_1),
            10 => Ok(Self::Q2K),
            11 => Ok(Self::Q3K),
            12 => Ok(Self::Q4K),
            13 => Ok(Self::Q5K),
            14 => Ok(Self::Q6K),
            15 => Ok(Self::Q8K),
            16 => Ok(Self::I8),
            17 => Ok(Self::I16),
            18 => Ok(Self::I32),
            19 => Ok(Self::I64),
            20 => Ok(Self::F64),
            21 => Ok(Self::IQ1S),
            22 => Ok(Self::IQ1M),
            30 => Ok(Self::BF16),
            other => Err(CoreError::Message(format!(
                "unknown GGML tensor type: {other}"
            ))),
        }
    }

    pub fn byte_len(self, element_count: u64) -> Result<usize> {
        let bytes = match self {
            Self::F32 | Self::I32 => element_count.checked_mul(4),
            Self::F16 | Self::I16 | Self::BF16 => element_count.checked_mul(2),
            Self::I8 => Some(element_count),
            Self::I64 | Self::F64 => element_count.checked_mul(8),
            Self::Q4_0 => block_bytes(element_count, 32, 18),
            Self::Q4_1 => block_bytes(element_count, 32, 20),
            Self::Q5_0 => block_bytes(element_count, 32, 22),
            Self::Q5_1 => block_bytes(element_count, 32, 24),
            Self::Q8_0 => block_bytes(element_count, 32, 34),
            Self::Q8_1 => block_bytes(element_count, 32, 40),
            Self::Q2K => block_bytes(element_count, 256, 84),
            Self::Q3K => block_bytes(element_count, 256, 110),
            Self::Q4K => block_bytes(element_count, 256, 144),
            Self::Q5K => block_bytes(element_count, 256, 176),
            Self::Q6K => block_bytes(element_count, 256, 210),
            Self::Q8K => block_bytes(element_count, 256, 292),
            Self::IQ1S => block_bytes(element_count, 256, 44),
            Self::IQ1M => block_bytes(element_count, 256, 56),
        }
        .ok_or_else(|| CoreError::Message("tensor byte length overflow".into()))?;

        usize::try_from(bytes)
            .map_err(|_| CoreError::Message("tensor byte length does not fit usize".into()))
    }
}

impl GgufSummary {
    pub fn inspect(path: impl AsRef<Path>) -> Result<Self> {
        let gguf = GgufFile::open(path)?;
        let metadata = gguf.metadata;

        let architecture = metadata
            .iter()
            .find(|(k, _)| k == "general.architecture")
            .map(|(_, v)| v.clone());
        let kind = metadata
            .iter()
            .find(|(k, _)| k.contains("model") || k.contains("codec"))
            .map(|(k, v)| format!("{k}={v}"));
        let parameter_count = metadata
            .iter()
            .find(|(k, _)| k == "general.parameter_count")
            .and_then(|(_, v)| v.parse().ok());

        Ok(Self {
            path: gguf.path,
            version: gguf.version,
            tensor_count: gguf.tensors.len() as u64,
            metadata,
            architecture,
            kind,
            parameter_count,
            file_size_bytes: gguf.file_size_bytes,
        })
    }

    pub fn display_name(&self) -> String {
        Path::new(&self.path)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown.gguf")
            .to_string()
    }

    pub fn size_human(&self) -> String {
        human_bytes(self.file_size_bytes)
    }
}

fn human_bytes(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    let mut size = bytes as f64;
    let mut unit = 0;
    while size >= 1024.0 && unit < UNITS.len() - 1 {
        size /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} B")
    } else {
        format!("{size:.2} {}", UNITS[unit])
    }
}

fn read_gguf_string(file: &mut File) -> Result<String> {
    let len = file.read_u64::<LittleEndian>().map_err(CoreError::Io)?;
    let mut buf = vec![0u8; len as usize];
    file.read_exact(&mut buf).map_err(CoreError::Io)?;
    String::from_utf8(buf).map_err(|e| CoreError::Message(e.to_string()))
}

fn read_tensor_info(file: &mut File) -> Result<GgufTensorInfo> {
    let name = read_gguf_string(file)?;
    let dimension_count = file.read_u32::<LittleEndian>().map_err(CoreError::Io)?;
    let mut dimensions = Vec::with_capacity(dimension_count as usize);
    for _ in 0..dimension_count {
        dimensions.push(file.read_u64::<LittleEndian>().map_err(CoreError::Io)?);
    }
    let ggml_type = GgmlType::from_u32(file.read_u32::<LittleEndian>().map_err(CoreError::Io)?)?;
    let relative_offset = file.read_u64::<LittleEndian>().map_err(CoreError::Io)?;
    Ok(GgufTensorInfo {
        name,
        dimensions,
        ggml_type,
        relative_offset,
    })
}

fn validate_tensor_bounds(
    tensors: &[GgufTensorInfo],
    tensor_data_start: u64,
    file_size_bytes: u64,
) -> Result<()> {
    for tensor in tensors {
        let byte_len = tensor.byte_len()? as u64;
        let start = tensor.absolute_offset(tensor_data_start);
        let end = start.checked_add(byte_len).ok_or_else(|| {
            CoreError::Message(format!("tensor offset overflow: {}", tensor.name))
        })?;
        if end > file_size_bytes {
            return Err(CoreError::Message(format!(
                "tensor {} extends past file end: {} > {}",
                tensor.name, end, file_size_bytes
            )));
        }
    }
    Ok(())
}

fn block_bytes(element_count: u64, block_size: u64, bytes_per_block: u64) -> Option<u64> {
    element_count
        .checked_add(block_size - 1)?
        .checked_div(block_size)?
        .checked_mul(bytes_per_block)
}

fn align_to(offset: u64) -> u64 {
    if offset.is_multiple_of(GGUF_ALIGNMENT) {
        offset
    } else {
        offset + (GGUF_ALIGNMENT - (offset % GGUF_ALIGNMENT))
    }
}

#[repr(u32)]
enum GgufValueType {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
}

fn read_gguf_value_as_string(file: &mut File) -> Result<String> {
    let value_type = file.read_u32::<LittleEndian>().map_err(CoreError::Io)?;
    match value_type {
        x if x == GgufValueType::Uint8 as u32 => {
            Ok(file.read_u8().map_err(CoreError::Io)?.to_string())
        }
        x if x == GgufValueType::Int8 as u32 => {
            Ok(file.read_i8().map_err(CoreError::Io)?.to_string())
        }
        x if x == GgufValueType::Uint16 as u32 => Ok(file
            .read_u16::<LittleEndian>()
            .map_err(CoreError::Io)?
            .to_string()),
        x if x == GgufValueType::Int16 as u32 => Ok(file
            .read_i16::<LittleEndian>()
            .map_err(CoreError::Io)?
            .to_string()),
        x if x == GgufValueType::Uint32 as u32 => Ok(file
            .read_u32::<LittleEndian>()
            .map_err(CoreError::Io)?
            .to_string()),
        x if x == GgufValueType::Int32 as u32 => Ok(file
            .read_i32::<LittleEndian>()
            .map_err(CoreError::Io)?
            .to_string()),
        x if x == GgufValueType::Float32 as u32 => Ok(file
            .read_f32::<LittleEndian>()
            .map_err(CoreError::Io)?
            .to_string()),
        x if x == GgufValueType::Bool as u32 => {
            Ok((file.read_u8().map_err(CoreError::Io)? != 0).to_string())
        }
        x if x == GgufValueType::String as u32 => read_gguf_string(file),
        x if x == GgufValueType::Uint64 as u32 => Ok(file
            .read_u64::<LittleEndian>()
            .map_err(CoreError::Io)?
            .to_string()),
        x if x == GgufValueType::Int64 as u32 => Ok(file
            .read_i64::<LittleEndian>()
            .map_err(CoreError::Io)?
            .to_string()),
        x if x == GgufValueType::Float64 as u32 => Ok(file
            .read_f64::<LittleEndian>()
            .map_err(CoreError::Io)?
            .to_string()),
        x if x == GgufValueType::Array as u32 => {
            let elem_type = file.read_u32::<LittleEndian>().map_err(CoreError::Io)?;
            let len = file.read_u64::<LittleEndian>().map_err(CoreError::Io)?;
            for _ in 0..len {
                read_gguf_value_with_type(file, elem_type)?;
            }
            Ok(format!("[array len={len}]"))
        }
        other => Err(CoreError::Message(format!(
            "unknown GGUF value type: {other}"
        ))),
    }
}

fn read_gguf_value_with_type(file: &mut File, value_type: u32) -> Result<()> {
    match value_type {
        0 => {
            file.read_u8().map_err(CoreError::Io)?;
        }
        1 => {
            file.read_i8().map_err(CoreError::Io)?;
        }
        2 => {
            file.read_u16::<LittleEndian>().map_err(CoreError::Io)?;
        }
        3 => {
            file.read_i16::<LittleEndian>().map_err(CoreError::Io)?;
        }
        4 => {
            file.read_u32::<LittleEndian>().map_err(CoreError::Io)?;
        }
        5 => {
            file.read_i32::<LittleEndian>().map_err(CoreError::Io)?;
        }
        6 => {
            file.read_f32::<LittleEndian>().map_err(CoreError::Io)?;
        }
        7 => {
            file.read_u8().map_err(CoreError::Io)?;
        }
        8 => {
            let _ = read_gguf_string(file)?;
        }
        9 => {
            let elem_type = file.read_u32::<LittleEndian>().map_err(CoreError::Io)?;
            let len = file.read_u64::<LittleEndian>().map_err(CoreError::Io)?;
            for _ in 0..len {
                read_gguf_value_with_type(file, elem_type)?;
            }
        }
        10 => {
            file.read_u64::<LittleEndian>().map_err(CoreError::Io)?;
        }
        11 => {
            file.read_i64::<LittleEndian>().map_err(CoreError::Io)?;
        }
        12 => {
            file.read_f64::<LittleEndian>().map_err(CoreError::Io)?;
        }
        other => return Err(CoreError::Message(format!("unknown type {other}"))),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn rejects_non_gguf() {
        let dir = std::env::temp_dir().join("fish_s2_gguf_test");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("bad.bin");
        std::fs::write(&path, b"notgguf").unwrap();
        assert!(GgufSummary::inspect(&path).is_err());
    }

    #[test]
    fn reads_tensor_directory_and_bytes() {
        let dir = std::env::temp_dir().join("fish_s2_gguf_test");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("tiny.gguf");
        write_tiny_gguf(&path);

        let gguf = GgufFile::open(&path).unwrap();
        assert_eq!(gguf.version, 3);
        assert_eq!(gguf.tensor_data_start % GGUF_ALIGNMENT, 0);
        assert_eq!(
            gguf.tensor_names().collect::<Vec<_>>(),
            vec!["blk.0.weight"]
        );

        let tensor = gguf.tensor("blk.0.weight").unwrap();
        assert_eq!(tensor.dimensions, vec![2, 2]);
        assert_eq!(tensor.ggml_type, GgmlType::F32);
        assert_eq!(tensor.relative_offset, 0);
        assert_eq!(tensor.byte_len().unwrap(), 16);

        let bytes = gguf.tensor_bytes("blk.0.weight").unwrap();
        let values = bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0]);

        let mapped = gguf.mapped_tensor_view("blk.0.weight").unwrap();
        assert_eq!(mapped.name, "blk.0.weight");
        assert_eq!(mapped.dimensions, vec![2, 2]);
        assert_eq!(mapped.ggml_type, GgmlType::F32);
        assert_eq!(mapped.byte_len, 16);
        assert_eq!(mapped.read_bytes().unwrap(), bytes);
        assert_eq!(mapped.read_bytes_prefix(4).unwrap(), 1.0_f32.to_le_bytes());

        let summary = GgufSummary::inspect(&path).unwrap();
        assert_eq!(summary.tensor_count, 1);
        assert_eq!(summary.architecture.as_deref(), Some("s2-pro-test"));
    }

    #[test]
    #[ignore = "requires local S2 Pro GGUF files in models/"]
    fn opens_local_s2_pro_model_pair_when_present() {
        let model_dir = local_model_dir();
        let transformer = find_one(&model_dir, "transformer-only.gguf");
        let codec = find_one(&model_dir, "codec-only.gguf");

        assert_transformer_gguf(&transformer);
        assert_codec_gguf(&codec);
    }

    fn assert_transformer_gguf(path: &Path) {
        let gguf = GgufFile::open(path).unwrap();
        assert_eq!(gguf.version, 3);
        assert_eq!(
            metadata_value(&gguf, "general.architecture"),
            Some("fish-speech")
        );
        assert_eq!(gguf.tensors.len(), 358);
        assert_layer_count(&gguf, "layers", 36);
        assert_layer_count(&gguf, "fast_layers", 4);
        assert_tensor(&gguf, "codebook_embeddings.weight", &[2560, 40960]);
        assert_tensor(&gguf, "embeddings.weight", &[2560, 155776]);
        assert_tensor(&gguf, "fast_embeddings.weight", &[2560, 4096]);
        assert_tensor(&gguf, "fast_layers.0.attention.wqkv.weight", &[2560, 6144]);
        assert_tensor(&gguf, "fast_layers.0.attention.wo.weight", &[4096, 2560]);
        assert_tensor(&gguf, "fast_output.weight", &[2560, 4096]);
        assert_tensor(&gguf, "layers.0.attention.q_norm.weight", &[128]);
        assert_tensor(&gguf, "layers.0.attention.wqkv.weight", &[2560, 6144]);
        assert_tensor(&gguf, "layers.0.attention.wo.weight", &[4096, 2560]);
        assert_tensor(&gguf, "layers.0.feed_forward.w1.weight", &[2560, 9728]);
        assert_tensor(&gguf, "norm.weight", &[2560]);
        assert!(!gguf
            .tensor_bytes_prefix("codebook_embeddings.weight", 4096)
            .unwrap()
            .is_empty());
    }

    fn assert_codec_gguf(path: &Path) {
        let gguf = GgufFile::open(path).unwrap();
        assert_eq!(gguf.version, 3);
        assert_eq!(
            metadata_value(&gguf, "general.architecture"),
            Some("fish-speech-codec")
        );
        assert_eq!(gguf.tensors.len(), 461);
        assert_tensor(&gguf, "encoder.block.0.conv.weight", &[7, 1, 64]);
        assert_tensor(
            &gguf,
            "encoder.block.4.block.5.causal_mask",
            &[16384, 16384],
        );
        assert_tensor(
            &gguf,
            "encoder.block.4.block.5.layers.0.attention.wqkv.weight",
            &[1024, 3072],
        );
        assert_tensor(
            &gguf,
            "quantizer.semantic_quantizer.quantizers.0.codebook.weight",
            &[8, 4096],
        );
        assert_tensor(
            &gguf,
            "quantizer.quantizer.quantizers.0.codebook.weight",
            &[8, 1024],
        );
        assert_tensor(
            &gguf,
            "quantizer.pre_module.layers.0.attention.wqkv.weight",
            &[1024, 3072],
        );
        assert_tensor(
            &gguf,
            "quantizer.post_module.layers.0.attention.wqkv.weight",
            &[1024, 3072],
        );
        assert_tensor(&gguf, "decoder.model.0.conv.weight", &[7, 1024, 1536]);
        assert_tensor(&gguf, "decoder.model.6.conv.weight", &[7, 96, 1]);
        assert!(!gguf
            .tensor_bytes_prefix("encoder.block.0.conv.weight", 4096)
            .unwrap()
            .is_empty());
    }

    fn assert_tensor(gguf: &GgufFile, name: &str, dimensions: &[u64]) {
        let tensor = gguf
            .tensor(name)
            .unwrap_or_else(|| panic!("missing tensor {name}"));
        assert_eq!(tensor.ggml_type, GgmlType::F16);
        assert_eq!(tensor.dimensions, dimensions);
    }

    fn assert_layer_count(gguf: &GgufFile, prefix: &str, expected: usize) {
        let mut layers = gguf
            .tensors
            .iter()
            .filter_map(|tensor| {
                tensor
                    .name
                    .strip_prefix(prefix)
                    .and_then(|rest| rest.strip_prefix('.'))
                    .and_then(|rest| rest.split_once('.'))
                    .and_then(|(layer, _)| layer.parse::<usize>().ok())
            })
            .collect::<Vec<_>>();
        layers.sort_unstable();
        layers.dedup();
        assert_eq!(layers.len(), expected, "{prefix} layer count");
    }

    fn metadata_value<'a>(gguf: &'a GgufFile, key: &str) -> Option<&'a str> {
        gguf.metadata
            .iter()
            .find(|(candidate, _)| candidate == key)
            .map(|(_, value)| value.as_str())
    }

    fn write_tiny_gguf(path: &Path) {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(GGUF_MAGIC);
        bytes.extend_from_slice(&3_u32.to_le_bytes());
        bytes.extend_from_slice(&1_u64.to_le_bytes());
        bytes.extend_from_slice(&1_u64.to_le_bytes());
        write_string(&mut bytes, "general.architecture");
        bytes.extend_from_slice(&(GgufValueType::String as u32).to_le_bytes());
        write_string(&mut bytes, "s2-pro-test");
        write_string(&mut bytes, "blk.0.weight");
        bytes.extend_from_slice(&2_u32.to_le_bytes());
        bytes.extend_from_slice(&2_u64.to_le_bytes());
        bytes.extend_from_slice(&2_u64.to_le_bytes());
        bytes.extend_from_slice(&(GgmlType::F32 as u32).to_le_bytes());
        bytes.extend_from_slice(&0_u64.to_le_bytes());
        let aligned = align_to(bytes.len() as u64) as usize;
        bytes.resize(aligned, 0);
        for value in [1.0_f32, 2.0, 3.0, 4.0] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        let mut file = File::create(path).unwrap();
        file.write_all(&bytes).unwrap();
    }

    fn write_string(bytes: &mut Vec<u8>, value: &str) {
        bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
        bytes.extend_from_slice(value.as_bytes());
    }

    fn find_one(root: &Path, suffix: &str) -> std::path::PathBuf {
        std::fs::read_dir(root)
            .unwrap_or_else(|err| panic!("cannot read {}: {err}", root.display()))
            .filter_map(std::result::Result::ok)
            .map(|entry| entry.path())
            .find(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.ends_with(suffix))
            })
            .unwrap_or_else(|| panic!("missing *{suffix} in {}", root.display()))
    }

    fn local_model_dir() -> std::path::PathBuf {
        std::env::var("FISH_S2_MODEL_DIR").map_or_else(
            |_| {
                Path::new(env!("CARGO_MANIFEST_DIR"))
                    .ancestors()
                    .nth(2)
                    .expect("workspace root")
                    .join("models")
            },
            std::path::PathBuf::from,
        )
    }
}
