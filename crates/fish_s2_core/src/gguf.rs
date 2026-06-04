use std::fs::File;
use std::io::Read;
use std::path::Path;

use byteorder::{LittleEndian, ReadBytesExt};

use crate::error::{CoreError, Result};

const GGUF_MAGIC: &[u8; 4] = b"GGUF";

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

impl GgufSummary {
    pub fn inspect(path: impl AsRef<Path>) -> Result<Self> {
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

        let mut metadata = Vec::new();
        for _ in 0..metadata_kv_count {
            let key = read_gguf_string(&mut file)?;
            let value = read_gguf_value_as_string(&mut file)?;
            metadata.push((key, value));
        }

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
            path: path.display().to_string(),
            version,
            tensor_count,
            metadata,
            architecture,
            kind,
            parameter_count,
            file_size_bytes,
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

    #[test]
    fn rejects_non_gguf() {
        let dir = std::env::temp_dir().join("fish_s2_gguf_test");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("bad.bin");
        std::fs::write(&path, b"notgguf").unwrap();
        assert!(GgufSummary::inspect(&path).is_err());
    }
}
