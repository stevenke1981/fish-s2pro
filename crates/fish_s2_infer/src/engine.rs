use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::UNIX_EPOCH;

use crate::error::{InferError, Result};
use crate::generate::GenerateParams;
#[cfg(feature = "legacy-s2-exe")]
use crate::paths::project_root;
use crate::paths::{default_tokenizer_path, ensure_project_dirs, server_workdir};
use crate::pipeline::{RustPipeline, RustPipelineConfig, RustSynthesisOptions};
use crate::prompt::PromptCodes;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngineBackend {
    RustPure,
    Ffi,
    FfiCuda,
    #[cfg(feature = "legacy-s2-exe")]
    Subprocess,
}

impl EngineBackend {
    pub fn parse(value: &str) -> Result<Self> {
        match value.to_ascii_lowercase().as_str() {
            "rust" | "rust-pure" | "pure-rust" => Ok(Self::RustPure),
            "ffi" | "cpp" | "native" => Ok(Self::Ffi),
            "cuda" | "ffi-cuda" | "cpp-cuda" | "native-cuda" => Ok(Self::FfiCuda),
            #[cfg(feature = "legacy-s2-exe")]
            "subprocess" | "s2" | "s2.exe" => Ok(Self::Subprocess),
            other => Err(InferError::Message(format!(
                "unknown backend: {other} (expected {})",
                Self::expected_values()
            ))),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::RustPure => "rust-pure",
            Self::Ffi => "ffi",
            Self::FfiCuda => "ffi-cuda",
            #[cfg(feature = "legacy-s2-exe")]
            Self::Subprocess => "subprocess",
        }
    }

    pub fn is_ffi(self) -> bool {
        matches!(self, Self::Ffi | Self::FfiCuda)
    }

    pub fn uses_cuda(self) -> bool {
        matches!(self, Self::FfiCuda)
    }

    pub fn expected_values() -> &'static str {
        #[cfg(feature = "legacy-s2-exe")]
        {
            "rust-pure, ffi, ffi-cuda, or subprocess"
        }
        #[cfg(not(feature = "legacy-s2-exe"))]
        {
            "rust-pure, ffi, or ffi-cuda"
        }
    }

    pub fn cli_values() -> &'static str {
        #[cfg(feature = "legacy-s2-exe")]
        {
            "rust-pure|ffi|ffi-cuda|subprocess"
        }
        #[cfg(not(feature = "legacy-s2-exe"))]
        {
            "rust-pure|ffi|ffi-cuda"
        }
    }
}

pub fn cpp_engine_linked() -> bool {
    cfg!(s2_cpp_linked)
}

#[derive(Debug, Clone)]
pub struct EngineConfig {
    pub transformer_gguf: PathBuf,
    pub codec_gguf: PathBuf,
    pub tokenizer_path: PathBuf,
    pub workdir: PathBuf,
    pub backend: EngineBackend,
    pub generate_params: GenerateParams,
    pub seed: u64,
    pub vulkan_device: i32,
    pub codec_vulkan_device: i32,
    pub cuda_device: i32,
    pub codec_cuda: bool,
}

impl EngineConfig {
    pub fn new(transformer_gguf: PathBuf, codec_gguf: PathBuf) -> Result<Self> {
        let _ = ensure_project_dirs();
        if !transformer_gguf.exists() {
            return Err(InferError::Message(format!(
                "transformer GGUF missing: {}",
                transformer_gguf.display()
            )));
        }
        if !codec_gguf.exists() {
            return Err(InferError::Message(format!(
                "codec GGUF missing: {}",
                codec_gguf.display()
            )));
        }
        Ok(Self {
            transformer_gguf,
            codec_gguf,
            tokenizer_path: default_tokenizer_path(),
            workdir: server_workdir(),
            backend: EngineBackend::RustPure,
            generate_params: GenerateParams::default(),
            seed: 0,
            vulkan_device: 0,
            codec_vulkan_device: 0,
            cuda_device: 0,
            codec_cuda: false,
        })
    }
}

#[derive(Debug, Clone)]
pub struct SynthesisRequest {
    pub text: String,
    pub reference_text: Option<String>,
    pub reference_wav: Option<PathBuf>,
}

#[derive(Debug, Clone)]
struct CachedReference {
    text: String,
    prompt_codes: PromptCodes,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ReferenceCacheKey {
    wav_path: PathBuf,
    wav_len: u64,
    wav_modified_nanos: Option<u128>,
    text: String,
}

#[derive(Debug, Clone)]
struct CachedRequestReference {
    key: ReferenceCacheKey,
    reference: CachedReference,
}

/// Rust-facing inference engine (replaces external `s2.exe` process).
pub struct InferenceEngine {
    config: EngineConfig,
    rust_pipeline: Option<Mutex<RustPipeline>>,
    default_reference: Option<CachedReference>,
    request_reference: Mutex<Option<CachedRequestReference>>,
    #[cfg(s2_cpp_linked)]
    native: Option<Mutex<native::NativeEngine>>,
}

impl InferenceEngine {
    pub fn load(config: EngineConfig) -> Result<Self> {
        std::fs::create_dir_all(&config.workdir)?;
        if !config.tokenizer_path.exists() {
            return Err(InferError::Message(format!(
                "tokenizer not found: {} — copy tokenizer.json from fishaudio/s2-pro into models/",
                config.tokenizer_path.display()
            )));
        }

        let (rust_pipeline, default_reference) = if config.backend == EngineBackend::RustPure {
            let pipeline = RustPipeline::load(
                RustPipelineConfig::new(&config.transformer_gguf, &config.codec_gguf)
                    .with_tokenizer(&config.tokenizer_path),
            )?;
            let default_reference = Self::load_workdir_reference(&pipeline, &config.workdir)?;
            (Some(Mutex::new(pipeline)), default_reference)
        } else {
            (None, None)
        };

        #[cfg(not(s2_cpp_linked))]
        if config.backend.is_ffi() {
            return Err(InferError::NativeNotLinked);
        }

        #[cfg(s2_cpp_linked)]
        let native = if config.backend.is_ffi() {
            Some(Mutex::new(native::NativeEngine::load(&config)?))
        } else {
            None
        };

        Ok(Self {
            config,
            rust_pipeline,
            default_reference,
            request_reference: Mutex::new(None),
            #[cfg(s2_cpp_linked)]
            native,
        })
    }

    pub fn backend(&self) -> EngineBackend {
        self.config.backend
    }

    pub fn backend_name(&self) -> &'static str {
        self.config.backend.as_str()
    }

    pub fn apply_reference(&self, wav: &Path, text: &str) -> Result<()> {
        std::fs::create_dir_all(&self.config.workdir)?;
        std::fs::copy(wav, self.config.workdir.join("reference.wav"))?;
        std::fs::write(self.config.workdir.join("reference.txt"), text)?;
        Ok(())
    }

    pub fn synthesize_wav(&self, request: &SynthesisRequest) -> Result<Vec<u8>> {
        if self.config.backend != EngineBackend::RustPure {
            if let (Some(wav), Some(text)) = (&request.reference_wav, &request.reference_text) {
                self.apply_reference(wav, text)?;
            }
        }

        match self.config.backend {
            EngineBackend::RustPure => self.synthesize_via_rust_pipeline(request),
            EngineBackend::Ffi | EngineBackend::FfiCuda => {
                #[cfg(s2_cpp_linked)]
                {
                    let native = self
                        .native
                        .as_ref()
                        .ok_or_else(|| InferError::Message("FFI backend not loaded".into()))?
                        .lock()
                        .map_err(|_| InferError::Message("engine lock poisoned".to_string()))?;
                    native.synthesize(&request.text, request.reference_text.as_deref())
                }

                #[cfg(not(s2_cpp_linked))]
                {
                    Err(InferError::NativeNotLinked)
                }
            }
            #[cfg(feature = "legacy-s2-exe")]
            EngineBackend::Subprocess => Self::synthesize_via_embedded_cli(&self.config, request),
        }
    }

    fn synthesize_via_rust_pipeline(&self, request: &SynthesisRequest) -> Result<Vec<u8>> {
        let mut pipeline = self
            .rust_pipeline
            .as_ref()
            .ok_or_else(|| InferError::Message("RustPure backend not loaded".into()))?
            .lock()
            .map_err(|_| InferError::Message("RustPure pipeline lock poisoned".into()))?;

        let mut options = RustSynthesisOptions::new(request.text.clone())
            .with_params(self.config.generate_params)
            .with_seed(self.config.seed);
        match (&request.reference_wav, &request.reference_text) {
            (Some(wav), Some(text)) => {
                let reference = self.cached_request_reference(&pipeline, wav, text)?;
                options = options.with_reference(reference.text, reference.prompt_codes);
            }
            (None, None) => {
                if let Some(reference) = &self.default_reference {
                    options = options
                        .with_reference(reference.text.clone(), reference.prompt_codes.clone());
                }
            }
            _ => {
                return Err(InferError::Message(
                    "reference_wav and reference_text must both be set for RustPure".into(),
                ));
            }
        }

        Ok(pipeline.synthesize(&options)?.wav_bytes)
    }

    fn cached_request_reference(
        &self,
        pipeline: &RustPipeline,
        wav: &Path,
        text: &str,
    ) -> Result<CachedReference> {
        let key = ReferenceCacheKey::from_path(wav, text)?;
        {
            let guard = self
                .request_reference
                .lock()
                .map_err(|_| InferError::Message("reference cache lock poisoned".into()))?;
            if let Some(cached) = guard.as_ref().filter(|cached| cached.key == key) {
                return Ok(cached.reference.clone());
            }
        }

        let prompt_codes = pipeline.encode_reference_wav(wav)?;
        let reference = CachedReference {
            text: text.to_string(),
            prompt_codes,
        };
        let mut guard = self
            .request_reference
            .lock()
            .map_err(|_| InferError::Message("reference cache lock poisoned".into()))?;
        *guard = Some(CachedRequestReference {
            key,
            reference: reference.clone(),
        });
        Ok(reference)
    }

    fn load_workdir_reference(
        pipeline: &RustPipeline,
        workdir: &Path,
    ) -> Result<Option<CachedReference>> {
        let wav = workdir.join("reference.wav");
        let text_path = workdir.join("reference.txt");
        match (wav.is_file(), text_path.is_file()) {
            (false, false) => Ok(None),
            (true, true) => {
                let text = std::fs::read_to_string(&text_path)?;
                let prompt_codes = pipeline.encode_reference_wav(&wav)?;
                Ok(Some(CachedReference { text, prompt_codes }))
            }
            _ => Err(InferError::Message(format!(
                "reference.wav and reference.txt must both exist in {}",
                workdir.display()
            ))),
        }
    }

    #[cfg(feature = "legacy-s2-exe")]
    fn synthesize_via_embedded_cli(
        config: &EngineConfig,
        request: &SynthesisRequest,
    ) -> Result<Vec<u8>> {
        let binary = Self::resolve_s2_binary();
        if !binary.exists() {
            return Err(InferError::Message(format!(
                "s2 binary not found: {}",
                project_root().join("bin").display()
            )));
        }

        Self::prepare_workdir_models(config)?;
        if let (Some(wav), Some(text)) = (&request.reference_wav, &request.reference_text) {
            std::fs::create_dir_all(&config.workdir)?;
            std::fs::copy(wav, config.workdir.join("reference.wav"))?;
            std::fs::write(config.workdir.join("reference.txt"), text)?;
        }

        let out_path = config.workdir.join("_rust_tts_out.wav");
        let status = std::process::Command::new(&binary)
            .current_dir(&config.workdir)
            .arg("-v")
            .arg(config.vulkan_device.to_string())
            .arg("--codec-vulkan")
            .arg(config.codec_vulkan_device.to_string())
            .arg("--model")
            .arg("model.gguf")
            .arg("--model-codec")
            .arg("codec.gguf")
            .arg("--text")
            .arg(&request.text)
            .arg("--output")
            .arg(&out_path)
            .status()
            .map_err(InferError::Io)?;

        if !status.success() {
            return Err(InferError::Message(format!(
                "s2 binary exited with {status}"
            )));
        }
        std::fs::read(&out_path).map_err(InferError::Io)
    }

    #[cfg(feature = "legacy-s2-exe")]
    fn resolve_s2_binary() -> PathBuf {
        let candidates = [
            project_root().join("bin").join("s2.exe"),
            project_root().join("bin").join("s2"),
            PathBuf::from("s2.exe"),
        ];
        for c in candidates {
            if c.exists() {
                return c;
            }
        }
        project_root().join("bin").join("s2.exe")
    }

    #[cfg(feature = "legacy-s2-exe")]
    fn prepare_workdir_models(config: &EngineConfig) -> Result<()> {
        std::fs::create_dir_all(&config.workdir)?;
        link_or_copy(&config.transformer_gguf, &config.workdir.join("model.gguf"))?;
        link_or_copy(&config.codec_gguf, &config.workdir.join("codec.gguf"))?;
        Ok(())
    }
}

impl ReferenceCacheKey {
    fn from_path(wav: &Path, text: &str) -> Result<Self> {
        let metadata = std::fs::metadata(wav)?;
        let wav_path = std::fs::canonicalize(wav).unwrap_or_else(|_| wav.to_path_buf());
        let wav_modified_nanos = metadata
            .modified()
            .ok()
            .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
            .map(|duration| duration.as_nanos());
        Ok(Self {
            wav_path,
            wav_len: metadata.len(),
            wav_modified_nanos,
            text: text.to_string(),
        })
    }
}

#[cfg(all(windows, feature = "legacy-s2-exe"))]
fn link_or_copy(from: &Path, to: &Path) -> Result<()> {
    let _ = std::fs::remove_file(to);
    std::fs::copy(from, to).map_err(InferError::Io)?;
    Ok(())
}

#[cfg(all(unix, feature = "legacy-s2-exe"))]
fn link_or_copy(from: &Path, to: &Path) -> Result<()> {
    use std::os::unix::fs::symlink;
    let _ = std::fs::remove_file(to);
    if symlink(from, to).is_err() {
        std::fs::copy(from, to).map_err(InferError::Io)?;
    }
    Ok(())
}

#[cfg(s2_cpp_linked)]
mod native {
    use std::ffi::{CStr, CString};
    use std::os::raw::c_char;
    use std::path::Path;

    use super::{EngineConfig, InferError, Result};

    #[repr(C)]
    struct S2EngineConfig {
        model_path: *const c_char,
        codec_path: *const c_char,
        tokenizer_path: *const c_char,
        workdir: *const c_char,
        vulkan_device: i32,
        codec_vulkan_device: i32,
        use_cuda: i32,
        cuda_device: i32,
        codec_use_cuda: i32,
        max_new_tokens: i32,
    }

    unsafe extern "C" {
        fn s2_engine_create(
            cfg: *const S2EngineConfig,
            err: *mut c_char,
            err_cap: usize,
        ) -> *mut std::ffi::c_void;
        fn s2_engine_destroy(handle: *mut std::ffi::c_void);
        fn s2_engine_synthesize_wav(
            handle: *mut std::ffi::c_void,
            text: *const c_char,
            reference_text: *const c_char,
            out_data: *mut *mut u8,
            out_len: *mut usize,
            err: *mut c_char,
            err_cap: usize,
        ) -> i32;
        fn s2_engine_free_buffer(ptr: *mut u8);
    }

    pub struct NativeEngine {
        handle: *mut std::ffi::c_void,
    }

    impl NativeEngine {
        pub fn load(config: &EngineConfig) -> Result<Self> {
            let model = cstr(&config.transformer_gguf)?;
            let codec = cstr(&config.codec_gguf)?;
            let tokenizer = cstr(&config.tokenizer_path)?;
            let workdir = cstr(&config.workdir)?;
            let max_new_tokens = i32::try_from(config.generate_params.max_new_tokens)
                .map_err(|_| InferError::Message("max_new_tokens overflows i32".into()))?;
            let mut err_buf = vec![0i8; 512];
            let cfg = S2EngineConfig {
                model_path: model.as_ptr(),
                codec_path: codec.as_ptr(),
                tokenizer_path: tokenizer.as_ptr(),
                workdir: workdir.as_ptr(),
                vulkan_device: config.vulkan_device,
                codec_vulkan_device: config.codec_vulkan_device,
                use_cuda: i32::from(config.backend.uses_cuda()),
                cuda_device: config.cuda_device,
                codec_use_cuda: i32::from(config.backend.uses_cuda() && config.codec_cuda),
                max_new_tokens,
            };
            let handle = unsafe { s2_engine_create(&cfg, err_buf.as_mut_ptr(), err_buf.len()) };
            if handle.is_null() {
                return Err(InferError::Message(read_cstr(&err_buf)));
            }
            Ok(Self { handle })
        }

        pub fn synthesize(&self, text: &str, reference_text: Option<&str>) -> Result<Vec<u8>> {
            let text_c =
                CString::new(text).map_err(|_| InferError::Message("invalid text".into()))?;
            let ref_c = reference_text
                .map(CString::new)
                .transpose()
                .map_err(|_| InferError::Message("invalid reference text".into()))?;
            let mut err_buf = vec![0i8; 512];
            let mut out_ptr: *mut u8 = std::ptr::null_mut();
            let mut out_len: usize = 0;
            let ok = unsafe {
                s2_engine_synthesize_wav(
                    self.handle,
                    text_c.as_ptr(),
                    ref_c.as_ref().map_or(std::ptr::null(), |c| c.as_ptr()),
                    &mut out_ptr,
                    &mut out_len,
                    err_buf.as_mut_ptr(),
                    err_buf.len(),
                )
            };
            if ok == 0 {
                return Err(InferError::Message(read_cstr(&err_buf)));
            }
            let slice = unsafe { std::slice::from_raw_parts(out_ptr, out_len) };
            let data = slice.to_vec();
            unsafe { s2_engine_free_buffer(out_ptr) };
            Ok(data)
        }
    }

    impl Drop for NativeEngine {
        fn drop(&mut self) {
            unsafe { s2_engine_destroy(self.handle) };
        }
    }

    unsafe impl Send for NativeEngine {}

    fn cstr(path: &Path) -> Result<CString> {
        CString::new(path.as_os_str().to_string_lossy().as_bytes())
            .map_err(|_| InferError::Message("path contains nul".into()))
    }

    fn read_cstr(buf: &[i8]) -> String {
        unsafe { CStr::from_ptr(buf.as_ptr()) }
            .to_string_lossy()
            .into_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn parses_backend_names() {
        assert_eq!(
            EngineBackend::parse("rust-pure").unwrap(),
            EngineBackend::RustPure
        );
        assert_eq!(EngineBackend::parse("ffi").unwrap(), EngineBackend::Ffi);
        assert_eq!(
            EngineBackend::parse("ffi-cuda").unwrap(),
            EngineBackend::FfiCuda
        );
        assert_eq!(
            EngineBackend::parse("cuda").unwrap(),
            EngineBackend::FfiCuda
        );
        #[cfg(feature = "legacy-s2-exe")]
        assert_eq!(
            EngineBackend::parse("subprocess").unwrap(),
            EngineBackend::Subprocess
        );
        #[cfg(not(feature = "legacy-s2-exe"))]
        assert!(EngineBackend::parse("subprocess").is_err());
        assert!(EngineBackend::parse("unknown").is_err());
    }

    #[test]
    fn reference_cache_key_tracks_text_and_file_metadata() {
        let path = std::env::temp_dir().join(format!(
            "fish_s2_reference_cache_key_{}_{}.wav",
            std::process::id(),
            "smoke"
        ));
        {
            let mut file = std::fs::File::create(&path).unwrap();
            file.write_all(b"one").unwrap();
        }
        let key_a = ReferenceCacheKey::from_path(&path, "same text").unwrap();
        let key_b = ReferenceCacheKey::from_path(&path, "different text").unwrap();
        assert_ne!(key_a, key_b);

        std::thread::sleep(std::time::Duration::from_millis(2));
        {
            let mut file = std::fs::File::create(&path).unwrap();
            file.write_all(b"longer").unwrap();
        }
        let key_c = ReferenceCacheKey::from_path(&path, "same text").unwrap();
        assert_ne!(key_a, key_c);
        let _ = std::fs::remove_file(path);
    }
}
