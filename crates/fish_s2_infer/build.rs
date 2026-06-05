fn main() {
    println!("cargo:rustc-check-cfg=cfg(s2_cpp_linked)");
    println!("cargo:rerun-if-env-changed=S2_CPP_LIB");
    println!("cargo:rerun-if-env-changed=S2_CPP_DLL_DIR");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_CPP_ENGINE");

    let lib_dir = std::env::var_os("S2_CPP_LIB")
        .map(std::path::PathBuf::from)
        .or_else(|| {
            if std::env::var("CARGO_FEATURE_CPP_ENGINE").is_ok() {
                std::env::var_os("S2_CPP_DIR").map(std::path::PathBuf::from)
            } else {
                None
            }
        });

    if let Some(lib) = lib_dir {
        let lib_str = lib.display().to_string();
        let msvc_lib = lib.join("fish_s2_cpp.lib");
        let unix_lib = lib.join("libfish_s2_cpp.a");
        println!("cargo:rerun-if-changed={}", msvc_lib.display());
        println!("cargo:rerun-if-changed={}", unix_lib.display());
        if msvc_lib.exists() || unix_lib.exists() {
            println!("cargo:rustc-cfg=s2_cpp_linked");
            println!("cargo:rustc-link-lib=static=fish_s2_cpp");
            println!("cargo:rustc-link-search=native={lib_str}");
            copy_runtime_dlls_next_to_exes(&lib);
        } else {
            println!(
                "cargo:warning=S2_CPP_LIB set but fish_s2_cpp static lib missing in {lib_str}"
            );
        }
    }
}

#[cfg(windows)]
fn copy_runtime_dlls_next_to_exes(lib_dir: &std::path::Path) {
    let Some(target_profile_dir) = target_profile_dir() else {
        println!("cargo:warning=unable to resolve target profile dir for GGML runtime DLL copy");
        return;
    };
    let Some(runtime_dir) = ggml_runtime_dll_dir(lib_dir) else {
        println!("cargo:warning=GGML runtime DLL dir not found; set S2_CPP_DLL_DIR or place ggml*.dll under the native build ggml\\bin\\Release");
        return;
    };

    let mut copied = 0usize;
    match std::fs::read_dir(&runtime_dir) {
        Ok(entries) => {
            for entry in entries.flatten() {
                let path = entry.path();
                let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
                    continue;
                };
                if !(name.starts_with("ggml") && name.ends_with(".dll")) {
                    continue;
                }
                println!("cargo:rerun-if-changed={}", path.display());
                let dest = target_profile_dir.join(name);
                match std::fs::copy(&path, &dest) {
                    Ok(_) => copied += 1,
                    Err(err) => println!(
                        "cargo:warning=failed to copy {} to {}: {err}",
                        path.display(),
                        dest.display()
                    ),
                }
            }
        }
        Err(err) => {
            println!(
                "cargo:warning=failed to enumerate GGML runtime DLL dir {}: {err}",
                runtime_dir.display()
            );
        }
    }
    if copied > 0 {
        println!(
            "cargo:warning=copied {copied} GGML runtime DLL(s) to {}",
            target_profile_dir.display()
        );
    }
}

#[cfg(not(windows))]
fn copy_runtime_dlls_next_to_exes(_lib_dir: &std::path::Path) {}

#[cfg(windows)]
fn target_profile_dir() -> Option<std::path::PathBuf> {
    let out_dir = std::path::PathBuf::from(std::env::var_os("OUT_DIR")?);
    out_dir.ancestors().nth(3).map(std::path::Path::to_path_buf)
}

#[cfg(windows)]
fn ggml_runtime_dll_dir(lib_dir: &std::path::Path) -> Option<std::path::PathBuf> {
    if let Some(dir) = std::env::var_os("S2_CPP_DLL_DIR").map(std::path::PathBuf::from) {
        if dir.is_dir() {
            return Some(dir);
        }
    }

    let native_root = lib_dir.parent()?;
    [
        native_root.join("ggml").join("bin").join("Release"),
        native_root.join("ggml").join("bin"),
    ]
    .into_iter()
    .find(|path| path.is_dir())
}
