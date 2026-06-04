fn main() {
    println!("cargo:rustc-check-cfg=cfg(s2_cpp_linked)");
    println!("cargo:rerun-if-env-changed=S2_CPP_LIB");
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
        if lib.join("fish_s2_cpp.lib").exists() || lib.join("libfish_s2_cpp.a").exists() {
            println!("cargo:rustc-cfg=s2_cpp_linked");
            println!("cargo:rustc-link-lib=static=fish_s2_cpp");
            println!("cargo:rustc-link-search=native={lib_str}");
        } else {
            println!(
                "cargo:warning=S2_CPP_LIB set but fish_s2_cpp static lib missing in {lib_str}"
            );
        }
    }
}
