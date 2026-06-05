mod app;
mod audio;

use std::sync::Arc;

use tracing_subscriber::EnvFilter;

fn main() -> eframe::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::from_default_env().add_directive("fish_s2_gui=info".parse().unwrap()),
        )
        .init();

    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1180.0, 760.0])
            .with_min_inner_size([960.0, 640.0]),
        ..Default::default()
    };
    eframe::run_native(
        "Fish S2 Pro Studio",
        native_options,
        Box::new(|cc| {
            install_cjk_font_fallback(&cc.egui_ctx);
            Ok(Box::new(app::FishS2App::new(cc)))
        }),
    )
}

fn install_cjk_font_fallback(ctx: &egui::Context) {
    let Some((font_name, font_bytes)) = load_cjk_font_bytes() else {
        return;
    };

    let mut fonts = egui::FontDefinitions::default();
    fonts.font_data.insert(
        font_name.clone(),
        Arc::new(egui::FontData::from_owned(font_bytes)),
    );
    for family in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
        fonts
            .families
            .entry(family)
            .or_default()
            .push(font_name.clone());
    }
    ctx.set_fonts(fonts);
}

#[cfg(windows)]
fn load_cjk_font_bytes() -> Option<(String, Vec<u8>)> {
    let fonts_dir = std::path::Path::new("C:\\Windows\\Fonts");
    [
        "NotoSansTC-VF.ttf",
        "NotoSansHK-VF.ttf",
        "NotoSansSC-VF.ttf",
        "simhei.ttf",
        "Deng.ttf",
        "simsunb.ttf",
        "kaiu.ttf",
        "msjh.ttc",
        "mingliu.ttc",
    ]
    .into_iter()
    .find_map(|file_name| {
        std::fs::read(fonts_dir.join(file_name))
            .ok()
            .map(|bytes| (format!("fish-s2pro-cjk-{file_name}"), bytes))
    })
}

#[cfg(not(windows))]
fn load_cjk_font_bytes() -> Option<(String, Vec<u8>)> {
    [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
    ]
    .into_iter()
    .find_map(|path| {
        std::fs::read(path)
            .ok()
            .map(|bytes| (format!("fish-s2pro-cjk-{path}"), bytes))
    })
}
