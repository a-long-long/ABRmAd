//! ABRmAd GUI 入口（需要 `gui` feature）

#![windows_subsystem = "windows"]

fn main() -> eframe::Result<()> {
    abrmad_rs_app::gui::run()
}
