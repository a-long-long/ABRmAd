//! 左侧文件列表面板

use egui::Context;

use super::super::state::SharedState;

/// 文件列表
pub fn show(ctx: &Context, state: &SharedState) {
    egui::SidePanel::left("file_panel")
        .default_width(250.0)
        .show(ctx, |ui| {
            ui.heading("文件列表");
            ui.separator();

            // 只快照渲染需要的少量字段，避免每帧克隆全部文件（含广告段/事件段）
            struct Row {
                name: String,
                detected: bool,
                ad_count: usize,
                marked: usize,
            }

            let (rows, selected) = {
                let s = state.lock().unwrap();
                let rows: Vec<Row> = s
                    .files
                    .iter()
                    .map(|f| Row {
                        name: f.name.clone(),
                        detected: f.detected,
                        ad_count: f.effective_ad_count(s.thresholds),
                        marked: f.marked_for_deletion.len(),
                    })
                    .collect();
                (rows, s.selected_file)
            };

            egui::ScrollArea::vertical().show(ui, |ui| {
                for (idx, file) in rows.iter().enumerate() {
                    let is_selected = selected == Some(idx);
                    let bg_color = if is_selected {
                        egui::Color32::from_rgb(60, 100, 160)
                    } else {
                        egui::Color32::from_rgb(40, 40, 40)
                    };

                    let response = egui::Frame::default()
                        .fill(bg_color)
                        .inner_margin(4.0)
                        .show(ui, |ui| {
                            ui.horizontal(|ui| {
                                let icon = if file.detected { "✓" } else { "○" };
                                let icon_color = if file.detected {
                                    egui::Color32::from_rgb(100, 200, 100)
                                } else {
                                    egui::Color32::from_rgb(150, 150, 150)
                                };
                                ui.colored_label(icon_color, icon);
                                ui.label(&file.name);
                            });

                            if file.detected {
                                ui.indent("ads", |ui| {
                                    // 与结果面板一致：只统计满足当前阈值的有效广告段
                                    if file.marked == 0 {
                                        ui.label(format!("广告: {} 个", file.ad_count));
                                    } else {
                                        ui.label(format!(
                                            "广告: {} 个（已标记删除 {}）",
                                            file.ad_count, file.marked
                                        ));
                                    }
                                });
                            }
                        })
                        .response;

                    if response.clicked() {
                        let mut s = state.lock().unwrap();
                        s.selected_file = Some(idx);
                        s.selected_ad = None;
                    }
                }
            });
        });
}
