//! 阈值设置窗口

use egui::Context;

use super::super::state::{SharedState, Thresholds};
use crate::core::viterbi::ViterbiConfig;

/// 显示阈值设置窗口
pub fn show(ctx: &Context, state: &SharedState) {
    if !state.lock().unwrap().show_settings {
        return;
    }

    let mut th = state.lock().unwrap().thresholds;
    let mut viterbi = state.lock().unwrap().viterbi_config;
    let mut changed = false;
    let mut open = true;

    egui::Window::new("检测参数设置")
        .open(&mut open)
        .resizable(false)
        .collapsible(false)
        .show(ctx, |ui| {
            ui.label(egui::RichText::new("广告段过滤与展示阈值").strong());
            changed |= show_thresholds(ui, &mut th);

            ui.separator();
            ui.label(egui::RichText::new("Viterbi DP 时序过滤（检测期间生效）").strong());
            changed |= show_viterbi(ui, &mut viterbi);

            ui.add_space(6.0);
            ui.horizontal(|ui| {
                if ui.button("全部恢复默认").clicked() {
                    th = Thresholds::default();
                    viterbi = None;
                    changed = true;
                }
                ui.label(
                    egui::RichText::new("修改后：阈值类立即影响显示与裁剪；Viterbi 需重新检测")
                        .small()
                        .color(egui::Color32::from_gray(150)),
                );
            });
        });

    if changed {
        let mut s = state.lock().unwrap();
        s.thresholds = th;
        s.viterbi_config = viterbi;
        // 阈值变化会改变有效广告段的数量与序号，清空旧选择与标记避免错位
        s.ads_to_delete.clear();
        for f in &mut s.files {
            f.marked_for_deletion.clear();
        }
    }
    if !open {
        state.lock().unwrap().show_settings = false;
    }
}

fn show_thresholds(ui: &mut egui::Ui, th: &mut Thresholds) -> bool {
    let mut changed = false;
    egui::Grid::new("threshold_grid")
        .num_columns(2)
        .spacing([12.0, 8.0])
        .show(ui, |ui| {
            ui.label("最小广告长度");
            changed |= ui
                .add(
                    egui::DragValue::new(&mut th.min_length_sec)
                        .range(0..=60)
                        .suffix(" 秒"),
                )
                .on_hover_text("0 表示不限制，过滤后不参与显示与裁剪")
                .changed();
            ui.end_row();

            ui.label("最小出现次数");
            changed |= ui
                .add(egui::DragValue::new(&mut th.min_count).range(2..=10))
                .on_hover_text("广告至少要在多少个文件中重复出现")
                .changed();
            ui.end_row();

            ui.label("极高置信度阈值");
            changed |= ui
                .add(
                    egui::DragValue::new(&mut th.score_top)
                        .range(th.score_high..=1.0)
                        .speed(0.01)
                        .fixed_decimals(2),
                )
                .on_hover_text("匹配度不低于此值显示为深红色")
                .changed();
            ui.end_row();

            ui.label("高置信度阈值");
            changed |= ui
                .add(
                    egui::DragValue::new(&mut th.score_high)
                        .range(th.score_low..=th.score_top)
                        .speed(0.01)
                        .fixed_decimals(2),
                )
                .on_hover_text("匹配度不低于此值显示为红色，达到极高阈值显示为深红")
                .changed();
            ui.end_row();

            ui.label("低置信度阈值");
            changed |= ui
                .add(
                    egui::DragValue::new(&mut th.score_low)
                        .range(0.0..=th.score_high)
                        .speed(0.01)
                        .fixed_decimals(2),
                )
                .on_hover_text("低于此值显示为黄色，介于两者之间为橙色")
                .changed();
            ui.end_row();
        });
    changed
}

fn show_viterbi(ui: &mut egui::Ui, viterbi: &mut Option<ViterbiConfig>) -> bool {
    let mut changed = false;
    let mut custom = viterbi.is_some();
    let prev_custom = custom;

    ui.horizontal(|ui| {
        changed |= ui
            .checkbox(&mut custom, "自定义 Viterbi 参数")
            .on_hover_text("关闭则使用内部默认值（max_drift=3, min_chain_score=2.5）")
            .changed();
    });

    if custom != prev_custom {
        // checkbox 切换：开 → 用 refine 内部默认值初始化；关 → 置 None（检测时也会走 refine 内部默认）
        if custom {
            *viterbi = Some(cfg_from_refine_default());
            changed = true;
        } else {
            *viterbi = None;
            changed = true;
        }
    }

    if let Some(ref mut cfg) = viterbi {
        egui::Grid::new("viterbi_grid")
            .num_columns(2)
            .spacing([12.0, 6.0])
            .show(ui, |ui| {
                ui.label("最大漂移 (帧)");
                changed |= ui
                    .add(
                        egui::DragValue::new(&mut cfg.max_drift)
                            .range(0..=20)
                            .speed(0.1),
                    )
                    .on_hover_text("相邻匹配允许的 pos1-pos2 偏移漂移，越大越宽松")
                    .changed();
                ui.end_row();

                ui.label("最小累计得分");
                changed |= ui
                    .add(
                        egui::DragValue::new(&mut cfg.min_chain_score)
                            .range(0.0..=10.0)
                            .speed(0.1)
                            .fixed_decimals(1),
                    )
                    .on_hover_text("保留链的最低累计相似度得分，越高越严格")
                    .changed();
                ui.end_row();
            });
    }

    changed
}

/// refine.rs 内部实际使用的 Viterbi 默认值（非 viterbi::ViterbiConfig::default()）
/// 保持与 CLI "未指定参数时走内部默认" 行为一致
fn cfg_from_refine_default() -> ViterbiConfig {
    ViterbiConfig {
        max_drift: 3,
        max_lookback: 200,
        min_chain_score: 2.5,
        min_chain_length: 2,
    }
}
