# Changelog

## v2.0

- English UI by default, with a persistent English / Chinese switch.
- SAM2, SAM3 and SAM3.1 options; SAM2 text prompts through GroundingDINO.
- Automatic 2D segmentation updates and separate model-loading / segmentation timing.
- Continuous multi-object segmentation, dense Gaussian ID persistence, naming,
  merging, renumbering and independent foreground / background layers.
- Visible-surface ID correction with accumulated clicks or box selection,
  highlight previews, explicit confirmation and undo / redo.
- Unified rasterized selection footprints and pixel coordinates for correction.
- RGB, Splats, Points and Depth render modes.
- Three-point ground calibration, persistent up vectors, upright 3D boxes,
  V-Drone paths, camera frustums and per-view segmentation previews.
- Compact, collapsible black / red controls and convenient scene selection.

The source release excludes pretrained weights, local environments, scenes,
and generated outputs. See `models/README.md` for checkpoint setup.

## v1.0 — 2026-09-24

首个 SAGO Web 发布版本（npm 包版本 `1.0.0`），基于 SuperSplat 3.3.1。

- 原版 SAM2 Hiera Large FP32，纯前端 WebGPU 推理。
- 正负点、原生框提示、四个候选 mask、边缘扩展及诊断导出。
- 单对象跨视角记忆，Virtual Drone 双向六视角细化、进度及取消。
- 模型下载确认、进度、并发分块、续传、完整性校验和本机目录复用。
- 分割前坐标系确认及地面三点校正。
- 场景面板内的分割入口；二维和三维前景/背景显示切换。
- 选区替换、增加、减去、交集、撤销/重做及过期画面保护。

模型单独作为 Release 附件发布。无需业务后端，静态主机即可运行；需要支持 WebGPU 的浏览器。文字提示、动态 TAG 和命名对象列表暂未实现。
