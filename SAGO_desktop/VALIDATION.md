# SAGO Desktop 1.1 validation

This release uses native Qt Widgets with a SAM2-only engine. The source and installer include no semantic text-detection or alternate segmentation runtime. SAGO and SAGO Web are unchanged.

Regression checks cover spatial prompts, candidate tracking state, model reuse and lazy loading, invalid model/text rejection, three-dimensional refinement, Gaussian IDs, corrections, ground calibration, render modes, GIF export, and idle layout stability.

See `validation/v1.1/` for this release's automated test output, native CUDA workflow and package verification. Earlier `validation/` subdirectories are historical records from 1.0 development.

The historical 1.1.0–1.1.6 installers bundle SAM2 weights and Qt; version 1.1.7 excludes weights and downloads SAM2 on first use. NVIDIA drivers and a compatible CUDA/PyTorch runtime with built rasterizers are still required. Package extraction and launch are tested without installing into the host system.

Verified results for 1.1.0:

- All 85 automated tests passed. Tests for removed integrations were replaced by SAM2 spatial-preview and unsupported-model/text rejection checks.
- Native Qt/CUDA test opened 1,352,080 Gaussians, generated four SAM2 candidates, and completed six-view refinement with 56,411 selected Gaussians. No scene knowledge was saved.
- Checked all 8,784 archive entries: only the SAM2 checkpoint and model README are present under the model directory; removed models and integration files are absent. The packaged engine matches the tested source.
- Extracted package opened the same real PLY using bundled Qt; the viewport stayed at 1022 × 752 while idle. No browser or HTTP server was loaded.
- Package size: 1,120,778,228 bytes (1.044 GiB). SHA256 is recorded in `dist/SHA256SUMS` and `validation/v1.1/package.json`.

## 1.1.1: C shortcut

- All 94 tests passed, including clearing prompts, suppressing stale preview callbacks, cancelling model loading, cancelling running refinement, rolling back a result finishing during cancellation, reverting an uncommitted result, preserving saved objects and Undo/Redo history, and ignoring shortcuts while typing.
- A real Qt/CUDA scene with 1,352,080 Gaussians produced four SAM2 candidates. After one refinement view, C stopped the route and returned to camera navigation with zero prompts and zero selected Gaussians, preserving the previous selection and undo availability.
- The installer was checked for SAM2-only model contents and exact equality of the packaged engine/window files to the tested source. See `validation/v1.1.1/`.

## 1.1.2: display rendering quality

- All 101 tests passed, covering automatic physical-pixel resolution, portrait and landscape geometry, manual sizes through 2560, size validation, single-step low-to-high quality navigation, idle stability, proportional resizing and frozen prompt captures.
- Native Qt/CUDA under 2× DPI opened 1,352,080 Gaussians at an automatic 2048-pixel long edge. A manual 1920 × 1413 frame switched to a 768-pixel navigation preview and returned to 1920 after interaction.
- Full-resolution SAM2 capture generated four candidates and completed all six refinement views (51,568 selected Gaussians), without saving scene knowledge.
- The 1.1.2 archive was checked for SAM2-only model contents, exact tested window/engine source, smooth image scaling and the new resolution selection. Hash and package details are in `validation/v1.1.2/package.json`.

## 1.1.7: first-use checkpoint download

- 132 automated checks passed (131 main suite checks plus the worker lifecycle check run separately). New checks cover verified first download, cache reuse without network, range resume, ignored ranges, malformed responses, checksum rejection, corrupt cache repair, incomplete HTTP reads, retries, cancellation, manual overrides, legacy paths, download-before-GPU ordering, progress dialogs, retry and automatic closure without stopping prompt updates.
- A real HTTPS request to Meta’s official endpoint resumed the last 1 MiB of the 897,952,466-byte SAM2 checkpoint. The completed file passed full SHA-256 verification, and a second invocation reused the cache without network access. The test used an isolated temporary cache and did not change user settings or scene knowledge.
- All package builds exclude model weights. Default cache: `~/.cache/sago-desktop/models/`, respecting `XDG_CACHE_HOME`. The verified 1.1.7 installer is distributed as an asset of the `desktop-v1.1.7` GitHub Release.
- NVIDIA drivers, CUDA/PyTorch and compiled rasterizers are still required. This change was verified with deterministic engine/Qt tests and a real checkpoint download; no new GPU segmentation benchmark was run.
