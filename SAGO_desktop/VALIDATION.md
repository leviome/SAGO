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
- All package builds exclude model weights. Default cache: `~/.cache/sago-desktop/models/`, respecting `XDG_CACHE_HOME`. Version 1.1.7 was published as `desktop-v1.1.7` after user approval.
- NVIDIA drivers, CUDA/PyTorch and compiled rasterizers are still required. This change was verified with deterministic engine/Qt tests and a real checkpoint download; no new GPU segmentation benchmark was run.

## 1.1.8: backgrounds and navigation

- Added one-click black/white backgrounds to the native Camera panel, continuous right/Shift drag panning, and held-key WASD roaming. Both eye and target translate together using the actual render-camera axes; the view direction and radius stay fixed.
- Added tests for translation geometry and aspect ratio, tilted camera axes, frozen captures, invalid motion, held keys and release, text-input exclusion, live panning before mouse release, and right-click background prompts. The model download and segmentation workflows remain covered by the regression suite.
- The local installer retains first-use SAM2 downloading and contains no model weights. It awaits user validation; no Git upload or release publication is authorized for these changes yet.

Validation for 1.1.8: 141 Desktop checks passed (140 main checks and the separately run worker lifecycle check). The Web suite passed 107 tests. A headless Chromium test against the actual Web interface and simulated backend passed black/white toggling, all four movement keys and release, text input, focus loss, panning before release, prompt locking, right-click background prompts and bilingual button labels. No new GPU segmentation benchmark was run for these UI/camera changes.

The local 1.1.8 build also maps Esc to application exit, including text-field focus, through the existing close handler. Regression checks cover clean exit, cancelling an unsaved-result close request, and cancelling/waiting for an active worker before exit.

## 1.1.9: Cube and cylinder foreground completion

- Both controls fit an upright volume from the current foreground and include every scene Gaussian center inside it. Cube completion reuses the same minimum horizontal-area box as the BBOX display; cylinder completion fits a minimum enclosing circle of the horizontal projection and the foreground height limits. Outside points, original geometry, saved IDs and visibility settings remain unchanged.
- Completed 155 Desktop checks (154 main checks plus the separate worker lifecycle test) and 117 Web tests. Added geometry and engine regressions for rotated horizontal boxes, tilted/flipped up vectors, non-centroid enclosing circles, single-point/flat/linear foregrounds, chunk boundaries, nonfinite background points, empty/frozen captures, ID ownership, no-op history, Undo and Redo. The native UI test covers button actions, foreground mode, empty/frozen state and bilingual labels.
- Headless Chromium exercised the actual Web interface against a simulated backend, including both completion requests, foreground display switching, prompt-mode locks and translations, alongside the navigation controls.
- Actual CUDA containment for both shapes passed on an NVIDIA GeForce RTX 4090 using a small synthetic point cloud. This verifies GPU execution, not real-scene timing or segmentation quality.
- The local 1.1.9 installer retains first-use SAM2 downloading, excludes checkpoints and awaits user validation. No Git push or Release publication was performed.


## 1.1.10–1.1.11: semantic knowledge and search highlighting

- Web: 132 backend tests passed. Desktop: 169 main-suite tests passed; the separate worker cancellation/shutdown check also passed. Native and mocked-browser tests cover building an index, GIF parameter reuse, text queries, bilingual labels, automatic All-mode selection, bbox activation, and clearing search highlights.
- Four-view indexing includes hidden object IDs and atomically stores both segmentation and normalized CLIP embeddings in scene-local knowledge. Tests cover cache reuse, rename/renumber preservation, changed-mask invalidation, deleted IDs, cancellation, failed writes, and preservation of unrelated knowledge fields.
- Actual CUDA rendering and CLIP image/text inference were checked on an RTX 4090 using two synthetic colored Gaussian spheres. Text searches ranked the corresponding sphere first, produced visible gold pixels and an upright bbox, and cleared the highlight without changing saved visibility, IDs, or the current selection. This is an integration check, not a real-scene accuracy benchmark.
- The v1.1.11 Debian archive is checked against the local backend, desktop and launcher sources, includes Qt, and excludes model checkpoints. The SHA-256 is supplied alongside the release installer.
