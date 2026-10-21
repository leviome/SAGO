# Third-party notices

The repository license does not replace licenses attached to vendored components.
Keep those notices when redistributing this project.

- Gaussian rendering and scene utilities: `gs/LICENSE.md`.
- Gaussian rasterizers: `vendor/rasterizer-depth/LICENSE.md`,
  `vendor/rasterizer-mask/LICENSE.md`, and `vendor/rasterizer-topk/LICENSE.md`.
- GLM: `third_party/glm/copying.txt` inside each rasterizer directory.
- SAM tracking library: `sago_core/CrossViewTracker/LICENSE`.

The full Debian installer includes the original SAM2 checkpoint. Its license and
checkpoint terms remain those of the [SAM2 project](https://github.com/facebookresearch/sam2).
Source-only distributions do not include model weights.

## Desktop runtime

The native interface uses PySide6/Qt (LGPLv3/GPLv3/commercial terms; the bundled distribution includes its license files). Qt source is available at https://code.qt.io/ and PySide source at https://code.qt.io/pyside/pyside-setup.git. Bundled libxcb-cursor comes from Ubuntu's xcb-util-cursor package; its copyright and license notice are in `packaging/licenses/libxcb-cursor.txt` (also bundled at `.runtime/licenses/libxcb-cursor.txt` in the installer).
