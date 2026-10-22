# Third-party notices

The repository license does not replace licenses attached to vendored components.
Keep those notices when redistributing this project.

- Gaussian rendering and scene utilities: `gs/LICENSE.md`.
- Gaussian rasterizers: `vendor/rasterizer-depth/LICENSE.md`,
  `vendor/rasterizer-mask/LICENSE.md`, and `vendor/rasterizer-topk/LICENSE.md`.
- GLM: `third_party/glm/copying.txt` inside each rasterizer directory.
- SAM tracking library: `sago_core/CrossViewTracker/LICENSE`.

Version 1.1.7 and later Debian installers and source distributions exclude model weights.
The original SAM2 checkpoint is downloaded from the official upstream endpoint on first use.
Its license and checkpoint terms remain those of the [SAM2 project](https://github.com/facebookresearch/sam2).

## Desktop runtime

The native interface uses PySide6/Qt (LGPLv3/GPLv3/commercial terms; the bundled distribution includes its license files). Qt source is available at https://code.qt.io/ and PySide source at https://code.qt.io/pyside/pyside-setup.git. Bundled libxcb-cursor comes from Ubuntu's xcb-util-cursor package; its copyright and license notice are in `packaging/licenses/libxcb-cursor.txt` (also bundled at `.runtime/licenses/libxcb-cursor.txt` in the installer).

## CLIP semantic retrieval

Semantic retrieval uses the [OpenAI CLIP](https://github.com/openai/CLIP) image/text encoder (MIT), distributed through [`openai/clip-vit-base-patch32`](https://huggingface.co/openai/clip-vit-base-patch32). The model revisions are pinned in `backend/semantic.py`; weights are cached separately and are not bundled. The runtime uses Hugging Face Transformers (Apache-2.0), huggingface-hub (Apache-2.0), and safetensors (Apache-2.0). See the upstream model card for model capabilities and limitations.
