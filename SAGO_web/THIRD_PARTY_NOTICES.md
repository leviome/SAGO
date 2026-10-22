# Third-party notices

The repository license does not replace licenses attached to vendored components.
Keep those notices when redistributing this project.

- Gaussian rendering and scene utilities: `gs/LICENSE.md`.
- Gaussian rasterizers: `vendor/rasterizer-depth/LICENSE.md`,
  `vendor/rasterizer-mask/LICENSE.md`, and `vendor/rasterizer-topk/LICENSE.md`.
- GLM: `third_party/glm/copying.txt` inside each rasterizer directory.
- SAM tracking library: `sago_core/CrossViewTracker/LICENSE`.
- SAM3 / SAM3.1: `vendor/sam3/LICENSE` and `vendor/SAM3_PROVENANCE.md`.
- GroundingDINO: `vendor/GroundingDINO/LICENSE` and
  `vendor/GROUNDINGDINO_PROVENANCE.md`.

Pretrained checkpoints are distributed separately by their owners and are not
included in this source release. Follow their checkpoint terms and access rules.

## CLIP semantic retrieval

Semantic retrieval uses the [OpenAI CLIP](https://github.com/openai/CLIP) image/text encoder (MIT), distributed through [`openai/clip-vit-base-patch32`](https://huggingface.co/openai/clip-vit-base-patch32). The model revisions are pinned in `backend/semantic.py`; weights are cached separately and are not bundled. The runtime uses Hugging Face Transformers (Apache-2.0), huggingface-hub (Apache-2.0), and safetensors (Apache-2.0). See the upstream model card for model capabilities and limitations.
