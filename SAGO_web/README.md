# SAGO Web

Part of the [SAGO repository](../README.md). For the native Linux application, see [SAGO Desktop](../SAGO_desktop/README.md).

A browser workspace for interactive segmentation of 3D Gaussian Splatting scenes, based on [SAGO](https://github.com/leviome/SAGO). The Python/CUDA backend renders the scene, runs segmentation models, and refines selections through multiple Virtual Drone views. The browser displays rendered images and sends interaction requests; it does not require WebGPU, ONNX, or a Node.js build.

The interface defaults to English. Use the button in the upper-right corner to switch to Chinese; your browser remembers the language preference.

**v2.0** includes SAM2, SAM3, and SAM3.1 support, text prompts, automatic 2D segmentation, multi-object IDs, ground calibration, visible-surface ID correction, and multiple rendering modes. See [CHANGELOG.md](CHANGELOG.md).

## Foreground completion

After creating a 3D foreground, use **Cube completion** or **Cylinder completion** in the 3D selection panel. Cube completion uses the same upright, minimum horizontal-area bounding box as the 3D BBOX display; its side lengths may differ. Cylinder completion fits the smallest enclosing circle of the foreground's horizontal projection and uses its minimum/maximum height. Both align to the current up vector.

All scene Gaussian centers inside the volume join the current foreground, including occluded centers. Gaussian size/ellipse intersection is not used. Existing selected centers remain selected; outside centers and saved object IDs are unchanged. Existing object visibility settings remain in effect. Undo/Redo restores the selection; Add ID and export use the normal workflow. The controls are disabled for an empty foreground or during prompts, calibration and correction.

## Environment installation

### Requirements

- Linux with an NVIDIA GPU and a working NVIDIA driver.
- Python 3.10 is recommended.
- A CUDA-enabled PyTorch installation and matching torchvision.
- A CUDA toolkit with `nvcc`, plus a compatible C++ compiler, to build the bundled extensions. The PyTorch wheel alone does not provide the compiler toolchain.
- Conda for the example below, or an equivalent isolated Python environment.

All application and CUDA extension sources are included in this repository. Model checkpoints, scenes, local environments, and generated outputs are distributed separately. No parent SAGO checkout is required.

### 1. Clone and create the main environment

```bash
git clone https://github.com/leviome/SAGO.git
cd SAGO/SAGO_web
conda create -n sago-web python=3.10 pip -y
conda activate sago-web
python -m pip install --upgrade pip
python -m pip install 'setuptools<81' wheel ninja
```

Run the remaining commands from the `SAGO_web/` application directory.

### 2. Install PyTorch and Python dependencies

Choose a PyTorch build compatible with your GPU driver and installed CUDA toolkit using the [PyTorch installation instructions](https://pytorch.org/get-started/previous-versions/). For example, this installs PyTorch 2.5.1 and torchvision 0.20.1 with CUDA 12.1 support:

```bash
python -m pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

Use this example with a compatible driver and CUDA 12.1 toolkit. For a different CUDA version, choose the corresponding PyTorch/torchvision pair instead. Check the installation before compiling:

```bash
nvidia-smi
nvcc --version
python -c "import torch; print('PyTorch:', torch.__version__); print('PyTorch CUDA:', torch.version.cuda); print('GPU available:', torch.cuda.is_available())"
```

`GPU available` must be `True` for scene rendering and segmentation.

### 3. Build the CUDA extensions

```bash
bash scripts/install-cuda.sh
```

This builds `simple-knn` and the depth, mask, and top-k Gaussian rasterizers from `vendor/`, including their bundled GLM headers. Build them in the same environment used to run the application. If CUDA is not discovered automatically, set `CUDA_HOME` to your toolkit installation and ensure its compiler is available on `PATH`. Rebuild the extensions after changing PyTorch or the CUDA toolchain.

### 4. Set up SAM2 weights and start

Download the original **SAM2 Hiera Large** checkpoint, `sam2_hiera_large.pt`, from the official [SAM2 repository](https://github.com/facebookresearch/sam2), and place it in `models/`:

```text
models/
  sam2_hiera_large.pt
```

```bash
python run.py
```

Open **http://127.0.0.1:8080**. To use a different port or checkpoint:

```bash
python run.py --port 8081 --checkpoint ./models/sam2_hiera_large.pt
```

The checkpoint can also be configured with `SAGO_CHECKPOINT`. Weights are not downloaded automatically. SAM2 is the default model and loads when you first enable **Enter segmentation prompts** or press **M**. Model switches show a separate load-time dialog that closes automatically after successful loading.

### Optional: SAM2 text prompts

Point and box prompts work with SAM2 alone. For text prompts, also download:

- [`groundingdino_swint_ogc.pth`](https://github.com/IDEA-Research/GroundingDINO/releases/download/weights/groundingdino_swint_ogc.pth) into `models/`.
- The model and tokenizer files from [`google-bert/bert-base-uncased`](https://huggingface.co/google-bert/bert-base-uncased) into `models/bert-base-uncased/`, including `model.safetensors`, `config.json`, `tokenizer.json`, `tokenizer_config.json`, and `vocab.txt`.

GroundingDINO source is bundled, and its Python dependencies are included in `requirements.txt`. The bundled implementation has a PyTorch deformable-attention fallback, so a separate GroundingDINO CUDA build is not required. Override asset locations with `SAGO_GROUNDING_CHECKPOINT` and `SAGO_GROUNDING_BERT` if needed. See [models/README.md](models/README.md) and [GroundingDINO provenance](vendor/GROUNDINGDINO_PROVENANCE.md).

GroundingDINO loads on the first SAM2 text prompt and is then reused. It locates candidate boxes in the initial view; SAM2 handles mask generation and subsequent multi-view tracking.

### Optional: SAM3 and SAM3.1 environment

SAM3 models run in a separate Python subprocess, allowing the renderer and SAM2 environment to remain independent. The bundled compatibility patches have been exercised with PyTorch 2.5.1 and CUDA 12.1; they do not require FlashAttention 3 or model compilation.

With the main environment active, create an isolated inference environment:

```bash
python -m venv .venv-sam3
.venv-sam3/bin/python -m pip install --upgrade pip
.venv-sam3/bin/python -m pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu121
.venv-sam3/bin/python -m pip install -r requirements-sam3.txt
.venv-sam3/bin/python -c "import torch, sam3; print('GPU available:', torch.cuda.is_available())"
```

Follow the official [SAM3 checkpoint access instructions](https://github.com/facebookresearch/sam3), then place either or both checkpoints in `models/`:

```text
models/
  sam3.pt
  sam3.1_multiplex.pt
```

Start the application from the main environment with `python run.py` and choose the model in the UI. The default worker interpreter is `.venv-sam3/bin/python`. You can explicitly configure the interpreter and checkpoints:

```bash
python run.py \
  --sam3-python ./.venv-sam3/bin/python \
  --sam3-checkpoint ./models/sam3.pt \
  --sam31-checkpoint ./models/sam3.1_multiplex.pt
```

Equivalent environment variables are `SAGO_SAM3_PYTHON`, `SAGO_SAM3_CHECKPOINT`, and `SAGO_SAM31_CHECKPOINT`. Worker errors are recorded in `outputs/sam3-worker.log`. Switching away from a SAM3 model releases its subprocess. See [SAM3 provenance](vendor/SAM3_PROVENANCE.md) for the pinned upstream revision and compatibility changes.

## Load a scene

Use **Open scene**, drag a PLY into the viewport, reopen a recent scene, or enter a scene path such as `./data/my-scene`.

Accepted inputs are a standard 3DGS `.ply` file or a training output directory containing `point_cloud/iteration_*/point_cloud.ply`; the latest iteration is selected automatically. Ordinary XYZ point clouds lack the required Gaussian attributes and cannot be loaded as 3DGS scenes. When a nearby `cameras.json` is available, its original views appear in the camera selector. Otherwise, the initial view is estimated from the Gaussians.

The native file picker reads the scene directly and requires `zenity` or `kdialog` on a Linux desktop. A browser file picker is available as a fallback. Browser-selected and dropped files are uploaded to `outputs/uploads/`, with progress and content-based caching. The upload limit is approximately 2 GB; use the native picker or a scene path for larger files. An uploaded PLY does not include neighboring files such as `cameras.json`, so use the native picker or directory path when you need those cameras.

The browser remembers the eight most recently loaded scenes. Removing a recent entry does not delete scene files or cached uploads.

## Segmentation workflow

1. Load a scene. Drag to orbit, right-drag or Shift-drag to pan continuously, hold **W/S** to move forward/backward along the view direction, **A/D** to strafe, and scroll to zoom. Releasing a key or leaving the window stops roaming; text inputs and segmentation/calibration/correction modes suspend it. The viewport fills the available rectangular space. Camera controls include up direction, target, field of view, and render resolution.
2. Frame the target and press **M**, or enable **Enter segmentation prompts**, to capture the view.
3. Add foreground points with the left mouse button, background points with the right button, or drag a box using the box tool. Points and boxes can be combined. If the selected model has text support configured, enter a short description such as `yellow rubber duck`.
4. The 2D result updates automatically as prompts change. Text input is debounced by approximately 450 ms. Select the candidate that covers the intended object. SAM2 supplies four mask levels per detected target; SAM3 models supply instance candidates.
5. Click **3D segmentation** or press **P** for six-view refinement. The operation selector supports replacement, addition, subtraction, and intersection. The default mask expansion is 2 px. When a foreground selection exists, subsequent prompt captures use that foreground's rendered image and depth. Clear the current selection to start from the scene again. Ground calibration still captures the full active scene.
6. Press **M** to leave prompt mode and inspect the result. Switch between **All**, **Current foreground**, and **Current background**.
7. Use **Add ID** to save the current selection as an object and continue segmenting. The completed object becomes hidden, and its Gaussians are excluded from subsequent selections until shown again.
8. Click **Save segmentation results** to persist assigned object IDs and metadata in the scene's `SAGO_knowledge.json`. Export a selection as PLY, a Gaussian-order boolean `.npy` mask, the current rendered image, or a 2D mask PNG.

Keyboard shortcuts do not trigger segmentation while you are typing in a text field. Undo and redo retain the most recent 12 editing steps. Exports are written to `outputs/` and downloaded through the browser.

| Model | Prompts | 3D tracking |
| --- | --- | --- |
| SAM2 | Points, boxes, optional GroundingDINO text | SAM2 memory tracking |
| SAM3 | Text, points, boxes | Official SAM3 video predictor |
| SAM3.1 | Text, points, boxes | Official Object Multiplex predictor |

Virtual Drones sample six views at +45°, -45°, +90°, -90°, +135°, and -135°, with a 30° elevation. The first move uses the mean of valid surface points reconstructed from the initial mask and rendered depth. After the first additional view, each next move uses the mean position of the retained foreground Gaussians. The two directions maintain independent tracking memories.

The drone gallery shows each rendered input and segmentation overlay, with downloads for the original image, overlay, and binary mask. Segmentation timings exclude initial model loading, browser network latency, and preview PNG saving. Model loading has its own timing dialog. Cancellation takes effect after submitted GPU work completes and does not commit an intermediate 3D selection.

## Foreground GIF export

Use **Export all object GIFs** to export every saved object ID as a separate GIF, regardless of visualization checkboxes or hidden status. Use **Export current GIF** to export the current foreground selection. Each object gets its own center and orbit distance, with original RGB colors against white. Downloads are named `ID_<number>_<object name>.gif`. The preview dialog lets you select, play, and download each generated GIF independently.

The camera orbits around the mean foreground position using the scene's up vector, with one full rotation at a configurable elevation (60° by default, adjustable from -89° to 89° in **GIF settings**). The distance fits the 99th percentile of foreground distances and three-sigma Gaussian extents to the field of view, with a margin, so sparse outliers do not make the object too small. All selected Gaussians are still rendered. Defaults are 512 × 512 pixels, 72 frames in one orbit, and 20 FPS for a 3.6-second looping animation. **GIF settings** also offers 256/768-pixel images and 36 frames per orbit.

Generation reports rendering/encoding progress and elapsed time and can be cancelled. A preview dialog provides playback and download; **View GIF** reopens the latest successful export. Files are saved in `outputs/`. Temporary frames are removed on completion, cancellation, or failure. Batch export reports the current object and overall progress. Cancelling or failing a batch cleans up every file generated by that batch, leaving previous exports intact. GIF encoding uses Pillow with a shared palette across views; FFmpeg is not required. Export does not load SAM or modify the camera, selection, IDs, or undo history.

## Object IDs and correction

Objects can be renamed, hidden, shown, selected, exported, merged, and renumbered. Merging retains the smaller ID and its name. Renumbering maps existing IDs to consecutive values starting at 1; background remains 0. These edits support undo and redo and require saving to persist.

In **All**, independently toggle background, object foreground, and foreground coloring by ID. **Current foreground** and **Current background** display the active round's selection and its complement. Loading a scene shows all saved objects by default.

Use **Correction** in **All** to assign visible Gaussians to a target ID. ID `0` means background; a new positive ID creates an object. **Ellipse correction** is the default, with **Point correction** also available. Click repeatedly to accumulate visible Gaussians, or drag a box to replace the pending selection. Selected Gaussians appear yellow; only **Confirm correction** applies the change. Cancel the selection or press Esc to discard the preview.

Correction uses the current rendered view's visible Gaussian ownership, so it does not select through occluded surfaces. Ellipse box selection includes visible rendered footprints intersecting the box; point box selection additionally requires the projected center to fall inside. Hidden layers are excluded. Correction does not run SAM.

Saved segmentation contains an `int32` ID vector with one entry per Gaussian in the original PLY order: `0` for background and positive integers for objects. The knowledge file also stores object names, metadata, and the next ID. Loading validates the Gaussian count and a position/order SHA-256 before restoring labels. An unassigned current selection must be saved as an object before it is included in this vector.

## Ground calibration and visualization

The **Camera** panel provides normalized RGB background controls (0–1) and a color picker, defaulting to **(0, 0, 0)** (black); **(1, 1, 1)** is white. The **Black / White** button switches directly to white for a dark background or black for a light background. The color applies to RGB, Splats, Points, Depth, prompt captures and image exports. Exit prompts or ground calibration before changing it. GIF exports keep their white background.

**Delete** removes the selected object record and returns all of its Gaussians to ID **0** (background), preserving their geometry. It also removes them from the current foreground. Undo/Redo restores or reapplies the edit; use **Save IDs** / **Save segmentation** to persist it to `SAGO_knowledge.json`. Other object IDs keep their numbers.

**Adversarial background** is enabled by default in the 3D segmentation panel. Before mask padding, SAGO samples the selected prompt mask’s one-pixel inner boundary in the original RGB capture. A mean brightness below 0.5 selects white; otherwise black. The same background is used for all six drone tracking inputs and preview renders, while the main viewport keeps its manual color. Turn the option off to use the manual background for drone views too.

**Three-point ground calibration** captures a view and lets you choose three distinct, non-collinear ground or tabletop positions. Rendered depth reconstructs their world coordinates. The unit normal is `normalize((P2 - P1) × (P3 - P1))`, initially oriented toward the camera; reverse it if needed, then apply the ground plane.

Applying the calibration updates the camera and drone up direction and saves the plane and up vector to `SAGO_knowledge.json`. Reversing an applied normal also saves it. Future scene loads restore the calibration. Selecting or previewing points does not overwrite saved knowledge. Ground calibration does not load SAM.

For standard training directories, the knowledge file is stored at the scene root; for standalone PLY files, it is stored beside the PLY. Uploaded files use the server-side upload directory. Entries are keyed by relative PLY path under `scenes`. Ground knowledge and segmentation coexist, and saves use atomic replacement.

Optional 3D overlays include an upright bounding box aligned with the up vector and a combined drone trajectory/frustum display. Frusta show the initial camera and six sampled cameras, including their orientation and field of view. Trajectories belong to the current session and do not modify segmentation data.

The viewport offers four independent rendering modes:

| Mode | Display |
| --- | --- |
| RGB | Original colored Gaussian rendering |
| Splats | Smaller opaque ellipses showing Gaussian shape |
| Points | One-pixel Gaussian centers with depth occlusion |
| Depth | Depth colors normalized over the valid 2nd–98th percentile range |

Entering segmentation prompts or ground calibration switches to RGB. Side-panel modules can be collapsed, and the browser remembers their expanded state.

## Project layout

```text
run.py                       Application entry point
frontend/                    Browser UI; no Node.js build or CDN dependency
backend/server.py            Flask API and serialized GPU task queue
backend/engine.py            Scene, camera, selection, object, and export adapter
backend/camera_math.py       Camera mathematics without desktop UI imports
sago_core/                   Bundled SAGO segmentation and SAM2 tracking code
gs/                          Gaussian scene model and rendering interfaces
vendor/                      CUDA extensions, GroundingDINO, and SAM3 sources
models/                      External checkpoint and tokenizer locations
scripts/install-cuda.sh      CUDA extension build helper
scripts/gpu-smoke.py         Optional GPU integration check
tests/                       Backend and browser checks
```

## Validation

Backend checks do not require a GPU:

```bash
python -m unittest discover -s tests -v
```

With Node.js installed, check frontend syntax and localization:

```bash
node --check frontend/app.js
node tests/i18n.cjs
```

For a real GPU integration check, supply your own scene and checkpoint:

```bash
python scripts/gpu-smoke.py \
  --scene ./data/my-scene \
  --checkpoint ./models/sam2_hiera_large.pt
```

Browser tests require Node.js, Playwright, and Chrome. Run the server on port 8080 first, then run these commands in another terminal from `SAGO_web/`:

```bash
npm install --no-save --package-lock=false --prefix ./.cache/browser-tests @playwright/test
PLAYWRIGHT_MODULE="$PWD/.cache/browser-tests/node_modules/@playwright/test" \
SAGO_CHROME="$(command -v google-chrome)" \
SAGO_TEST_SCENE=./data/my-scene node tests/browser-smoke.cjs
```

`PLAYWRIGHT_MODULE` and `SAGO_CHROME` are resolved from your current checkout and installed browser. Scene-input checks are also available in `tests/scene-inputs.cjs`. Screenshots and test downloads are stored in `test-results/`. These checks validate application behavior, not segmentation accuracy. Node.js is not needed to run the application itself.

## Scope

SAGO Web provides the interactive segmentation workflow for static 3DGS scenes. The original desktop Python expression editor, training connection, video export, dynamic TAG panels, RAG interface, and 2DGS UI are not exposed as web controls.

The server binds to `127.0.0.1` and uses a single shared workspace with serialized GPU operations. Images are transferred as PNG responses. Remote or multi-user deployment requires additional authentication, session isolation, and streaming design.

## Citation and licenses

SAGO is based on **Online Segment 3D Gaussians via Launching Virtual Drones**, Liwei Liao, Rongjie Wang, and Ronggang Wang, ECCV 2026 ([paper](https://arxiv.org/abs/2607.01628)). See [CITATION.cff](CITATION.cff).

The repository [LICENSE](LICENSE) does not replace licenses attached to bundled components or checkpoints. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for component licenses and [models/README.md](models/README.md) for checkpoint setup.

### Semantic object knowledge

After assigning object IDs, open **Semantic knowledge** and select **Build / update semantic knowledge**. This indexes every object, including hidden IDs, and saves both the current ID segmentation and its features atomically in the scene's `SAGO_knowledge.json`.

Each object is rendered alone against white at four azimuths 90° apart. The camera uses the GIF settings for resolution and orbit elevation (60° by default), the same object-fit distance calculation, scene FOV, and calibrated up vector. Four normalized 512-dimensional CLIP image embeddings are stored per object, along with the exact model revisions and rendering parameters. Rendered views are temporary in-memory images; model weights remain in the external Hugging Face cache (`HF_HOME`, if configured) and are downloaded only on first use. No checkpoint is added to the project or installer.

Enter a description and select **Find object IDs**. The same `openai/clip-vit-base-patch32` model encodes the text; results list object IDs, names, and the highest cosine similarity across their four views. These scores indicate relative similarity, not confidence probabilities. English descriptions work best with this CLIP model; queries must fit its 77-token context.

Indices are reused after reopening a scene. Object membership changes invalidate the affected entries, deleted IDs are omitted, and renaming or renumbering preserves valid embeddings. Search reports IDs that need rebuilding and excludes their old features. A changed up vector also requires rebuilding. Updating the index refreshes changed entries and reuses unchanged ones with identical render settings. Cancellation or failure leaves the previous knowledge file intact.

After a successful semantic search, the viewport switches to **All**, highlights the top-ranked object in gold, and enables its up-aligned 3D bounding box. A hidden or unchecked match is temporarily revealed without changing its visibility setting, ID membership, or current segmentation selection. The highlight follows camera movement. Use **Clear search highlight** to restore normal rendering; starting segmentation, editing objects, or loading a scene also clears it.

### GaussianRAG (API prototype)

Open **GaussianRAG** to build spatial knowledge and ask natural-language questions about saved object IDs. This prototype uses a configurable **Chat Completions-compatible API** with function/tool calling. It does not load a local LLM. Provider credentials and a model name are required for questions; no provider or model is silently selected. A vision-capable model is additionally required for optional image descriptions.

In **API settings**, enter the provider's Base URL (ending in `/v1` for providers that use that prefix), model name, and key, then select **Test API connection**. The application appends `/chat/completions`. These fields are session-only; keys are not stored in browser storage, Desktop preferences, or scene knowledge. Alternatively, configure the backend process:

```bash
export SAGO_RAG_BASE_URL='https://your-provider.example/v1'
export SAGO_RAG_MODEL='your-tool-capable-model'
# Set SAGO_RAG_API_KEY securely in your shell/session before starting the app.
```

`OPENAI_BASE_URL` and `OPENAI_API_KEY` are accepted as fallback environment variables. Environment variables must be available to the running backend/Desktop process; this feature does not automatically read `.env` files. The API connection test checks basic text access; asking spatial questions also requires tool-call support.

1. Name/assign object IDs and calibrate the ground if available. Hidden objects are included.
2. Select **Build / update spatial knowledge**. This works offline using existing object names and geometry, and also saves current IDs. To enrich names with semantic labels, colors and descriptions, enable **Generate object descriptions with a vision API**. This sends four isolated object images per changed object to the configured API, using the GIF elevation, resolution and auto-fit distance. Unchanged compatible descriptions are reused.
3. Ask a question such as “Which object is closest to ID 2?”, “What is above the table?”, or “How many chairs are in the catalog?” The model sees the object catalog and can call a local spatial tool to obtain requested geometry and all other objects ordered by centroid distance. Answers cite `[ID N]`; the UI also lists cited IDs and time spent. Each question starts a fresh context.

Knowledge is stored in the current scene's `SAGO_knowledge.json` under `gaussian_rag` and exported to a readable `SAGO_spatial_<scene-key-hash>.json` beside it. The document records names, optional API descriptions and their model provenance, Gaussian counts, centroids, up-aligned boxes, common-frame bounds, signed ground-height ranges (when calibrated), and the six nearest saved objects' spatial relationships. Arbitrary requested relationships are computed from the same document during questioning. CLIP embeddings are not interpreted as textual class labels. The catalog covers positive object IDs only; it does not describe background or unsegmented objects.

Distances are **scene units**, not guaranteed metres. Nearest-object ranking uses centroid distance. Box-gap values are lower bounds from common-frame bounds, and box overlap/containment does not prove physical contact, support, or containment of real surfaces. Left/right/front/back require an explicit reference frame. The model is instructed to state uncertainty, but answers and visual descriptions still require review. Updating IDs, names, scene identity, up direction or ground calibration invalidates the saved spatial document until rebuilt; visibility changes alone do not. API failures or cancellation before persistence preserve existing knowledge. Network requests time out after 60 seconds; cancellation is checked between requests and render steps.

Enabling visual descriptions sends images to your chosen endpoint. Asking questions sends the catalog, question and retrieved spatial facts. No raw Gaussian arrays or CLIP vectors are sent. Provider data handling and API charges follow the configured provider's terms. A real provider/model must be configured to validate the application's automatic answer quality; deterministic API fixtures verify transport and integration only.

### GaussianRAG validation with Codex (no API)

To test grounded reasoning directly in a Codex conversation, export an existing saved scene using the project's Python environment:

```bash
python scripts/rag-local.py export --scene /path/to/scene.ply --output outputs/rag-validation \
  --question "Which saved object is closest to ID 1?"
python scripts/rag-local.py facts --document outputs/rag-validation/spatial.json --ids 1
```

Ask Codex to read the exported `request.json` and `spatial.json`, query local facts, and answer with object IDs. A scene folder is also accepted when its knowledge file identifies exactly one saved segmented PLY. The exporter validates Gaussian ordering, includes hidden objects, and leaves the original PLY and knowledge unchanged. It uses CPU geometry without SAM, CLIP, rendering, or network requests. Distances retain scene units; physical support and viewer-relative directions require further evidence.

Codex supplies the natural-language reasoning in the conversation. This workflow does not connect the application's question box to Codex; automatic in-app answers still use the configured API. Optional rendered images can be reviewed separately and must be identified as visual observations rather than measured geometry.

### Export foreground objects as meshes

In **Export**, choose **Export object meshes**, select saved foreground IDs, and start reconstruction. Each selected ID produces a separate **OBJ triangle mesh with vertex colors**, restored to the original scene coordinates. Hidden IDs can also be exported. This does not change segmentation IDs or scene knowledge.

The pipeline follows SAGO's `mega_use()` workflow: render four orbits at 0°, 20°, 30° and 60°, export COLMAP cameras and Gaussian centers, prepare masked RGBA images, optimize NeuS2, extract a marching-cubes surface, and remove unused vertices. Masks come from accumulated opacity of the selected object's splats, thresholded at 0.5, so exporting saved IDs does not require an earlier SAM tracking session. Only the selected object's Gaussians are rendered. Views of missing surfaces cannot recover real unseen geometry reliably.

Mesh extraction requires a separate, working **CUDA NeuS2 installation and matching Python environment**. NeuS2 and its initialization weights are not bundled or downloaded by SAGO. Set the **Mesh exporting directory**, **Mesh exporting Python executable**, and **network configuration** in the mesh dialog, or configure:

```bash
export SAGO_MESH_ROOT=/path/to/NeuS2
export SAGO_MESH_PYTHON=/path/to/neus2/environment/bin/python
```

The NeuS2 directory must contain its compiled `pyngp` extension under `build*`, `utils/mlp_weights*.txt`, and the selected network configuration (default `configs/nerf/workspace_mask002.json`). Install and compile NeuS2 according to its upstream instructions before exporting. If paths are blank, SAGO checks these environment variables, then `~/NeuS2` and common local NeuS2 virtual/Conda environments. It checks the extension before rendering. Custom builds must provide the same `pyngp` training/extraction API and network schema as the original SAGO NeuS2 workflow.

Defaults are **512 × 512 images, 72 views per orbit (288 total), 15,000 training steps, and a 512³ extraction grid**. These settings are independent of GIF settings and require substantially more time and GPU memory than segmentation; lower view counts, training steps or grid resolution can be used for a quick test. Multiple selected objects are reconstructed in parallel by independent workers (default **2**, configurable **1–4** under **Parallel exports**). More concurrent jobs use more GPU memory; rendering remains on the main engine thread and overlaps reconstruction. Results remain ordered by object ID. Progress aggregates all objects; **Cancel** terminates every active worker, and a failed worker cancels the rest of the batch. The `SAGO_NEUS2_ROOT` and `SAGO_NEUS2_PYTHON` environment variables remain supported as compatibility aliases.

Meshes are saved under the application's output directory as `mesh-<run-id>/ID_<number>/ID_<number>_<name>.obj`. Web shows download links and saved paths; Desktop offers **Save as…** and **Open output folder**. The run directory also contains rendered inputs, camera data, normalization metadata, training logs and a result manifest. Failed/cancelled runs keep diagnostic files but remove final OBJ exports from that batch; their temporary raw mesh files are not offered as completed exports. No NeuS2 training checkpoints are bundled into the application.
