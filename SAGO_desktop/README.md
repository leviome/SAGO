# SAGO Desktop

Part of the [SAGO repository](../README.md). For the browser application, see [SAGO Web](../SAGO_web/README.md).

A native Linux desktop application for interactive 3D Gaussian segmentation. The interface uses **PySide6 / Qt Widgets**, a QWidget viewport painted by QPainter, and native dialogs. There is no browser, WebEngine, Flask server, listening port, HTML, JavaScript, or dependency on a SAGO Web checkout. The Gaussian renderer and segmentation algorithms are vendored into this project.

## Run from source

An NVIDIA GPU, a compatible driver, and a CUDA-enabled PyTorch runtime are required for scene rendering. Python 3.10+ and a Linux x86-64 desktop are supported. Qt itself can start without a GPU, so settings and help remain accessible when the CUDA environment is incomplete.

On Debian/Ubuntu, the Qt X11 plugin may also require `sudo apt install libxcb-cursor0`.

```bash
git clone https://github.com/leviome/SAGO.git
cd SAGO/SAGO_desktop

# Reuse a compatible CUDA Python installation without changing that environment:
/path/to/cuda/python -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install --no-build-isolation \
  ./vendor/simple-knn ./vendor/rasterizer-depth ./vendor/rasterizer-mask ./vendor/rasterizer-topk
./sago
./sago path/to/gaussians.ply
```

Run source and build commands from `SAGO_desktop/`. Checkpoints and virtual environments are not included in Git. The SAM2 checkpoint downloads automatically on first use as described below. No SAGO Web checkout is required.

## Debian package

The local v1.5.0 build adds parallel selected-object mesh export and labels the UI as **Mesh exporting**, with configurable parallel jobs, reconstruction settings, progress, cancellation, and native save dialogs. It also includes the GaussianRAG API prototype from v1.1.12. NeuS2 remains a separate runtime dependency and is not bundled. It awaits user validation and is not published. The latest published installer remains v1.1.11.

Desktop **1.1.11** adds four-view CLIP semantic object retrieval with automatic top-match coloring and an up-aligned 3D bbox. It also includes Cube/cylinder foreground completion, Esc to exit, black/white backgrounds, live right-drag panning and WASD roaming. Model weights remain external and are downloaded on first use.

Download `sago-desktop_1.1.11_amd64.deb` and its checksum from the [Desktop release](https://github.com/leviome/SAGO/releases/tag/desktop-v1.1.11). Run the installation command from the download directory. A local build writes the installer under `dist/`.

```bash
sudo dpkg -i ./sago-desktop_1.1.11_amd64.deb
# If system libraries are missing, let apt resolve them:
sudo apt-get -f install
sago
sago path/to/gaussians.ply
```

Installation provides a desktop-menu entry, the `sago` command, and a PLY MIME association. It installs resources under `/opt/sago-desktop` and writes application data only to user-owned XDG directories. It does not change your default PLY handler without your desktop's confirmation.

**Version 1.1.7 and later package the app, algorithm source and Qt (when built with `--qt-site`), without model weights. SAM2 downloads on first use. NVIDIA drivers and CUDA Python runtimes are not bundled.** The older published 1.1.2 installer includes weights. On a new machine, configure a compatible CUDA Python environment first:

```bash
/opt/sago-desktop/scripts/setup-runtime.sh --python /path/to/cuda/python
# Or use an existing fully configured environment:
SAGO_PYTHON=/path/to/cuda/python sago scene.ply
```

The launcher checks the project virtual environment, the user runtime created by the setup script, and local Conda environments. Set `SAGO_PYTHON` to choose explicitly. A CUDA toolkit compatible with PyTorch is required to compile the bundled rasterizers. PyTorch and torchvision should be installed as a matching CUDA pair before installing other requirements.

Build the package:

```bash
python3 scripts/build-deb.py --qt-site .venv/lib/python3.10/site-packages
```

The build runs without root, sets archive ownership to root, and excludes virtual environments, outputs, caches, and test files. All builds exclude model weights; `--without-models` remains a compatibility option.

## Models

The Desktop version supports **SAM2 only**, using point and box prompts. The default checkpoint is `$XDG_CACHE_HOME/sago-desktop/models/sam2_hiera_large.pt` (normally `~/.cache/sago-desktop/models/sam2_hiera_large.pt`). Use **Preferences** or `--checkpoint` to select an existing compatible SAM2 checkpoint. Clear the Preferences checkpoint field to restore automatic download. Existing custom files are never overwritten.

The model downloads and loads lazily when first entering prompt mode (**M**). The first download requires internet access and about 857 MiB of disk space. The dialog shows download progress, transferred MiB, verification and loading; interrupted downloads retain partial data for resume. A failed download offers **Retry download** and a cancel button stops the request after the current network read (up to 20 seconds). The official SAM2 file is checked against its expected size and SHA-256 before becoming usable. Later launches reuse the verified cache without downloading. The timing dialog closes automatically after loading; GPU loading time is reported separately from downloading. For offline use, supply an existing checkpoint in Preferences or via `--checkpoint`. Image features are reused while updating prompts in the same view.

The installer includes SAM2 tracking code; its checkpoint is downloaded from Meta’s official SAM2 endpoint. Model terms remain those of the original owner; see `THIRD_PARTY_NOTICES.md` and `models/README.md`.

## Foreground completion

**Cube completion** uses the same upright bounding box as the 3D BBOX display (minimum horizontal area, independent side lengths). **Cylinder completion** fits the minimum enclosing circle of the horizontal foreground projection and uses the foreground's lowest/highest positions for its caps. Both align to the current up vector.

All Gaussian centers inside the fitted volume join the current foreground, including occluded centers. Gaussian ellipse intersections are not used. Outside centers, Gaussian geometry and saved object IDs stay unchanged, and existing object visibility settings remain in effect. The result appears as Current foreground, supports Undo/Redo, and can be exported or assigned through the normal Add ID workflow. Empty foregrounds, prompts, ground calibration and correction disable completion.

## Desktop workflow

The **Camera** panel provides normalized RGB background controls (0–1) and a color picker, defaulting to **(0, 0, 0)** (black); **(1, 1, 1)** is white. The **Black / White** button switches directly to white for a dark background or black for a light background. The color applies to RGB, Splats, Points, Depth, prompt captures and image exports. Exit prompts or ground calibration before changing it. GIF exports keep their white background.

**Delete** removes the selected object record and returns all of its Gaussians to ID **0** (background), preserving their geometry. It also removes them from the current foreground. Undo/Redo restores or reapplies the edit; use **Save IDs** / **Save segmentation** to persist it to `SAGO_knowledge.json`. Other object IDs keep their numbers.

**Adversarial background** is enabled by default in the 3D segmentation panel. Before mask padding, SAGO samples the selected prompt mask’s one-pixel inner boundary in the original RGB capture. A mean brightness below 0.5 selects white; otherwise black. The same background is used for all six drone tracking inputs and preview renders, while the main viewport keeps its manual color. Turn the option off to use the manual background for drone views too.

- Open a PLY or training scene directory with native file dialogs, drag a PLY into the viewport, choose a recent scene, or pass a path to `sago`.
- Drag to orbit, right-drag or Shift-drag to pan continuously, and scroll to zoom. Hold **W/S** to move forward/backward along the viewing direction and **A/D** to strafe. Eye and target translate together; orbit radius and viewing angles stay fixed. Speed scales with the current orbit radius, and diagonal movement is normalized. Releasing a key or leaving the window stops roaming. Text fields, dialogs, prompts, calibration and correction suspend roaming. The viewport fills available space while preserving the rendered aspect ratio. Original scene cameras, FOV, resolution, target and up vector can be selected. Render size defaults to Auto, matching the viewport in physical screen pixels (512–2560 long edge, rounded up in steps of 64). Manual sizes include 512, 768, 1024, 1536, 1920 and 2560. Navigation uses up to 768 pixels and refreshes at the selected quality after 450 ms of inactivity. Prompt and ground capture use the selected quality with fixed pixel coordinates. Image display uses smooth scaling.
- Press **M** to enter segmentation prompt mode. Left-click foreground points, right-click background points, or draw a box. 2D segmentation updates automatically after each change. Choose the desired candidate and press **P** for six-view Virtual Drone refinement. Padding defaults to 2 pixels.
- Press **Esc** to exit the application, including when a text field has focus. Running jobs use the existing cancellation/wait flow, and unsaved object IDs retain the Save/Discard/Cancel check. Dialogs keep their normal Esc-to-close behavior.
- Press **C** to cancel the current round and return to camera navigation. Prompts and 2D candidates are cleared. Running 3D refinement stops after the current GPU step; a completed, uncommitted 3D result reverts to the selection before that refinement. Saved object IDs are preserved. Shortcuts do not activate while typing in input fields.
- Use **Add ID** to save and hide the current object and begin the next round. Save selections, rename, select, hide, delete, merge and renumber objects. **Save IDs** persists the integer label vector (0 = background) to the scene's `SAGO_knowledge.json`.
- Use three ground points to calibrate a plane, flip its up vector if needed, and apply. Ground knowledge is persisted by the backend and restored on load.
- In **All**, show background/foreground independently, filter object IDs, and optionally color foreground by ID. **Current foreground/background** refer to the current selection. RGB, Splats, Points and Depth are available above the viewport.
- **Correction** supports ellipse or point selection, a target ID (default 0), repeated clicks to accumulate visible Gaussians and a drag box to replace the pending selection. Only visible Gaussians are selected. Confirm highlighted Gaussians before committing. Undo/redo remains available.
- Optional upright bounding boxes and a single route/frustum toggle show the initial camera and Virtual Drone trajectory. The gallery provides each drone render, segmentation overlay and binary mask, with native save dialogs.
- Export foreground PLY, per-Gaussian masks, the rendered image, 2D masks, the current GIF, or all saved object GIFs. Each object has an independent single camera orbit, configurable elevation (default 60°), white background and original RGB. GIF defaults: 512 × 512, 72 frames, 20 FPS. QMovie provides native playback, object selection and named save dialogs.
- Every control module can collapse. English is the default, with a Chinese toggle. Computation uses one serialized background worker. Segmentation/GIF jobs support cancellation; the window remains responsive.
- Closing with dirty object IDs asks whether to save. Closing during a job offers cancellation and waits for the current GPU operation before cleaning up the worker. Undo history and incomplete prompts are session state and are not restored after exiting.

## Data locations

- Settings and recent paths: `$XDG_CONFIG_HOME/sago-desktop/settings.json` (default `~/.config/sago-desktop/`).
- Exports and diagnostics: `$XDG_DATA_HOME/sago-desktop/outputs/` (default `~/.local/share/sago-desktop/outputs/`).
- Cache: `$XDG_CACHE_HOME/sago-desktop/` (default `~/.cache/sago-desktop/`); downloaded weights and resumable `.part` files are in `models/`.
- Scene ground and labels: scene-local `SAGO_knowledge.json` and its associated data, using the backend's existing format.

**Preferences** can change the export directory and SAM2 checkpoint path. The source tree and installed `/opt` directory remain read-only during normal use. Opening an existing scene reads its knowledge; only explicit save/ground-apply operations persist scene edits.

## Validation

```bash
QT_QPA_PLATFORM=offscreen .venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/gpu-smoke.py /path/to/gaussians.ply
```

The tests exercise native viewport coordinate mapping, worker serialization/cancellation, model timing, automatic prompts, candidate selection, correction preview/commit, multi-ID operations, GIF settings, language switching, CLI and package contents. GPU smoke checks load a real PLY, render all four styles and generate a GIF using the independent backend. See `VALIDATION.md` for the actual checks run on the development machine.

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

Enabling visual descriptions sends images to your chosen endpoint. Asking questions sends the catalog, question and retrieved spatial facts. No raw Gaussian arrays or CLIP vectors are sent. Provider data handling and API charges follow the configured provider's terms. A real provider/model must be configured to validate answer quality; deterministic API fixtures verify transport and integration only.

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
