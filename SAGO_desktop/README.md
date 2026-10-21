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

The current local build is `dist/sago-desktop_1.1.7_amd64.deb`, with checkpoint-free packaging and automatic, verified first-use SAM2 downloads. This version removes bundled model weights and retains object deletion, normalized RGB backgrounds and adaptive drone backgrounds.

Download `sago-desktop_1.1.7_amd64.deb` and its checksum from the [Desktop release](https://github.com/leviome/SAGO/releases/tag/desktop-v1.1.7). Run the installation command from the download directory. A local build writes the installer under `dist/`.

```bash
sudo dpkg -i ./sago-desktop_1.1.7_amd64.deb
# If system libraries are missing, let apt resolve them:
sudo apt-get -f install
sago
sago path/to/gaussians.ply
```

Installation provides a desktop-menu entry, the `sago` command, and a PLY MIME association. It installs resources under `/opt/sago-desktop` and writes application data only to user-owned XDG directories. It does not change your default PLY handler without your desktop's confirmation.

**Version 1.1.7 packages the app, algorithm source and Qt (when built with `--qt-site`), without model weights. SAM2 downloads on first use. NVIDIA drivers and CUDA Python runtimes are not bundled.** The older published 1.1.2 installer includes weights. On a new machine, configure a compatible CUDA Python environment first:

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

## Desktop workflow

The **Camera** panel provides normalized RGB background controls (0–1) and a color picker, defaulting to **(0, 0, 0)** (black); **(1, 1, 1)** is white. The color applies to RGB, Splats, Points, Depth, prompt captures and image exports. Exit prompts or ground calibration before changing it. GIF exports keep their white background.

**Delete** removes the selected object record and returns all of its Gaussians to ID **0** (background), preserving their geometry. It also removes them from the current foreground. Undo/Redo restores or reapplies the edit; use **Save IDs** / **Save segmentation** to persist it to `SAGO_knowledge.json`. Other object IDs keep their numbers.

**Adversarial background** is enabled by default in the 3D segmentation panel. Before mask padding, SAGO samples the selected prompt mask’s one-pixel inner boundary in the original RGB capture. A mean brightness below 0.5 selects white; otherwise black. The same background is used for all six drone tracking inputs and preview renders, while the main viewport keeps its manual color. Turn the option off to use the manual background for drone views too.

- Open a PLY or training scene directory with native file dialogs, drag a PLY into the viewport, choose a recent scene, or pass a path to `sago`.
- Drag to orbit, Shift/right-drag to pan, and scroll to zoom. The viewport fills available space while preserving the rendered aspect ratio. Original scene cameras, FOV, resolution, target and up vector can be selected. Render size defaults to Auto, matching the viewport in physical screen pixels (512–2560 long edge, rounded up in steps of 64). Manual sizes include 512, 768, 1024, 1536, 1920 and 2560. Navigation uses up to 768 pixels and refreshes at the selected quality after 450 ms of inactivity. Prompt and ground capture use the selected quality with fixed pixel coordinates. Image display uses smooth scaling.
- Press **M** to enter segmentation prompt mode. Left-click foreground points, right-click background points, or draw a box. 2D segmentation updates automatically after each change. Choose the desired candidate and press **P** for six-view Virtual Drone refinement. Padding defaults to 2 pixels.
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
