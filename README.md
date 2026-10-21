# SAGO — Segment Any Gaussians Online

Official code for **Online Segment 3D Gaussians via Launching Virtual Drones** (ECCV 2026), by Liwei Liao, Rongjie Wang, and Ronggang Wang.

[Paper](https://arxiv.org/abs/2607.01628) · [Web installation](SAGO_web/README.md) · [Desktop installation](SAGO_desktop/README.md) · [Releases](https://github.com/leviome/SAGO/releases)

SAGO turns 2D object prompts into 3D Gaussian selections, then refines them from multiple Virtual Drone viewpoints. This repository contains two complete applications. Each includes its own renderer and segmentation backend and can run independently of the other.

## Object gallery

<table>
  <tr>
    <td width="16.66%"><img src="assets/0b4c9976e02a4d8da2b2897886e6e58c.gif" width="160" alt="Segmented 3D object 1"></td>
    <td width="16.66%"><img src="assets/0b60fefbe2494ac48984b66bd344537f.gif" width="160" alt="Segmented 3D object 2"></td>
    <td width="16.66%"><img src="assets/1bdf9bbb6d9b4099a736c5ccbd0659e3.gif" width="160" alt="Segmented 3D object 3"></td>
    <td width="16.66%"><img src="assets/260f2a84d6fe4aefb7ee8cf5e467229b.gif" width="160" alt="Segmented 3D object 4"></td>
    <td width="16.66%"><img src="assets/3232c6cbb3124a6a982f18f23505a487.gif" width="160" alt="Segmented 3D object 5"></td>
    <td width="16.66%"><img src="assets/3627202b115c4de584c16fd9f5369aca.gif" width="160" alt="Segmented 3D object 6"></td>
  </tr>
  <tr>
    <td width="16.66%"><img src="assets/36fb5dd3ebf549ebabe27f75b155985d.gif" width="160" alt="Segmented 3D object 7"></td>
    <td width="16.66%"><img src="assets/39f0430de80c4d24a3aa6acbbff7961a.gif" width="160" alt="Segmented 3D object 8"></td>
    <td width="16.66%"><img src="assets/4bdbb4f7ed8444bebd4f413c8c4ea744.gif" width="160" alt="Segmented 3D object 9"></td>
    <td width="16.66%"><img src="assets/62bc4504d0494522bd2753cff21df917.gif" width="160" alt="Segmented 3D object 10"></td>
    <td width="16.66%"><img src="assets/7027f197493f4062b1dd87028dd44bfe.gif" width="160" alt="Segmented 3D object 11"></td>
    <td width="16.66%"><img src="assets/71929db371b74eb1a6b962621bffc0fd.gif" width="160" alt="Segmented 3D object 12"></td>
  </tr>
  <tr>
    <td width="16.66%"><img src="assets/731420c049df412f8bc22e10b2da5ff3.gif" width="160" alt="Segmented 3D object 13"></td>
    <td width="16.66%"><img src="assets/7807bf298e4b490abaf4a3375239160d.gif" width="160" alt="Segmented 3D object 14"></td>
    <td width="16.66%"><img src="assets/784f28e275634dd6b46ce6b043a94eed.gif" width="160" alt="Segmented 3D object 15"></td>
    <td width="16.66%"><img src="assets/960801a4bb6d40bfbf276d131a0d6d77.gif" width="160" alt="Segmented 3D object 16"></td>
    <td width="16.66%"><img src="assets/9b8fefcd793c4f2ba100580ba927f915.gif" width="160" alt="Segmented 3D object 17"></td>
    <td width="16.66%"><img src="assets/9e74eb4f1e64474f89a0890bb26ce4bc.gif" width="160" alt="Segmented 3D object 18"></td>
  </tr>
  <tr>
    <td width="16.66%"><img src="assets/a66cec219a8f460abfa15c7b7bf01b2e.gif" width="160" alt="Segmented 3D object 19"></td>
    <td width="16.66%"><img src="assets/a70d14c9d68f4f91b28b84465c109114.gif" width="160" alt="Segmented 3D object 20"></td>
    <td width="16.66%"><img src="assets/cabd39bdbebd44a8ad61fd9c9429bf83.gif" width="160" alt="Segmented 3D object 21"></td>
    <td width="16.66%"><img src="assets/d31af010b39d4e3eb604a63cb5780512.gif" width="160" alt="Segmented 3D object 22"></td>
    <td width="16.66%"><img src="assets/f0479c90b1bc4c66aeca48e6edffd754.gif" width="160" alt="Segmented 3D object 23"></td>
    <td width="16.66%"><img src="assets/f0c5a52135ac4633a1c03644850e76c7.gif" width="160" alt="Segmented 3D object 24"></td>
  </tr>
</table>

## Applications

| | SAGO Web | SAGO Desktop |
|---|---|---|
| Source | [`SAGO_web/`](SAGO_web/) | [`SAGO_desktop/`](SAGO_desktop/) |
| Interface | Browser UI with a local Python/CUDA backend | Native PySide6 / Qt Widgets; no browser or HTTP server |
| Segmentation | SAM2; optional SAM3 / SAM3.1 | SAM2 only |
| Prompts | Foreground/background points and boxes; optional text prompts | Foreground/background points and boxes |
| Platform | Linux NVIDIA/CUDA backend | Linux x86-64, NVIDIA/CUDA |
| Start | `python run.py` inside `SAGO_web/` | `./sago scene.ply` inside `SAGO_desktop/`, or install the Debian package |

Both interfaces default to English and support Chinese. The current Desktop release does not support macOS / Apple Silicon.

## Get the code

```bash
git clone https://github.com/leviome/SAGO.git
cd SAGO
```

Then follow the complete environment instructions for your application:

- **[SAGO Web](SAGO_web/README.md#environment-installation):** install a compatible CUDA-enabled PyTorch environment, build the included rasterizers, obtain the model checkpoint, and launch the local web interface.
- **[SAGO Desktop](SAGO_desktop/README.md):** install the Linux Debian package from [Releases](https://github.com/leviome/SAGO/releases), or prepare a CUDA Python environment and run the native application from source.

An NVIDIA GPU, compatible drivers, and a CUDA/PyTorch runtime are required for both backends. A matching CUDA toolkit and C++ compiler are needed to build the bundled extensions. Installing the Desktop package does not install NVIDIA drivers or PyTorch.

## Repository layout

```text
SAGO/
├── assets/         # Animated object gallery
├── SAGO_web/       # Independent browser application and Python/CUDA backend
├── SAGO_desktop/   # Independent native Linux application and Python/CUDA backend
├── CITATION.cff
├── THIRD_PARTY_NOTICES.md
└── README.md
```

Run application commands from the corresponding application directory. The two backends are intentionally self-contained; neither imports code from its sibling directory.

## Main features

- Automatic 2D mask updates after point or box prompts, followed by six-view 3D refinement.
- Multiple object IDs, editable names, merging, renumbering, foreground coloring, and scene-local persistence.
- Ground calibration from three points and upright bounding boxes.
- Visible-Gaussian correction with preview and confirmation.
- RGB, Splats, Points, and Depth rendering, plus Virtual Drone trajectories and camera frustums.
- Foreground PLY/mask exports and single-orbit GIFs for the current selection or all saved objects.

Desktop additionally provides automatic high-DPI rendering and **C** to cancel the current segmentation round. See each application's README for supported shortcuts and detailed behavior.

## Models and releases

Model checkpoints, scenes, virtual environments, caches, generated outputs (apart from the curated gallery in `assets/`), and Debian packages are **not stored in Git**. Download model weights from the upstream projects as described in each application's model guide:

- [Web model setup](SAGO_web/models/README.md)
- [Desktop SAM2 setup](SAGO_desktop/models/README.md)

Installers are published as GitHub Release assets. Release tags use application prefixes, such as `web-v2.0` and `desktop-v1.1.2`, so Web and Desktop can be versioned independently. The full Desktop installer includes SAM2 weights and Qt; source archives do not.

The former standalone Web repository has been consolidated here. Its commit history is retained in this repository; the previous v2.0 snapshot is available under `legacy-web-v2.0`. Earlier commits remain accessible through the retained history. New development and releases belong to `leviome/SAGO`.

## Citation and licenses

If you use SAGO in your research, please cite [the paper](https://arxiv.org/abs/2607.01628); machine-readable metadata is in [CITATION.cff](CITATION.cff).

License files and third-party notices are preserved in both application directories. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) before redistributing code, model weights, or installers; vendored components retain their own licenses.
