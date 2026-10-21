# Validation

The native application is exercised with the automated tests in `tests/` and the optional real-scene checks in `scripts/`. Run these from `SAGO_desktop/` after setting up the environment described in README.md:

```bash
QT_QPA_PLATFORM=offscreen .venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/gpu-smoke.py path/to/scene.ply
QT_QPA_PLATFORM=offscreen .venv/bin/python scripts/native-gpu-smoke.py path/to/scene.ply
```

The 1.1.2 implementation passed 101 automated checks during development. Checks cover rendering modes, high-DPI sizing, idle stability, spatial prompts, candidate tracking, C-key cancellation, frozen interaction coordinates, multi-object IDs, correction, ground calibration, persistence, and GIF export.

A development-machine Qt/CUDA test opened 1,352,080 Gaussians at an automatic 2048-pixel long edge under 2× DPI. Manual 1920 rendering switched to a 768-pixel navigation preview and returned to full quality. At 1920 resolution, SAM2 generated four candidates and six-view refinement completed.

The Debian installer was checked for SAM2-only model assets and for matching the tested application source. Drivers, CUDA/PyTorch environments and rasterizer builds remain runtime prerequisites. Host installation and portability across arbitrary Linux distributions are not claimed by these checks.

Generated logs, screenshots, model weights and machine-specific validation artifacts are not included in Git. The scripts that load real scenes require local inputs and do not contact SAGO Web.
