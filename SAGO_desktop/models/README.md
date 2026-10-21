# SAM2 model weights

SAGO Desktop supports SAM2 point and box prompts only. Version 1.1.7 and later installers contain no model weights.

On first entering segmentation prompts, the original `sam2_hiera_large.pt` downloads from the official [SAM2 project](https://github.com/facebookresearch/sam2) to `$XDG_CACHE_HOME/sago-desktop/models/` (default `~/.cache/sago-desktop/models/`). The download is approximately 857 MiB. Downloads support progress, cancellation, resume and retry; expected size and SHA-256 are verified before the model is used. The cache is reused on later launches.

For offline use, select an existing checkpoint in Preferences or use `sago --checkpoint path/to/sam2_hiera_large.pt`. Clear the checkpoint field to restore automatic download. Custom checkpoint files are not overwritten. A missing legacy installation-relative checkpoint falls back to the cache automatically.

Model weights are excluded from Git and Debian packages; original upstream license terms apply.
