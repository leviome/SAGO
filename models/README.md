# Model weights

Checkpoints are external data and are not included in the GitHub source release.

- **SAM2:** place the original `sam2_hiera_large.pt` in this directory, or pass
  `python run.py --checkpoint /path/to/sam2_hiera_large.pt`.
  Obtain the checkpoint from the official [SAM2 repository](https://github.com/facebookresearch/sam2).
- **SAM3 / SAM3.1 (optional):** place `sam3.pt` and/or `sam3.1_multiplex.pt` here,
  or use `--sam3-checkpoint` / `--sam31-checkpoint`. Follow the official
  [SAM3](https://github.com/facebookresearch/sam3) checkpoint access instructions.
  See the main README for the separate inference environment.
- **SAM2 text prompts (optional):** download
  [`groundingdino_swint_ogc.pth`](https://github.com/IDEA-Research/GroundingDINO/releases/download/weights/groundingdino_swint_ogc.pth)
  into this directory. Download the model and tokenizer files from
  [`google-bert/bert-base-uncased`](https://huggingface.co/google-bert/bert-base-uncased)
  into `bert-base-uncased/`, including `model.safetensors`, `config.json`,
  `tokenizer.json`, `tokenizer_config.json`, and `vocab.txt`.
  Override the paths with `SAGO_GROUNDING_CHECKPOINT` and `SAGO_GROUNDING_BERT`.

Checkpoint licenses and access conditions remain those of their respective owners.
