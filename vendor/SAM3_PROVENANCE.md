SAM3 / SAM3.1 source: https://github.com/facebookresearch/sam3
Pinned revision: 7f473cfd983d5eb43040ff608f66bff0d7bdcb53
Upstream license: sam3/LICENSE; checkpoints retain their own license.

Local inference compatibility changes:
- sam3/model/sam3_base_predictor.py: filter shared session kwargs against each model's init_state signature (multiplex does not accept offload_state_to_cpu).
- sam3/model/sam3_multiplex_base.py: cast boolean sort keys to int32 for PyTorch 2.5 CUDA compatibility, preserving the positive-first ordering.
- sam3/model_builder.py: initialize the bare multiplex tracker without the merged checkpoint, then load the full merged checkpoint once, strictly, into the final model. The combined predictor builds its tracker with `with_backbone=False` on CPU: detector features replace that backbone, so avoid constructing and transferring a visual backbone only to delete it. Standalone tracker construction retains `with_backbone=True` by default; the fully assembled model still moves to CUDA after strict checkpoint loading.

SAGO uses the single-GPU SAM3 predictor (no distributed socket) and the official SAM3.1 multiplex builder. FlashAttention 3 and compilation are disabled; complex RoPE is used with the existing checkpoints. SAM3.1 uses per-frame grounding (`use_batched_grounding=False`) for the short, incrementally generated drone sessions; this avoids empty padded feature slices from video batching. No weight files are modified or converted.

Incremental drone tracking additions:
- sam3/model/sam3_image.py and sam3/model/sam3_multiplex_detector.py: optional `prefetch_frames=False` disables future-frame image inference. The adapter reserves future input slots and overwrites each immediately before its own inference; those placeholders must never enter the feature cache.
- sam3/model/sam3_multiplex_detector.py: use inclusive propagation-window bounds, matching the video loop's `max_frame_num_to_track=0` convention for one current frame.
- backend/sam3_worker.py retains independent committed states for both directions and computes proposed steps on deep copies, preserving aliases within each state. It invokes the native one-frame video loop, avoiding the interactive wrapper's cache-fetch behavior for repeated propagation calls. s0 video memory is finalized once before branching.
