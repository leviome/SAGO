"""Local GroundingDINO boxes for SAM2; RGB input, normalized xyxy output."""
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def candidate_boxes(boxes, scores, foreground=(), region=(), threshold=.35, limit=8):
    """Convert cxcywh, reject invalid boxes, constrain hints, then suppress duplicates."""
    result = []
    boxes, scores = np.asarray(boxes), np.asarray(scores)
    for index in np.argsort(-scores):
        cx, cy, width, height = boxes[index]
        xyxy = np.clip([cx-width/2, cy-height/2, cx+width/2, cy+height/2], 0., 1.)
        x1, y1, x2, y2 = xyxy
        if not np.isfinite(xyxy).all() or x2 <= x1 or y2 <= y1 or not np.isfinite(scores[index]) or scores[index] < threshold:
            continue
        if region:
            # Explicit region restricts detection by its center, while SAM uses its intersection.
            (rx1, ry1), (rx2, ry2) = region
            if not rx1 <= cx <= rx2 or not ry1 <= cy <= ry2:
                continue
            x1, y1, x2, y2 = max(x1, rx1), max(y1, ry1), min(x2, rx2), min(y2, ry2)
            if x2 <= x1 or y2 <= y1:
                continue
        if any(not (x1 <= x <= x2 and y1 <= y <= y2) for x, y in foreground):
            continue
        area = (x2-x1)*(y2-y1)
        duplicate = False
        for previous in result:
            (a, b), (c, d) = previous['box']
            intersection = max(0., min(x2, c)-max(x1, a))*max(0., min(y2, d)-max(y1, b))
            if intersection/(area+(c-a)*(d-b)-intersection) > .5:
                duplicate = True
                break
        if not duplicate:
            result.append(dict(box=[[float(x1), float(y1)], [float(x2), float(y2)]], score=float(scores[index])))
        if len(result) >= limit:
            break
    return result


class GroundingDetector:
    def __init__(self):
        self.model = None

    def close(self):
        self.model = None

    def load(self):
        if self.model is not None:
            return 0.
        started = time.perf_counter()
        sys.path.insert(0, str(ROOT/'vendor/GroundingDINO'))
        import torch
        from groundingdino.models import build_model
        from groundingdino.util.slconfig import SLConfig
        from groundingdino.util.misc import clean_state_dict
        checkpoint = Path(os.environ.get('SAGO_GROUNDING_CHECKPOINT', str(ROOT/'models/groundingdino_swint_ogc.pth'))).expanduser()
        bert = Path(os.environ.get('SAGO_GROUNDING_BERT', str(ROOT/'models/bert-base-uncased'))).expanduser()
        if not checkpoint.is_file() or not (bert/'config.json').is_file():
            raise ValueError('缺少 GroundingDINO 或 BERT 本地权重，请按 README 配置 SAM2 文本提示')
        config = SLConfig.fromfile(str(ROOT/'vendor/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py'))
        config.device = 'cuda'
        config.text_encoder_type = str(bert.resolve())
        # The local checkpoint includes BERT weights; all assets are local.
        model = build_model(config)
        state = clean_state_dict(torch.load(checkpoint, map_location='cpu')['model'])
        incompatible = model.load_state_dict(state, strict=False)
        # Training-only denoising embedding and legacy BERT position buffer are unused.
        unexpected = set(incompatible.unexpected_keys)-{'label_enc.weight', 'bert.embeddings.position_ids'}
        if incompatible.missing_keys or unexpected:
            raise RuntimeError(f'GroundingDINO 权重不匹配：{incompatible}')
        self.model = model.eval().to('cuda')
        torch.cuda.synchronize()
        return time.perf_counter()-started

    def detect(self, rgb, text, foreground=(), background=(), region=()):
        import torch
        from PIL import Image
        import groundingdino.datasets.transforms as transforms
        transform = transforms.Compose([transforms.RandomResize([800], max_size=1333), transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
        image, _ = transform(Image.fromarray(rgb), None)
        caption = text.lower().strip()
        if not caption.endswith('.'):
            caption += '.'
        with torch.inference_mode():
            output = self.model(image[None].to('cuda'), captions=[caption])
        scores = output['pred_logits'][0].sigmoid().max(dim=-1).values.cpu().numpy()
        boxes = output['pred_boxes'][0].cpu().numpy()
        # Background points refine masks in SAM2; a background point inside a box
        # must not discard the box, which may contain holes and other objects.
        return candidate_boxes(boxes, scores, foreground, region)
