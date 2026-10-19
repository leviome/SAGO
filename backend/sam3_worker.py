"""Private JSON-line worker for official SAM3 / SAM3.1 (separate PyTorch runtime)."""
import contextlib
import copy
import uuid
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT/'vendor/sam3'))

import numpy as np
from PIL import Image
import torch


def arrays(output):
    def cpu(value):
        return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
    ids = cpu(output['out_obj_ids'])
    masks = cpu(output['out_binary_masks']).astype(bool)
    scores = cpu(output.get('out_probs', np.ones(len(ids))))
    return ids, masks, scores


def seed(predictor, session, prompts):
    """Text/box detects instances; point-only prompts use the instance tracker."""
    text, box = prompts['text'], prompts['box']
    fg, bg = prompts['foreground'], prompts['background']
    request = dict(type='add_prompt', session_id=session, frame_index=0)
    if text or box:
        if text:
            request['text'] = text
        if box:
            (x0, y0), (x1, y1) = box
            request.update(bounding_boxes=[[x0, y0, x1-x0, y1-y0]], bounding_box_labels=[1])
    else:
        request.update(points=fg+bg, point_labels=[1]*len(fg)+[0]*len(bg), obj_id=1)
    output = predictor.handle_request(request)['outputs']
    ids, masks, scores = arrays(output)
    all_ids = ids.copy()
    # The concept detector and instance point decoder are separate official APIs.
    # Spatial points disambiguate text/box detections without changing identity.
    if text or box:
        h, w = masks.shape[-2:]
        keep = np.ones(len(ids), dtype=bool)
        for points, wanted in ((fg, True), (bg, False)):
            for x, y in points:
                keep &= masks[:, min(h-1, int(y*h)), min(w-1, int(x*w))] == wanted
        ids, masks, scores = ids[keep], masks[keep], scores[keep]
    order = np.argsort(-scores, kind='stable')
    return ids[order], masks[order], scores[order], all_ids


def infer(predictor, request):
    frames = [Image.open(path).convert('RGB') for path in request['frames']]
    session = predictor.handle_request(dict(type='start_session', resource_path=frames,
                                            offload_video_to_cpu=True))['session_id']
    try:
        ids, masks, scores, all_ids = seed(predictor, session, request['prompts'])
        np.savez_compressed(request['output'], masks=masks, scores=scores)
        return dict(count=len(masks))
    finally:
        predictor.handle_request(dict(type='close_session', session_id=session))
        for image in frames:
            image.close()


def prepare_frame(state, image, model):
    # Match the official loader, but prepare only the newly rendered frame.
    from sam3.model.io_utils import load_resource_as_video_frames
    tensor, _, _ = load_resource_as_video_frames([image], state['image_size'],
        offload_video_to_cpu=True, img_mean=model.image_mean, img_std=model.image_std)
    return tensor[0]


def advance_frame(model, state, index):
    # The interactive wrappers normally treat repeated propagation as a cache
    # fetch. Invoke their native video loop for precisely one new frame instead.
    from sam3.model.sam3_video_inference import Sam3VideoInference
    from sam3.model.sam3_multiplex_tracking import Sam3MultiplexTracking
    base = Sam3MultiplexTracking if isinstance(model, Sam3MultiplexTracking) else Sam3VideoInference
    output = None
    for frame_index, masks in base.propagate_in_video(model, state,
            start_frame_idx=index, max_frame_num_to_track=0, reverse=False):
        if frame_index != index:
            raise RuntimeError('增量跟踪处理了非当前视角')
        output = masks
    if output is None:
        raise RuntimeError('增量跟踪没有返回当前视角')
    return output


class DroneSessions:
    """Two persistent states, with speculative steps committed after 3D filtering.

    Cloning state preserves tensor aliases within a branch. Only the proposed
    state is mutated; rejection and errors leave the committed memory untouched.
    No past image is encoded or propagated again.
    """
    def __init__(self, predictor, prepare=prepare_frame, advance=advance_frame):
        self.predictor = predictor
        self.prepare = prepare
        self.advance = advance
        self.routes = {}
        self.image_encodes = 0
        if hasattr(predictor.model, 'detector'):
            def count_images(module, args, output):
                batch = args[0].tensors if hasattr(args[0], 'tensors') else args[0]
                self.image_encodes += int(batch.shape[0])
            predictor.model.detector.backbone.vision_backbone.trunk.register_forward_hook(count_images)

    def start(self, image_path, prompts, target_path):
        initial_encodes = self.image_encodes
        with Image.open(image_path) as image_file:
            image = image_file.convert('RGB')
        # Reserve s0 + three future slots per direction. Placeholders are only
        # storage: each is overwritten before its single-frame inference.
        session = self.predictor.handle_request(dict(type='start_session',
            resource_path=[image]*4, offload_video_to_cpu=True))['session_id']
        try:
            ids, masks, _, _ = seed(self.predictor, session, prompts)
            target = np.load(target_path, allow_pickle=False).astype(bool)
            if not len(ids):
                raise ValueError('无法重建首视角目标，请重新预览分割')
            overlap = (masks & target).sum(axis=(1, 2)) / np.maximum((masks | target).sum(axis=(1, 2)), 1)
            chosen = int(np.argmax(overlap))
            if overlap[chosen] < .9:
                raise ValueError('首视角目标与预览不一致，请重新预览分割')
            state = self.predictor._get_session(session)['state']
            # Finalize the seed's video memory once before branching, matching
            # the first-frame propagation performed by the previous video path.
            self.advance(self.predictor.model, state, 0)
            route = uuid.uuid4().hex
            self.routes[route] = dict(states=[copy.deepcopy(state), copy.deepcopy(state)],
                pending={}, next_index=[1, 1], obj_id=int(ids[chosen]),
                size=image.size, seed_runs=1, new_frames=0, initial_encodes=initial_encodes,
                accepted=[0, 0], rejected=[0, 0])
            return dict(route=route, **self.stats(route))
        finally:
            self.predictor.handle_request(dict(type='close_session', session_id=session))
            image.close()

    def stats(self, route):
        r = self.routes[route]
        return dict(seed_runs=r['seed_runs'], new_frames=r['new_frames'],
                    image_encodes=self.image_encodes-r['initial_encodes'],
                    accepted=list(r['accepted']), rejected=list(r['rejected']))

    def step(self, route, side, image_path, output_path):
        r = self.routes[route]
        if side not in (0, 1) or side in r['pending']:
            raise ValueError('跟踪方向无效或上一步尚未提交')
        index = r['next_index'][side]
        if index > 3:
            raise ValueError('该方向已完成三个新视角')
        proposed = copy.deepcopy(r['states'][side])
        with Image.open(image_path) as image_file:
            image = image_file.convert('RGB')
        try:
            if image.size != r['size']:
                raise ValueError('跟踪视角尺寸与首视角不一致')
            frame = self.prepare(proposed, image, self.predictor.model)
            batch = proposed['input_batch'].img_batch
            batch = batch.tensors if hasattr(batch, 'tensors') else batch
            batch[index].copy_(frame.to(device=batch.device, dtype=batch.dtype))
            output = self.advance(self.predictor.model, proposed, index)
            ids, masks, _ = arrays(output)
            result = np.zeros((1, image.height, image.width), dtype=bool)
            found = np.flatnonzero(ids == r['obj_id'])
            if len(found):
                result[0] = masks[found[0]]
            np.savez_compressed(output_path, masks=result, scores=np.ones(1))
            r['pending'][side] = proposed
            r['new_frames'] += 1
            return dict(frame_index=index, **self.stats(route))
        finally:
            image.close()

    def finish(self, route, side, accepted):
        r = self.routes[route]
        if side not in r['pending']:
            raise ValueError('没有待提交的跟踪步骤')
        proposed = r['pending'].pop(side)
        if accepted:
            r['states'][side] = proposed
            r['next_index'][side] += 1
            r['accepted'][side] += 1
        else:
            r['rejected'][side] += 1
        return self.stats(route)

    def close(self, route):
        r = self.routes.pop(route, None)
        if r is not None:
            r.clear()
        return dict(closed=True)


def main():
    predictor = None
    sessions = None
    for line in sys.stdin:
        try:
            request = json.loads(line)
            with contextlib.redirect_stdout(sys.stderr), torch.inference_mode():
                if request['action'] == 'load':
                    started = time.perf_counter()
                    if request['version'] == 'sam3':
                        from sam3.model.sam3_video_predictor import Sam3VideoPredictor
                        predictor = Sam3VideoPredictor(checkpoint_path=request['checkpoint'],
                                                       async_loading_frames=False)
                    else:
                        from sam3.model_builder import build_sam3_predictor
                        predictor = build_sam3_predictor(checkpoint_path=request['checkpoint'],
                            version='sam3.1', use_fa3=False, use_rope_real=False,
                            compile=False, async_loading_frames=False)
                        # Drone routes are very short and arrive one view at a time.
                        # Use the official per-frame grounding path, avoiding video
                        # batch padding/cached-feature slices on 2-4 frame sessions.
                        predictor.model.use_batched_grounding = False
                    # Future camera images do not exist yet. Never encode their
                    # reserved slots or retain stale prefetched features.
                    predictor.model.detector.prefetch_frames = False
                    sessions = DroneSessions(predictor)
                    torch.cuda.synchronize()
                    result = dict(model_load_seconds=time.perf_counter()-started)
                else:
                    started = time.perf_counter()
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        action = request['action']
                        if action == 'start_route':
                            result = sessions.start(request['image'], request['prompts'], request['target'])
                        elif action == 'step_route':
                            result = sessions.step(request['route'], request['side'], request['image'], request['output'])
                        elif action == 'finish_step':
                            result = sessions.finish(request['route'], request['side'], request['accepted'])
                        elif action == 'close_route':
                            result = sessions.close(request['route'])
                        elif action == 'preview':
                            result = infer(predictor, request)
                        else:
                            raise ValueError('无效 SAM3 推理操作')
                    torch.cuda.synchronize()
                    result['sam_seconds'] = time.perf_counter()-started
            print(json.dumps(dict(result=result)), flush=True)
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            print(json.dumps(dict(error=str(exc))), flush=True)


if __name__ == '__main__':
    main()
