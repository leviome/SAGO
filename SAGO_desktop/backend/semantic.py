"""Four-view CLIP retrieval. Scene-local features, external immutable model cache."""
import hashlib
import math
import time

import numpy as np
from PIL import Image

from backend.gif_export import orbit_geometry

MODEL = 'openai/clip-vit-base-patch32'
REVISION = '3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268'
# Official Hub safetensors conversion, pinned independently of the model metadata.
WEIGHTS_REVISION = 'c237dc49a33fc61debc9276459120b7eac67e7ef'
MODEL_INFO = dict(model=MODEL, revision=REVISION, weights_revision=WEIGHTS_REVISION, dimension=512)


def check_cancel(cancel):
    if cancel.is_set():
        from backend.engine import Cancelled
        raise Cancelled('Semantic indexing cancelled')


def normalized(values, rows=None):
    values = np.asarray(values, dtype=np.float32)
    expected = (512,) if rows is None else (rows, 512)
    if values.shape != expected or not np.isfinite(values).all():
        raise ValueError('Invalid CLIP embeddings; rebuild semantic knowledge')
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms < 1e-8):
        raise ValueError('Empty CLIP embeddings; rebuild semantic knowledge')
    return values / norms


def mask_fingerprint(engine, obj):
    mask = engine.labels() == obj['number']
    digest = hashlib.sha256(str(engine.gaussian_fingerprint).encode())
    digest.update(len(mask).to_bytes(8, 'little'))
    digest.update(np.packbits(mask).tobytes())
    return digest.hexdigest()


class ClipEncoder:
    """Load safe weights only, on first use; reuse the same image/text model."""
    def __init__(self, progress, cancel):
        import torch
        from huggingface_hub import hf_hub_download
        from transformers import CLIPConfig, CLIPModel, CLIPProcessor
        from tqdm.auto import tqdm

        class DownloadProgress(tqdm):
            def update(self, n=1):
                check_cancel(cancel)
                result = super().update(n)
                if self.total:
                    progress(0, f'Downloading CLIP: {self.n / self.total * 100:.0f}%')
                return result

        def download(name, revision):
            check_cancel(cancel)
            try:
                return hf_hub_download(MODEL, name, revision=revision, local_files_only=True)
            except FileNotFoundError:
                progress(0, 'Downloading CLIP model (first use)…')
                return hf_hub_download(MODEL, name, revision=revision, tqdm_class=DownloadProgress)

        progress(0, 'Loading CLIP image and text encoders…')
        from pathlib import Path
        config_path = Path(download('config.json', REVISION)).parent
        for name in ('preprocessor_config.json', 'tokenizer_config.json', 'special_tokens_map.json', 'vocab.json', 'merges.txt'):
            download(name, REVISION)
        weights_path = Path(download('model.safetensors', WEIGHTS_REVISION)).parent
        check_cancel(cancel)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.processor = CLIPProcessor.from_pretrained(config_path, local_files_only=True, use_fast=False)
        self.model = CLIPModel.from_pretrained(weights_path,
            config=CLIPConfig.from_pretrained(config_path, local_files_only=True),
            local_files_only=True, use_safetensors=True).eval().to(self.device)
        self.torch = torch
        check_cancel(cancel)

    def images(self, images):
        inputs = self.processor(images=images, return_tensors='pt').to(self.device)
        with self.torch.inference_mode():
            return normalized(self.model.get_image_features(**inputs).float().cpu().numpy(), 4)

    def text(self, text):
        inputs = self.processor(text=[text], return_tensors='pt', padding=True)
        if inputs['input_ids'].shape[-1] > self.model.config.text_config.max_position_embeddings:
            raise ValueError('Query is too long for CLIP (maximum 77 tokens)')
        with self.torch.inference_mode():
            return normalized(self.model.get_text_features(**inputs.to(self.device)).float().cpu().numpy()[0])


def encoder(engine, progress, cancel):
    if getattr(engine, '_clip_encoder', None) is None:
        try:
            instance = ClipEncoder(progress, cancel)
        except ImportError as error:
            raise ValueError('CLIP dependencies are missing; install this project requirements.txt in the active Python runtime') from error
        engine._clip_encoder = instance
    check_cancel(cancel)
    return engine._clip_encoder


def settings(engine, data):
    elevation = data.get('elevation_degrees', 60)
    size = data.get('size', 512)
    if type(elevation) not in (int, float) or not math.isfinite(elevation) or not -89 <= elevation <= 89:
        raise ValueError('Orbit elevation must be between -89 and 89 degrees')
    if type(size) is not int or not 128 <= size <= 768:
        raise ValueError('Render resolution must be between 128 and 768')
    up = np.asarray(engine.view['up'], dtype=float)
    if up.shape != (3,) or not np.isfinite(up).all() or np.linalg.norm(up) < 1e-8:
        raise ValueError('Invalid up vector')
    return dict(size=size, elevation_degrees=elevation, up=(up / np.linalg.norm(up)).tolist(),
        fov=engine.view['fov'], yaw=engine.view['yaw'], background='white', views=4, orbit_fit_version=1)


def render_views(engine, obj, options, cancel):
    images = []
    try:
        with engine.torch.no_grad():
            target = engine.utils.masking_for_fg(engine.model, engine.object_mask(obj['number']))
            center, up, distance = orbit_geometry(target.get_xyz.detach().cpu().numpy(),
                target.get_scaling.detach().cpu().numpy(), options['up'], options['fov'])
            center = engine.torch.tensor(center, dtype=engine.torch.float32)
            up = engine.torch.tensor(up, dtype=engine.torch.float32)
            for i in range(4):
                check_cancel(cancel)
                cam = engine.utils.sphere2params(distance, math.radians(options['yaw']) + i * math.pi/2,
                    math.radians(options['elevation_degrees']), center, up, math.radians(options['fov']),
                    width=options['size'], height=options['size'])
                cam.camera_center = cam.world_view_transform.inverse()[3, :3]
                rendered = engine.render_simple(cam, target, bg='white')
                images.append(Image.fromarray(engine.rgb(rendered['render'])))
        return images
    except BaseException:
        for image in images:
            image.close()
        raise


def load_index(engine):
    scene = engine.knowledge.read().get('scenes', {}).get(engine.knowledge.key, {})
    index = scene.get('semantic_index', {}) if isinstance(scene, dict) else {}
    if not isinstance(index, dict) or index.get('encoder') != MODEL_INFO or index.get('schema_version') != 1 or not isinstance(index.get('render'), dict):
        return {}
    return index


def valid_record(engine, obj, record):
    return (isinstance(record, dict) and record.get('mask_sha256') == mask_fingerprint(engine, obj)
        and record.get('gaussian_count') == int(np.count_nonzero(engine.labels() == obj['number'])))


def semantic_action(engine, data, progress, cancel):
    engine.require_scene()
    if engine.knowledge is None:
        raise ValueError('Load a scene before using semantic knowledge')
    if engine.capture is not None or engine.ground_capture is not None or engine.correction_pending is not None:
        raise ValueError('Exit prompts, calibration or correction before using semantic knowledge')
    action = data.get('operation')
    if action == 'clear':
        engine.semantic_highlight = engine._semantic_box = None
        return dict(cleared=True)
    if action not in ('build', 'search'):
        raise ValueError('Unknown semantic operation')
    objects = sorted(engine.objects, key=lambda obj: obj['number'])
    if not objects:
        raise ValueError('Add object IDs before building semantic knowledge')
    index = load_index(engine)
    old_records = index.get('objects', {})
    if not isinstance(old_records, dict):
        old_records = {}
    started = time.perf_counter()
    check_cancel(cancel)
    if action == 'build':
        options = settings(engine, data)
        records, rendered_count = {}, 0
        clip = None
        for i, obj in enumerate(objects):
            check_cancel(cancel)
            record = old_records.get(obj['id'])
            reusable = valid_record(engine, obj, record) and index.get('render') == options
            if reusable:
                try:
                    normalized(record['embeddings'], 4)
                except (KeyError, TypeError, ValueError):
                    reusable = False
            if reusable:
                records[obj['id']] = record
            else:
                if clip is None:
                    clip = encoder(engine, progress, cancel)
                progress(round(i/len(objects)*100), f'Encoding object ID {obj["number"]}: four views')
                images = render_views(engine, obj, options, cancel)
                try:
                    features = normalized(clip.images(images), 4)
                finally:
                    for image in images:
                        image.close()
                records[obj['id']] = dict(mask_sha256=mask_fingerprint(engine, obj),
                    gaussian_count=int(np.count_nonzero(engine.labels() == obj['number'])), embeddings=features.tolist())
                rendered_count += 1
            records[obj['id']] = dict(records[obj['id']], number=obj['number'], name=obj['name'])
            progress(round((i+1)/len(objects)*100), f'Indexed objects: {i+1}/{len(objects)}')
        # Commit IDs and features together so freshly added objects survive scene reopen.
        labels = engine.labels()
        segmentation = dict(gaussian_count=len(labels), gaussian_order_sha256=engine.gaussian_fingerprint,
            order='original_ply_vertex_order', gaussian_ids=labels.tolist(), objects=[dict(o) for o in engine.objects],
            next_id=engine.next_object_id)
        engine.knowledge.validate_segmentation(segmentation, len(labels), engine.gaussian_fingerprint)
        check_cancel(cancel)
        engine.knowledge.save_record(segmentation=segmentation, semantic_index=dict(schema_version=1,
            encoder=MODEL_INFO, render=options, objects=records))
        engine.ids_dirty = False
        return dict(indexed=len(records), updated=rendered_count, saved_path=str(engine.knowledge.path),
            seconds=time.perf_counter()-started)
    text = data.get('text')
    if not isinstance(text, str) or not text.strip() or len(text) > 512:
        raise ValueError('Enter a text query (1–512 characters)')
    if not index:
        raise ValueError('Build semantic knowledge first')
    candidates, stale = [], []
    current_up = settings(engine, {})['up']
    stored_up = np.asarray(index.get('render', {}).get('up', []), dtype=float)
    up_matches = stored_up.shape == (3,) and np.isfinite(stored_up).all() and np.allclose(stored_up, current_up)
    for obj in objects:
        record = old_records.get(obj['id'])
        try:
            if not up_matches or not valid_record(engine, obj, record):
                raise ValueError()
            features = normalized(record['embeddings'], 4)
            candidates.append((obj, features))
        except (KeyError, TypeError, ValueError):
            stale.append(obj['number'])
    if not candidates:
        raise ValueError('Semantic knowledge is outdated; rebuild it before searching')
    clip = encoder(engine, progress, cancel)
    progress(50, 'Encoding text and ranking object IDs…')
    query = normalized(clip.text(text.strip()))
    check_cancel(cancel)
    results = [dict(id=obj['id'], number=obj['number'], name=obj['name'],
        score=float(np.max(features @ query))) for obj, features in candidates]
    results.sort(key=lambda item: (-item['score'], item['number']))
    engine.semantic_highlight = results[0]['id']
    engine._semantic_box = None
    progress(100, 'Semantic search complete')
    return dict(results=results, stale_ids=stale, query=text.strip(), score_type='max_view_cosine',
        seconds=time.perf_counter()-started)
