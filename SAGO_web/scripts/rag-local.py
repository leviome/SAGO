#!/usr/bin/env python3
"""Export real saved scene geometry for a human/Codex LLM; never calls an API.

Original PLY and knowledge files are read only. No renderer, SAM or CLIP loads.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))

from backend.gaussian_rag import SYSTEM, facts, geometry_document, relation
from backend.knowledge import SceneKnowledge


def export_scene(scene,output,question):
    import numpy as np
    from plyfile import PlyData
    import torch

    scene=Path(scene).expanduser().resolve()
    if scene.is_file():
        knowledge=SceneKnowledge(scene)
    else:
        # Select only an existing knowledge record, never guess an iteration.
        knowledge_file=scene/'SAGO_knowledge.json'
        records=json.loads(knowledge_file.read_text(encoding='utf-8')).get('scenes',{})
        valid=[scene/key for key,record in records.items() if isinstance(record,dict) and record.get('segmentation') and (scene/key).is_file()]
        if len(valid)!=1:
            raise ValueError('Scene folder must contain exactly one saved segmented PLY; pass a PLY path to select explicitly')
        scene=valid[0]
        knowledge=SceneKnowledge(scene)
    record=knowledge.read().get('scenes',{}).get(knowledge.key,{})
    vertices=PlyData.read(str(scene),mmap='r')['vertex'].data
    xyz=np.column_stack([vertices[key] for key in ('x','y','z')]).astype('<f4')
    if not len(xyz) or not np.isfinite(xyz).all():raise ValueError('Invalid Gaussian coordinates')
    fingerprint=hashlib.sha256(xyz.tobytes()).hexdigest()
    saved=knowledge.load_segmentation(len(xyz),fingerprint)
    if saved is None:raise ValueError('Save object IDs before exporting local RAG data')
    labels,objects,next_id=saved
    if not objects:raise ValueError('No positive object IDs')
    plane=knowledge.load_ground()
    up=plane['normal'] if plane else record.get('up_vector',[0,-1,0])
    engine=SimpleNamespace(model=SimpleNamespace(get_xyz=torch.from_numpy(xyz)),objects=objects,
        gaussian_fingerprint=fingerprint,knowledge=knowledge,view=dict(up=up),ground_plane=plane,
        labels=lambda:labels,object_mask=lambda number:torch.from_numpy(labels==number))
    document=geometry_document(engine,threading.Event())
    # Reuse previously generated descriptions only for the same Gaussian membership.
    old=record.get('gaussian_rag',{}).get('objects',[])
    by_id={obj['id']:obj for obj in old}
    for obj in document['objects']:
        previous=by_id.get(obj['id'],{})
        if previous.get('mask_sha256')==obj['mask_sha256'] and previous.get('semantic',{}).get('source')=='api_vision':
            obj['semantic']=previous['semantic']
        neighbors=[relation(obj,other) for other in document['objects'] if obj is not other]
        obj['nearest_relations']=sorted(neighbors,key=lambda row:row['centroid_distance'])[:6]
    output=Path(output).expanduser().resolve();output.mkdir(parents=True,exist_ok=True)
    path=output/'spatial.json';path.write_text(json.dumps(document,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    packet=dict(schema_version=1,mode='codex_manual_no_api',question=question,instructions=SYSTEM,
        spatial_document=str(path),source_scene=str(scene),source_knowledge=str(knowledge.path),
        scene_signature=document['signature'],object_catalog=[dict(number=o['number'],name=o['name']) for o in document['objects']],
        note='Read spatial_document and query computed facts with rag-local.py facts. This export does not invoke or impersonate a running LLM.')
    (output/'request.json').write_text(json.dumps(packet,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (output/'README.md').write_text('# GaussianRAG — Codex validation without API\n\n'+
        'The PLY and original knowledge file were read only. `spatial.json` contains measured scene facts; '+
        '`request.json` is the question and grounding instructions for Codex. No API, model authentication, '+
        'SAM, CLIP or renderer was used. Object labels are user names unless a verified saved description was available.\n\n'+
        'Ask Codex to read `request.json`, inspect `spatial.json`, and use the local facts command for spatial evidence. '+
        'Answers appear in the Codex conversation; this does not automatically connect the SAGO question box to Codex.\n\n'+
        'Distances are scene units, not guaranteed metres. Box relationships do not establish physical support or contact.\n',encoding='utf-8')
    return packet


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    export=commands.add_parser('export');export.add_argument('--scene',required=True);export.add_argument('--output',required=True)
    export.add_argument('--question',default='Summarize the saved objects and give grounded spatial relationships with object ID citations.')
    query=commands.add_parser('facts');query.add_argument('--document',required=True);query.add_argument('--ids',type=int,nargs='+',required=True)
    args=parser.parse_args()
    try:
        if args.command=='export':result=export_scene(args.scene,args.output,args.question)
        else:
            document=json.loads(Path(args.document).read_text(encoding='utf-8'))
            result=facts(document,args.ids)
        print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
    except (OSError,ValueError,KeyError) as error:
        parser.exit(1,str(error)+'\n')


if __name__=='__main__':main()
