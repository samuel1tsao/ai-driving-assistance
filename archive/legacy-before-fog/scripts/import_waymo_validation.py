"""Extract only verified Occ3D-Waymo 0.4 m validation labels from local ZIPs."""
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import tarfile
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def import_labels():
    sources = json.loads((ROOT/'reports/occupancy/waymo_validation_sources.json').read_text())
    destination = ROOT/'data/occupancy/datasets/occ3d_waymo/validation'
    records=[]
    for index, (member, archives) in enumerate(sorted(sources.items())):
        archive=Path(archives[0][0])
        scene=PurePosixPath(member).stem
        with zipfile.ZipFile(archive) as outer:
            with outer.open(member) as stream, tarfile.open(fileobj=stream,mode='r|*') as inner:
                for item in inner:
                    if not item.isfile(): continue
                    parts=PurePosixPath(item.name).parts
                    if len(parts)!=2 or parts[0]!=scene or not re.fullmatch(r'\d+_04\.npz',parts[1]):
                        raise ValueError(f'Unexpected archive entry: {item.name}')
                    raw=inner.extractfile(item).read()
                    with np.load(io.BytesIO(raw),allow_pickle=False) as data:
                        for key in ('voxel_label','final_voxel_state','infov'):
                            if data[key].shape!=(200,200,16): raise ValueError(f'Wrong grid: {member}/{item.name}')
                        if not np.isin(data['voxel_label'],list(range(15))+[15,23]).all():
                            raise ValueError(f'Unknown label ID: {item.name}')
                    path=(destination/scene/parts[1]).resolve()
                    if not path.is_relative_to(destination.resolve()): raise ValueError('Unsafe destination')
                    digest=hashlib.sha256(raw).hexdigest()
                    if path.exists():
                        if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
                            raise ValueError(f'Existing file differs: {path}')
                    else:
                        path.parent.mkdir(parents=True,exist_ok=True)
                        with path.open('xb') as output: output.write(raw)
                    records.append({'scene':scene,'frame':path.stem.removesuffix('_04'),
                                    'path':str(path.relative_to(ROOT)),'sha256':digest,'archive':str(archive),'member':member})
        if index%10==0: print(f'Imported {index+1}/{len(sources)} scenes; {len(records)} frames',flush=True)
    scenes={r['scene'] for r in records}
    report={'resolution_m':.4,'scenes':len(scenes),'frames':len(records),'expected_scenes':202,
            'missing_scenes':sorted(set(f'{i:03}' for i in range(202))-scenes),'records':records}
    local=list(destination.glob('*/*_04.npz'))
    local_scenes={p.parent.name for p in local}
    report.update(local_frames=len(local),local_scenes=len(local_scenes),
                  local_missing_scenes=sorted(set(f'{i:03}' for i in range(202))-local_scenes))
    (ROOT/'reports/occupancy/waymo_import.json').write_text(json.dumps(report,indent=2))
    print({k:v for k,v in report.items() if k!='records'},flush=True)


if __name__=='__main__': import_labels()
