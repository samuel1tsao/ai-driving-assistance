"""Metric depth comparison with bounded memory and explicit relative-depth treatment."""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from common import select_samples, latency_summary

MODELS = [
    {'name':'Depth Anything V2 Small','repo':'depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf','mode':'metric',
     'description':'Small outdoor metric model; predicts camera-axis depth in meters.'},
    {'name':'Depth Anything V2 Base','repo':'depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf','mode':'metric',
     'description':'Larger outdoor metric model; compare accuracy against latency.'},
    {'name':'MiDaS DPT-Hybrid','repo':'Intel/dpt-hybrid-midas','mode':'relative',
     'description':'Relative inverse-depth model; oracle-aligned diagnostic, not deployable metric distance.'},
]


def load_samples(manifest, limit=100, seed=42):
    manifest = Path(manifest)
    if not manifest.is_file():
        print('Depth labels are not prepared. See README: RGB + calibrated LiDAR depth in meters are required.')
        return []
    table = pd.read_csv(manifest)
    required = {'image','depth','split','depth_scale'}
    if not required.issubset(table): raise ValueError(f'Required manifest columns: {sorted(required)}')
    table = table[table.split == 'val']
    if table.image.duplicated().any(): raise ValueError('Duplicate image rows would bias the evaluation.')
    samples = []
    for row in table.itertuples():
        samples.append({'id':str(row.image),'image':manifest.parent/row.image,
                        'depth':manifest.parent/row.depth,'scale':float(row.depth_scale)})
    selected = select_samples(samples,limit,seed)
    for sample in selected:
        if not sample['image'].is_file() or not sample['depth'].is_file():
            raise FileNotFoundError(f'Missing paired data for {sample["id"]}')
        if not np.isfinite(sample['scale']) or sample['scale']<=0: raise ValueError('depth_scale must be positive.')
    return selected


def read_target(sample):
    path = sample['depth']
    target = np.load(path,allow_pickle=False) if path.suffix=='.npy' else np.asarray(Image.open(path))
    target = target.astype(float)/sample['scale']
    if target.ndim != 2: raise ValueError('Depth target must be a 2D map.')
    return target


def metrics(prediction, target, mode='metric', maximum=80):
    if prediction.shape != target.shape: raise ValueError('Depth prediction and label dimensions differ.')
    valid = np.isfinite(target) & (target>0) & (target<=maximum)
    if valid.sum()<10: raise ValueError('Fewer than 10 valid labeled pixels.')
    if not np.isfinite(prediction[valid]).all(): raise ValueError('Non-finite predictions at labeled pixels.')
    if mode=='relative':
        # MiDaS predicts inverse depth. Fit inverse ground truth, then invert.
        design = np.stack([prediction[valid],np.ones(valid.sum())],axis=1)
        scale,shift = np.linalg.lstsq(design,1/target[valid],rcond=None)[0]
        prediction = 1/np.maximum(scale*prediction+shift,1/maximum)
    estimate = np.clip(prediction,.001,maximum)
    p,t = estimate[valid],target[valid]
    return {'abs_rel':float(np.mean(abs(p-t)/t)), 'rmse_m':float(np.sqrt(np.mean((p-t)**2))),
            'delta1':float(np.mean(np.maximum(p/t,t/p)<1.25)), 'valid_pixels':int(valid.sum())},estimate,valid


def evaluate_model(root, spec, samples, device='cpu', examples=3):
    row = {'model':spec['name'], 'description':spec['description'], 'mode':spec['mode'], 'frames':len(samples), 'device':device}
    if not samples:
        return {**row,'status':'needs_depth_labels','reason':'Prepare data/depth/manifest.csv; see README.'},[]
    import torch
    import torch.nn.functional as F
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    visuals=[]
    try:
        cache=str(Path(root)/'data/models/huggingface')
        processor=AutoImageProcessor.from_pretrained(spec['repo'],cache_dir=cache)
        model=AutoModelForDepthEstimation.from_pretrained(spec['repo'],cache_dir=cache).to(device).eval()
        def predict(image):
            inputs={k:v.to(device) for k,v in processor(images=image,return_tensors='pt').items()}
            with torch.inference_mode(): output=model(**inputs).predicted_depth
            return F.interpolate(output[:,None],size=image.size[::-1],mode='bicubic',align_corners=False)[0,0].cpu().numpy()
        def sync():
            if device.startswith('cuda'): torch.cuda.synchronize()
        first=Image.open(samples[0]['image']).convert('RGB')
        for _ in range(3): predict(first)
        scores,times=[],[]
        for index,sample in enumerate(samples):
            image=Image.open(sample['image']).convert('RGB')
            target=read_target(sample)
            sync(); start=time.perf_counter(); prediction=predict(image); sync()
            times.append((time.perf_counter()-start)*1000)
            score,estimate,valid=metrics(prediction,target,spec['mode'])
            scores.append(score)
            if index<examples: visuals.append((sample,target,estimate,valid))
        row.update(status='ok',**{key:float(np.mean([s[key] for s in scores])) for key in ('abs_rel','rmse_m','delta1')},
                   valid_pixels=sum(s['valid_pixels'] for s in scores),**latency_summary(times))
        del model
    except Exception as error:
        row.update(status='error',reason=f'{type(error).__name__}: {error}')
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return row,visuals


def show_examples(spec, examples):
    import matplotlib.pyplot as plt
    for sample,target,prediction,valid in examples:
        fig,axes=plt.subplots(1,4,figsize=(16,4))
        axes[0].imshow(Image.open(sample['image'])); axes[0].set_title('Camera image')
        im=axes[1].imshow(np.ma.masked_where(~valid,target),vmin=0,vmax=80,cmap='magma')
        axes[1].set_title('Depth labels (m)')
        axes[2].imshow(prediction,vmin=0,vmax=80,cmap='magma')
        axes[2].set_title('Oracle-aligned MiDaS' if spec['mode']=='relative' else spec['name'])
        error=axes[3].imshow(np.ma.masked_where(~valid,abs(target-prediction)),vmin=0,vmax=20,cmap='inferno')
        axes[3].set_title('Error at labeled pixels (m)')
        for axis in axes: axis.axis('off')
        fig.colorbar(im,ax=list(axes[1:3]),label='Depth (m)',shrink=.7)
        fig.colorbar(error,ax=axes[3],label='Absolute error (m)',shrink=.7)
        fig.suptitle(sample['id']); plt.show()
