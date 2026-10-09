"""Validate metric edge cases and execute the three primary notebooks."""
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'notebooks/_support'))
import numpy as np
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
import occupancy
import detection
import depth


def check_metrics():
    # Waymo has a different taxonomy and raw free-space value. Keep it isolated.
    waymo_target=np.array([0,15,15],dtype=np.uint8)
    waymo_counts=occupancy.confusion_counts(waymo_target,waymo_target,np.ones(3,bool),'occ3d_waymo')
    assert waymo_counts.shape==(16,16)
    assert occupancy.summarize_counts(waymo_counts)['free_space_recall']==1
    assert occupancy.summarize_counts(waymo_counts)['mIoU']==1
    try: occupancy.confusion_counts(waymo_target,np.array([0,23,15]),np.ones(3,bool),'occ3d_waymo')
    except ValueError: pass
    else: raise AssertionError('Unnormalized Waymo predictions accepted')
    assert '/occ3d_waymo/' in occupancy.prediction_path(ROOT,{'slug':'flashocc'}, {'id':'000/000','dataset':'occ3d_waymo'}).as_posix()
    # Invisible voxels must contribute nothing, including to free-space counts.
    target=np.array([0,17,17],dtype=np.int64)
    pred=np.array([17,17,0],dtype=np.int64)
    scores=occupancy.summarize_counts(occupancy.confusion_counts(target,pred,np.array([True,True,False])))
    assert scores['occupied_recall']==0 and scores['free_space_recall']==1
    assert scores['mIoU']==0
    assert occupancy.summarize_counts(occupancy.confusion_counts(target,target,np.ones(3,bool)))['mIoU']==1
    try: occupancy.confusion_counts(target,np.array([18,0,0]),np.ones(3,bool))
    except ValueError: pass
    else: raise AssertionError('Invalid class IDs accepted')
    t={'boxes':np.array([[0,0,10,10]]),'labels':np.array([0])}
    p={**t,'scores':np.array([.9])}
    assert detection.score_predictions([p],[t])['mAP50_95']==1
    wrong={**p,'boxes':np.array([[20,20,30,30]])}
    assert detection.score_predictions([wrong],[t])['mAP50']==0
    truth=np.arange(1,17,dtype=float).reshape(4,4)
    scores,_,_=depth.metrics(truth,truth)
    assert scores['rmse_m']==0 and scores['delta1']==1
    scores,_,_=depth.metrics(2/truth+3,truth,'relative')
    assert scores['rmse_m']<1e-8
    try: depth.metrics(np.full((4,4),np.nan),truth)
    except ValueError: pass
    else: raise AssertionError('Invalid depth silently excluded')
    print('Metric checks passed.',flush=True)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    check_metrics()
    kernels=ROOT/'.runtime/kernels'
    (ROOT/'.runtime/ipython').mkdir(parents=True,exist_ok=True)
    kernel=kernels/'project'
    kernel.mkdir(parents=True,exist_ok=True)
    (kernel/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','--ipython-dir='+str(ROOT/'.runtime/ipython'),'-f','{connection_file}'],
                                               'display_name':'Driving notebooks','language':'python'}))
    names=sys.argv[1:] or ['occupancy_evaluation','monocular_depth_evaluation','model_evaluation_basics']
    for name in names:
        path=ROOT/'notebooks'/f'{name}.ipynb'
        nb=nbformat.read(path,as_version=4)
        nbformat.validate(nb)
        manager=KernelManager(kernel_name='project',kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernels)]))
        print('Executing',name,flush=True)
        try:
            NotebookClient(nb,km=manager,timeout=1800,resources={'metadata':{'path':str(ROOT/'notebooks')}}).execute()
        finally:
            nbformat.write(nb,path)
        for cell in nb.cells:
            for output in cell.get('outputs',[]):
                if output.output_type=='stream': print(output.text[:1500],flush=True)
        print('Completed',name,flush=True)
