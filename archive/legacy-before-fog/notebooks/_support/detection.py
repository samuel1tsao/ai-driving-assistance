"""One BDD100K label policy and one evaluation loop for every detector."""
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
from common import select_samples, latency_summary

# Both BDD person categories map to COCO person. Traffic signs have no COCO counterpart.
CLASSES = ['person', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle', 'traffic light']
ALIASES = {'pedestrian': 'person', 'rider': 'person'}
MODELS = [
    {'name':'YOLOv8n','checkpoint':'yolov8n.pt','family':'yolo','description':'Small convolutional detector; older lightweight baseline.'},
    {'name':'YOLO11n','checkpoint':'yolo11n.pt','family':'yolo','description':'Small YOLO detector for a lightweight comparison.'},
    {'name':'YOLO26n','checkpoint':'yolo26n.pt','family':'yolo','description':'Compact YOLO26 model.'},
    {'name':'YOLO26s','checkpoint':'yolo26s.pt','family':'yolo','description':'Larger YOLO26 model; accuracy/speed tradeoff.'},
    {'name':'RT-DETR-L','checkpoint':'rtdetr-l.pt','family':'rtdetr','description':'Transformer detector with end-to-end box prediction.'},
    {'name':'Faster R-CNN','family':'fasterrcnn','description':'Two-stage proposal and classification detector.'},
    {'name':'RetinaNet','family':'retinanet','description':'One-stage detector trained using focal loss.'},
    {'name':'SSD300','family':'ssd','description':'Fixed 300-pixel single-shot detector; lower-resolution baseline.'},
]


def load_samples(root, limit=100, seed=42):
    root = Path(root)
    label = root / 'bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json'
    image_dir = root / 'datasets/bdd100k/images/val'
    entries = json.loads(label.read_text())
    samples = []
    for entry in entries:
        boxes, labels = [], []
        for item in entry.get('labels', []):
            name = ALIASES.get(item['category'], item['category'])
            if name in CLASSES and 'box2d' in item:
                box = item['box2d']
                boxes.append([box[k] for k in ('x1','y1','x2','y2')])
                labels.append(CLASSES.index(name))
        samples.append({'id':entry['name'], 'image':image_dir / entry['name'],
                        'target':{'boxes':np.asarray(boxes,dtype=np.float32).reshape(-1,4),
                                  'labels':np.asarray(labels,dtype=np.int64)}})
    chosen = select_samples(samples, limit, seed)
    missing = [s['image'] for s in chosen if not s['image'].is_file()]
    if missing:
        raise FileNotFoundError(f'{len(missing)} selected BDD images missing; first: {missing[0]}')
    print(f'{len(entries):,} validation records; comparing every model on the same {len(chosen)} images.')
    return chosen


def load_model(root, spec, device):
    import torch
    if spec['family'] in ('yolo','rtdetr'):
        from ultralytics import YOLO, RTDETR
        cls = RTDETR if spec['family'] == 'rtdetr' else YOLO
        model = cls(str(Path(root) / spec['checkpoint']))
        if spec['family'] == 'rtdetr':
            model.model.model[-1].dynamic = True
        return {'model':model, 'names':model.names, 'family':spec['family'], 'device':device}
    from torchvision.models import detection as tv
    choices = {'fasterrcnn':('fasterrcnn_resnet50_fpn','FasterRCNN_ResNet50_FPN_Weights'),
               'retinanet':('retinanet_resnet50_fpn','RetinaNet_ResNet50_FPN_Weights'),
               'ssd':('ssd300_vgg16','SSD300_VGG16_Weights')}
    constructor, enum = choices[spec['family']]
    weights = getattr(tv,enum).DEFAULT
    # Explicit checkpoint directory; no environment variables are changed.
    state = torch.hub.load_state_dict_from_url(weights.url, model_dir=str(Path(root)/'data/models/torchvision'), progress=True)
    kwargs = {'weights':None, 'weights_backbone':None, 'num_classes':len(weights.meta['categories'])}
    if spec['family'] != 'ssd':
        kwargs.update(min_size=640, max_size=640)
    model = getattr(tv,constructor)(**kwargs)
    model.load_state_dict(state)
    return {'model':model.to(device).eval(), 'names':weights.meta['categories'], 'family':spec['family'], 'device':device}


def predict(bundle, image, size=640):
    import torch
    from torchvision.transforms.functional import pil_to_tensor
    if bundle['family'] in ('yolo','rtdetr'):
        out = bundle['model'].predict(image, imgsz=size, device=bundle['device'], conf=.001, max_det=300, verbose=False, save=False)[0]
        boxes, labels, scores = out.boxes.xyxy.cpu().numpy(), out.boxes.cls.cpu().numpy().astype(int), out.boxes.conf.cpu().numpy()
    else:
        tensor = pil_to_tensor(image).float().div(255).to(bundle['device'])
        with torch.inference_mode():
            out = bundle['model']([tensor])[0]
        boxes, labels, scores = (out[k].cpu().numpy() for k in ('boxes','labels','scores'))
    mapped = np.array([CLASSES.index(bundle['names'][int(i)]) if bundle['names'][int(i)] in CLASSES else -1 for i in labels])
    keep = (mapped >= 0) & (scores >= .001)
    order = np.argsort(-scores[keep])[:300]
    return {'boxes':boxes[keep][order], 'labels':mapped[keep][order], 'scores':scores[keep][order]}


def iou(box, boxes):
    area = np.maximum(0, np.minimum(box[2:],boxes[:,2:])-np.maximum(box[:2],boxes[:,:2])).prod(1)
    union = np.maximum(0,box[2:]-box[:2]).prod()+np.maximum(0,boxes[:,2:]-boxes[:,:2]).prod(1)-area
    return area / np.maximum(union,1e-12)


def score_predictions(predictions, targets):
    aps, operating = [], []
    for threshold in np.arange(.5,.96,.05):
        class_aps = []
        for cls in range(len(CLASSES)):
            truth = [t['boxes'][t['labels']==cls] for t in targets]
            total = sum(map(len,truth))
            if not total:
                if threshold == .5:
                    false_positives = sum(int(np.sum((p['labels']==cls) & (p['scores']>=.25))) for p in predictions)
                    operating.append((0,false_positives,0))
                continue
            candidates = sorted([(float(s),i,b) for i,p in enumerate(predictions)
                                 for b,c,s in zip(p['boxes'],p['labels'],p['scores']) if c==cls], key=lambda x:-x[0])
            used = [np.zeros(len(t),bool) for t in truth]
            tp, fp = [], []
            for confidence, index, box in candidates:
                overlaps = iou(box,truth[index]) if len(truth[index]) else np.array([])
                overlaps[used[index]] = -1
                match = int(overlaps.argmax()) if len(overlaps) else -1
                hit = match >= 0 and overlaps[match] >= threshold
                if hit:
                    used[index][match] = True
                tp.append(int(hit)); fp.append(int(not hit))
            cum_tp, cum_fp = np.cumsum(tp), np.cumsum(fp)
            recall = cum_tp / total
            precision = cum_tp / np.maximum(cum_tp+cum_fp,1)
            class_aps.append(np.mean([precision[recall>=r].max() if np.any(recall>=r) else 0 for r in np.linspace(0,1,101)]))
            if threshold == .5:
                keep = np.array([c[0]>=.25 for c in candidates])
                operating.append((int(np.array(tp)[keep].sum()),int(np.array(fp)[keep].sum()),total))
        aps.append(float(np.mean(class_aps)) if class_aps else np.nan)
    tp,fp,total = np.sum(operating,axis=0) if operating else (0,0,0)
    return {'mAP50':aps[0], 'mAP50_95':float(np.mean(aps)),
            'precision_at_025':float(tp/(tp+fp)) if tp+fp else 0,
            'recall_at_025':float(tp/total) if total else np.nan}


def evaluate_model(root, spec, samples, device='cpu', examples=3):
    import torch
    row = {'model':spec['name'], 'description':spec['description'], 'frames':len(samples), 'device':device}
    visuals = []
    try:
        bundle = load_model(root,spec,device)
        def sync():
            if device.startswith('cuda'): torch.cuda.synchronize()
        if not samples: raise ValueError('No labeled images selected.')
        first = Image.open(samples[0]['image']).convert('RGB')
        for _ in range(3): predict(bundle,first)
        predictions, times = [], []
        for index,sample in enumerate(samples):
            image = Image.open(sample['image']).convert('RGB')
            sync(); start = time.perf_counter()
            prediction = predict(bundle,image)
            sync(); times.append((time.perf_counter()-start)*1000)
            predictions.append(prediction)
            if index < examples: visuals.append((sample,prediction))
        row.update(status='ok', **score_predictions(predictions,[s['target'] for s in samples]), **latency_summary(times))
        del bundle
    except Exception as error:
        row.update(status='error',reason=f'{type(error).__name__}: {error}')
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return row, visuals


def show_examples(name, examples):
    import matplotlib.pyplot as plt
    for sample,prediction in examples:
        image = Image.open(sample['image']).convert('RGB')
        fig,axes = plt.subplots(1,2,figsize=(14,4))
        for axis,labels,title,color in zip(axes,[sample['target'],prediction],['Ground truth',name],['cyan','lime']):
            axis.imshow(image)
            for i,(box,cls) in enumerate(zip(labels['boxes'],labels['labels'])):
                if 'scores' in labels and labels['scores'][i]<.25: continue
                x1,y1,x2,y2=box
                axis.add_patch(plt.Rectangle((x1,y1),x2-x1,y2-y1,fill=False,edgecolor=color))
                axis.text(x1,y1,CLASSES[cls],fontsize=8,backgroundcolor=color)
            axis.set_title(title); axis.axis('off')
        fig.suptitle(sample['id']); fig.tight_layout(); plt.show()
