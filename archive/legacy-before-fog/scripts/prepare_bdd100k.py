import os
import json
import zipfile
import shutil
from pathlib import Path

# Paths
ROOT_DIR = Path(r"C:\Users\tsao1\OneDrive\Documents\workspace\Ai Driving Assistance app")
DATASETS_DIR = ROOT_DIR / "datasets"
RAW_DIR = DATASETS_DIR / "bdd100k_raw"
FINAL_DIR = DATASETS_DIR / "bdd100k"
ZIP_PATH = Path(r"C:\Users\tsao1\Downloads\archive.zip")

# Class mapping (from configs/bdd100k.local.yaml)
CLASS_MAP = {
    "pedestrian": 0,
    "rider": 1,
    "car": 2,
    "truck": 3,
    "bus": 4,
    "train": 5,
    "motorcycle": 6,
    "bicycle": 7,
    "traffic light": 8,
    "traffic sign": 9
}

# Image dimensions for normalization
IMG_WIDTH = 1280.0
IMG_HEIGHT = 720.0

def extract_archive():
    print(f"Extracting {ZIP_PATH} to {RAW_DIR}...")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(RAW_DIR)
    print("Extraction complete.")

def convert_labels(json_path, output_dir):
    print(f"Converting labels from {json_path} to {output_dir}...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    for item in data:
        img_name = item['name']
        txt_name = img_name.replace('.jpg', '.txt')
        txt_path = output_dir / txt_name
        
        lines = []
        if 'labels' in item:
            for label in item['labels']:
                category = label.get('category')
                if category not in CLASS_MAP:
                    continue
                class_id = CLASS_MAP[category]
                
                if 'box2d' not in label:
                    continue
                    
                box = label['box2d']
                x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
                
                # YOLO format: class x_center y_center width height (normalized)
                x_center = ((x1 + x2) / 2) / IMG_WIDTH
                y_center = ((y1 + y2) / 2) / IMG_HEIGHT
                w = (x2 - x1) / IMG_WIDTH
                h = (y2 - y1) / IMG_HEIGHT
                
                # Ensure values are within [0, 1]
                x_center = max(0.0, min(1.0, x_center))
                y_center = max(0.0, min(1.0, y_center))
                w = max(0.0, min(1.0, w))
                h = max(0.0, min(1.0, h))
                
                lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
                
        if lines:
            with open(txt_path, 'w', encoding='utf-8') as out_f:
                out_f.write("\n".join(lines) + "\n")

def process_dataset():
    # 1. Extract
    extract_archive()
    
    # 2. Setup final directories
    print("Setting up YOLO directories...")
    images_train_dir = FINAL_DIR / "images" / "train"
    images_val_dir = FINAL_DIR / "images" / "val"
    labels_train_dir = FINAL_DIR / "labels" / "train"
    labels_val_dir = FINAL_DIR / "labels" / "val"
    
    images_train_dir.mkdir(parents=True, exist_ok=True)
    images_val_dir.mkdir(parents=True, exist_ok=True)
    
    # 3. Convert labels
    train_json = RAW_DIR / "bdd100k_labels_release" / "bdd100k" / "labels" / "bdd100k_labels_images_train.json"
    val_json = RAW_DIR / "bdd100k_labels_release" / "bdd100k" / "labels" / "bdd100k_labels_images_val.json"
    
    convert_labels(train_json, labels_train_dir)
    convert_labels(val_json, labels_val_dir)
    
    # 4. Move images
    print("Moving images...")
    raw_images_train = RAW_DIR / "bdd100k" / "bdd100k" / "images" / "100k" / "train"
    raw_images_val = RAW_DIR / "bdd100k" / "bdd100k" / "images" / "100k" / "val"
    
    # Use shutil.move for fast moving (same drive)
    for img in raw_images_train.glob("*.jpg"):
        shutil.move(str(img), str(images_train_dir / img.name))
        
    for img in raw_images_val.glob("*.jpg"):
        shutil.move(str(img), str(images_val_dir / img.name))
        
    print("Images moved.")
    
    # 5. Cleanup
    print("Cleaning up raw extraction directory...")
    shutil.rmtree(RAW_DIR, ignore_errors=True)
    
    print("BDD100K Dataset Preparation Complete!")

if __name__ == "__main__":
    process_dataset()
