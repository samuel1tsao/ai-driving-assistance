import os
import zipfile
import tarfile
from pathlib import Path
import shutil

ROOT_DIR = Path(r"C:\Users\tsao1\OneDrive\Documents\workspace\Ai Driving Assistance app")
DATASETS_DIR = ROOT_DIR / "data" / "occupancy" / "datasets"
OCC3D_DIR = DATASETS_DIR / "occ3d_waymo"
RAW_DIR = DATASETS_DIR / "occ3d_waymo_raw"
DOWNLOADS_DIR = Path(r"C:\Users\tsao1\Downloads")

def process_occ3d():
    print("Finding zip batches in Downloads...")
    zips = list(DOWNLOADS_DIR.glob("drive-download-*.zip"))
    
    if not zips:
        print("No drive-download-*.zip files found!")
        return

    print(f"Found {len(zips)} zip batches. Extracting to staging area...")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    
    for z in zips:
        print(f"Extracting {z.name}...")
        with zipfile.ZipFile(z, 'r') as zip_ref:
            zip_ref.extractall(RAW_DIR)
            
    print("Zip extraction complete. Now untarring archives to final destination...")
    
    # Extract training tars
    training_raw = RAW_DIR / "voxel01" / "training"
    if training_raw.exists():
        train_dest = OCC3D_DIR / "training"
        train_dest.mkdir(parents=True, exist_ok=True)
        for tar_path in training_raw.glob("*.tar"):
            print(f"Untarring {tar_path.name}...")
            with tarfile.open(tar_path, 'r') as tar:
                # Need to use 'data_filter' to avoid DeprecationWarning and be secure if possible,
                # but we'll use simple extractall since it's a known local file
                tar.extractall(path=train_dest)
                
    # Extract validation tars
    validation_raw = RAW_DIR / "voxel01" / "validation"
    if validation_raw.exists():
        val_dest = OCC3D_DIR / "validation"
        val_dest.mkdir(parents=True, exist_ok=True)
        for tar_path in validation_raw.glob("*.tar"):
            print(f"Untarring {tar_path.name}...")
            with tarfile.open(tar_path, 'r') as tar:
                tar.extractall(path=val_dest)

    print("Cleaning up staging area...")
    shutil.rmtree(RAW_DIR, ignore_errors=True)
    
    print("Occ3D Dataset Preparation Complete!")

if __name__ == "__main__":
    process_occ3d()
